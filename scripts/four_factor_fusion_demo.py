#!/usr/bin/env python3
"""Four-factor fusion: star + VO + Skyline + TRN, with complementary cliffs.

Phase 6 of Skyline Lock. Phase 5 fused the star tracker (attitude), VO (relative
motion) and Skyline (absolute horizon position) into one honest pose graph, each
trusted only as far as its confidence earned. This adds the project's fourth
localization modality -- terrain-relative navigation (TRN) -- as a second
*absolute position* factor, and the point is NOT "more sensors -> better." It is
that Skyline and TRN have **complementary failure modes**, so their union stays
localized where either one alone collapses:

  - Skyline reads the FAR horizon. It aliases where the terrain is rotationally
    symmetric -- e.g. anywhere on Tycho's circular rim the 360-deg skyline looks
    nearly identical, so equal-radius positions are indistinguishable.
  - TRN reads the GROUND directly below: it matches a nadir image patch (here the
    LROC WAC ortho) against the orbital map. It locks wherever the surface is
    texture-rich and locally distinctive -- the very rim/ejecta that aliases the
    skyline -- and instead starves on smooth, featureless mare.

Over Tycho a radial traverse drives out of the distinctive interior into the
self-similar exterior: the skyline locks at the start, then aliases by tens of
kilometres; TRN holds a tight lock the whole way. Fusing both keeps the rover
localized across the entire traverse.

Honest envelope (all visible / reproducible from public data):
  - The two absolute factors come from DIFFERENT real datasets at DIFFERENT
    scales: Skyline from the LOLA LDEM elevation profile (far field, coarse is
    fine), TRN from the LROC WAC ortho image (near field, needs resolution).
  - Each factor's information is set by its OWN real uniqueness margin (a 1-D
    horizon-profile margin for Skyline, a 2-D image template margin for TRN), so
    an aliased fix is down-weighted automatically, not trusted.
  - The converse cliff is real, but NOT the tidy mirror image this once claimed.
    Running this same stack over the Apollo 11 mare (scripts/converse_cliff_demo.py)
    shows the opposite of the intuition that "TRN starves on mare": with the real
    LROC WAC ortho the mare's albedo speckle is texture-rich enough that this NCC
    matcher locks 20/20, while the FLAT mare horizon leaves Skyline almost nothing
    to lock onto (2/15 unique vs 7/15 over Tycho). So the honest version is
    asymmetric -- Skyline is a relief-dependent cue, TRN a texture-dependent one,
    and the Moon feeds them unequally per terrain. Fusion wins not because the
    cliffs are symmetric but because each factor is margin-weighted, so whichever
    cue the terrain starves is discounted automatically -- no sensor is globally
    best, and no terrain classifier is in the loop.

Reuses scripts/factor_graph_fusion_demo.py for the scenario, the Skyline matcher
and the linear pose-graph solver (positions decouple per axis given the absolute
star attitude, so the TRN unary factor slots into the same solver unchanged).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from skyline_lock_demo import build_dem  # noqa: E402
from factor_graph_fusion_demo import (  # noqa: E402
    make_truth_trajectory, true_yaw_series, simulate_vo, integrate,
    SkylineLocalizer, margin_to_sigma, fuse_positions,
)
from predictive_localizability import (  # noqa: E402
    fuse_positions_directional,
    normalized_information,
    skyline_observability,
)


# --------------------------------------------------------------------------- #
# Appearance map (what the nadir TRN camera matches against)                  #
# --------------------------------------------------------------------------- #
def hillshade(dem: np.ndarray, px_to_m: float, *, az_deg=315.0, alt_deg=45.0) -> np.ndarray:
    """Shaded-relief of the DEM in [0, 1] -- a nadir-appearance proxy for synth.

    Used only for the synthetic source. For real terrain we match against the
    actual LROC WAC ortho image (an independent sensor), which is both more
    honest and far richer than a coarse-DEM hillshade.
    """
    gy, gx = np.gradient(dem.astype(np.float64), px_to_m)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)  # x east, y north
    az, alt = math.radians(az_deg), math.radians(alt_deg)
    hs = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    return np.clip(hs, 0.0, 1.0).astype(np.float32)


def build_appearance_map(args, dem, px_to_m):
    """Return (appearance_map float32, app_px_to_m, kind_str).

    The appearance map lives in the SAME world metric frame as the DEM and the
    trajectory: row 0 = y 0 (north-up via origin="lower"), col 0 = x 0; the box
    spans [0, extent_m] on both axes. So a TRN fix in appearance pixels maps to
    world metres directly.
    """
    extent_m = dem.shape[0] * px_to_m
    if args.source != "lola":
        up = max(1, args.trn_upsample)
        size = dem.shape[0]
        app = cv2.resize(hillshade(dem, px_to_m), (size * up, size * up),
                         interpolation=cv2.INTER_LINEAR)
        return _standardize(app), extent_m / app.shape[0], f"hillshade x{up}"

    # Real: co-register the LROC WAC ortho to the LOLA DEM's geographic box.
    from lro_trn_demo import TARGETS, fetch_wac_mosaic
    if args.target not in TARGETS:
        raise SystemExit(f"unknown target '{args.target}'")
    lat, lon = TARGETS[args.target]
    # The DEM square (scripts/skyline_lock_demo.load_lola_dem) keeps the full
    # +/-half_width longitude span and a cos(lat)-shrunk, centred latitude span.
    lat_half = args.half_width_deg * math.cos(math.radians(lat))
    dem_lat = (lat - lat_half, lat + lat_half)
    dem_lon = (lon - args.half_width_deg, lon + args.half_width_deg)

    mosaic, wlat0, wlat1, wlon0, wlon1 = fetch_wac_mosaic(
        lat, lon, zoom=args.trn_wac_zoom, tile_radius=args.trn_wac_tile_radius,
        cache_dir=args.cache_dir)
    # The WAC tile window can fall a hair short of the DEM box on one edge (tile
    # rounding). A small overhang is clamped by BORDER_REPLICATE below and only
    # touches a thin edge strip the traverse never visits; a large shortfall
    # means the mosaic is genuinely too small, so error and ask for more tiles.
    lat_span, lon_span = dem_lat[1] - dem_lat[0], dem_lon[1] - dem_lon[0]
    short = max((wlat0 - dem_lat[0]) / lat_span, (dem_lat[1] - wlat1) / lat_span,
                (wlon0 - dem_lon[0]) / lon_span, (dem_lon[1] - wlon1) / lon_span)
    if short > 0.05:
        raise SystemExit(
            "WAC mosaic does not cover the DEM box; raise --trn-wac-tile-radius "
            f"(wac lat[{wlat0:.2f},{wlat1:.2f}] lon[{wlon0:.2f},{wlon1:.2f}] vs "
            f"dem lat[{dem_lat[0]:.2f},{dem_lat[1]:.2f}] lon[{dem_lon[0]:.2f},{dem_lon[1]:.2f}])")
    if short > 0.0:
        print(f"  note: WAC short by {short*100:.1f}% on one edge; clamped (traverse avoids it)")

    sa = args.trn_app_px
    xs = np.linspace(dem_lon[0], dem_lon[1], sa)   # lon per column (x east)
    ys = np.linspace(dem_lat[0], dem_lat[1], sa)   # lat per row    (y north-up)
    lon_grid, lat_grid = np.meshgrid(xs, ys)
    mh, mw = mosaic.shape
    col = (lon_grid - wlon0) / (wlon1 - wlon0) * (mw - 1)
    row = (wlat1 - lat_grid) / (wlat1 - wlat0) * (mh - 1)  # lat_max at row 0
    app = cv2.remap(mosaic.astype(np.float32), col.astype(np.float32),
                    row.astype(np.float32), cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE)
    return _standardize(app), extent_m / sa, f"WAC z{args.trn_wac_zoom}"


def _standardize(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    return (img - img.mean()) / (img.std() + 1e-6)


# --------------------------------------------------------------------------- #
# TRN localizer: 2-D nadir-patch template match + real uniqueness margin      #
# --------------------------------------------------------------------------- #
class TrnLocalizer:
    """Match a nadir image patch against the orbital map (the image-domain
    analogue of SkylineLocalizer's horizon match over a position grid).

    The fix is the normalized cross-correlation peak over all positions; the
    uniqueness margin is the peak minus the best *competing* peak farther than
    one patch away -- small where the surface is self-similar / texture-poor
    (the fix can jump and is down-weighted), large where it is locally distinct.
    """

    def __init__(self, app: np.ndarray, app_px_to_m: float, *, patch_m: float):
        self.app = app
        self.app_px_to_m = app_px_to_m
        self.half = max(2, int(round(0.5 * patch_m / app_px_to_m)))
        self.patch_m = patch_m

    def _match(self, truth_xy, rng, noise_frac):
        cx = int(round(truth_xy[0] / self.app_px_to_m))
        cy = int(round(truth_xy[1] / self.app_px_to_m))
        h, w = self.app.shape
        r0, r1 = cy - self.half, cy + self.half
        c0, c1 = cx - self.half, cx + self.half
        # Clamp the patch window into the map (edge poses still get a fix).
        r0, r1 = max(0, r0), min(h, r1)
        c0, c1 = max(0, c0), min(w, c1)
        patch = self.app[r0:r1, c0:c1].copy()
        patch = patch + rng.normal(0.0, noise_frac, patch.shape).astype(np.float32)
        res = cv2.matchTemplate(self.app, patch, cv2.TM_CCOEFF_NORMED)
        th, tw = patch.shape
        pk = np.unravel_index(int(np.argmax(res)), res.shape)
        peak = float(res[pk])
        est_x = (pk[1] + tw / 2.0) * self.app_px_to_m
        est_y = (pk[0] + th / 2.0) * self.app_px_to_m
        # 2-D uniqueness margin: best score outside ~one patch of the peak.
        rr, cc = np.mgrid[0:res.shape[0], 0:res.shape[1]]
        dist = np.hypot((cc + tw / 2.0) - (pk[1] + tw / 2.0),
                        (rr + th / 2.0) - (pk[0] + th / 2.0)) * self.app_px_to_m
        far = res[dist > 1.5 * self.patch_m]
        second = float(far.max()) if far.size else peak
        return (
            np.array([est_x, est_y], dtype=np.float64),
            float(peak - second),
            peak,
            res,
            pk,
        )

    def fix(self, truth_xy, rng, noise_frac):
        """Return the legacy scalar-margin TRN fix tuple."""
        estimate, margin, peak, _, _ = self._match(truth_xy, rng, noise_frac)
        return estimate, margin, peak

    def fix_with_observability(self, truth_xy, rng, noise_frac, *,
                               score_sigma=0.01, position_scale_m=None):
        """Return ``(estimate, margin, peak, XY ObservabilityReport)``.

        Directional information comes from the raw 2-D match response before
        the factor graph sees the absolute fix.
        """
        from predictive_localizability import trn_observability

        estimate, margin, peak, response, peak_rc = self._match(
            truth_xy, rng, noise_frac
        )
        report = trn_observability(
            response,
            peak_rc,
            px_to_m=self.app_px_to_m,
            score_sigma=score_sigma,
            position_scale_m=position_scale_m or self.patch_m,
        )
        return estimate, margin, peak, report


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # DEM source (shared with skyline_lock_demo's build_dem).
    ap.add_argument("--source", choices=["synth", "lola"], default="lola")
    ap.add_argument("--terrain", choices=["hills", "craters", "flat"], default="craters")
    ap.add_argument("--target", default="tycho")
    ap.add_argument("--ldem-ppd", type=int, default=16, choices=[4, 16, 64])
    ap.add_argument("--half-width-deg", type=float, default=3.0)
    ap.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "datasets" / "lro_cache")
    ap.add_argument("--size-px", type=int, default=512)
    ap.add_argument("--px-to-m", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=7)
    # Trajectory + sensors.
    ap.add_argument("--trajectory", choices=["radial", "s_curve", "diag"], default="radial")
    ap.add_argument("--n-poses", type=int, default=60)
    ap.add_argument("--skyline-every", type=int, default=4)
    ap.add_argument("--trn-every", type=int, default=3)
    ap.add_argument("--mast-height-m", type=float, default=2.0)
    ap.add_argument("--n-az", type=int, default=180)
    ap.add_argument("--n-range", type=int, default=160)
    ap.add_argument("--grid", type=int, default=41)
    ap.add_argument("--grid-margin-frac", type=float, default=0.12)
    ap.add_argument("--noise-arcmin", type=float, default=2.0)
    ap.add_argument("--yaw-sigma-deg", type=float, default=2.0)
    # TRN appearance + matcher.
    ap.add_argument("--trn-app-px", type=int, default=480,
                    help="Side length (px) of the co-registered appearance map (lola).")
    ap.add_argument("--trn-upsample", type=int, default=4,
                    help="Hillshade upsample factor (synth appearance map).")
    ap.add_argument("--trn-wac-zoom", type=int, default=6)
    ap.add_argument("--trn-wac-tile-radius", type=int, default=1)
    ap.add_argument("--trn-patch-frac", type=float, default=0.09,
                    help="Nadir patch side as a fraction of the box extent.")
    ap.add_argument("--trn-noise-frac", type=float, default=0.05)
    # VO drift model.
    ap.add_argument("--vo-scale-bias", type=float, default=0.04)
    ap.add_argument("--vo-yaw-bias-deg", type=float, default=0.25)
    ap.add_argument("--vo-sigma-frac", type=float, default=0.06)
    # Fusion information model.
    ap.add_argument("--sigma-prior-m", type=float, default=20.0)
    ap.add_argument("--sigma-vo-m", type=float, default=120.0)
    ap.add_argument("--sigma-lock-m", type=float, default=400.0)
    ap.add_argument("--margin-ref", type=float, default=0.15)
    ap.add_argument("--margin-floor", type=float, default=0.01)
    ap.add_argument("--sigma-cap-m", type=float, default=40000.0)
    # TRN information model (margin scale differs: 2-D image NCC ~0.3-0.7).
    ap.add_argument("--trn-sigma-lock-m", type=float, default=150.0)
    ap.add_argument("--trn-margin-ref", type=float, default=0.4)
    ap.add_argument("--trn-margin-floor", type=float, default=0.02)
    ap.add_argument("--trn-unique-margin", type=float, default=0.1)
    ap.add_argument("--directional-information", action="store_true",
                    help="Also solve with raw-measurement XY information matrices "
                    "instead of scalar isotropic fix sigmas.")
    ap.add_argument("--output", type=Path,
                    default=REPO_ROOT / "outputs" / "factor_graph_fusion" / "four_factor_fusion.png")
    ap.add_argument("--output-json", type=Path, default=None)
    args = ap.parse_args()

    dem, px_to_m = build_dem(args)
    extent_m = dem.shape[0] * px_to_m
    scene = f"lola:{args.target}" if args.source == "lola" else f"synth:{args.terrain}"
    print(f"scene={scene}  extent={extent_m/1000:.1f} km  px_to_m={px_to_m:.1f}")

    rng = np.random.default_rng(args.seed)
    truth = make_truth_trajectory(extent_m, args.n_poses, args.trajectory)
    truth_yaw = true_yaw_series(truth)
    star_yaw = truth_yaw + rng.normal(0.0, math.radians(args.yaw_sigma_deg), size=truth_yaw.shape)
    vo_deltas = simulate_vo(truth, star_yaw, rng, scale_bias=args.vo_scale_bias,
                            yaw_bias_rad=math.radians(args.vo_yaw_bias_deg),
                            sigma_frac=args.vo_sigma_frac)
    vo_only = integrate(truth[0], vo_deltas)

    print("building appearance map + skyline candidate grid ...")
    app, app_px_to_m, app_kind = build_appearance_map(args, dem, px_to_m)
    print(f"  appearance: {app_kind}  {app.shape}  {app_px_to_m:.0f} m/px")
    sky = SkylineLocalizer(dem, px_to_m, grid=args.grid, margin_frac=args.grid_margin_frac,
                           n_az=args.n_az, n_range=args.n_range,
                           mast_height_m=args.mast_height_m, yaw_sigma_deg=args.yaw_sigma_deg)
    trn = TrnLocalizer(app, app_px_to_m, patch_m=args.trn_patch_frac * extent_m)

    # Skyline fixes (1-D horizon margin -> sigma).
    sky_fixes, sky_directional_fixes, sky_records = [], [], []
    for k in range(0, args.n_poses, args.skyline_every):
        est_xy, margin, ncc = sky.fix(tuple(truth[k]), truth_yaw[k], star_yaw[k],
                                      rng, args.noise_arcmin)
        sigma = margin_to_sigma(margin, sigma_lock_m=args.sigma_lock_m,
                                margin_ref=args.margin_ref, margin_floor=args.margin_floor,
                                sigma_cap_m=args.sigma_cap_m)
        err = float(math.hypot(est_xy[0] - truth[k, 0], est_xy[1] - truth[k, 1]))
        sky_fixes.append((k, est_xy, sigma))
        if args.directional_information:
            _, obs = skyline_observability(
                sky._horizon, tuple(truth[k]), truth_yaw[k],
                position_delta_m=max(0.5 * sky.grid_step_m, px_to_m),
                yaw_delta_rad=math.radians(0.5),
                horizon_sigma_rad=math.radians(args.noise_arcmin / 60.0),
                position_scale_m=sky.grid_step_m,
                yaw_scale_rad=math.radians(args.yaw_sigma_deg),
            )
            info = normalized_information(
                obs, ("x", "y"), sigma_best=sigma,
                reliability=1.0, information_floor=1e-3,
            )
            sky_directional_fixes.append((k, est_xy, info))
        sky_records.append({"pose": k, "margin": round(margin, 4), "best_ncc": round(ncc, 4),
                            "sigma_m": round(sigma, 1), "fix_err_m": round(err, 1),
                            "unique": bool(margin >= 0.05)})

    # TRN fixes (2-D image margin -> sigma).
    trn_fixes, trn_directional_fixes, trn_records = [], [], []
    for k in range(0, args.n_poses, args.trn_every):
        if args.directional_information:
            est_xy, margin, peak, obs = trn.fix_with_observability(
                tuple(truth[k]), rng, args.trn_noise_frac,
                position_scale_m=args.trn_patch_frac * extent_m,
            )
        else:
            est_xy, margin, peak = trn.fix(
                tuple(truth[k]), rng, args.trn_noise_frac
            )
        sigma = margin_to_sigma(margin, sigma_lock_m=args.trn_sigma_lock_m,
                                margin_ref=args.trn_margin_ref,
                                margin_floor=args.trn_margin_floor, sigma_cap_m=args.sigma_cap_m)
        err = float(math.hypot(est_xy[0] - truth[k, 0], est_xy[1] - truth[k, 1]))
        trn_fixes.append((k, est_xy, sigma))
        if args.directional_information:
            info = normalized_information(
                obs, ("x", "y"), sigma_best=sigma,
                reliability=1.0, information_floor=1e-3,
            )
            trn_directional_fixes.append((k, est_xy, info))
        trn_records.append({"pose": k, "margin": round(margin, 4), "peak_ncc": round(peak, 4),
                           "sigma_m": round(sigma, 1), "fix_err_m": round(err, 1),
                           "unique": bool(margin >= args.trn_unique_margin)})

    # Three estimators sharing the same solver. The TRN unary factor uses the
    # exact same (k, xy, sigma) form as the skyline one, so adding it is just a
    # longer fix list -- no solver change.
    est3, cov3 = fuse_positions(args.n_poses, vo_deltas, truth[0],
                                args.sigma_prior_m, args.sigma_vo_m, sky_fixes)
    est4, cov4 = fuse_positions(args.n_poses, vo_deltas, truth[0],
                                args.sigma_prior_m, args.sigma_vo_m, sky_fixes + trn_fixes)
    est_directional = cov_directional = None
    if args.directional_information:
        est_directional, cov_directional = fuse_positions_directional(
            args.n_poses,
            vo_deltas,
            truth[0],
            sigma_prior=args.sigma_prior_m,
            sigma_vo=args.sigma_vo_m,
            fixes=sky_directional_fixes + trn_directional_fixes,
        )

    def rmse(a):
        return float(np.sqrt(np.mean(np.sum((a - truth) ** 2, axis=1))))
    vo_err = np.linalg.norm(vo_only - truth, axis=1)
    err3 = np.linalg.norm(est3 - truth, axis=1)
    err4 = np.linalg.norm(est4 - truth, axis=1)
    err_directional = (
        np.linalg.norm(est_directional - truth, axis=1)
        if est_directional is not None else None
    )
    pos_std4 = np.sqrt(cov4[:, 0, 0] + cov4[:, 1, 1])

    summary = {
        "scene": scene,
        "appearance": app_kind,
        "n_poses": args.n_poses,
        "n_skyline_fixes": len(sky_fixes),
        "n_skyline_unique": sum(1 for r in sky_records if r["unique"]),
        "n_trn_fixes": len(trn_fixes),
        "n_trn_unique": sum(1 for r in trn_records if r["unique"]),
        "vo_only_rmse_m": round(rmse(vo_only), 1),
        "fused3_star_vo_skyline_rmse_m": round(rmse(est3), 1),
        "fused4_plus_trn_rmse_m": round(rmse(est4), 1),
        "vo_only_final_err_m": round(float(vo_err[-1]), 1),
        "fused3_final_err_m": round(float(err3[-1]), 1),
        "fused4_final_err_m": round(float(err4[-1]), 1),
        "fused4_max_err_m": round(float(err4.max()), 1),
        "fused4_vs_vo_x": round(rmse(vo_only) / max(rmse(est4), 1e-9), 2),
        "fused4_vs_fused3_x": round(rmse(est3) / max(rmse(est4), 1e-9), 2),
        "directional_information_enabled": args.directional_information,
        "skyline_fixes": sky_records,
        "trn_fixes": trn_records,
    }
    if est_directional is not None:
        summary.update({
            "fused4_directional_rmse_m": round(rmse(est_directional), 1),
            "fused4_directional_final_err_m":
                round(float(err_directional[-1]), 1),
            "directional_vs_scalar_rmse_x":
                round(rmse(est4) / max(rmse(est_directional), 1e-9), 3),
        })
    print(json.dumps(summary, indent=2))

    _render(args, dem, app, app_px_to_m, extent_m, truth, vo_only, est3, est4,
            cov4, pos_std4, sky_fixes, sky_records, trn_fixes, trn_records,
            vo_err, err3, err4, scene, app_kind,
            est_directional=est_directional,
            err_directional=err_directional,
            cov_directional=cov_directional)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2))
        print(f"wrote {args.output_json}")
    return 0


def _render(args, dem, app, app_px_to_m, extent_m, truth, vo_only, est3, est4,
            cov4, pos_std4, sky_fixes, sky_records, trn_fixes, trn_records,
            vo_err, err3, err4, scene, app_kind, *,
            est_directional=None, err_directional=None, cov_directional=None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    SKY_U, SKY_A = "#19c819", "#ff8c00"   # skyline unique / aliased
    TRN_U, TRN_A = "#1f77ff", "#9aa0a6"   # trn unique / starved

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))
    fig.suptitle(
        f"Four-factor fusion (star + VO + Skyline + TRN) over {scene}  -  "
        "Skyline and TRN cover each other's localizability cliffs",
        fontsize=13)

    # Panel 1: appearance map (what TRN matches) + trajectories + fixes.
    ax = axes[0]
    im = ax.imshow(app, origin="lower", extent=[0, extent_m, 0, extent_m], cmap="gray")
    ax.plot(truth[:, 0], truth[:, 1], "-", color="white", lw=2.2, label="ground truth")
    ax.plot(vo_only[:, 0], vo_only[:, 1], "--", color="red", lw=1.6, label="VO only")
    ax.plot(est3[:, 0], est3[:, 1], "-", color="#ffd24d", lw=1.6, label="fused (no TRN)")
    ax.plot(est4[:, 0], est4[:, 1], "-", color="cyan", lw=2.0, label="fused + TRN")
    if est_directional is not None:
        ax.plot(est_directional[:, 0], est_directional[:, 1], "-",
                color="#ff66cc", lw=1.8, label="directional information")
    for k in range(0, len(est4), max(1, len(est4) // 12)):
        sx = 2.0 * math.sqrt(max(cov4[k, 0, 0], 0.0))
        sy = 2.0 * math.sqrt(max(cov4[k, 1, 1], 0.0))
        ax.add_patch(Ellipse((est4[k, 0], est4[k, 1]), 2 * sx, 2 * sy,
                             fill=False, color="cyan", lw=0.7, alpha=0.6))
    for (k, fxy, _), rec in zip(sky_fixes, sky_records):
        ax.scatter(*fxy, c=(SKY_U if rec["unique"] else SKY_A), marker="P",
                   s=64, edgecolors="k", lw=0.5, zorder=6)
    for (k, fxy, _), rec in zip(trn_fixes, trn_records):
        ax.scatter(*fxy, c=(TRN_U if rec["unique"] else TRN_A), marker="o",
                   s=26, edgecolors="k", lw=0.4, zorder=5)
    ax.scatter([], [], c=SKY_U, marker="P", s=64, edgecolors="k", label="skyline (locks)")
    ax.scatter([], [], c=SKY_A, marker="P", s=64, edgecolors="k", label="skyline (aliased)")
    ax.scatter([], [], c=TRN_U, marker="o", s=26, edgecolors="k", label="TRN (locks)")
    ax.set_xlim(0, extent_m); ax.set_ylim(0, extent_m)
    ax.set_title(f"appearance map ({app_kind}) + trajectories + 2σ")
    ax.set_xlabel("x east (m)"); ax.set_ylabel("y north (m)")
    ax.legend(loc="upper left", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Panel 2: error vs pose for the three estimators.
    ax = axes[1]
    ax.plot(vo_err, "--", color="red", lw=1.8, label="VO only")
    ax.plot(err3, "-", color="#e0a800", lw=1.8, label="fused (star+VO+skyline)")
    ax.plot(err4, "-", color="cyan", lw=2.2, label="fused + TRN")
    if err_directional is not None:
        ax.plot(err_directional, "-", color="#ff66cc", lw=2.0,
                label="directional information")
    for (k, _, _), rec in zip(sky_fixes, sky_records):
        ax.axvline(k, color=(SKY_U if rec["unique"] else SKY_A), lw=0.8, alpha=0.35)
    ax.set_title("position error vs pose")
    ax.set_xlabel("pose index"); ax.set_ylabel("error (m)")
    ax.set_yscale("log")
    ax.legend(fontsize=8)

    # Panel 3: the money panel -- complementary margins.
    ax = axes[2]
    sk_k = [r["pose"] for r in sky_records]; sk_m = [r["margin"] for r in sky_records]
    tr_k = [r["pose"] for r in trn_records]; tr_m = [r["margin"] for r in trn_records]
    ax.plot(sk_k, sk_m, "-P", color="#137333", lw=1.4, ms=7, label="Skyline margin (horizon)")
    ax.plot(tr_k, tr_m, "-o", color="#1f77ff", lw=1.4, ms=5, label="TRN margin (nadir image)")
    ax.axhline(0.05, color="#137333", ls=":", lw=1.0, alpha=0.7)
    ax.axhline(args.trn_unique_margin, color="#1f77ff", ls=":", lw=1.0, alpha=0.7)
    ax.set_ylim(bottom=0.0)
    ax.set_title("uniqueness margins: where each modality locks vs aliases")
    ax.set_xlabel("pose index"); ax.set_ylabel("uniqueness margin")
    ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=120)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    sys.exit(main())
