#!/usr/bin/env python3
"""Predict Skyline alignment cliffs before localization over three lunar scenes.

For every candidate rover position this demo computes:

1. local (x, y, yaw) observability from finite differences of the raw predicted
   horizon, before any factor-graph solve;
2. the existing global uniqueness margin;
3. empirical localization error under noisy horizon observations; and
4. a combined risk that is high for either a weak local direction or a strong
   distant alias.

Large inputs and full-resolution outputs default to the external data roots
resolved by ``astro_data_paths.py``. Use ``--cache-dir``/``--output-dir`` to
override them.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from astro_data_paths import data_root, output_root  # noqa: E402
from predictive_localizability import skyline_observability  # noqa: E402
from lro_trn_demo import LUNAR_RADIUS_M, TARGETS, fetch_ldem  # noqa: E402
from skyline_lock_demo import best_yaw_ncc, render_horizon, uniqueness_margin  # noqa: E402


SCENES = (
    ("tycho", "Tycho highland"),
    ("apollo11", "Apollo 11 mare"),
    ("shackleton", "Shackleton polar"),
)


def load_tangent_dem(target: str, half_width_deg: float, ppd: int,
                     cache_dir: Path) -> tuple[np.ndarray, float]:
    """Resample cylindrical LOLA onto a local east/north tangent grid.

    Unlike the legacy rectangular crop, the spherical forward-geodesic mapping
    remains well-conditioned at Shackleton (89.66 S), where longitude pixels
    collapse and an equirectangular crop can become only one pixel wide.
    """
    lat0_deg, lon0_deg = TARGETS[target]
    source = fetch_ldem(ppd, cache_dir)
    px_to_m = math.radians(1.0 / ppd) * LUNAR_RADIUS_M
    half_m = math.radians(half_width_deg) * LUNAR_RADIUS_M
    size = max(8, int(round(2.0 * half_m / px_to_m)))
    east = np.linspace(-half_m, half_m, size, dtype=np.float64)
    north = np.linspace(-half_m, half_m, size, dtype=np.float64)
    xx, yy = np.meshgrid(east, north)
    distance = np.hypot(xx, yy) / LUNAR_RADIUS_M
    bearing = np.arctan2(xx, yy)
    lat0 = math.radians(lat0_deg)
    lon0 = math.radians(lon0_deg)
    lat = np.arcsin(
        math.sin(lat0) * np.cos(distance)
        + math.cos(lat0) * np.sin(distance) * np.cos(bearing)
    )
    lon = lon0 + np.arctan2(
        np.sin(bearing) * np.sin(distance) * math.cos(lat0),
        np.cos(distance) - math.sin(lat0) * np.sin(lat),
    )
    map_y = ((math.pi / 2.0 - lat) * 180.0 / math.pi * ppd).astype(np.float32)
    map_x = ((lon * 180.0 / math.pi) % 360.0 * ppd).astype(np.float32)
    patch = cv2.remap(
        source, map_x, map_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    ).astype(np.float32)
    return patch, float(2.0 * half_m / size)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def evaluate_scene(args, target: str, label: str, rng: np.random.Generator) -> dict:
    dem, px_to_m = load_tangent_dem(
        target, args.half_width_deg, args.ldem_ppd, args.cache_dir
    )
    extent_m = dem.shape[0] * px_to_m
    axis = np.linspace(args.grid_margin_frac * extent_m,
                       (1.0 - args.grid_margin_frac) * extent_m, args.grid)
    candidates = [(float(x), float(y)) for y in axis for x in axis]
    step_m = float(axis[1] - axis[0]) if args.grid > 1 else extent_m
    r_min = max(60.0, 2.0 * px_to_m)
    r_max = 0.9 * extent_m

    def horizon(xy):
        return render_horizon(
            dem, px_to_m, xy, args.mast_height_m,
            n_az=args.n_az, r_min_m=r_min, r_max_m=r_max,
            n_range=args.n_range,
        )

    predictions = np.stack([horizon(xy) for xy in candidates])
    window_bins = max(
        1, int(math.ceil(3.0 * args.yaw_sigma_deg * args.n_az / 360.0))
    )
    records = []
    for i, xy in enumerate(candidates):
        scores, _ = best_yaw_ncc(
            predictions[i], predictions, prior_lag=0, window_bins=window_bins
        )
        margin, second = uniqueness_margin(
            scores, candidates, xy, radius_m=1.5 * step_m
        )
        _, report = skyline_observability(
            horizon, xy, 0.0,
            position_delta_m=max(args.position_delta_m, 0.5 * px_to_m),
            yaw_delta_rad=math.radians(args.yaw_delta_deg),
            horizon_sigma_rad=math.radians(args.noise_arcmin / 60.0),
            position_scale_m=step_m,
            yaw_scale_rad=math.radians(args.yaw_scale_deg),
        )

        errors = []
        wrong = 0
        sigma = math.radians(args.noise_arcmin / 60.0)
        for _ in range(args.trials):
            jitter = rng.uniform(
                -args.truth_jitter_frac * step_m,
                args.truth_jitter_frac * step_m,
                size=2,
            )
            truth_xy = np.clip(
                np.asarray(xy) + jitter,
                args.grid_margin_frac * extent_m,
                (1.0 - args.grid_margin_frac) * extent_m,
            )
            truth_profile = horizon((float(truth_xy[0]), float(truth_xy[1])))
            yaw_error = rng.normal(0.0, math.radians(args.yaw_sigma_deg))
            heading_bins = int(round(yaw_error * args.n_az / (2.0 * math.pi)))
            noisy = np.roll(truth_profile, -heading_bins)
            noisy = noisy + rng.normal(0.0, sigma, args.n_az)
            noisy_scores, _ = best_yaw_ncc(
                noisy, predictions, prior_lag=0, window_bins=window_bins
            )
            estimate = candidates[int(np.argmax(noisy_scores))]
            error = float(np.linalg.norm(np.asarray(estimate) - truth_xy))
            errors.append(error)
            wrong += int(error > 1.5 * step_m)

        margin_reliability = float(np.clip(margin / args.margin_reference, 0.0, 1.0))
        combined_reliability = (1.0 - report.alignment_risk) * margin_reliability
        records.append({
            "grid_index": i,
            "xy_m": [round(xy[0], 3), round(xy[1], 3)],
            "local_alignment_risk": round(report.alignment_risk, 6),
            "combined_alignment_risk": round(1.0 - combined_reliability, 6),
            "directional_confidence": {
                name: round(float(value), 6)
                for name, value in zip(report.state_labels, report.directional_confidence)
            },
            "weak_direction_scaled": {
                name: round(float(value), 6)
                for name, value in zip(report.state_labels, report.weak_direction)
            },
            "eigenvalues": [round(float(v), 6) for v in report.eigenvalues],
            "spectral_balance": round(report.spectral_balance, 6),
            "measurement_strength": round(report.measurement_strength, 6),
            "uniqueness_margin": round(float(margin), 6),
            "second_mode_ncc": round(float(second), 6),
            "median_error_m": round(float(np.median(errors)), 3),
            "p90_error_m": round(float(np.percentile(errors, 90)), 3),
            "wrong_lock_rate": round(wrong / args.trials, 6),
        })

    local_risk = np.array([r["local_alignment_risk"] for r in records])
    combined_risk = np.array([r["combined_alignment_risk"] for r in records])
    margin_risk = 1.0 - np.clip(
        np.array([r["uniqueness_margin"] for r in records]) / args.margin_reference,
        0.0, 1.0,
    )
    errors = np.array([r["p90_error_m"] for r in records])
    return {
        "target": target,
        "label": label,
        "grid": args.grid,
        "extent_m": round(extent_m, 3),
        "grid_step_m": round(step_m, 3),
        "records": records,
        "correlation_with_p90_error": {
            "local_observability_risk": round(_safe_corr(local_risk, errors), 4),
            "uniqueness_margin_risk": round(_safe_corr(margin_risk, errors), 4),
            "combined_risk": round(_safe_corr(combined_risk, errors), 4),
        },
        "mean_wrong_lock_rate": round(
            float(np.mean([r["wrong_lock_rate"] for r in records])), 6
        ),
    }


def _grid(scene: dict, key: str, subkey: str | None = None) -> np.ndarray:
    if subkey is None:
        values = [r[key] for r in scene["records"]]
    else:
        values = [r[key][subkey] for r in scene["records"]]
    return np.asarray(values, dtype=np.float64).reshape(scene["grid"], scene["grid"])


def render_summary(scenes: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(len(scenes), 4, figsize=(15.5, 10.0), constrained_layout=True)
    column_titles = (
        "combined predicted risk",
        "relative direction confidence (RGB = x / y / yaw)",
        "uniqueness margin",
        "empirical p90 error (m)",
    )
    for col, title in enumerate(column_titles):
        axes[0, col].set_title(title, fontsize=11, fontweight="bold")
    for row, scene in enumerate(scenes):
        direction_channels = np.stack([
            _grid(scene, "directional_confidence", "x"),
            _grid(scene, "directional_confidence", "y"),
            _grid(scene, "directional_confidence", "yaw"),
        ], axis=-1)
        direction = direction_channels / np.maximum(
            direction_channels.max(axis=-1, keepdims=True), 1e-12
        )
        panels = (
            (_grid(scene, "combined_alignment_risk"), "magma", 0.0, 1.0),
            (direction, None, None, None),
            (_grid(scene, "uniqueness_margin"), "YlGn", 0.0, None),
            (_grid(scene, "p90_error_m"), "inferno", 0.0, None),
        )
        for col, (image, cmap, vmin, vmax) in enumerate(panels):
            shown = axes[row, col].imshow(
                image, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax
            )
            if col != 1:
                fig.colorbar(shown, ax=axes[row, col], shrink=0.78)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
        corr = scene["correlation_with_p90_error"]
        axes[row, 0].set_ylabel(scene["label"], fontsize=10, fontweight="bold")
        axes[row, 0].text(
            0.02, 0.02,
            f"corr risk/error: local {corr['local_observability_risk']:+.2f}, "
            f"margin {corr['uniqueness_margin_risk']:+.2f}, "
            f"combined {corr['combined_risk']:+.2f}",
            transform=axes[row, 0].transAxes, fontsize=7,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
    fig.suptitle(
        "Predict before you drift: Skyline directional observability vs actual lock error",
        fontsize=14, fontweight="bold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--targets", nargs="+", default=[s[0] for s in SCENES])
    ap.add_argument("--ldem-ppd", type=int, default=16, choices=(4, 16, 64))
    ap.add_argument("--half-width-deg", type=float, default=2.0)
    ap.add_argument("--cache-dir", type=Path, default=data_root() / "lro_cache")
    ap.add_argument("--output-dir", type=Path,
                    default=output_root() / "predictive_localizability")
    ap.add_argument("--summary-json", type=Path, default=None,
                    help="Optional small record-free JSON suitable for Git.")
    ap.add_argument("--grid", type=int, default=7)
    ap.add_argument("--grid-margin-frac", type=float, default=0.18)
    ap.add_argument("--n-az", type=int, default=120)
    ap.add_argument("--n-range", type=int, default=100)
    ap.add_argument("--mast-height-m", type=float, default=2.0)
    ap.add_argument("--position-delta-m", type=float, default=150.0)
    ap.add_argument("--yaw-delta-deg", type=float, default=0.5)
    ap.add_argument("--yaw-scale-deg", type=float, default=5.0)
    ap.add_argument("--yaw-sigma-deg", type=float, default=2.0)
    ap.add_argument("--noise-arcmin", type=float, default=8.0)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--truth-jitter-frac", type=float, default=0.35,
                    help="Off-grid truth perturbation as a fraction of grid spacing.")
    ap.add_argument("--margin-reference", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=23)
    args = ap.parse_args()

    labels = dict(SCENES)
    unknown = sorted(set(args.targets) - set(labels))
    if unknown:
        ap.error(f"unknown target(s): {', '.join(unknown)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    scenes = []
    for target in args.targets:
        print(f"evaluate {target} ...", flush=True)
        scenes.append(evaluate_scene(args, target, labels[target], rng))

    payload = {
        "schema_version": 1,
        "method": {
            "local": "finite-difference raw-horizon Fisher information",
            "global": "best-vs-competing-position uniqueness margin",
            "combined": "product of local and global reliability",
            "state": ["x", "y", "yaw"],
            "factor_graph_used": False,
        },
        "settings": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "scenes": scenes,
    }
    json_path = args.output_dir / "skyline_predictive_localizability.json"
    figure_path = args.output_dir / "skyline_predictive_localizability.png"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    render_summary(scenes, figure_path)
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "schema_version": payload["schema_version"],
            "method": payload["method"],
            "settings": {
                key: value for key, value in payload["settings"].items()
                if key not in {"cache_dir", "output_dir", "summary_json"}
            },
            "scenes": [{
                "target": scene["target"],
                "label": scene["label"],
                "grid": scene["grid"],
                "extent_m": scene["extent_m"],
                "grid_step_m": scene["grid_step_m"],
                "correlation_with_p90_error":
                    scene["correlation_with_p90_error"],
                "mean_wrong_lock_rate": scene["mean_wrong_lock_rate"],
            } for scene in scenes],
            "full_artifact": "external SSD; regenerate with the command in docs/predictive_localizability.md",
        }
        args.summary_json.write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps({
        "json": str(json_path),
        "figure": str(figure_path),
        "summary_json": str(args.summary_json) if args.summary_json else None,
        "scene_correlations": {
            s["target"]: s["correlation_with_p90_error"] for s in scenes
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
