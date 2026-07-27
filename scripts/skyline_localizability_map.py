#!/usr/bin/env python3
"""Skyline localizability map + don't-get-lost routing over real LOLA terrain.

Phase 4 of Skyline Lock. Turns the per-position horizon uniqueness_margin into a
*map*: for every cell on a grid, how unambiguously could a rover there pin its
own position from the horizon alone (heading known from the star tracker)?

Because the candidate horizons are already predicted on the grid, the map costs
no extra rendering: cell i's "observation" IS preds[i], scored against every
other cell. localizability(i) = 1 - (best competing match outside a small
exclusion radius). High where the terrain is distinctive (e.g. a crater seen
from near its centre); low where it is self-similar or rotationally symmetric.

The map is then used as a routing cost. A baseline A* takes the shortest path;
a localizability-aware A* adds cost for low-localizability cells, so it detours
to stay on terrain where the rover can keep a horizon fix -- the "don't get
lost" route. This is the horizon-driven sibling of the existing TRN-confidence
localizability-aware routing demo.

Reuses scripts/skyline_lock_demo.py (DEM + matcher) and the A* from
scripts/render_hazard_aware_navigation_demo.py.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from skyline_lock_demo import (  # noqa: E402
    load_lola_dem, render_horizon, best_yaw_ncc, uniqueness_margin,
)
from predictive_localizability import skyline_grid_observability  # noqa: E402
from render_hazard_aware_navigation_demo import (  # noqa: E402
    GridPoint, astar,
)


def compute_localizability_map(
    preds: np.ndarray, cand_xy: list, grid: int, *,
    prior_lag: int, window_bins: int, excl_radius_m: float,
) -> np.ndarray:
    """Return a (grid, grid) localizability map in [0, 1] (row=y, col=x).

    localizability(i) = 1 - strongest competing horizon match beyond
    excl_radius_m of cell i. Heading is assumed known (prior_lag), so only
    near-zero rotations are admitted -- a rover with a star-tracker attitude.
    """
    loc = np.zeros(grid * grid, dtype=np.float64)
    for i in range(len(cand_xy)):
        scores, _ = best_yaw_ncc(preds[i], preds,
                                 prior_lag=prior_lag, window_bins=window_bins)
        # est is cell i itself (its self-match is the global max); the margin is
        # 1 minus the best *other* position's score outside the exclusion radius.
        margin, _ = uniqueness_margin(scores, cand_xy, cand_xy[i], radius_m=excl_radius_m)
        loc[i] = max(0.0, margin)
    return loc.reshape(grid, grid)


def route_localizability(route, loc_map) -> dict:
    vals = np.asarray([float(loc_map[p.y, p.x]) for p in route], dtype=np.float64)
    length_cells = sum(math.hypot(b.x - a.x, b.y - a.y) for a, b in zip(route[:-1], route[1:]))
    straight = math.hypot(route[-1].x - route[0].x, route[-1].y - route[0].y)
    return {
        "mean_localizability": round(float(vals.mean()), 4),
        "min_localizability": round(float(vals.min()), 4),
        "low_loc_fraction": round(float(np.mean(vals < 0.05)), 4),
        "length_cells": round(length_cells, 2),
        "detour_ratio": round(length_cells / straight if straight > 0 else 1.0, 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default="tycho")
    ap.add_argument("--ldem-ppd", type=int, default=16, choices=[4, 16, 64])
    ap.add_argument("--half-width-deg", type=float, default=3.0)
    ap.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "datasets" / "lro_cache")
    ap.add_argument("--grid", type=int, default=55, help="grid x grid cells (cost ~ grid^2).")
    ap.add_argument("--grid-margin-frac", type=float, default=0.1)
    ap.add_argument("--n-az", type=int, default=180)
    ap.add_argument("--n-range", type=int, default=160)
    ap.add_argument("--mast-height-m", type=float, default=2.0)
    ap.add_argument("--yaw-sigma-deg", type=float, default=5.0,
                    help="Star-tracker heading 1-sigma; lag window is +/- 3 sigma around known heading.")
    ap.add_argument("--loc-weight", type=float, default=12.0,
                    help="How strongly the aware route avoids low-localizability terrain.")
    ap.add_argument("--routing-signal", choices=("margin", "directional"),
                    default="margin",
                    help="margin keeps the original global-alias route; directional "
                    "uses pre-solve (x,y,yaw) observability.")
    ap.add_argument("--noise-arcmin", type=float, default=8.0,
                    help="Horizon extraction sigma for directional information.")
    ap.add_argument("--yaw-scale-deg", type=float, default=5.0,
                    help="Yaw perturbation scale used to compare yaw with one grid step.")
    # Default start/goal cross the dark exterior above Tycho's rim, so the
    # shortest path stays in aliased terrain and the aware route must detour
    # down onto the distinctive rim ring.
    ap.add_argument("--start-frac", type=float, nargs=2, default=(0.1, 0.82))
    ap.add_argument("--goal-frac", type=float, nargs=2, default=(0.9, 0.82))
    ap.add_argument("--output", type=Path,
                    default=REPO_ROOT / "docs" / "figures" / "skyline_lock" / "skyline_localizability_route.png")
    ap.add_argument("--output-json", type=Path, default=None)
    args = ap.parse_args()

    dem, px_to_m = load_lola_dem(args.target, args.half_width_deg, args.ldem_ppd, args.cache_dir)
    size = dem.shape[0]
    extent_m = size * px_to_m
    r_max = 0.9 * extent_m
    r_min = max(60.0, 2.0 * px_to_m)
    A = args.n_az
    bins_per_deg = A / 360.0

    m = args.grid_margin_frac
    gx = np.linspace(m * extent_m, (1.0 - m) * extent_m, args.grid)
    gy = np.linspace(m * extent_m, (1.0 - m) * extent_m, args.grid)
    cand_xy = [(x, y) for y in gy for x in gx]
    grid_step_m = (gx[1] - gx[0]) if args.grid > 1 else extent_m

    print(f"predicting {len(cand_xy)} horizons + localizability map ...")
    preds = np.stack([
        render_horizon(dem, px_to_m, xy, args.mast_height_m,
                       n_az=A, r_min_m=r_min, r_max_m=r_max, n_range=args.n_range)
        for xy in cand_xy
    ])
    # Heading known (star tracker): predictions and observations share the world
    # frame, so the true lag is 0; admit only +/- 3 sigma around it.
    window_bins = max(1, int(math.ceil(3.0 * args.yaw_sigma_deg * bins_per_deg)))
    loc_map = compute_localizability_map(
        preds, cand_xy, args.grid,
        prior_lag=0, window_bins=window_bins, excl_radius_m=2.0 * grid_step_m)
    directional_map, _ = skyline_grid_observability(
        preds,
        grid_shape=(args.grid, args.grid),
        grid_step_m=grid_step_m,
        horizon_sigma_rad=math.radians(args.noise_arcmin / 60.0),
        position_scale_m=grid_step_m,
        yaw_scale_rad=math.radians(args.yaw_scale_deg),
    )

    # Routing on the candidate grid (GridPoint.x = col, .y = row).
    signal_map = loc_map if args.routing_signal == "margin" else directional_map
    loc_norm = (
        (signal_map - signal_map.min())
        / max(1e-9, signal_map.max() - signal_map.min())
    )
    base_cost = np.ones((args.grid, args.grid), dtype=np.float64)
    aware_cost = 1.0 + args.loc_weight * (1.0 - loc_norm)

    def frac_to_grid(fx, fy):
        return GridPoint(int(round(fx * (args.grid - 1))), int(round(fy * (args.grid - 1))))

    start = frac_to_grid(*args.start_frac)
    goal = frac_to_grid(*args.goal_frac)
    base_route = astar(base_cost, start, goal)
    aware_route = astar(aware_cost, start, goal)

    base_m = route_localizability(base_route, signal_map)
    aware_m = route_localizability(aware_route, signal_map)
    summary = {
        "scene": f"lola:{args.target}",
        "grid": args.grid,
        "grid_step_m": round(float(grid_step_m), 2),
        "loc_weight": args.loc_weight,
        "routing_signal": args.routing_signal,
        "localizable_cell_fraction": round(float(np.mean(loc_map >= 0.05)), 4),
        "mean_directional_reliability": round(float(directional_map.mean()), 6),
        "shortest_route_signal_metrics": base_m,
        "predictive_aware_route_signal_metrics": aware_m,
    }
    print(json.dumps(summary, indent=2))

    _render(args, dem, extent_m, gx, gy, signal_map,
            base_route, aware_route, start, goal)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2))
        print(f"wrote {args.output_json}")
    return 0


def _route_world(route, gx, gy):
    return np.array([gx[p.x] for p in route]), np.array([gy[p.y] for p in route])


def _render(args, dem, extent_m, gx, gy, loc_map, base_route, aware_route, start, goal) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bx, by = _route_world(base_route, gx, gy)
    ax_, ay = _route_world(aware_route, gx, gy)
    sx, sy = gx[start.x], gy[start.y]
    e_x, e_y = gx[goal.x], gy[goal.y]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    aware_name = (
        "localizability-aware"
        if args.routing_signal == "margin"
        else "predictive-observability-aware"
    )
    fig.suptitle(
        f"Skyline localizability routing over {args.target.title()}  -  "
        f"{aware_name} route avoids weak horizon geometry", fontsize=13)

    ax = axes[0]
    im = ax.imshow(dem, origin="lower", extent=[0, extent_m, 0, extent_m], cmap="terrain")
    ax.plot(bx, by, "--", color="red", lw=2.0, label="shortest")
    ax.plot(ax_, ay, "-", color="cyan", lw=2.0, label=aware_name)
    ax.scatter([sx, e_x], [sy, e_y], c="white", marker="o", s=60, edgecolors="k", zorder=5)
    ax.set_title("DEM (m) + routes"); ax.set_xlabel("x east (m)"); ax.set_ylabel("y north (m)")
    ax.legend(loc="upper left", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1]
    im = ax.imshow(loc_map, origin="lower", extent=[gx[0], gx[-1], gy[0], gy[-1]],
                   cmap="magma", vmin=0.0, aspect="auto")
    ax.plot(bx, by, "--", color="red", lw=2.0)
    ax.plot(ax_, ay, "-", color="cyan", lw=2.0)
    signal_title = (
        "global uniqueness margin"
        if args.routing_signal == "margin"
        else "pre-solve directional reliability"
    )
    ax.set_title(f"horizon routing signal: {signal_title}")
    ax.set_xlabel("x east (m)"); ax.set_ylabel("y north (m)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[2]
    for route, color, label in ((base_route, "red", "shortest"),
                                (aware_route, "cyan", "aware")):
        vals = [float(loc_map[p.y, p.x]) for p in route]
        s = np.concatenate([[0.0], np.cumsum([math.hypot(b.x - a.x, b.y - a.y)
                                              for a, b in zip(route[:-1], route[1:])])])
        ax.plot(s / s[-1], vals, color=color, lw=1.6, label=label)
    if args.routing_signal == "margin":
        ax.axhline(0.05, color="gray", ls=":", lw=1.0, label="aliased (<0.05)")
    ax.set_title(f"{args.routing_signal} signal along route")
    ax.set_xlabel("route fraction"); ax.set_ylabel("routing reliability")
    ax.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=120)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    sys.exit(main())
