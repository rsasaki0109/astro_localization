#!/usr/bin/env python3
"""Animate the don't-get-lost route: watch A* detour onto horizon-distinctive rim.

Phase 4 of Skyline Lock made a static figure (scripts/skyline_localizability_map.py);
this promotes it to a GIF/MP4 in two acts over the real LOLA localizability map
(bright = a rover there can pin itself from the horizon; dark = self-similar /
rotationally symmetric terrain where the fix aliases):

  Act 1 -- search. The localizability-aware A* expansion floods outward from the
  start. Because low-localizability cells carry extra cost, the frontier hugs the
  bright rim ring and shies away from the dark interior, visibly routing AROUND
  the aliased terrain rather than straight across it.

  Act 2 -- traverse. Both finished routes are revealed and a rover walks each in
  step: the shortest path (red) cuts straight through the dark and its
  localizability dives below the aliased threshold; the aware path (cyan) takes a
  longer arc that stays on terrain where the horizon keeps locking.

Reuses scripts/skyline_localizability_map.py (map + scoring) and a small traced
copy of the A* so we can replay the expansion order frame by frame.
"""

from __future__ import annotations

import argparse
import heapq
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from skyline_lock_demo import load_lola_dem, render_horizon  # noqa: E402
from skyline_localizability_map import (  # noqa: E402
    compute_localizability_map, route_localizability,
)
from predictive_localizability import skyline_grid_observability  # noqa: E402
from render_hazard_aware_navigation_demo import GridPoint  # noqa: E402


def astar_traced(cost, start, goal):
    """A* that also returns the order cells were expanded (popped), so the search
    can be animated. Same transition model as the shared astar."""
    h, w = cost.shape
    neighbors = [(-1, -1, math.sqrt(2.0)), (0, -1, 1.0), (1, -1, math.sqrt(2.0)),
                 (-1, 0, 1.0), (1, 0, 1.0), (-1, 1, math.sqrt(2.0)),
                 (0, 1, 1.0), (1, 1, math.sqrt(2.0))]

    def heuristic(a, b):
        return math.hypot(a.x - b.x, a.y - b.y)

    frontier = [(0.0, 0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0.0}
    counter = 0
    expanded = []
    seen_expanded = set()
    while frontier:
        _p, _c, current = heapq.heappop(frontier)
        if current in seen_expanded:
            continue
        seen_expanded.add(current)
        expanded.append(current)
        if current == goal:
            break
        for dx, dy, step in neighbors:
            nx, ny = current.x + dx, current.y + dy
            if nx < 0 or ny < 0 or nx >= w or ny >= h:
                continue
            nxt = GridPoint(nx, ny)
            tc = step * (0.5 * float(cost[current.y, current.x]) + 0.5 * float(cost[ny, nx]))
            new_cost = cost_so_far[current] + tc
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                counter += 1
                heapq.heappush(frontier, (new_cost + 1.3 * heuristic(nxt, goal), counter, nxt))
                came_from[nxt] = current
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = came_from.get(cur)
    path.reverse()
    return path, expanded


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default="tycho")
    ap.add_argument("--ldem-ppd", type=int, default=16, choices=[4, 16, 64])
    ap.add_argument("--half-width-deg", type=float, default=3.0)
    ap.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "datasets" / "lro_cache")
    ap.add_argument("--grid", type=int, default=55)
    ap.add_argument("--grid-margin-frac", type=float, default=0.1)
    ap.add_argument("--n-az", type=int, default=180)
    ap.add_argument("--n-range", type=int, default=160)
    ap.add_argument("--mast-height-m", type=float, default=2.0)
    ap.add_argument("--yaw-sigma-deg", type=float, default=5.0)
    ap.add_argument("--loc-weight", type=float, default=12.0)
    ap.add_argument("--routing-signal", choices=("margin", "directional"),
                    default="margin")
    ap.add_argument("--noise-arcmin", type=float, default=8.0)
    ap.add_argument("--yaw-scale-deg", type=float, default=5.0)
    ap.add_argument("--start-frac", type=float, nargs=2, default=(0.1, 0.82))
    ap.add_argument("--goal-frac", type=float, nargs=2, default=(0.9, 0.82))
    ap.add_argument("--search-frames", type=int, default=20)
    ap.add_argument("--traverse-frames", type=int, default=16)
    ap.add_argument("--duration-ms", type=int, default=140)
    ap.add_argument("--mp4", action="store_true")
    ap.add_argument("--output", type=Path,
                    default=REPO_ROOT / "docs" / "figures" / "skyline_lock"
                    / "skyline_localizability_route_demo.gif")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
        for xy in cand_xy])
    window_bins = max(1, int(math.ceil(3.0 * args.yaw_sigma_deg * bins_per_deg)))
    margin_map = compute_localizability_map(
        preds, cand_xy, args.grid, prior_lag=0, window_bins=window_bins,
        excl_radius_m=2.0 * grid_step_m)
    directional_map, _ = skyline_grid_observability(
        preds,
        grid_shape=(args.grid, args.grid),
        grid_step_m=grid_step_m,
        horizon_sigma_rad=math.radians(args.noise_arcmin / 60.0),
        position_scale_m=grid_step_m,
        yaw_scale_rad=math.radians(args.yaw_scale_deg),
    )
    loc_map = margin_map if args.routing_signal == "margin" else directional_map
    signal_name = (
        "global uniqueness margin"
        if args.routing_signal == "margin"
        else "pre-solve directional reliability"
    )

    loc_norm = (loc_map - loc_map.min()) / max(1e-9, loc_map.max() - loc_map.min())
    base_cost = np.ones((args.grid, args.grid), dtype=np.float64)
    aware_cost = 1.0 + args.loc_weight * (1.0 - loc_norm)

    def frac_to_grid(fx, fy):
        return GridPoint(int(round(fx * (args.grid - 1))), int(round(fy * (args.grid - 1))))

    start = frac_to_grid(*args.start_frac)
    goal = frac_to_grid(*args.goal_frac)
    base_route, _ = astar_traced(base_cost, start, goal)
    aware_route, aware_expanded = astar_traced(aware_cost, start, goal)
    base_m = route_localizability(base_route, loc_map)
    aware_m = route_localizability(aware_route, loc_map)
    print(f"  shortest mean signal {base_m['mean_localizability']:.3f} (weak frac "
          f"{base_m['low_loc_fraction']:.2f}); aware mean loc "
          f"{aware_m['mean_localizability']:.3f} (weak frac {aware_m['low_loc_fraction']:.2f}); "
          f"detour x{aware_m['detour_ratio']:.2f}; expanded {len(aware_expanded)} cells")

    bx = np.array([gx[p.x] for p in base_route]); byr = np.array([gy[p.y] for p in base_route])
    ax_ = np.array([gx[p.x] for p in aware_route]); ay = np.array([gy[p.y] for p in aware_route])
    sx, sy = gx[start.x], gy[start.y]
    e_x, e_y = gx[goal.x], gy[goal.y]
    ex = np.array([gx[p.x] for p in aware_expanded])
    ey = np.array([gy[p.y] for p in aware_expanded])

    def loc_along(route):
        vals = np.array([float(loc_map[p.y, p.x]) for p in route])
        s = np.concatenate([[0.0], np.cumsum([math.hypot(b.x - a.x, b.y - a.y)
                            for a, b in zip(route[:-1], route[1:])])])
        return s / s[-1], vals

    bf, bv = loc_along(base_route)
    af, av = loc_along(aware_route)
    extent = [gx[0], gx[-1], gy[0], gy[-1]]

    def base_fig():
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.0),
                                 gridspec_kw={"width_ratios": [1.15, 1.0]})
        ax = axes[0]
        ax.imshow(loc_map, origin="lower", extent=extent, cmap="magma", vmin=0.0, aspect="auto")
        ax.scatter([sx, e_x], [sy, e_y], c="white", marker="o", s=70, edgecolors="k", zorder=8)
        ax.text(sx, sy, " start", color="white", fontsize=8, va="center", zorder=9)
        ax.text(e_x, e_y, " goal", color="white", fontsize=8, va="center", ha="right", zorder=9)
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
        ax.set_xlabel("x east (m)"); ax.set_ylabel("y north (m)")
        return fig, axes

    frames: list[Image.Image] = []
    n_exp = len(aware_expanded)

    def grab(fig):
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
        plt.close(fig)

    # --- Act 1: aware A* expansion ---
    print(f"rendering {args.search_frames} search frames ...")
    for i in range(1, args.search_frames + 1):
        k = max(1, int(round(n_exp * i / args.search_frames)))
        fig, axes = base_fig()
        fig.suptitle(f"Skyline routing — Act 1: A* expands through stronger {signal_name}",
                     fontsize=12.0, color="#1f77ff")
        ax = axes[0]
        ax.scatter(ex[:k], ey[:k], c="#19c819", marker="s", s=10, alpha=0.5, zorder=4,
                   label="expanded cells")
        ax.scatter(ex[k - 1], ey[k - 1], c="cyan", marker="s", s=40, edgecolors="k", zorder=6)
        ax.set_title(f"{signal_name} (bright = reliable) — expanded {k}/{n_exp}", fontsize=10)
        ax.legend(loc="upper left", fontsize=8)
        ax = axes[1]
        ax.axhline(0.05, color="gray", ls=":", lw=1.0, label="weak (<0.05)")
        ax.set_xlim(0, 1); ax.set_ylim(0.0, max(0.2, float(loc_map.max()) * 1.05))
        ax.set_title("routing signal along route (fills in Act 2)", fontsize=10)
        ax.set_xlabel("route fraction"); ax.set_ylabel("routing reliability")
        ax.legend(fontsize=8, loc="upper right")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        grab(fig)

    # --- Act 2: traverse both routes ---
    print(f"rendering {args.traverse_frames} traverse frames ...")
    for i in range(1, args.traverse_frames + 1):
        t = i / args.traverse_frames
        kb = max(1, int(round(len(base_route) * t)))
        ka = max(1, int(round(len(aware_route) * t)))
        fig, axes = base_fig()
        fig.suptitle(f"Skyline routing — Act 2: shortest vs {signal_name}-aware",
                     fontsize=11.5, color="#137333")
        ax = axes[0]
        ax.scatter(ex, ey, c="#19c819", marker="s", s=8, alpha=0.18, zorder=3)
        ax.plot(bx[:kb], byr[:kb], "--", color="red", lw=2.2, label="shortest", zorder=5)
        ax.plot(ax_[:ka], ay[:ka], "-", color="cyan", lw=2.2,
                label="predictive-aware" if args.routing_signal == "directional"
                else "localizability-aware", zorder=5)
        ax.scatter(bx[kb - 1], byr[kb - 1], c="red", s=55, edgecolors="k", zorder=7)
        ax.scatter(ax_[ka - 1], ay[ka - 1], c="cyan", s=55, edgecolors="k", zorder=7)
        ax.set_title(f"routes (aware detour ×{aware_m['detour_ratio']:.2f})", fontsize=10)
        ax.legend(loc="upper left", fontsize=8)
        ax = axes[1]
        ax.axhline(0.05, color="gray", ls=":", lw=1.0, label="weak (<0.05)")
        nb = max(1, int(round(len(bf) * t))); na = max(1, int(round(len(af) * t)))
        ax.plot(bf[:nb], bv[:nb], "-", color="red", lw=1.8,
                label=f"shortest (weak {base_m['low_loc_fraction']*100:.0f}%)")
        ax.plot(af[:na], av[:na], "-", color="cyan", lw=1.8,
                label=f"aware (weak {aware_m['low_loc_fraction']*100:.0f}%)")
        ax.set_xlim(0, 1); ax.set_ylim(0.0, max(0.2, float(loc_map.max()) * 1.05))
        ax.set_title(f"{signal_name} along route", fontsize=10)
        ax.set_xlabel("route fraction"); ax.set_ylabel("routing reliability")
        ax.legend(fontsize=8, loc="upper right")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        grab(fig)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    seq = frames + [frames[-1]] * 10
    seq[0].save(args.output, save_all=True, append_images=seq[1:],
                duration=args.duration_ms, loop=0, optimize=True)
    print(f"wrote {args.output}  ({len(seq)} frames)")

    if args.mp4:
        mp4 = args.output.with_suffix(".mp4")
        cmd = ["ffmpeg", "-y", "-i", str(args.output), "-movflags", "faststart",
               "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", str(mp4)]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"wrote {mp4}")
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            print(f"mp4 conversion skipped: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
