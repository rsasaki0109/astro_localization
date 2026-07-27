# Predictive Localizability

This research line asks a pre-solve question: **which `(x, y, yaw)` directions
does the raw lunar skyline constrain, and can that predict a bad alignment
before the factor graph trusts it?**

The existing `uniqueness_margin` measures the gap to a distant competing
position. It is a global alias test. The new metric finite-differences the raw
horizon prediction and forms a scaled Fisher information matrix. It is a local
directional observability test. Neither is a substitute for the other.

## Data placement

Large datasets, caches, and full benchmark artifacts live outside Git. On the
current development machine they are configured in the ignored file
`.astro_navigation.local.json`:

```json
{
  "data_root": "/media/sasaki/aiueo/datasets/astro_navigation",
  "output_root": "/media/sasaki/aiueo/benchmarks/astro_navigation"
}
```

`ASTRO_NAV_DATA_ROOT` and `ASTRO_NAV_OUTPUT_ROOT` override that file. A fresh
clone without either setting falls back to `datasets/` and `outputs/`.

## Reproduce

```bash
python3 scripts/predictive_localizability_demo.py \
  --summary-json docs/figures/skyline_lock/predictive_localizability_summary.json

python3 -m unittest tests/test_predictive_localizability.py -v

python3 scripts/skyline_localizability_map.py \
  --target tycho \
  --cache-dir /media/sasaki/aiueo/datasets/astro_navigation/lro_cache \
  --grid 25 --n-az 90 --n-range 80 \
  --routing-signal directional --loc-weight 20 \
  --output /media/sasaki/aiueo/benchmarks/astro_navigation/predictive_localizability/tycho_directional_route.png \
  --output-json docs/figures/skyline_lock/predictive_directional_route_summary.json
```

The full JSON and PNG are written under the configured external output root.
Only the record-free summary JSON is retained in Git.

## Initial honest result

The first coarse 7x7 study uses the same settings over Tycho highland, Apollo
11 mare, and a spherical tangent-plane resampling around Shackleton. The polar
resampling matters: a longitude crop becomes almost one pixel wide at 89.66 S
and creates a false flat-horizon result.

Local directional risk had positive correlation with empirical p90 lock error
in all three scenes. The scalar uniqueness-margin risk was weak over Tycho and
anti-correlated over the mare and Shackleton in this particular off-grid/noisy
trial. Multiplying the two reliabilities did not consistently improve on local
observability alone. This is evidence that local directional degeneracy and
distant spatial aliasing are distinct failure modes; it is not yet evidence for
a final calibrated risk formula.

Next evaluation gates are denser grids, repeated seeds, calibration metrics
(Brier score and reliability curves), and independent perturbations for DEM
resolution and horizon extraction noise.

The first directional-routing run increased mean route reliability from 0.1485
to 0.2640 at a 1.124x path-length cost. Its minimum reliability remained almost
unchanged (0.0440 vs 0.0468), so this is a mean-exposure improvement, not yet a
guarantee against the weakest cell.

## Directional factor-graph result

`four_factor_fusion_demo.py --directional-information` sends Skyline and TRN
raw-measurement information shapes into a full-XY linear pose graph. For a fair
ablation, the strongest eigenvalue is fixed to the legacy scalar factor's
`1 / sigma^2`; only directionality changes.

The synthetic crater RMSE improved from 68.0 m to 55.2 m. Real Tycho was
287.2 m vs 289.0 m, and Apollo 11 mare was 392.7 m vs 393.8 m. The real runs
are effectively ties. Directionality is therefore useful now as a failure
warning and routing signal, but the current experiment does not support a claim
of better real-terrain fusion accuracy. The record-free comparison is in
`docs/figures/skyline_lock/predictive_directional_fusion_summary.json`.
