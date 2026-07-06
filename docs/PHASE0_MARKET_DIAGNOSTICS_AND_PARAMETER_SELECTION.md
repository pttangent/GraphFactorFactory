# Phase 0 Market Diagnostics and Parameter Selection

## Scope

This baseline validation uses two regular-session, 60-minute samples:

- active day: `2026-06-09`
- inactive day: `2026-06-16`
- universe: 100 high-liquidity symbols
- interval: 09:30-10:29 America/New_York
- raw edge rows evaluated: 1,215,818
- community checkpoints: 1,196 layer-scale snapshots at the repository's 5-minute theme cadence
- graph families evaluated: 30 non-return-correlation layer-scale graphs

The three return-correlation layers were not included in this empirical selection because the supplied daily smoke packages contained only one-symbol `node_factors_1m`. They inherit the strict baseline until a full-universe NFF source is used.

## Implemented Phase 0 diagnostics

The canonical Phase 0 graph output now supports:

1. strong-edge ratios at training-period q80, q90 and q95 thresholds;
2. strong-edge node coverage;
3. weight p50, p75, p90, p95 and p99;
4. top-1% mean weight and p95 tail mass;
5. active-node coverage and isolated-node ratio;
6. degree-cap saturation;
7. connected-component and giant-component coverage;
8. Leiden community count, size HHI, top-community coverage, weighted modularity, CPM quality and within-community weight ratio when evaluation extras are installed;
9. all-edge and strong-edge birth, death, persistence and Jaccard rates;
10. cross-layer and cross-family edge resonance;
11. node layer participation and cross-family core-node counts.

Run:

```bash
python scripts/run_phase0_diagnostics.py \
  --graph-root data/graph_store \
  --output data/phase0_diagnostics \
  --train-dates 2026-06-09 \
  --dates 2026-06-09 2026-06-16
```

## Market-state result

The original claim is supported: total edge count is a poor standalone activity indicator under fixed reciprocal top-k and degree caps.

Under strict arm B, active versus inactive day differences remained small in total density, but stronger structural measures changed independently. The diagnostics therefore separate graph capacity from market structure and should be used jointly rather than replacing edge count with one new scalar.

Observed daily averages:

| Metric | 2026-06-09 A | 2026-06-09 B | 2026-06-16 A | 2026-06-16 B |
|---|---:|---:|---:|---:|
| edge count | 244.35 | 147.27 | 245.00 | 144.09 |
| node coverage | 0.914 | 0.835 | 0.914 | 0.828 |
| strong-edge ratio using A-q90 | 0.126 | 0.176 | 0.126 | 0.175 |
| strong-node coverage | 0.272 | 0.270 | 0.263 | 0.261 |
| weight p95 | 0.763 | 0.789 | 0.759 | 0.786 |
| weight p99 | 0.816 | 0.830 | 0.813 | 0.829 |
| weighted modularity | 0.661 | 0.738 | 0.661 | 0.739 |
| within-community weight ratio | 0.826 | 0.879 | 0.828 | 0.885 |
| strong-edge persistence | 0.287 | 0.284 | 0.285 | 0.284 |

Arm B removed weak edges and raised strong-edge purity, upper-tail weight, modularity and within-community concentration. It did not materially improve strong-edge persistence, so persistence remains an independent diagnostic rather than a reason to select B by itself.

## Parameter selection

Candidates:

- A: `top_k=8`, `degree_cap=6`, `minimum_similarity=0.10`
- B: `top_k=6`, `degree_cap=4`, `minimum_similarity=0.30`

The selector uses a multi-objective score containing:

- strong-edge purity and strong-node coverage;
- p95/p99 edge weight;
- strong-edge persistence;
- weighted modularity and within-community weight ratio;
- active/inactive effect size;
- penalties for degree-cap saturation and empty graphs.

Result:

- B selected for 26 of 30 evaluated layer-scale graphs;
- A retained for four short-horizon or breadth-sensitive graphs:
  - `volume_expansion:5`
  - `volume_expansion:15`
  - `signed_flow:5`
  - `flow_return_alignment:5`

The committed registry is `configs/phase0_ab_selected_v1.yaml`.

## Hierarchical parameter resolution

`BuildConfig.graph_parameters_for()` resolves parameters in this order:

1. global defaults;
2. `family:<family>`;
3. `layer:<layer_name>`;
4. `scale:<layer_name>:<lookback_minutes>`.

This keeps Phase 0 backward compatible while allowing family, layer and scale selection without creating 35 independent pipelines.

## Reproducible selection

Diagnostics for multiple candidate graph stores can be compared with:

```bash
python scripts/select_phase0_parameters.py \
  --candidate A=/path/to/A/diagnostics \
  --candidate B=/path/to/B/diagnostics \
  --parameters-json candidate_parameters.json \
  --layers data/graph_store/dimensions/layers.parquet \
  --active-dates 2026-06-09 \
  --inactive-dates 2026-06-16 \
  --output data/phase0_parameter_selection
```

The output contains candidate scores, one winner per layer-scale and a versioned YAML parameter registry.

## Validation boundary

This completes the baseline Phase 0 diagnostic and parameter-selection chain for the two-day active/inactive smoke comparison. It does not claim final 14-day generalization. The next validation should run the same unchanged code on blocked high, medium and low activity dates, then freeze the registry before Phase 1 theme discovery.
