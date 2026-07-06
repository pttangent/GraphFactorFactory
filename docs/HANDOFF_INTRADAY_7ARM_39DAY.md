# Handoff: ReturnCorr Rebuild + Phase 1/2 Rerun

## Repository and branch

Primary repository:

- `pttangent/GraphFactorFactory`
- branch: `codex/intraday-7arm-layer-analysis`

Reference repositories that must be read before changing behavior:

1. `pttangent/StockNet_V2` (`main`)
   - `src/stocknetv2/domain/graph/return_corr.py`
   - `src/stocknetv2/domain/graph/series_utils.py`
   - `src/stocknetv2/domain/graph/layer_config.py`
   - relevant ReturnCorr tests and graph evaluation code
2. `pttangent/US-Stock`
   - current NodeFactorFactory/Phase 0 source implementation
   - verify `ret_5m`, `timestamp`, `available_time`, SPY/QQQ/IWM coverage, and the exact local source configuration used for `Smoke_Test_Output/graph_store`

## Mandatory files in GraphFactorFactory

Read these before running anything:

- `src/graphfactorfactory/domain/layers.py`
- `src/graphfactorfactory/domain/config.py`
- `src/graphfactorfactory/application/math_utils.py`
- `src/graphfactorfactory/application/graph.py`
- `src/graphfactorfactory/application/pipeline.py`
- `src/graphfactorfactory/infrastructure/store.py`
- `src/graphfactorfactory/themes/pipeline.py`
- `src/graphfactorfactory/themes/consensus.py`
- `scripts/run_intraday_7arm_5m.py`
- `docs/INTRADAY_7ARM_39DAY_REPORT.md`
- `artifacts/intraday_7arm_39day/recommended_arm_by_layer.csv`

## Why the graph logic changed

The former GFF `return_corr` layer treated `ret_5m` as a generic normalized trajectory and sent it through reciprocal LSH. This differs from StockNet_V2, where returns are first stripped of a broad cross-sectional baseline before pairwise correlation/top-k selection.

The January-February Phase 1 review found that GFF ReturnCorr did not reproduce the clear industry/theme communities seen in StockNet. The likely causes are:

- broad SPY/QQQ/small-cap beta remains in raw returns;
- generic reciprocal LSH/top-k and degree caps fragment or distort the correlation graph;
- raw market co-movement and industry-specific co-movement were mixed into one layer;
- adding multiple price layers without consensus controls would triple-count the price family.

## Code already changed on this branch

### Phase 0 layer specification

The layer registry now contains 15 layers:

- layer 1: `return_corr` — raw returns, retained for backward compatibility and macro/market-mode observation;
- layer 14: `return_corr_market_residual` — each stock is regressed on available SPY/QQQ/IWM returns, then residual trajectories are used for graph construction;
- layer 15: `return_corr_cross_sectional_residual` — per-timestamp cross-sectional median return is removed, matching the broad StockNet approach and acting as a deterministic fallback/reference.

Do not rename or overwrite layer 1. Raw and residual structures answer different financial questions.

### Phase 0 configuration

`BuildConfig` now contains:

- `return_corr_benchmarks = ("SPY", "QQQ", "IWM")`
- `return_corr_min_benchmark_points = 8`
- `return_corr_ridge = 1e-6`

The YAML loader and config hash include these fields.

### Phase 0 trajectory transformation

`math_utils.return_trajectory()` now implements:

1. raw return trajectory;
2. benchmark regression residuals using an intercept plus available SPY/QQQ/IWM columns;
3. cross-sectional median residuals;
4. deterministic median-residual fallback when benchmark symbols are missing;
5. row standardization only after residualization.

`MultilayerGraphBuilder` passes the configured benchmark parameters and stores the transform name in snapshot metadata.

### Phase 1 consensus protection

All three ReturnCorr layers remain visible as independent layer communities.

However, only `return_corr_market_residual` is allowed to vote in consensus themes. `return_corr` and `return_corr_cross_sectional_residual` are excluded in `themes/consensus.py` so the price family is not counted three times.

This is intentional for the first validation run. Do not re-enable all three price layers in consensus without a normalized family-level weighting design.

## Existing Phase 0 specification

Current graph store layout:

```text
graph_store/
  dimensions/
    symbols.parquet
    layers.parquet
  canonical/
    date=YYYY-MM-DD/
      edges.parquet
      node_features.parquet
      snapshots.parquet
      labels.parquet
  graphfactorfactory.duckdb
  manifest.json
```

The January-February universe mapping contains 5,590 symbols in `dimensions/symbols.parquet`.

Existing Phase 0 graph parameters must be preserved for the first A/B comparison unless a test proves a parameter is invalid:

- graph window: current production value;
- graph step: current production value;
- reciprocal top-k: current config;
- degree cap: current config;
- minimum similarity and minimum window points: current config;
- PIT filtering: `available_time <= decision_time` and `timestamp <= decision_time`;
- regular-session filtering and existing label semantics unchanged.

The first rerun is a ReturnCorr logic change, not a simultaneous global tuning exercise.

## Existing Phase 1 specification

Input: Phase 0 daily `edges.parquet` and `node_features.parquet`.

Outputs per date:

- `layer_communities.parquet`
- `subcommunities.parquet`
- `themes.parquet`
- `semantics.parquet`
- lifecycle/read-model outputs

Current major settings:

- Leiden hierarchy per layer;
- market-mode threshold;
- temporal edge replay with enter/exit hysteresis and smoothing;
- consensus requires multiple families;
- sequential lifecycle assignment after parallel community detection.

Because ReturnCorr edges and the registered layer set changed, Phase 1 is affected and must be rerun for every rebuilt Phase 0 date.

A layer-only Phase 1 patch is sufficient only for `layer_communities` and `subcommunities`. It is not sufficient for `themes`, lifecycle, semantics, or quality because consensus inputs changed. Therefore:

- if the deliverable includes consensus themes, rerun full Phase 1 for each rebuilt date;
- if doing a quick diagnostic first, a temporary layer-only ReturnCorr community run is allowed, but store it in a separate diagnostic directory and never merge it into production Phase 1 outputs.

## Existing Phase 2 / intraday 7-arm specification

39 trading days:

- 2026-01-02 through 2026-02-27;
- 5-minute evaluation grid;
- minimum community size 20;
- arms A, B, C, D9, D11, D13, D15.

Current arm definitions are in `scripts/run_intraday_7arm_5m.py`.

Existing result warning:

- D9 wins 12/13 layers mechanically because dormant/revival permissions increase structural persistence;
- this is not an economic optimum;
- matched controls, identity drift, core retention, member turnover, false-revival, split/merge and forward returns remain required.

After Phase 1 is rebuilt, Phase 2 must be rerun because all ReturnCorr community identities and consensus themes may change.

At minimum rerun:

- full `Layer × Arm` intraday table;
- raw vs market-residual vs cross-sectional-residual ReturnCorr comparison;
- S15/S30/S60;
- revival and confirmed-revival rates;
- core-retention/member-turnover;
- sector/industry purity once metadata is joined.

## Required execution order

### Step 0 — protect current outputs

Before any rebuild:

1. copy or rename the current graph store and Phase 1 outputs;
2. record source fingerprint, config hash, branch SHA and date coverage;
3. never run against the only copy of `Smoke_Test_Output/graph_store`;
4. use a new target such as:

```text
Smoke_Test_Output/graph_store_returncorr_v2
Smoke_Test_Output/theme_discovery_phase1_returncorr_v2
```

### Step 1 — validate source coverage

For every date, confirm SPY, QQQ and IWM exist in the PIT panel with enough return observations.

Produce a CSV with:

- date;
- snapshot count;
- benchmark symbol;
- available points;
- missing ratio;
- fallback-used flag.

If all benchmarks are missing, stop and fix source/universe handling. Do not silently accept median fallback for the entire run.

### Step 2 — smoke test one date

Use 2026-01-02 first.

Build Phase 0 and verify:

- layer IDs 1, 14 and 15 are present;
- no NaN/inf edge weights;
- no self edges;
- min-overlap/window-points rules hold;
- symbol IDs match the existing 5,590-symbol dimension mapping;
- raw and residual layers are not byte-identical;
- benchmark residual variance is materially reduced;
- edge count, giant-component share and community-size distribution are plausible.

Run Phase 1 for the same date and verify:

- all three ReturnCorr layer communities exist;
- only market-residual ReturnCorr appears as price support in consensus themes;
- raw and cross-sectional layers remain available for diagnostics;
- no duplicate price-family voting.

### Step 3 — decide patch versus full Phase 0 rerun

The current `CanonicalGraphStore.open_day()` deletes the complete daily directory. Therefore it is unsafe for a layer-only patch.

Patch mode is allowed only after implementing an atomic merge workflow:

1. build layers 1/14/15 into a temporary daily directory;
2. read existing `edges`, `node_features`, `snapshots`;
3. remove only layer IDs 1, 14 and 15 from the old tables;
4. append rebuilt rows;
5. preserve labels unchanged;
6. validate schemas and row counts;
7. write temporary parquet files;
8. atomically replace the originals;
9. refresh `dimensions/layers.parquet`, DuckDB views and manifest;
10. record patch history and hashes.

Do not update multiplex layer 0 in an incomplete layer patch. Phase 1 ignores layer 0, but any downstream multiplex analysis would be stale. Either:

- rebuild all layers and multiplex; or
- explicitly mark multiplex stale/disabled in the manifest until a full rebuild.

Decision rule:

- preferred fastest safe path: atomic patch of layer 1 plus new layers 14/15, with multiplex marked stale;
- preferred production-complete path: full Phase 0 rerun for the 39 dates;
- if atomic merge is not fully tested, use the full rerun. Data integrity has priority over runtime.

### Step 4 — complete January-February Phase 0

Process all 39 dates with resume/checkpoint support.

For each date write a success marker only after:

- parquet metadata is readable;
- all required layers are present;
- row-count and uniqueness checks pass;
- the date manifest is updated.

### Step 5 — compare ReturnCorr variants before Phase 1 full run

Create a report for raw, market-residual and cross-sectional-residual layers with:

- active nodes;
- edge count and density;
- connected components;
- giant-component share;
- mean/median degree;
- clustering/transitivity;
- assortativity;
- Leiden modularity;
- number of communities;
- largest-community share;
- Top-5 share;
- community-size entropy/Gini;
- sector and industry purity;
- SPY/QQQ/IWM beta remaining in community returns;
- degree-distribution model comparison.

Do not call a network power-law based on log-log OLS. Use MLE/KS/bootstrap and compare power law against lognormal, exponential and truncated power law.

### Step 6 — rerun Phase 1

Full Phase 1 rerun is required for all 39 rebuilt dates because consensus themes use the new market-residual price layer.

Use date-level checkpoints. A date is complete only when all expected parquet outputs and `_SUCCESS` exist.

Do not overwrite the old Phase 1 directory.

### Step 7 — rerun Phase 2

Run `scripts/run_intraday_7arm_5m.py` against the new Phase 1 root.

Add or preserve these outputs:

- `daily_layer_arm.csv`
- `layer_arm_summary.csv`
- S15/S30/S60 matrices;
- revival matrices;
- recommended arm by layer;
- raw vs residual ReturnCorr arm comparison.

Then shortlist C, D9 and D11 for a 1-minute run with matched controls and identity-drift penalties.

## Acceptance criteria

The work is not complete until all conditions below are met:

1. all 39 dates have valid Phase 0 outputs;
2. raw and two residual ReturnCorr layers are present;
3. benchmark coverage/fallback report is complete;
4. market-residual communities show better sector/industry purity or a clearly documented negative result;
5. full Phase 1 has been rerun for all 39 dates;
6. consensus does not triple-count the price family;
7. Phase 2 7-arm tables are regenerated from the new Phase 1 outputs;
8. old and new outputs remain separately reproducible;
9. tests pass;
10. all code, reports and manifests are committed and pushed to the same branch.

## Exact instruction to the local agent

Work continuously until the acceptance criteria are met. Do not stop after planning, a one-day smoke test, or a partial January run. Prefer the atomic ReturnCorr patch only if it is fully validated and preserves all non-ReturnCorr rows exactly. Otherwise run the complete 39-day Phase 0 rebuild. After Phase 0, rerun full Phase 1 because the consensus price input changed, then rerun Phase 2 and regenerate the Layer × Arm tables. Commit and push every completed, tested implementation and final report to `pttangent/GraphFactorFactory`, branch `codex/intraday-7arm-layer-analysis`.
