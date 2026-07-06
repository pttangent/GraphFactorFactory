# Local Agent Handoff: ReturnCorr Graph Rebuild, Phase 1 Rebuild, and Phase 2 Re-evaluation

## 1. Goal

Rebuild the January-February 2026 ReturnCorr graph family so that GraphFactorFactory can recover financially meaningful industry/theme communities instead of mixing broad market beta with local co-movement.

The local agent must complete all of the following before stopping:

1. rebuild Phase 0 ReturnCorr layers for all 39 trading dates;
2. rerun full Phase 1 for all 39 dates;
3. rerun the intraday 7-arm Phase 2 analysis;
4. compare raw and residual ReturnCorr network quality and sector/industry purity;
5. commit and push code, QA reports, manifests, and final findings to the same branch.

Do not stop after planning, a one-date smoke test, or a partial month.

## 2. Repositories and branch

Primary repository:

- `pttangent/GraphFactorFactory`
- branch: `codex/intraday-7arm-layer-analysis`

Reference repositories that must be read:

### StockNet_V2

Repository: `pttangent/StockNet_V2`, branch `main`.

Read at minimum:

- `src/stocknetv2/domain/graph/return_corr.py`
- `src/stocknetv2/domain/graph/series_utils.py`
- `src/stocknetv2/domain/graph/layer_config.py`
- `tests/test_return_corr_graph.py`
- graph evaluation and community evaluation code

Important StockNet behavior:

- remove the per-timestamp broad cross-sectional baseline before correlation;
- calculate actual pairwise correlations with overlap checks;
- apply top-k/reciprocal-top-k/degree-cap only after correlation;
- do not infer industry structure from raw market beta alone.

### US-Stock

Repository: `pttangent/US-Stock`.

Read the current NodeFactorFactory implementation and verify:

- `ret_5m` calculation and point-in-time semantics;
- `timestamp` and `available_time` semantics;
- SPY, QQQ, and IWM coverage;
- the exact source root used for the current `Smoke_Test_Output/graph_store`;
- symbol metadata and sector/industry metadata sources.

## 3. Mandatory GraphFactorFactory files

Read before execution:

- `src/graphfactorfactory/domain/layers.py`
- `src/graphfactorfactory/domain/config.py`
- `src/graphfactorfactory/application/math_utils.py`
- `src/graphfactorfactory/application/graph.py`
- `src/graphfactorfactory/application/pipeline.py`
- `src/graphfactorfactory/application/return_corr_patch.py`
- `src/graphfactorfactory/infrastructure/store.py`
- `src/graphfactorfactory/themes/pipeline.py`
- `src/graphfactorfactory/themes/consensus.py`
- `src/graphfactorfactory/interfaces/cli.py`
- `scripts/run_return_corr_patch_range.py`
- `scripts/validate_stable_universe_for_full_rebuild.py`
- `scripts/run_intraday_7arm_5m.py`
- `scripts/run_intraday_7arm_5m_local.py`
- `docs/HANDOFF_INTRADAY_7ARM_39DAY.md`
- `docs/INTRADAY_7ARM_39DAY_REPORT.md`
- `artifacts/intraday_7arm_39day/recommended_arm_by_layer.csv`

## 4. Code already implemented on the branch

### 4.1 Phase 0 layer registry

The layer registry now contains 15 layers.

ReturnCorr family:

- layer 1: `return_corr`
  - transform: `return_corr_raw`
  - retained for backward compatibility and market/macro co-movement;
- layer 14: `return_corr_market_residual`
  - regress each stock on available SPY/QQQ/IWM returns;
  - use the residual trajectory for graph construction;
- layer 15: `return_corr_cross_sectional_residual`
  - remove the per-timestamp cross-sectional median;
  - acts as the StockNet-compatible deterministic reference/fallback.

Do not overwrite or rename layer 1. Raw and residual graphs answer different financial questions.

### 4.2 ReturnCorr configuration

`BuildConfig` and `configs/default.yaml` now contain:

- `return_corr_benchmarks: [SPY, QQQ, IWM]`
- `return_corr_min_benchmark_points: 8`
- `return_corr_ridge: 1e-6`

These fields are included in the config hash.

### 4.3 Benchmark residualization correction

`application/math_utils.py` now requires real, non-null benchmark observations inside each graph window.

A benchmark is eligible only when:

- it exists in the stable universe;
- it has at least `return_corr_min_benchmark_points` real observations;
- its variance is non-zero.

Missing benchmark columns filled by a cross-sectional median are no longer falsely treated as real benchmark data.

The regression includes an intercept. Ridge regularization is applied to benchmark coefficients but not to the intercept.

If no benchmark is valid in a window, the code uses the cross-sectional-median residual fallback.

The date-level patch pipeline stops if all configured benchmarks are unavailable for the entire date.

### 4.4 Selective graph construction

`MultilayerGraphBuilder` now accepts:

- an explicit tuple of layers;
- `include_multiplex=False`.

This allows rebuilding only layers 1, 14, and 15 without spending time on the other 12 layers.

The normal full Phase 0 path remains unchanged and still builds all layers plus multiplex layer 0.

### 4.5 Atomic ReturnCorr patch pipeline

New module:

- `src/graphfactorfactory/application/return_corr_patch.py`

New CLI command:

```text
graphfactorfactory patch-return-corr-date
```

The patch pipeline:

1. requires separate source and output graph stores;
2. copies the baseline date into the output store;
3. reuses the existing `dimensions/symbols.parquet` mapping exactly;
4. rebuilds only layers 1/14/15;
5. removes only layers 1/14/15 from old edges, node features, and snapshots;
6. appends rebuilt rows;
7. validates duplicate keys and required layer presence;
8. writes temporary parquet files and atomically replaces the output files;
9. verifies `labels.parquet` SHA-256 is unchanged;
10. updates `dimensions/layers.parquet`, DuckDB views, manifest, patch history, and benchmark coverage report.

The output manifest marks layer 0 as:

```text
stale_disabled_until_full_rebuild
```

The patch does not rebuild multiplex layer 0. Phase 1 explicitly skips layer 0, so this is acceptable for the required Phase 1/2 rerun. Any downstream multiplex analysis must remain disabled until a complete all-layer Phase 0 rebuild.

### 4.6 Resume-safe 39-date runner

New script:

- `scripts/run_return_corr_patch_range.py`

It discovers dates from the baseline graph store, writes one success marker per date, and resumes completed dates.

Use one date process at a time. Use snapshot-level workers within the date. Do not use parallel date writers against the same output manifest.

### 4.7 Full rebuild universe safety

Current legacy `GraphFactorPipeline.build_date()` can reduce the supplied universe to the symbols present on a date and regenerate integer IDs. This is unsafe if a full rebuild date is missing any symbol from the existing 5,590-symbol dimension.

New validator:

- `scripts/validate_stable_universe_for_full_rebuild.py`

It blocks the full rebuild fallback if any date would change the existing symbol mapping.

Therefore:

- preferred path: atomic ReturnCorr patch;
- full rebuild fallback: allowed only after the validator passes for every date, or after stable-universe handling in the full pipeline is explicitly fixed and tested.

Never run a full fallback that silently changes `symbol_id`.

### 4.8 Phase 1 consensus behavior

All three ReturnCorr layers remain visible as independent layer communities.

Consensus behavior:

- `return_corr_market_residual` can vote as the price-family layer;
- raw `return_corr` is excluded from consensus voting;
- `return_corr_cross_sectional_residual` is excluded from consensus voting.

This prevents the price family from being counted three times.

### 4.9 Phase 2 current status

The existing 39-day report is a pre-residual baseline.

Current Phase 2 specification:

- dates: 2026-01-02 through 2026-02-27;
- 39 trading days;
- input: Phase 1 `layer_communities.parquet`;
- 5-minute evaluation grid from every fifth one-minute snapshot;
- minimum community size: 20;
- arms: A, B, C, D9, D11, D13, D15;
- metrics: confirmed paths, median life, S5/S15/S30/S60, revival rate, confirmed revival rate.

The old result selected D9 for most layers because D9 mechanically permits dormant recovery. This is not an economic optimum.

The old report has been marked as a baseline that must be regenerated.

A local path wrapper is available:

- `scripts/run_intraday_7arm_5m_local.py`

## 5. Why Phase 1 must be rerun

A production Phase 1 rerun is mandatory for every rebuilt date.

Reasons:

1. ReturnCorr edges change;
2. layer IDs 14 and 15 are newly available;
3. layer communities and subcommunities change;
4. market-residual ReturnCorr now participates in consensus;
5. raw and cross-sectional ReturnCorr are excluded from consensus;
6. theme identities, lifecycle, semantics, and quality scores can change.

A ReturnCorr-only community diagnostic is allowed only as a temporary smoke test in a separate directory.

It is not sufficient for production because it cannot correctly rebuild:

- `themes.parquet`;
- lifecycle outputs;
- `semantics.parquet`;
- quality outputs;
- read models.

Therefore the production instruction is:

> Patch Phase 0 ReturnCorr only, then rerun full Phase 1 for every patched date.

## 6. Output roots

Never overwrite the current outputs.

Recommended roots:

```text
Smoke_Test_Output/graph_store
Smoke_Test_Output/graph_store_returncorr_v2
Smoke_Test_Output/theme_discovery_phase1
Smoke_Test_Output/theme_discovery_phase1_returncorr_v2
Smoke_Test_Output/intraday_7arm_returncorr_v2
```

The old roots are the baseline. The `_returncorr_v2` roots are the rebuilt outputs.

## 7. Local setup

Example PowerShell setup:

```powershell
git clone https://github.com/pttangent/GraphFactorFactory.git
cd GraphFactorFactory
git fetch origin
git checkout codex/intraday-7arm-layer-analysis
git pull --ff-only origin codex/intraday-7arm-layer-analysis

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev,themes,evaluation]"
pytest -q
```

Also clone/read the two reference repositories:

```powershell
git clone https://github.com/pttangent/StockNet_V2.git
git clone https://github.com/pttangent/US-Stock.git
```

Define local paths. Replace the examples with the actual machine paths:

```powershell
$NODE_FACTORS = "D:\DEV\US-Stock\Smoke_Test_Output\node_factors"
$OLD_GRAPH = "D:\DEV\US-Stock\Smoke_Test_Output\graph_store"
$NEW_GRAPH = "D:\DEV\US-Stock\Smoke_Test_Output\graph_store_returncorr_v2"
$OLD_PHASE1 = "D:\DEV\US-Stock\Smoke_Test_Output\theme_discovery_phase1"
$NEW_PHASE1 = "D:\DEV\US-Stock\Smoke_Test_Output\theme_discovery_phase1_returncorr_v2"
$PHASE2_OUT = "D:\DEV\US-Stock\Smoke_Test_Output\intraday_7arm_returncorr_v2"
$CONFIG = ".\configs\default.yaml"
$METADATA = "D:\DEV\US-Stock\metadata\symbol_metadata.csv"
```

Verify these paths from the actual local project. Do not assume the examples are correct.

## 8. Execution order

### Step 1: protect and fingerprint the baseline

Record:

- current branch SHA;
- source fingerprint;
- old config hash;
- dates present;
- row counts by date;
- SHA-256 for dimensions files and representative daily files.

Do not modify `$OLD_GRAPH`.

### Step 2: run tests

```powershell
pytest -q
```

Tests must include:

- layer registry tests;
- benchmark residual tests;
- atomic merge tests;
- existing Phase 0 and Phase 1 tests.

If a test fails, fix and push before running 39 dates.

### Step 3: one-date Phase 0 patch smoke test

Use 2026-01-02 first:

```powershell
graphfactorfactory patch-return-corr-date `
  --node-factors $NODE_FACTORS `
  --date 2026-01-02 `
  --source-graph-store $OLD_GRAPH `
  --output-graph-store $NEW_GRAPH `
  --config $CONFIG `
  --max-workers 26
```

Smoke-test acceptance:

- layer IDs 1, 14, and 15 exist in edges, node features, and snapshots;
- non-ReturnCorr layer row counts equal the baseline exactly;
- labels SHA-256 equals the baseline exactly;
- no duplicate edge/node/snapshot keys;
- no NaN/inf weights;
- no self edges;
- raw and residual layers are not identical;
- SPY/QQQ/IWM coverage report is present;
- all-benchmark fallback did not occur;
- manifest marks multiplex layer 0 stale;
- DuckDB views open successfully.

### Step 4: run one-date Phase 1 smoke test

Use a separate output directory:

```powershell
graphfactorfactory-theme `
  --graph-store $NEW_GRAPH `
  --theme-store $NEW_PHASE1 `
  --metadata-csv $METADATA `
  --date-start 2026-01-02 `
  --date-end 2026-01-02 `
  --run-id returncorr_v2 `
  --frame-minutes 1 `
  --min-consensus-score 0.35 `
  --min-distinct-families 2 `
  --min-overlap 0.5
```

Verify:

- `layer_communities.parquet` contains `return_corr`, `return_corr_market_residual`, and `return_corr_cross_sectional_residual`;
- only market-residual ReturnCorr contributes to price-family consensus support;
- layer 0 is absent from community detection;
- outputs are readable and `_SUCCESS` is written;
- expected one-minute snapshot coverage is preserved.

### Step 5: run the 39-date Phase 0 patch

```powershell
python .\scripts\run_return_corr_patch_range.py `
  --node-factors $NODE_FACTORS `
  --source-graph-store $OLD_GRAPH `
  --output-graph-store $NEW_GRAPH `
  --config $CONFIG `
  --start 2026-01-02 `
  --end 2026-02-27 `
  --max-workers 26 `
  --date-workers 1
```

The script is resume-safe. Restart the same command after interruption.

Do not use multiple date writers against one output root unless manifest updates are made process-safe first.

### Step 6: Phase 0 QA for all dates

Produce a table by date and ReturnCorr variant with:

- snapshot count;
- benchmark points and fallback flags;
- active nodes;
- edge count;
- edge density;
- connected components;
- giant-component share;
- mean and median degree;
- clustering/transitivity;
- assortativity;
- Leiden modularity;
- community count;
- largest-community share;
- Top-5 share;
- community-size entropy and Gini;
- sector purity;
- industry purity;
- remaining SPY/QQQ/IWM beta;
- degree-distribution model comparison.

Do not claim a power law from log-log OLS. Use MLE, KS, bootstrap, and comparison against lognormal, exponential, and truncated power law.

### Step 7: full Phase 1 rerun for all 39 dates

Delete only the incomplete `_returncorr_v2` Phase 1 output if restarting from scratch. Never delete the baseline.

```powershell
graphfactorfactory-theme `
  --graph-store $NEW_GRAPH `
  --theme-store $NEW_PHASE1 `
  --metadata-csv $METADATA `
  --date-start 2026-01-02 `
  --date-end 2026-02-27 `
  --run-id returncorr_v2 `
  --frame-minutes 1 `
  --min-consensus-score 0.35 `
  --min-distinct-families 2 `
  --min-overlap 0.5
```

Use the same Phase 1 settings as the baseline unless a tested defect requires a change. This run is intended to isolate the ReturnCorr logic change, not tune every Phase 1 parameter simultaneously.

A date is complete only when all expected parquet files and success markers exist.

### Step 8: rerun Phase 2

```powershell
python .\scripts\run_intraday_7arm_5m_local.py `
  --phase1-root $NEW_PHASE1 `
  --output $PHASE2_OUT `
  --min-size 20 `
  --sample-every 5
```

Required outputs:

- `daily_layer_arm.csv`;
- `layer_arm_summary.csv`;
- `s5_matrix.csv`;
- `s15_matrix.csv`;
- `s30_matrix.csv`;
- `s60_matrix.csv`;
- `revival_rate_matrix.csv`;
- `confirmed_revival_rate_matrix.csv`;
- `median_life_matrix.csv`;
- `recommended_arm_by_layer.csv`.

The new comparison must show separately:

- `return_corr`;
- `return_corr_market_residual`;
- `return_corr_cross_sectional_residual`.

Do not automatically accept D9 because it has the highest raw persistence. Report:

- raw persistence winner;
- conservative winner after false-revival penalty;
- coverage/path-count trade-off;
- whether different layers prefer different arms.

After the 5-minute run, shortlist C, D9, and D11 for a one-minute controlled run.

### Step 9: compare old versus new

Compare baseline and rebuilt outputs for:

- ReturnCorr sector/industry purity;
- market beta contamination;
- community size distribution;
- modularity with component diagnostics;
- persistence and revival by arm;
- consensus theme count and family support;
- representative communities with concrete tickers.

A lower modularity is not automatically worse. Extremely high modularity can result from graph fragmentation caused by reciprocal top-k and degree caps.

The primary success criterion is clearer financially meaningful communities with controlled market beta, not maximizing modularity.

## 9. Patch versus full rebuild decision

### Preferred path

Use the atomic ReturnCorr patch when the one-date smoke test passes.

This is the minimum-runtime path because it rebuilds only three layers and preserves:

- the other 12 layers;
- labels;
- existing symbol mapping;
- existing date coverage.

### Full Phase 0 fallback

Use a complete Phase 0 rebuild only when:

- atomic merge tests fail;
- old daily parquet schemas are incompatible;
- non-ReturnCorr rows cannot be proven unchanged;
- multiplex layer 0 is required immediately for downstream analysis.

Before a full fallback, run:

```powershell
python .\scripts\validate_stable_universe_for_full_rebuild.py `
  --node-factors $NODE_FACTORS `
  --symbols-parquet "$OLD_GRAPH\dimensions\symbols.parquet" `
  --source-graph-store $OLD_GRAPH `
  --start 2026-01-02 `
  --end 2026-02-27 `
  --output ".\artifacts\full_rebuild_universe_validation.csv"
```

If any date is unsafe, do not use the current full rebuild path. Fix stable-universe handling first or return to the patch path.

## 10. Phase 1 rerun decision

The decision is already resolved:

- diagnostic layer-only Phase 1: optional for the one-date smoke test;
- production Phase 1: full rerun required for every patched date.

The production Phase 1 rerun cannot be skipped because consensus inputs changed.

## 11. Final acceptance criteria

The task is complete only when:

1. all 39 dates have successful ReturnCorr Phase 0 patches or safe full rebuilds;
2. layers 1/14/15 exist for every date;
3. benchmark coverage report covers all dates and benchmarks;
4. non-ReturnCorr rows are preserved exactly in patch mode;
5. labels are unchanged in patch mode;
6. symbol mapping remains identical to the existing 5,590-symbol mapping;
7. full Phase 1 is complete for all 39 dates;
8. consensus does not triple-count the price family;
9. Phase 2 tables are regenerated from the rebuilt Phase 1 root;
10. raw and residual ReturnCorr are compared for sector/industry purity and market beta;
11. network metrics and proper degree-distribution tests are produced;
12. tests pass;
13. code, reports, QA artifacts, manifests, and final conclusions are committed and pushed to `codex/intraday-7arm-layer-analysis`.

## 12. Required final Git actions

Before finishing:

```powershell
git status --short
pytest -q
git add src scripts tests configs docs artifacts
git commit -m "complete ReturnCorr rebuild and rerun analysis"
git push origin codex/intraday-7arm-layer-analysis
```

Do not commit raw graph stores or large Phase 1 parquet outputs unless the repository policy explicitly requires them. Commit compact QA tables, reports, manifests, and reproducible scripts.

## 13. Exact goal for the local agent

> Complete the full January-February 2026 ReturnCorr rebuild workflow on `pttangent/GraphFactorFactory`, branch `codex/intraday-7arm-layer-analysis`. First validate and use the atomic layer 1/14/15 Phase 0 patch for minimum runtime; preserve every non-ReturnCorr row, label, and symbol ID exactly. If the patch is not safe, use a full Phase 0 rebuild only after stable-universe validation or after fixing the full rebuild mapping logic. Then rerun full Phase 1 for all 39 dates because consensus now uses market-residual ReturnCorr, rerun Phase 2 7-arm analysis, compare raw versus residual ReturnCorr financial and network quality, update all reports, run tests, commit, and push. Do not stop at planning, a smoke test, or partial date coverage.
