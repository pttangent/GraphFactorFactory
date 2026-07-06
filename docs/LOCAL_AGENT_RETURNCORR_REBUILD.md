# Local agent: ReturnCorr rebuild and rerun instructions

## Target

Repository: `pttangent/GraphFactorFactory`

Branch: `codex/intraday-7arm-layer-analysis`

Complete the January-February 2026 ReturnCorr rebuild, rerun full Phase 1, then regenerate Phase 2. Runtime matters, but data integrity has priority.

## Read before running

GraphFactorFactory:

- `docs/HANDOFF_INTRADAY_7ARM_39DAY.md`
- `docs/INTRADAY_7ARM_39DAY_REPORT.md`
- `src/graphfactorfactory/domain/layers.py`
- `src/graphfactorfactory/domain/config.py`
- `src/graphfactorfactory/application/math_utils.py`
- `src/graphfactorfactory/application/correlation.py`
- `src/graphfactorfactory/application/graph.py`
- `src/graphfactorfactory/application/pipeline.py`
- `src/graphfactorfactory/infrastructure/store.py`
- `src/graphfactorfactory/themes/consensus.py`
- `scripts/rebuild_return_corr_patch.py`
- `scripts/run_phase0.py`
- `scripts/run_intraday_7arm_5m.py`
- `artifacts/intraday_7arm_39day/recommended_arm_by_layer.csv`

Reference repository `pttangent/StockNet_V2`, branch `main`:

- `src/stocknetv2/domain/graph/return_corr.py`
- `src/stocknetv2/domain/graph/series_utils.py`
- `src/stocknetv2/domain/graph/layer_config.py`

Source repository `pttangent/US-Stock`:

- active NodeFactorFactory MonthPack implementation;
- verify `ret_5m`, `timestamp`, `available_time`, SPY/QQQ/IWM coverage, and the exact local source path.

## Phase 0 changes already on this branch

ReturnCorr now has three layers:

- layer 1 `return_corr`: raw returns, retained for macro/market-mode diagnostics;
- layer 14 `return_corr_market_residual`: intercept plus available SPY/QQQ/IWM regression residuals;
- layer 15 `return_corr_cross_sectional_residual`: per-timestamp cross-sectional median residuals.

All three ReturnCorr layers now use exact all-pairs correlation-equivalent scores followed by minimum similarity, per-symbol top-k, reciprocal top-k, and degree cap. Non-ReturnCorr layers still use reciprocal LSH.

Only layer 14 may vote as the price family in Phase 1 consensus. Layers 1 and 15 remain visible diagnostics but are excluded from consensus to prevent triple counting.

Stable node mapping is mandatory. Always reuse:

`Smoke_Test_Output/graph_store/dimensions/symbols.parquet`

Expected January-February universe: 5,590 symbols. Do not regenerate per-date symbol IDs.

## Fastest safe Phase 0 path

Protect the current store first. Work on a copy such as:

- `Smoke_Test_Output/graph_store_returncorr_v2`
- `Smoke_Test_Output/theme_discovery_phase1_returncorr_v2`

Run a one-day smoke test on `2026-01-02` with:

```powershell
python scripts/rebuild_return_corr_patch.py `
  --source-monthpack-root "C:\nodefactor_work\month_packs" `
  --graph-root "D:\DEV\US-Stock\Smoke_Test_Output\graph_store_returncorr_v2" `
  --config "configs\default.yaml" `
  --workers 8 `
  --dates 2026-01-02
```

The patch script rebuilds only layers 1, 14, and 15, keeps labels and all other layers, writes backups, performs atomic parquet replacement, refreshes layer dimensions and DuckDB views, and marks multiplex layer 0 stale.

Patch mode may continue for all 39 dates only after proving:

- symbols mapping is unchanged;
- only layers 1/14/15 changed;
- all other edge/node/snapshot rows are content-identical;
- labels are unchanged;
- no self edges or NaN/inf weights;
- all three ReturnCorr layers exist;
- raw and residual graphs are not identical;
- SPY/QQQ/IWM residual variance or beta is reduced;
- backups and rollback work.

If any check fails, stop patch mode and run a full 39-day Phase 0 rebuild. If multiplex is needed downstream, rebuild it fully rather than using the stale patched version.

## Required benchmark report

For each date and snapshot, report:

- SPY, QQQ, and IWM available points;
- missing ratio;
- fallback-used flag.

If all benchmarks are missing for most snapshots, fix source/universe handling. Do not silently accept median fallback for the full run.

## Phase 1 must be rerun

Full Phase 1 is required for every rebuilt date because ReturnCorr communities change, two new layers exist, and consensus now uses market-residual ReturnCorr as its price input.

A layer-only diagnostic may be written to a separate directory, but it cannot replace production Phase 1 because themes, lifecycle, semantics, and quality outputs may all change.

Expected Phase 1 outputs per date:

- `layer_communities.parquet`
- `subcommunities.parquet`
- `themes.parquet`
- `semantics.parquet`
- lifecycle/read-model outputs
- `_SUCCESS`

Do not overwrite the original Phase 1 directory.

## ReturnCorr comparison required before and after Phase 1

Compare raw, market-residual, and cross-sectional-residual ReturnCorr using:

- active nodes, edges, density;
- connected components and giant-component share;
- mean/median degree;
- clustering/transitivity and assortativity;
- Leiden modularity;
- community count, largest share, Top-5 share;
- community-size entropy and Gini;
- sector and industry purity;
- remaining SPY/QQQ/IWM beta;
- degree-distribution model comparison.

Do not claim power-law behavior from log-log OLS. Use MLE, KS/bootstrap, and compare power law with lognormal, exponential, and truncated power law.

## Phase 2 current specification

- 39 trading days: 2026-01-02 through 2026-02-27;
- 5-minute grid;
- minimum community size 20;
- arms A, B, C, D9, D11, D13, D15;
- runner: `scripts/run_intraday_7arm_5m.py`.

Existing Phase 2 results become obsolete after Phase 1 is rebuilt.

Regenerate:

- `daily_layer_arm.csv`
- `layer_arm_summary.csv`
- S15/S30/S60 matrices
- revival and confirmed-revival matrices
- recommended arm by layer
- raw vs residual ReturnCorr arm comparison
- core-retention and member-turnover tables

Do not select an arm only by raw survival. Penalize identity drift, member turnover, false revival, immediate post-revival death, and excessive split/merge. D9 often wins mechanically because revival permissions extend paths.

## Execution order

1. Pull the branch and record SHA, config hash, source path, and file hashes.
2. Copy old Phase 0 and Phase 1 roots.
3. Validate benchmark coverage.
4. Patch-smoke-test 2026-01-02.
5. Prove non-ReturnCorr preservation.
6. If safe, patch all 39 dates with resume/checkpoints; otherwise full rebuild.
7. Compare the three ReturnCorr variants.
8. Rerun full Phase 1 for all 39 dates.
9. Rerun Phase 2 and regenerate Layer by Arm tables.
10. Run tests, update reports/manifests, commit, and push.

## Completion rule

Do not stop after planning, code changes, one smoke-test day, or January only. Finish all 39 Phase 0 dates, full Phase 1, and regenerated Phase 2. Commit and push every tested implementation and final report to the same branch.
