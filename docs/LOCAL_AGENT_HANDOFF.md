# Local Agent Handoff: ReturnCorr rebuild

Work in `pttangent/GraphFactorFactory`, branch `codex/intraday-7arm-layer-analysis`.

## Read first

GraphFactorFactory:
- `README.md`
- `docs/pipeline.md`
- `docs/storage_design.md`
- `docs/HANDOFF_INTRADAY_7ARM_39DAY.md`
- `docs/INTRADAY_7ARM_39DAY_REPORT.md`
- `src/graphfactorfactory/domain/layers.py`
- `src/graphfactorfactory/domain/config.py`
- `src/graphfactorfactory/application/math_utils.py`
- `src/graphfactorfactory/application/correlation.py`
- `src/graphfactorfactory/application/graph.py`
- `src/graphfactorfactory/application/pipeline.py`
- `src/graphfactorfactory/application/return_corr_patch.py`
- `src/graphfactorfactory/interfaces/cli.py`
- `src/graphfactorfactory/themes/pipeline.py`
- `src/graphfactorfactory/themes/community.py`
- `src/graphfactorfactory/themes/store.py`
- `src/graphfactorfactory/themes/temporal.py`

Reference repository: `pttangent/StockNet_V2`, main branch. Review its ReturnCorr layer configuration, common-timestamp trajectory construction, correlation graph, sparsification, and community discovery. Use it to verify semantics, not to copy blindly.

## Current Phase 0 specification

The branch preserves raw ReturnCorr and adds two residual layers:
- Layer 1: `return_corr`, transform `return_corr_raw`
- Layer 14: `return_corr_market_residual`
- Layer 15: `return_corr_cross_sectional_residual`

Market residual uses available SPY, QQQ, and IWM 5-minute trajectories with intercept and ridge regression. If benchmark coverage is inadequate, it falls back to cross-sectional median residualization and records QA.

ReturnCorr uses a common timestamp pivot. It no longer uses the generic LSH candidate path. It computes an exact Pearson-equivalent correlation matrix, then applies minimum similarity, top-k, reciprocal filtering, and symmetric degree cap. This is intended to restore StockNet-like industry structure and avoid LSH fragmentation.

The command `patch-return-corr-date` rebuilds only layers 1, 14, and 15 into a separate output graph store. It atomically replaces ReturnCorr edge, node, and snapshot rows; verifies that labels are unchanged; updates layer dimensions and benchmark coverage; and records patch metadata.

The patch does not rebuild Layer 0 multiplex. The manifest marks multiplex as stale. Phase 1 must continue to ignore Layer 0. Any multiplex consumer requires a later full Phase 0 rebuild.

## Phase 1 decision

Phase 1 must be rerun. The old `layer_communities.parquet` files are invalid for the patched graph store because:
- Layer 1 edges change;
- Layers 14 and 15 are new;
- Leiden partitions and downstream consensus inputs therefore change.

The fastest safe route is:
1. Patch only ReturnCorr in Phase 0.
2. Rerun Phase 1 for the target 39 dates into a new output root.

A Phase 1 layer-only patch may be used only after proving that replacing layers 1, 14, and 15 is identical to a full same-day Phase 1 rerun. It must use atomic replacement, uniqueness checks, deletion of old affected rows before merge, and content hashes showing untouched layers are unchanged. Until that proof exists, full Phase 1 date reruns are mandatory.

## Phase 2 decision

Phase 2 includes cross-layer consensus, semantic labels, quality scoring, lifecycle, and later path tracking. It must be rerun after Phase 1 because residual layers change family support, consensus membership, quality, and identity continuity.

Run Phase 2 chronologically from the earliest date. Do not splice old lifecycle records onto new communities. Per-snapshot Leiden work may be parallelized, but lifecycle and path state must preserve deterministic time order.

## Required execution order

1. Run the test suite.
2. Validate one date, starting with 2026-01-02.
3. Confirm layers 1, 14, and 15 exist; benchmark coverage is valid; labels are unchanged; non-ReturnCorr layers are unchanged; no duplicate keys exist; multiplex is stale-marked.
4. Patch Phase 0 for all 39 dates from 2026-01-02 through 2026-02-27. Dates may run in parallel, but each date must write independently and atomically.
5. Rerun Phase 1 for all 39 dates into a new directory. Do not overwrite old results before QA.
6. Rerun Phase 2 from the earliest date in chronological order.
7. Rerun the intraday 7-arm analysis, matched controls, identity retention, false revival, and alpha validation using only the new outputs.

## Required QA and comparisons

Verify common timestamp alignment, benchmark fallback behavior, constant benchmark exclusion, exact correlation against hand-computed Pearson values, reciprocal top-k, degree cap, unchanged labels, unchanged untouched layers, unique edge/node/snapshot keys, stale multiplex marking, Layer 0 exclusion, and deterministic output under fixed seed.

Compare raw and residual ReturnCorr using community count, modularity, largest-community share, Top-5 share, sector purity, industry purity, entropy, S15/S30, core retention, market beta, and within-community minus outside-community correlation.

## Deliverables

Push code, tests, runners, QA CSVs, raw-versus-residual comparison, Excel workbook, Chinese report, run logs, config hash, source fingerprint, date coverage, and final conclusions back to the current branch.

Do not stop at a plan or one-day prototype. Do not mix overnight and intraday 7-arm analyses. Do not call structural persistence alpha.