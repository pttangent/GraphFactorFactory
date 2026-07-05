# Next Agent Handoff

## Repositories

Primary:
- `pttangent/GraphFactorFactory`
- branch: `phase1/realdata-60min-temporal-validation`

Reference lifecycle implementation:
- `pttangent/stocknet-research`
- branch: `main`
- inspect `src/stocknetwork/temporal_labels.py`
- inspect `src/stocknetwork/graph_snapshots.py`

## Relevant commits

- `6928fd439a305ccec1264dd7a9c42232f55e6993`: effective-state continuous tracker
- `0585da0c22f1aa3dca91067cb116634d488d0126`: tracker tests
- `a0e92f3f938121362dd5dc090da84f25d415acaa`: structural macro matcher
- `de67cbfeb6f2ed5aa4fde6885ccfc0d0cb45e87a`: structural matcher tests
- `1fa6f147bcb0a87bcc8ddddf8508d73ed8baf67c`: original macro matcher
- `4b11cf3686a2b955da2d694649c5868b58c86c95`: cross-day A/B notes

## Google Drive

Folder:
- name: `daily_cano_packs`
- id: `1CPAcuN714jjwEDkqyXFrVaSnaT5_U8Yr`

Daily files contain:
- `edges.parquet`
- `snapshots.parquet`
- `node_features.parquet`
- `labels.parquet`
- `row_counts.json`

Expected trading dates are 2026-01-02 through 2026-01-23, excluding weekends and 2026-01-19.

## Required work

### 1. One-minute intraday tracking

Re-download complete daily canonical files and reproduce A/B on at least three days.

Compare:
- `stocknet_j035`
- `overlap_c050`
- `hybrid_strict`
- `hybrid`

Use `graph_state_hash`. Repeated minute states must not increment lifecycle age.

### 2. Macro scales

Prioritize 5m and 15m. Add 30m and 60m as sensitivity tests.

Do not simply sample endpoints. Aggregate unique graph states and edge persistence inside each window, then run layer Leiden and sparse cross-family consensus.

### 3. Three-session tracking

Use a consecutive sequence such as:
- 2026-01-05 -> 2026-01-06 -> 2026-01-07
- or 2026-01-07 -> 2026-01-08 -> 2026-01-09

Do not reset the tracker at day boundaries. Mark close-to-open transitions as `overnight`.

## Metrics

Report:
- raw-minute continuation
- effective-state continuation
- path length distribution
- split and merge events
- overnight inheritance
- two-session and three-session survival
- shifted-time null
- member-permutation null
- day-order null

## Rules

- Metadata may be used only after discovery for naming and validation.
- Trust final Parquet metadata over stale `row_counts.json`.
- Save after every completed day and scale.
- Rename downloaded `edges.parquet` immediately with its date.
- Never claim cross-day results from incomplete daily data.
- Continuous tracking must process adjacent states chronologically.
