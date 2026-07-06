# Phase 0 35-scale two-day continuous smoke

Branch: `codex/phase0-35-layer-scales`

## Execution

- Real one-minute NFF data for 2026-06-01 and 2026-06-02
- 09:30 through 10:29 ET
- 60 consecutive decision minutes per day
- All 35 layer/lookback configurations evaluated each minute
- 600-symbol high-coverage universe
- Six process workers
- One checkpoint set per decision minute

The run begins at the opening minute and therefore covers insufficient, partially mature, and mature windows. NFF was not rerun.

## Result

| Metric | Result |
|---|---:|
| Trading days | 2 |
| Minutes per day | 60 |
| Checkpoints | 120/120 |
| Configurations per minute | 35 |
| Total attempts | 4,200 |
| Mature attempts | 3,564 |
| Mature successful builds | 3,564 |
| Correct insufficient-point results | 636 |
| Runtime errors | 0 |
| Retained edges | 5,067,202 |
| Day 1 initial execution | 27.98 s |
| Day 2 initial execution | 28.17 s |

**Continuous technical gate: PASS.**

Every mature invocation built successfully. Before minimum support was available, the same path returned `insufficient_points` instead of crashing or emitting a false mature graph.

## Maturity behavior

| Lookback | First standard-layer eligibility |
|---:|---|
| 5m | 09:33 ET, minimum 3 observations |
| 15m | 09:38 ET, minimum 8 observations |
| 30m | 09:42 ET, minimum 12 observations |

Return-correlation configurations use their stricter minimums. Every record includes window points, minimum points, maturity state, status, edge count, and elapsed time.

## Resume and parallel validation

The runner writes snapshot, edge, and node Parquet shards plus one completion marker per minute. A restart schedules only incomplete minutes.

After all 60 first-day checkpoints existed, a second invocation recomputed zero minutes and completed report consolidation in about 3.6 seconds.

The runner uses `ProcessPoolExecutor` at decision-minute granularity. The existing production pipeline process-pool architecture remains in place.

## Runner

`scripts/run_phase0_35_continuous_smoke.py`

Example:

```text
python scripts/run_phase0_35_continuous_smoke.py --input-root NODE_FACTOR_ROOT --output OUTPUT_ROOT --dates 2026-06-01 2026-06-02 --workers 6 --universe-limit 600
```

Running the same command again resumes from existing minute checkpoints.

## Scope

This validates continuous time progression, opening warm-up behavior, all 35 layer-scale paths, point-in-time filtering, checkpoint recovery, and process parallelism. It is not an alpha or long-horizon stability test.