# Phase 0/1 2026H1 local optimization handoff

Updated: 2026-07-07 (Asia/Taipei)

## Scope and hard gate

The approved scope is:

1. Optimize and validate Phase 0 and Phase 1 on 2026-06-16.
2. Push the optimized implementation for human review.
3. Only after explicit approval, run Phase 0 for 2026-01 through 2026-06.
4. Run Phase 1 after all required Phase 0 dates are complete.
5. Do not run Phase 2.

The six-month run has **not** started.

## Repository and data

- Branch: `codex/phase0-35-layer-scales`
- Starting commit: `59429eaf46b96ff14c1598322be6e64d6c78e395`
- Working copy: `C:\Users\A001\.config\superpowers\worktrees\GraphFactorFactory\phase0-35-layer-scales`
- NFF MonthPack root: `D:\DEV\US-Stock\GraphFactorFactory\data\nodefactor\month_packs`
- Available months: `month=2026-01` through `month=2026-06`
- Available trading dates: 122 total (20, 19, 22, 21, 20, 20 by month)
- Total source size observed: 188.815 GiB

For the 35 layer-scale graph build, use `node_factors_1m`. The existing
`scripts/run_phase0.py` is not a valid production entry point for this run: it
loads `node_factors_5m`, starts a dashboard thread that can mutate Git state,
and automatically invokes the older theme pipeline.

## Bugs found and fixed locally

1. `decision_grid` used late `available_time` values and generated 81 decision
   times, including 20:05, 20:10, and 20:15 UTC. It is now capped at the RTH
   close and generates 78 five-minute decisions (13:35 through 20:00 UTC).
2. Edge, node, and snapshot Parquet schemas silently discarded layer-scale,
   parameter, and diagnostic fields emitted by `MultilayerGraphBuilder`.
   Schemas now retain these fields.
3. The Parquet writer now raises on undeclared columns instead of silently
   dropping new details.
4. Exact return-correlation graphs used the old per-node union cap and could
   exceed the cap at the opposite endpoint. They now use the same deterministic
   strict bilateral cap as LSH graphs.
5. Phase 0 used one static contiguous chunk per worker. Early-session chunks
   finished almost immediately while mature-session workers remained busy for
   hundreds of seconds. Decisions are now split into configurable small chunks
   (default 3), dynamically scheduled, and buffered for chronological writes.

## Reproducible single-day measurements

Common workload:

- Date: 2026-06-16
- Input: full `node_factors_1m` market data
- Universe after RTH filtering: 5,557 symbols
- Frequency: 5 minutes
- Config: `configs/phase0_ab_selected_v1.yaml`
- Labels disabled only for graph-stage profiling
- Task chunk size: 3 for optimized runs

| Run | Workers | Seconds | Edge rows | Notes |
|---|---:|---:|---:|---|
| Failed launch record | 4 | n/a | 0 | Windows multiline `python -c` quoting failed before work began |
| Original architecture | 4 | 460.603 | 14,646,993 | 81 times, lost scale fields, non-strict exact-correlation cap |
| Corrected architecture | 4 | 316.063 | 14,094,924 | 78 times, lossless schema, strict cap |
| Corrected architecture | 8 | 244.894 | 14,094,924 | Same output row counts |

Output roots:

- `D:\DEV\US-Stock\GraphFactorFactory\outputs\phase01_tuning\baseline_w4_20260707_003431`
- `D:\DEV\US-Stock\GraphFactorFactory\outputs\phase01_tuning\optimized_w4_20260707_004518`
- `D:\DEV\US-Stock\GraphFactorFactory\outputs\phase01_tuning\optimized_w8_20260707_005148`

The corrected 4-worker run was 31.4% faster than the original 4-worker run,
despite writing more schema detail and enforcing stricter correctness.

## Corrected Phase 0 QA (2026-06-16)

- Decision times: 78
- Universe: 5,557
- Edge rows: 14,094,924
- Snapshot/layer-scale rows with sufficient data: 2,687
- Node feature rows: 15,353,991
- Maximum actual degree: 6
- Degree-cap violating nodes: 0
- Duplicate undirected edges within `(time, layer, lookback)`: 0
- Non-finite weights: 0
- Self edges: 0
- Missing endpoints: 0
- Edge, node, and snapshot Parquet order: chronological
- Scale, graph parameters, and snapshot diagnostics: retained in Parquet

The 338,421-edge handoff figure is not reproducible with the full 5,557-symbol
universe. Earlier repository smoke documentation used a 600-symbol universe,
which can plausibly explain the difference. Do not silently reduce the universe
to match that historical count; first establish the intended universe contract.

## Tests

Latest local result before this handoff: `28 passed, 1 skipped`.

New regression coverage verifies:

- strict bilateral LSH cap;
- strict bilateral exact-correlation cap;
- inferred five-minute cadence;
- RTH-close decision-grid cap;
- preservation of scale/parameter/QA schema fields;
- rejection of silent Parquet detail loss;
- small deterministic Phase 0 task partitioning.

## Required next work, in order

1. Run corrected 12-worker and 16-worker sweeps on the identical 2026-06-16
   graph workload. Select the lowest worker count near peak stable throughput;
   keep at least 20 GiB memory headroom. Do not guess 24 workers.
2. Add a dedicated `scripts/run_phase0_production.py` using `node_factors_1m`,
   five-minute decisions, configurable workers/chunk size, new output roots,
   atomic per-date `running/complete/failed` markers, and complete-date resume.
3. Make Phase 0 daily promotion atomic. The current `CanonicalGraphStore.open_day`
   deletes and writes the final date directory in place and is not sufficient
   for a six-month resumable production run.
4. Persist per-date source fingerprint, config hash, parameter-set ID, row counts,
   resource measurements, and the QA invariants listed above.
5. Fix Phase 1 before running it:
   - `production_worker.detect_snapshot` currently groups only by `layer_id` and
     would mix multiple lookbacks; group by the full layer-scale identity.
   - `run_phase01_production.py` writes `running` and `complete` but does not write
     an atomic `failed` marker on exceptions.
   - Confirm that lifecycle assignment is excluded from Phase 1 output, because
     the governing requirement assigns continuation/split/merge/birth/death to
     Phase 2.
6. Run Phase 0 and Phase 1 once for 2026-06-16 with the production runners,
   validate all output contracts, then run each a second time to prove complete
   dates are skipped.
7. Commit and push that complete single-day implementation for human review.
8. Only after explicit approval, start six-month Phase 0 in a new output root.
   After all dates pass, start Phase 1 in another new output root. Do not start
   Phase 2.

## Monitoring contract for the eventual long run

- First hour: sample every five minutes.
- After one healthy hour: sample hourly.
- Record PID/job identity, elapsed time, completed dates, throughput, total and
  per-process CPU, RSS, available memory, disk throughput/free space, retries,
  and heartbeat age.
- Scale up only after several safe samples with ready work and low CPU; scale
  down immediately on memory pressure, swap growth, falling throughput, or
  repeated failures.
- A live PID or growing `.part` file is not completion. Only validated atomic
  `complete` markers permit resume skipping.
