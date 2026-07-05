# Local Agent Handoff: 2025 Phase 2 Rebuild and Carry-over A/B

## Repository state

- Repository: `pttangent/GraphFactorFactory`
- Branch: `phase1/realdata-60min-temporal-validation`
- Do not return to the old smoke-run branch for production research.

The branch now separates the pipeline into immutable upstream work and restartable temporal work:

```text
Phase 0: graph generation
Phase 1: snapshot-local, same-snapshot community discovery
Phase 2: chronological same-layer temporal linking
Carry-over A/B: independent overnight boundary studies over Phase 1 communities
```

## Required invariant

Once 2025 Phase 0 and Phase 1 are complete, they must not be rerun merely because Phase 2 logic changes.

Phase 2 reads only:

```text
<phase1-root>/date=YYYY-MM-DD/layer_communities.parquet
```

Phase 2 serializes the final active state of every layer at the end of every trading day under:

```text
<phase2-root>/_state/date=YYYY-MM-DD/layer=<layer_name>/
```

The state includes active candidates and lifecycle records. This allows a later run to load the previous trading day's state and continue across day, month, and year boundaries.

## What changed

### Stateful Phase 2

`src/graphfactorfactory/themes/phase2_pipeline.py` now:

- reads `layer_communities.parquet`, not cross-layer consensus themes;
- tracks each `layer_name` independently;
- sorts by `layer_name`, `snapshot_time`, and `community_id` using stable ordering;
- links snapshots sequentially inside each layer;
- carries the last active state into the next trading day;
- writes day-level outputs and a separate serialized state checkpoint;
- supports invalidation and chronological rebuild from any date.

### Earliest-invalid launcher

`scripts/run_phase2_from_earliest.py` finds the first Phase 1 date where:

- Phase 2 output is missing;
- Phase 2 serialized state is missing; or
- `layer_communities.parquet` is newer than the Phase 2 success markers.

It then invokes Phase 2 with `--rebuild-from <earliest-date>` and recomputes that date and every later date in chronological order.

### Seven-arm carry-over worker

`scripts/run_monthly_carryover_task.py` now evaluates the configured arms rather than interpreting the digits in `D9/D11/D13/D15` as dormant duration.

The intended arms are:

- A: containment baseline;
- B: fingerprint-confirmed bridge;
- C: fingerprint-assisted short persistence;
- D9: base revival;
- D11: revival plus 10% breadth expansion;
- D13: fingerprint >= 0.22 plus breadth expansion;
- D15: fingerprint >= 0.30 plus breadth expansion and post-revival confirmation.

All bridge, continuation, control, and revival matching is restricted to the same `layer_id`.

### Cross-month orchestration

`scripts/run_monthly_carryover_ab.py`:

- discovers all Phase 1 dates in one sorted index;
- creates boundaries with `zip(dates[:-1], dates[1:])`;
- therefore includes month-end to next-month-start boundaries automatically;
- pre-generates deterministic null mappings shared by all arms;
- runs one subprocess per boundary;
- validates every expected arm/control `_SUCCESS` before marking a boundary successful.

Do not run each month as an isolated date list if cross-month boundaries are required. Use one date range covering the entire study period.

## First-time local setup

```powershell
git fetch origin
git checkout phase1/realdata-60min-temporal-validation
git pull --ff-only origin phase1/realdata-60min-temporal-validation
$env:PYTHONPATH="src"
python -m pip install -e .
```

## Step 1: verify Phase 1 coverage

Expected structure:

```text
outputs/theme_discovery_phase1/
  date=2025-01-02/layer_communities.parquet
  ...
  date=2025-12-31/layer_communities.parquet
```

Run a dry inspection and confirm dates are strictly sorted and there are no duplicate `snapshot_time/layer_id/community_id` rows.

Phase 1 is snapshot-local. It may use snapshot parallelism, but its canonicalized output must be identical for `max_workers=1` and the production worker count.

## Step 2: rebuild Phase 2 automatically from the earliest affected date

Dry run:

```powershell
python scripts/run_phase2_from_earliest.py `
  --phase1-root outputs/theme_discovery_phase1 `
  --phase2-root outputs/theme_temporal_phase2 `
  --dry-run
```

Execute:

```powershell
python scripts/run_phase2_from_earliest.py `
  --phase1-root outputs/theme_discovery_phase1 `
  --phase2-root outputs/theme_temporal_phase2
```

When 2025 is inserted before existing 2026 data, the earliest invalid date should be the first available 2025 trading date. The script must rebuild Phase 2 from that date through the latest Phase 1 date. Phase 0 and Phase 1 remain untouched.

Manual equivalent:

```powershell
python scripts/run_theme_temporal_phase2.py `
  --phase1-root outputs/theme_discovery_phase1 `
  --out-root outputs/theme_temporal_phase2 `
  --date-start 2025-01-02 `
  --date-end 2026-06-30 `
  --rebuild-from 2025-01-02
```

## Step 3: Phase 2 acceptance checks

Before starting carry-over A/B, verify:

1. Every Phase 1 date has:
   - `<phase2-root>/date=<date>/_SUCCESS`
   - `<phase2-root>/_state/date=<date>/_SUCCESS`
2. Within every layer, `snapshot_time` is monotonically increasing.
3. A path never changes `source_layers`.
4. The first snapshot of a day can continue a path from the previous trading day.
5. The last trading day of a month can continue into the first trading day of the next month.
6. Repeating Phase 2 over identical Phase 1 input produces identical canonical outputs.
7. Interrupting after one day and resuming produces the same result as an uninterrupted run.

If an upstream Phase 1 date is modified, rerun `run_phase2_from_earliest.py`; do not delete arbitrary later state directories manually.

## Step 4: full-year 2025 seven-arm study

Dry run:

```powershell
python scripts/run_monthly_carryover_ab.py `
  --config configs/monthly_carryover_ab_2025.json `
  --phase1-root outputs/theme_discovery_phase1 `
  --max-workers 4 `
  --dry-run
```

Production:

```powershell
python scripts/run_monthly_carryover_ab.py `
  --config configs/monthly_carryover_ab_2025.json `
  --phase1-root outputs/theme_discovery_phase1 `
  --max-workers 4 `
  --resume
```

Start with four boundary workers. Increase only after measuring peak RSS and I/O throughput. A boundary worker loads the previous close and current opening horizon once, then evaluates all seven arms and fixed controls sequentially.

## Checkpoint rules

The scheduling unit is one overnight boundary. The atomic completion unit remains:

```text
boundary x arm x control x replicate
```

Each unit must contain:

```text
bridge_candidates.parquet
path_states.parquet
revival_events.parquet
matched_controls.parquet
outcomes.parquet
task_manifest.json
_SUCCESS
```

A boundary is successful only when every expected unit has both `_SUCCESS` and `task_manifest.json`.

On resume:

- valid units are skipped;
- missing units are computed;
- a partially completed boundary is not treated as complete;
- all arms use the same deterministic `null_mapping.parquet`.

## Cross-month and cross-year operation

The date index must cover the full continuous study range. For example, a combined 2025-2026 run must include all dates in one index so these boundaries are generated:

```text
2025-01 last day -> 2025-02 first day
...
2025-12 last day -> 2026-01 first day
```

A configuration ending on `2025-12-31` validates 2025 internal boundaries. To test the 2025-to-2026 boundary, create a configuration whose `date_end` includes the first 2026 trading day.

## Mandatory tests before unattended execution

1. Phase 1 determinism: one worker vs production workers.
2. Phase 2 determinism: same fixed Phase 1 input twice.
3. Resume equivalence: interrupted vs uninterrupted Phase 2.
4. Same-layer assertion: no path or revival changes layer.
5. Seven-arm differentiation: D9, D11, D13, and D15 event counts must not be identical by construction.
6. D15 thresholds: inspect sampled events and confirm fingerprint >= 0.30 and breadth expansion >= 0.10.
7. Cross-month boundary: confirm at least one month-end boundary appears in `run_manifest.json`.
8. Boundary completeness: remove one unit `_SUCCESS`; `--resume` must recompute it and must not skip the boundary.

## Stop conditions

Stop the unattended run only for:

- non-deterministic Phase 1 or Phase 2 output;
- time ordering violation;
- cross-layer path continuation;
- future leakage;
- corrupt checkpoint accepted as successful;
- schema failure;
- insufficient disk space.

Do not stop merely because one arm is insignificant, one month is weak, or D15 fails to outperform a control. Preserve all successful and failed-event outputs for later analysis.
