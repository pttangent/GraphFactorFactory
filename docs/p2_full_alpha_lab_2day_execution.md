# Full P2 Alpha Lab: 2-Day Local Execution Plan

Branch: `codex/p2-alpha-partition-safe`

This document records the full alpha-lab architecture and the real-data validation performed before the 2-day local run.

## Input sufficiency check

Google Drive folder: `p2_2day_pack`

Required inputs are present:

- `labels/date=2026-01-07/labels.parquet`
- `labels/date=2026-01-08/labels.parquet`
- `p0/date=2026-01-07.tar.gz.part*`
- `p0/date=2026-01-08.tar.gz.part*`
- `p1/date=2026-01-07.tar.gz.part*`
- `p1/date=2026-01-08.tar.gz.part*`
- `symbol_mapping/symbols.parquet`

The labels and symbol files were fetched and inspected from Drive:

- `labels.parquet` for 2026-01-07: 108,386,360 bytes, 1,875,727 rows, 35 columns, 2 row groups.
- Label time range: `2026-01-07 14:31:00+00:00` to `2026-01-07 21:00:00+00:00`.
- Unique `symbol_id`: 5,511.
- `symbols.parquet`: 48,176 bytes, 5,590 rows, columns `symbol_id`, `symbol`.

## Architecture

The P2 lab now has three layers:

```text
P2A: P0 direct graph alpha
  - p0_node_features
  - p0_edge_spillover
  - p0_graph_state
  - p0_alpha_eval

P2B: P1 theme alpha
  - theme_returns

P2C: P1 relation alpha
  - relation_spillover
  - daily_relation_features
  - daily_relation_eval
```

## Important fix

The previous P2 implementation dropped `decision_time` from P1 membership/relation joins, which caused near-390x repeated full-partition joins.

The fixed implementation restores `decision_time` alignment:

```text
membership[decision_time=t] joins labels[decision_time=t]
relation_edges[decision_time=t] joins theme_returns[decision_time=t]
```

This removes the 390x repeated merge and preserves the correct intraday graph semantics.

## Real-data validation

Validation was performed with actual Drive-derived data available in the execution sandbox.

### P1 theme_returns validation

Input:

```text
P1: date=2026-01-07 / layer_id=9 / scale=30m / theme_memberships.parquet
Labels: date=2026-01-07 / labels.parquet
```

P1 membership partition:

```text
rows: 3,357,416
row_groups: 379
columns: decision_time, layer_id, scale, theme_id, member_id, level, root_b50_theme_id, rank_in_theme, core_score
```

Aligned benchmark on first 20 row groups:

```json
{
  "input_rows": 180980,
  "time_groups": 20,
  "output_rows": 46486,
  "batches": 20,
  "elapsed_sec": 4.19
}
```

This validates that the fixed implementation uses time-aligned chunk joins and streaming Parquet output. It no longer performs `390 x full_membership` repeated joins.

### P1 relation_spillover validation

Input:

```text
P1 relation: date=2026-01-07 / layer_id=9 / scale=30m / theme_relation_edges.parquet
Theme returns: validated first-20-row-group output above
```

Relation partition:

```text
rows: 769,755
row_groups: 379
```

Aligned benchmark on first 20 row groups:

```json
{
  "rel_rows": 40145,
  "past_rows": 46486,
  "output_rows": 24631,
  "batches": 20,
  "elapsed_sec": 0.45
}
```

### P0 edge_spillover validation

Input:

```text
P0 edges: actual 2026-01-07 edges.parquet
Labels: actual 2026-01-07 labels.parquet
```

P0 edge file inspected in sandbox:

```text
rows: 48,963,776
row_groups: 370
columns: decision_time, window_start, window_end, layer_id, src_id, dst_id, weight, src_rank, dst_rank, directed, lag_bars, window_points, vector_dimension
```

P0 direct edge spillover benchmark:

```json
{
  "max_row_groups": 1,
  "output_rows": 39495,
  "batches": 1,
  "elapsed_sec": 3.60
}
```

```json
{
  "max_row_groups": 5,
  "output_rows": 199864,
  "batches": 5,
  "elapsed_sec": 9.08
}
```

This validates that direct P0 graph alpha is now part of the lab and can be processed row-group-by-row-group without full-table loading.

## Local 2-day command

After extracting the Drive tar parts into local folders:

```text
C:\GFF_Cache\p2_2day_pack\p0
C:\GFF_Cache\p2_2day_pack\p1
C:\GFF_Cache\p2_2day_pack\labels
```

Run the full 2-day alpha lab:

```powershell
python scripts/run_p2_24core_scheduler.py ^
  --p0-root C:\GFF_Cache\p2_2day_pack\p0 ^
  --p1-root C:\GFF_Cache\p2_2day_pack\p1 ^
  --labels-root C:\GFF_Cache\p2_2day_pack\labels ^
  --p2-root C:\GFF_Cache\p2_alpha_lab_full ^
  --dates 2026-01-07,2026-01-08 ^
  --layers 3,6,8,9,11 ^
  --scales 15m,30m ^
  --profile max ^
  --cores 24 ^
  --target-cpu 1.0 ^
  --inner-workers 1 ^
  --skip-existing
```

If CPU is not saturated and memory is stable, increase nested stage concurrency:

```powershell
--inner-workers 2
```

If RAM remains stable, test:

```powershell
--inner-workers 4
```

The scheduler records its actual plan at:

```text
<p2-root>/p2_24core_schedule_plan.json
```

## Stage-specific restart commands

P0 only:

```powershell
python scripts/run_p2_24core_scheduler.py ^
  --p0-root C:\GFF_Cache\p2_2day_pack\p0 ^
  --labels-root C:\GFF_Cache\p2_2day_pack\labels ^
  --p2-root C:\GFF_Cache\p2_alpha_lab_full ^
  --dates 2026-01-07,2026-01-08 ^
  --profile max ^
  --stage p0 ^
  --skip-existing
```

P1 theme only:

```powershell
python scripts/run_p2_24core_scheduler.py ^
  --p1-root C:\GFF_Cache\p2_2day_pack\p1 ^
  --labels-root C:\GFF_Cache\p2_2day_pack\labels ^
  --p2-root C:\GFF_Cache\p2_alpha_lab_full ^
  --dates 2026-01-07,2026-01-08 ^
  --profile max ^
  --inner-workers 1 ^
  --stage theme ^
  --skip-existing
```

Relation + daily only:

```powershell
python scripts/run_p2_24core_scheduler.py ^
  --p1-root C:\GFF_Cache\p2_2day_pack\p1 ^
  --labels-root C:\GFF_Cache\p2_2day_pack\labels ^
  --p2-root C:\GFF_Cache\p2_alpha_lab_full ^
  --dates 2026-01-07,2026-01-08 ^
  --profile max ^
  --inner-workers 1 ^
  --stage relation ^
  --skip-existing
```

Then:

```powershell
python scripts/run_p2_24core_scheduler.py ^
  --labels-root C:\GFF_Cache\p2_2day_pack\labels ^
  --p2-root C:\GFF_Cache\p2_alpha_lab_full ^
  --dates 2026-01-07,2026-01-08 ^
  --profile max ^
  --stage daily ^
  --skip-existing
```

Finally:

```powershell
python scripts/run_p2_24core_scheduler.py ^
  --labels-root C:\GFF_Cache\p2_2day_pack\labels ^
  --p2-root C:\GFF_Cache\p2_alpha_lab_full ^
  --stage eval
```

## Expected outputs

```text
p2_alpha_lab_full/
  p0_node_features/
  p0_edge_spillover/
  p0_graph_state/
  p0_alpha_eval/
  theme_returns/
  relation_spillover/
  daily_relation_features/
  daily_relation_eval/
  p2_24core_schedule_plan.json
```
