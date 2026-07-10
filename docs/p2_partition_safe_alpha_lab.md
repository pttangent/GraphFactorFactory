# P2 partition-safe alpha lab

This branch replaces the previous date-level P2 alpha scripts with a partition-level runner.

## Why

The previous scripts used `ProcessPoolExecutor(max_workers=6)` at date granularity. Each worker still loaded full-date `theme_memberships`, `labels`, or `theme_relation_edges` DataFrames, and some scripts returned full DataFrames to the parent for a final concat. A lock around `read_parquet` only serialized allocation during read; it did not limit resident DataFrames after the read finished. That design can still OOM and does not truly use 24 cores efficiently.

## New execution model

Worker unit:

```text
date + layer_id + scale
```

Each worker:

- reads one P1 partition, not the whole date;
- streams membership or relation parquet by row group;
- reads only required label columns;
- writes output parquet atomically via a `.tmp` file and final rename;
- writes a `manifest.json` with `status`, input paths, output path, row count and elapsed seconds;
- returns metadata only, never a large DataFrame.

## Entrypoints

The old script names now route to the partition-safe runner:

```bash
python scripts/p2_build_theme_returns.py \
  --p1-root artifacts/p1_b50_b35_sharded \
  --labels-root data/graph_store_6m/canonical \
  --out-root artifacts/p2_alpha_lab/theme_returns \
  --workers 20 \
  --skip-existing

python scripts/p2_relation_spillover_alpha.py \
  --p1-root artifacts/p1_b50_b35_sharded \
  --theme-returns-root artifacts/p2_alpha_lab/theme_returns \
  --out-root artifacts/p2_alpha_lab/relation_spillover \
  --workers 20 \
  --past-horizon 15m \
  --skip-existing

python scripts/p2_core_peripheral_alpha.py \
  --p1-root artifacts/p1_b50_b35_sharded \
  --labels-root data/graph_store_6m/canonical \
  --out-root artifacts/p2_alpha_lab/core_peripheral \
  --workers 20 \
  --past-horizon 15m \
  --skip-existing

python scripts/p2_reduce_alpha_metrics.py \
  --signals-root artifacts/p2_alpha_lab \
  --out-dir artifacts/p2_alpha_lab/metrics \
  --horizons 5m,15m,30m,60m
```

The unified implementation is `scripts/p2_alpha_partitioned_lab.py`.

## Important validity fix

P1 theme IDs are snapshot-local. Relation spillover must not shift theme IDs across time and then join by `theme_id`.

The new workflow builds same-timestamp past-return columns in `theme_returns`:

```text
past_eq_5m, past_eq_15m, ...
past_core_5m, past_core_15m, ...
```

These are created by shifting member-level P0 labels before aggregating to themes. Relation spillover then joins current relation edges to current source-theme `past_eq_*` at the same `decision_time`.

## Smoke test that was run

Inputs downloaded / used locally:

- Drive P1 package: `2026-01-07_p1.tar.gz.part000` through `part007`, extracted to real P1 partitions.
- P0 labels: `labels.parquet` for 2026-01-07.

Smoke command shape:

```bash
python scripts/p2_build_theme_returns.py \
  --p1-root /mnt/data/drive_p1_0107 \
  --labels-root /mnt/data/labels.parquet \
  --out-root /mnt/data/alpha_test_unified/theme_returns \
  --dates 2026-01-07 --layers 2 --scales 5m \
  --workers 2 --max-partitions 1 --max-row-groups 20

python scripts/p2_relation_spillover_alpha.py \
  --p1-root /mnt/data/drive_p1_0107 \
  --theme-returns-root /mnt/data/alpha_test_unified/theme_returns \
  --out-root /mnt/data/alpha_test_unified/relation_spillover \
  --dates 2026-01-07 --layers 2 --scales 5m \
  --workers 2 --max-partitions 1 --max-row-groups 20 --past-horizon 5m

python scripts/p2_core_peripheral_alpha.py \
  --p1-root /mnt/data/drive_p1_0107 \
  --labels-root /mnt/data/labels.parquet \
  --out-root /mnt/data/alpha_test_unified/core_peripheral \
  --dates 2026-01-07 --layers 2 --scales 5m \
  --workers 2 --max-partitions 1 --max-row-groups 20 --past-horizon 5m

python scripts/p2_reduce_alpha_metrics.py \
  --signals-root /mnt/data/alpha_test_unified \
  --out-dir /mnt/data/alpha_test_unified/metrics \
  --horizons 5m,15m,30m,60m
```

Observed smoke outputs:

- `theme_returns`: 26,690 rows.
- `relation_spillover_signals`: 18,699 rows.
- `core_peripheral_signals`: 30,257 rows.
- reducer: 208 partition metric rows and 16 summary rows.

## Are P0 and P1 enough?

Yes, for first-pass alpha research, if P0 includes `labels.parquet` or enough price data to reconstruct labels, and P1 includes:

- `theme_memberships.parquet`
- `theme_relation_edges.parquet`
- optional `temporal_theme_edges.parquet`

For stronger validation, later add market/sector-neutral labels, shuffled relation baselines, and daily IC aggregation.
