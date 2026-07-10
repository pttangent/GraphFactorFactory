# P1 sharded streaming production pipeline

This production path fixes the date-level OOM/slow-worker problem by changing the worker grain.

## Old failing grain

```text
1 worker = 1 whole date
```

Large dates can contain 100M+ P0 edges. Date-level workers force pandas to load all layers/scales for that date and make multi-worker execution unsafe.

## New production grain

```text
1 worker = 1 date + 1 layer_id + 1 scale shard
```

The pipeline has two stages:

1. `scripts/shard_p0_edges_by_layer_scale.py`
   - reads P0 `edges.parquet` in batches
   - physically writes `date/layer/scale` shards
2. `scripts/run_p1_sharded_parallel.py`
   - schedules shard-local P1 jobs with multiple workers
   - largest shards run first
   - skips existing `manifest.json` when `--skip-existing` is enabled

The shard-local builder is:

```text
scripts/build_b50_b35_theme_forest_streaming.py
```

It preserves the P1 design:

- B50 stable theme forest
- B35 local refinement under each B50 leaf
- fuzzy same-layer relation graph
- fuzzy temporal continuation graph

## Required fuzzy outputs

`theme_relation_edges` includes:

- `relation_strength`
- `relation_tier`
- `hard_keep`
- `normalized_weight`
- `inter_leaf_weight`
- `edge_count`

`temporal_theme_edges` includes:

- `continuation_strength`
- `jaccard`
- `containment`
- `hard_continue`
- `fuzzy_continue`

## Example

```powershell
python scripts/shard_p0_edges_by_layer_scale.py ^
  --p0-root D:\DEV\US-Stock\GraphFactorFactory\data\graph_store_6m\canonical ^
  --out-root D:\DEV\US-Stock\GraphFactorFactory\data\p0_edge_shards ^
  --dates 2026-03-26

python scripts/run_p1_sharded_parallel.py ^
  --shard-root D:\DEV\US-Stock\GraphFactorFactory\data\p0_edge_shards ^
  --out-root artifacts\p1_b50_b35_sharded ^
  --workers 8 ^
  --skip-existing
```

Smoke test:

```powershell
python scripts/shard_p0_edges_by_layer_scale.py ^
  --p0-root D:\path\to\edges.parquet ^
  --date 2026-01-07 ^
  --out-root data\p0_edge_shards_smoke ^
  --max-batches 1

python scripts/run_p1_sharded_parallel.py ^
  --shard-root data\p0_edge_shards_smoke ^
  --out-root artifacts\p1_b50_b35_sharded_smoke ^
  --workers 4 ^
  --max-shards 4 ^
  --max-snapshots 3 ^
  --output-format csv
```
