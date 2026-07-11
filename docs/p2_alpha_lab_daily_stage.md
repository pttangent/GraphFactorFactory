# P2 Alpha Lab Daily Stage

This stage stays inside the existing P2 alpha lab. It is not a separate P3 pipeline.

## Lineage

```text
P0 labels.parquet + P1 theme_memberships.parquet
  -> theme_returns.parquet

P1 theme_relation_edges.parquet + theme_returns.parquet
  -> relation_spillover_signals.parquet

relation_spillover_signals.parquet
  -> daily_relation_features.parquet
  -> daily_alpha_metrics.csv / daily_alpha_summary.csv
```

The daily stage reuses the same `theme_returns`, `relation_spillover`, `date/layer_id/scale` partitioning, and `manifest.json` lineage. It does not copy completed relation packs or recompute P1.

## CPU and memory model

The runner keeps the partition-safe rule:

```text
one worker = one date/layer_id/scale partition
```

Parent processes only receive small metadata dictionaries. Large intermediate rows are written by the worker to partitioned parquet.

Default thread caps are set inside the script:

```text
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
ARROW_NUM_THREADS=2
```

For a 24-core machine:

```text
--workers 16  conservative
--workers 20  normal production
--workers 24  only after checking RAM headroom
```

## Example from P0 + P1

```powershell
python scripts/p2_alpha_daily_features.py build-theme-returns ^
  --p1-root artifacts\p1_b50_b35_sharded ^
  --labels-root data\graph_store_6m\canonical ^
  --out-root artifacts\p2_alpha_lab\theme_returns ^
  --dates 2026-01-07 ^
  --layers 9 ^
  --scales 30m ^
  --levels B50,B35 ^
  --workers 20

python scripts/p2_alpha_daily_features.py relation-spillover ^
  --p1-root artifacts\p1_b50_b35_sharded ^
  --theme-returns-root artifacts\p2_alpha_lab\theme_returns ^
  --out-root artifacts\p2_alpha_lab\relation_spillover ^
  --dates 2026-01-07 ^
  --layers 9 ^
  --scales 30m ^
  --levels B50,B35 ^
  --past-horizon 15m ^
  --workers 20

python scripts/p2_alpha_daily_features.py daily-relation-features ^
  --signals-root artifacts\p2_alpha_lab\relation_spillover ^
  --out-root artifacts\p2_alpha_lab\daily_relation_features ^
  --dates 2026-01-07 ^
  --layers 9 ^
  --scales 30m ^
  --workers 20

python scripts/p2_alpha_daily_features.py evaluate-daily ^
  --features-root artifacts\p2_alpha_lab\daily_relation_features ^
  --out-dir artifacts\p2_alpha_lab\daily_relation_eval
```

## Validation run performed

Validated using real downloaded P0/P1 artifacts, not a precomputed completed relation pack.

```text
P0 label source:
  /mnt/data/labels.parquet

P1 source:
  /mnt/data/2026-01-07_p1.tar.gz.part000 ... part007
  reconstructed and extracted before running.

Partition:
  date=2026-01-07
  layer_id=9
  scale=30m
  level=B50,B35
```

Observed outputs:

```text
theme_returns rows:              886,014
relation_spillover signal rows:  479,713
daily_relation_features rows:    479,713
daily eval metric rows:          32
daily eval summary rows:         32
```

Observed peak memory in direct stage validation:

```text
theme returns:        about 3.5 GB RSS
relation spillover:   about 1.3 GB RSS
daily features:       about 0.8 GB RSS
daily evaluate:       about 1.1 GB RSS
```

This validates that the daily-stage module can be built from P0 + P1 while preserving the partition-safe design.
