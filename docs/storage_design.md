# Storage design: preserve graph detail, avoid Qlib duplication

## Canonical source of truth

GraphFactorFactory does not copy NodeFactorFactory data. The source Parquet remains external and is referenced by a manifest fingerprint.

The only retained GraphFactorFactory products are:

- `dimensions/symbols.parquet`: integer symbol dictionary.
- `dimensions/layers.parquet`: layer registry and feature lineage.
- `canonical/date=YYYY-MM-DD/edges.parquet`: retained graph edges with weight, reciprocal ranks, window, direction, lag, observation count and vector dimension.
- `canonical/date=YYYY-MM-DD/node_features.parquet`: long-form layer node metrics, including multiplex participation.
- `canonical/date=YYYY-MM-DD/snapshots.parquet`: graph-level QA, timing and window metadata.
- `canonical/date=YYYY-MM-DD/labels.parquet`: strict next-bar-entry labels, if enabled.
- `graphfactorfactory.duckdb`: small catalog containing external views, not duplicated tables.

## Why Qlib is not the canonical format

Standard Qlib model input is a rectangular `(datetime, instrument, feature)` table. Flattening graph edges into node factors is useful for LightGBM and Transformers, but it cannot represent the full edge list, reciprocal ranks, layer topology or graph-native batches without loss.

Therefore Qlib is an on-demand view:

- `CanonicalQlibDataLoader` reads external node factors and canonical graph-node metrics.
- Qlib processors perform normalization in memory.
- No Qlib `.bin` or duplicate Qlib Parquet is written by default.
- `materialize-qlib-cache` is optional and disposable.
- `GraphBatchProvider` exposes the retained edge list to custom GNN/TGNN Qlib models.

## Disk minimization

- Symbols are stored as `int32 symbol_id` in graph tables instead of repeated ticker strings.
- Layer IDs use `int16`.
- Edge and graph metrics use `float32`.
- Parquet uses Zstandard, dictionary encoding and large row groups.
- PIT node panels are not persisted.
- NodeFactorFactory source is not duplicated.
- Qlib caches are not persisted unless explicitly requested.

## Measured full-universe capacity

Real validation used 5,247 symbols on 2025-07-22. Three full-universe snapshots produced:

- edges: 3.34 MB
- long-form graph node factors: 4.45 MB
- snapshot QA: 0.01 MB
- strict labels: 12.75 MB
- Qlib cache: 0 MB, because the Qlib view was generated on demand

Scaling the graph portion to a conservative 27 snapshots per day at the default 15-minute cadence gives approximately 83 MB per trading day, 20.9 GB per 252-day year, and 94.1 GB for three years plus 50% operational headroom. With labels disabled and recomputed on demand, the same three-year headroom estimate is approximately 79.6 GB.

Recommended allocations:

- minimum canonical GraphFactorFactory store for three years: **100 GB free**
- practical research headroom, including temporary Qlib caches and model predictions: **250 GB free**
- NodeFactorFactory source data is external and must be budgeted separately
