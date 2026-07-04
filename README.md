# GraphFactorFactory

A NodeFactorFactory-native, point-in-time multilayer graph factor factory with lossless canonical graph storage and real Microsoft Qlib integration.

## Design

- **NodeFactorFactory remains the external node-factor source of truth.** It is not copied.
- **GraphFactorFactory stores maximum graph detail:** retained edges, layer-node metrics, graph snapshots, lineage and strict labels.
- **Qlib is an on-demand consumer, not the canonical storage format.** This avoids irreversible graph flattening and duplicate disk usage.
- **Full universe is the default.** A universe CSV is optional, not required.

## Complete pipeline

```text
NodeFactorFactory Parquet
  -> source/PIT audits
  -> 5-minute decision panel
  -> strict next-bar labels
  -> 13 causal graph layers
  -> lossless canonical Parquet graph store
  -> DuckDB external-view catalog
  -> on-demand Qlib DataLoader / graph-native provider
```

## Canonical graph detail

Edges preserve decision/window time, layer, integer source/destination IDs, weight, reciprocal top-k ranks, direction, lag, window observations and vector dimension. Node metrics preserve degree, strength, core score, neighbor signals and multiplex participation.

## Qlib

`CanonicalQlibDataLoader` dynamically combines every numeric NodeFactorFactory feature, flattened graph-node factors and strict labels into Qlib's `(datetime, instrument)` format. `DataHandlerLP` can apply cross-sectional normalization without storing another dataset. `GraphBatchProvider` serves the original edge list to custom graph models.

Qlib conversion is **not** performed by default. A disposable Parquet cache can be explicitly materialized when repeated training speed is more important than disk space.

See `docs/pipeline.md` and `docs/storage_design.md`.

## Measured disk use

The lossless canonical design projects to about **20.9 GB/year** for 5,247 symbols, 13 layers and 15-minute graph snapshots, including strict labels and excluding the external NodeFactorFactory source. Three years plus 50% headroom is about **94 GB**, so 100 GB is the minimum and 250 GB is the practical allocation.
