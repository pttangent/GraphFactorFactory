# GraphFactorFactory Dashboard

## Pipeline Outputs (Phase 0)

The pipeline has been successfully configured to run and output the full US Stock Market graph topology and semantic themes.

### 1. Canonical Graph Database (DuckDB)
**Path:** `D:\DEV\US-Stock\GraphFactorFactory\data\graph_store\graphfactorfactory.duckdb`
*Description:* A zero-copy SQL catalog pointing to the underlying Parquet files. Use this file in Python (`duckdb.connect()`) or DBeaver to instantly query the graph data across all dates.

### 2. Canonical Graph Parquet Files
**Path:** `D:\DEV\US-Stock\GraphFactorFactory\data\graph_store\canonical\`
*Description:* The raw partitioned Parquet files containing:
- `edges.parquet`: The highly sparse Top-K topological edges.
- `node_features.parquet`: The truncated node feature matrix.
- `snapshots.parquet`: The global graph statistics per 15-minute snapshot.
- `labels.parquet`: The forward return labels.

### 3. Theme Discovery Artifacts
**Path:** `D:\DEV\US-Stock\GraphFactorFactory\data\graph_store\themes\`
*Description:* The sequential output of the Leiden community detection and consensus clustering. Contains:
- `themes.parquet`: The consolidated and temporally smoothed themes.
- `subcommunities.parquet`: The granular sub-clusters within the themes.
- `lifecycle.parquet`: The entry, exit, and tracking logs for temporal themes.
- `semantics.parquet`: Semantic labels attached to themes based on metadata.

---
*Note: Due to the efficient downsampling (15-minute snapshots) and topological truncation (Top-8 edges), the footprint for 1 month is ~2.7 GB (instead of NFF's 38 GB).*
