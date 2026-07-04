# End-to-end pipeline

1. Read a NodeFactorFactory 5-minute Parquet file, directory or glob.
2. Verify `available_time >= timestamp` and unique `(symbol, timestamp)`.
3. Filter regular-session rows in `America/New_York`, automatically respecting daylight saving time.
4. Construct a fixed 5-minute point-in-time panel using only rows with `available_time <= decision_time`.
5. Build labels with signal at `t`, entry at `t+5m`, and exact exit at `entry+horizon`.
6. Build 13 graph layers over a trailing causal window using deterministic LSH candidates, reciprocal top-k and a symmetric degree cap.
7. Persist full retained edges, long-form node metrics and snapshot QA through streaming Parquet writers.
8. Register DuckDB views over Parquet without materializing duplicate tables.
9. Serve node and flattened graph factors to Qlib on demand; preserve edge lists for graph-native models.

CLI:

```bash
graphfactorfactory build-date \
  --node-factors /path/to/node_factors_5m \
  --date 2025-07-22 \
  --output /path/to/graph_store \
  --config configs/default.yaml
```
