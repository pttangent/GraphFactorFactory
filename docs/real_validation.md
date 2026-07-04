# Real Qlib and full-universe validation

Validation used the user-supplied Microsoft Qlib source archive, installed locally as `pyqlib 0.9.8.dev0`, and the real NodeFactorFactory 5-minute v5 shard for 2025-07-22.

## Results

- Source universe: 5,248 symbols
- Validation window: 18:15–20:00 UTC
- Point-in-time samples: 117,401
- Automatically discovered NodeFactorFactory numeric features: 67
- Missing node factors in Qlib: 0
- Graph layers: 13 plus multiplex
- Graph edges: 138,590
- Flattened graph factor columns: 71
- Qlib table: 117,401 rows × 143 columns
- `DataHandlerLP`: passed
- `DatasetH.prepare`: passed
- Availability violations: 0
- Duplicate `(decision_time, symbol)` rows: 0
- Prefix invariance: passed after appending 7,899 rows unavailable at the decision time

## Strict timing

A feature is permitted only when both `source_timestamp <= decision_time` and `source_available_time <= decision_time`.

Labels use:

- signal time: `decision_time`
- entry time: `decision_time + 5 minutes`
- exit time: `entry_time + horizon`
- missing exact entry or exit bar: missing label

This prevents same-close execution and row-shift leakage across missing bars.

## Performance

The dense full-market correlation matrix was replaced by deterministic LSH candidate generation, reciprocal top-k filtering, a symmetric degree cap, and sparse CSR neighbor aggregation. The full-universe 13-layer validation completed in approximately 17 seconds in the validation environment.
