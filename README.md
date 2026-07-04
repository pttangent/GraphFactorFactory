# GraphFactorFactory

NodeFactorFactory-native multilayer graph feature factory with strict point-in-time semantics and real Microsoft Qlib integration.

## Responsibilities

- NodeFactorFactory owns raw trade processing and node factors.
- GraphFactorFactory owns multilayer graph construction and graph factors.
- Qlib owns datasets, processors, models, experiments, portfolios, and backtests.

## Verified real-data integration

Validated with the user-supplied Microsoft Qlib source, installed locally as `pyqlib 0.9.8.dev0`, and a real NodeFactorFactory 5-minute v5 shard for 2025-07-22.

- Full source universe: 5,248 symbols
- PIT samples: 117,401
- Automatically discovered NodeFactorFactory numeric factors: 67
- Missing node factors in Qlib: 0
- Graph layers: 13 plus multiplex
- Graph edges: 138,590
- Flattened graph feature columns: 71
- Qlib table: 117,401 rows × 143 columns
- `DataHandlerLP`: passed
- `DatasetH.prepare`: passed
- Availability violations: 0
- Duplicate `(decision_time, symbol)` keys: 0
- Prefix invariance: passed after adding 7,899 future-available rows

Strict labels use signal time `t`, entry at `t+5m`, and exit at `entry+horizon`; exact bars are required.

See `docs/real_validation.md`.
