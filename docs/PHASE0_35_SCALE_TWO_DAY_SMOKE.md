# Phase 0 35-scale two-day smoke

Branch: `codex/phase0-35-layer-scales`

## Inputs

- `node_factors_1m/date=2026-06-01/data.parquet`
- `node_factors_1m/date=2026-06-02/data.parquet`
- Decision time per day: 10:00 ET (`14:00 UTC`)
- Universe: 1,200 symbols with at least 20 valid one-minute observations in the 30-minute source window
- Input rows per day: 34,800

The smoke uses the real 1-minute NFF fields and constructs every registered layer/lookback combination. It does not rerun NFF.

## Result

| Metric | Result |
|---|---:|
| Expected graphs per day | 35 |
| Days | 2 |
| Total graph attempts | 70 |
| Successful graphs | 70 |
| Failed graphs | 0 |
| Total retained edges | 209,101 |
| 2026-06-01 elapsed | 4.86 s |
| 2026-06-02 elapsed | 4.55 s |

**Technical gate: PASS.**

All layer-scale configurations generated trajectories and retained edges on both days. This validates the 1-minute NFF schema path, 5/15/30-minute window slicing, internal rolling-5-minute return derivation, cross-sectional residual transforms, correlation graph path, LSH graph path, and sparse-layer execution.

## Scale matrix exercised

- 5-minute trigger graphs: 10
- 15-minute confirmation graphs: 11
- 30-minute structural graphs: 14
- Total: 35

## Important scope

This is a technical smoke, not a statistical or alpha validation. It deliberately uses one mature decision time per trading day so every configured lookback is available. Full-session production runs remain a separate performance and research validation step.
