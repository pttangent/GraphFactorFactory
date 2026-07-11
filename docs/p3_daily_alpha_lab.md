# P3 Daily Alpha Lab

P3 converts P2 intraday relation-spillover pulses into modular daily alpha features.

## Why P3 exists

P2 answers a short-horizon question:

```text
At snapshot t, does relation spillover predict target return over the next 5m/15m/etc.?
```

P3 answers a daily question:

```text
Across the full trading day, was a target theme path repeatedly and persistently pointed to by relation spillover,
and did that pressure remain into the close without the target fully reacting?
```

This keeps the lab modular:

- P2 remains the intraday signal factory.
- P3 consumes P2 shards and builds daily path-level features.
- Future daily alpha ideas can be added without changing P2.

## Inputs

Current supported input:

```text
artifacts/p2_alpha_lab/relation_spillover/
  date=YYYY-MM-DD/layer_id=X/scale=30m/relation_spillover_signals.parquet
```

Expected columns include:

```text
date
decision_time
layer_id
scale
level
dst_theme_id
signal
relation_strength_mean
relation_edge_count
target_5m / target_15m / target_30m / target_60m / target_120m
```

If a persistent `target_path_id` does not exist, the module creates a fallback path id by stripping the `ts=...` token from `dst_theme_id`. This is deterministic but not a perfect temporal identity. A future improvement should use `temporal_theme_edges` to create stronger path IDs.

## Outputs

```text
<out-dir>/daily_relation_features.parquet
<out-dir>/manifest.json
```

Feature columns include:

```text
daily_pressure
positive_pressure
negative_pressure
absolute_pressure
pressure_intensity
positive_observation_rate
negative_observation_rate
persistence_proxy
relation_edge_count_sum
avg_relation_strength
late_signal_sum
late_pos_signal_sum
late_abs_signal_sum
late_absolute_share
late_confirmation_score
target_response_z
target_underreaction_z
daily_pressure_score
daily_underreaction_score
daily_consensus_score
```

## Core feature interpretation

### daily_pressure

Total intraday spillover pressure into the target path.

### persistence_proxy

Positive observation rate minus negative observation rate. It estimates whether relation pressure was directionally persistent through the day.

### late_confirmation_score

Measures whether pressure remains in the last N minutes of the partition session. Default is 60 minutes.

### target_underreaction_z

A proxy for whether relation pressure is high relative to the target's same-day short-horizon response. This is only a proxy until next-day labels are joined.

### daily_consensus_score

Pressure score multiplied by relation edge-count breadth. This approximates multi-source / multi-relation consensus when explicit source theme counts are unavailable.

## Example: build daily features

```powershell
python scripts/p3_daily_alpha_lab.py build-reaction-features ^
  --signals-root artifacts\p2_alpha_lab\relation_spillover ^
  --out-dir artifacts\p3_daily_alpha_lab\relation_daily_features ^
  --layers 3,8,9 ^
  --scales 30m ^
  --levels B50,B35 ^
  --late-minutes 60
```

## Example: proxy evaluation

The current completed pack contains intraday targets inside the P2 signal rows. These are useful for smoke testing the feature builder but should not be treated as final daily alpha validation.

```powershell
python scripts/p3_daily_alpha_lab.py evaluate-features ^
  --features artifacts\p3_daily_alpha_lab\relation_daily_features\daily_relation_features.parquet ^
  --out-dir artifacts\p3_daily_alpha_lab\relation_daily_eval ^
  --target-cols target_5m_mean_proxy,target_15m_mean_proxy,target_30m_mean_proxy,target_60m_mean_proxy
```

## Real validation run performed during development

Validated against downloaded real completed pack relation-spillover outputs:

```text
Dates: 2026-01-02, 2026-01-05, 2026-01-06, 2026-01-07, 2026-01-08,
       2026-01-09, 2026-01-12, 2026-01-13, 2026-01-14, 2026-01-15
Layer: 3
Scale: 30m
Levels: B50, B35
Output daily feature rows: 27,293
Proxy metric rows: 240
Proxy summary rows: 24
```

This confirms the module can read real P2 partition outputs and emit daily path-level features.

## What is still needed for true daily alpha validation

To validate day-level / overnight alpha, add a daily label join stage:

```text
T close -> T+1 open
T+1 open -> T+1 close
T close -> T+1 close
T close -> T+3 close
T close -> T+5 close
```

The next module should merge `daily_relation_features.parquet` to stock/theme-level next-day labels and then evaluate daily IC, quintile spread, positive-day ratio, and capacity/liquidity filters.

## Recommended next alpha variants

1. `daily_relation_pressure`: raw daily pressure score.
2. `daily_relation_underreaction`: daily pressure with target underreaction.
3. `daily_relation_consensus`: daily pressure weighted by relation breadth.
4. `close_imprint`: only last-60m relation pressure.
5. `multi_day_path_pressure`: pressure persistence across multiple trading days.
