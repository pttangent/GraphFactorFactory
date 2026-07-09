# D1/D2/D3 penalty-start experiment

**Important:** `top_sector_share` is evaluation only. Sector/industry metadata is not used in clustering, tree construction, or split scoring. The tree uses graph edges only.

Window approximation: three P0-derived graph snapshots inside one hour: 14:44, 15:14, 15:44 UTC on 2026-01-06. Tested selected layer-scales: return corr, rolling return corr, block activity, large trade flow, absorption, flow-return alignment.

For tractability in this chat runtime, each graph is pruned to top-5 weighted neighbors per node before tree construction.

## Aggregate

|   penalty_start_depth |   groups |   avg_leaf_count |   median_leaf_size |   p90_leaf_size |   max_leaf_size |   mean_top_sector_share |   median_top_sector_share |   sector_pure_60 |   sector_pure_80 |
|----------------------:|---------:|-----------------:|-------------------:|----------------:|----------------:|------------------------:|--------------------------:|-----------------:|-----------------:|
|                     1 |       16 |          127.062 |                 10 |              18 |              61 |                0.295447 |                  0.282015 |               10 |                0 |
|                     2 |       16 |          138.812 |                 11 |              20 |              44 |                0.289162 |                  0.273932 |                9 |                0 |
|                     3 |       16 |          138.812 |                 11 |              20 |              44 |                0.289162 |                  0.273932 |                9 |                0 |

## Best layer cases

|   penalty_start_depth | decision_time             | layer_name                             |   lookback_minutes |   leaf_count |   leaf_median |   leaf_max |   mean_top_sector_share |   sector_pure_60_count |   sector_pure_80_count |
|----------------------:|:--------------------------|:---------------------------------------|-------------------:|-------------:|--------------:|-----------:|------------------------:|-----------------------:|-----------------------:|
|                     1 | 2026-01-06 14:44:00+00:00 | flow_return_alignment                  |                 30 |          153 |          10   |         49 |                0.303138 |                      5 |                      0 |
|                     1 | 2026-01-06 15:14:00+00:00 | return_corr_raw_1m                     |                 30 |          206 |           9   |         59 |                0.308461 |                      1 |                      0 |
|                     1 | 2026-01-06 15:44:00+00:00 | block_activity                         |                 30 |           42 |          10.5 |         39 |                0.303702 |                      1 |                      0 |
|                     1 | 2026-01-06 15:44:00+00:00 | return_corr_raw_1m                     |                 30 |          185 |           9   |         60 |                0.300931 |                      1 |                      0 |
|                     1 | 2026-01-06 15:44:00+00:00 | flow_return_alignment                  |                 30 |          209 |          10   |         56 |                0.299985 |                      1 |                      0 |
|                     1 | 2026-01-06 14:44:00+00:00 | absorption                             |                 30 |          202 |          10   |         36 |                0.29925  |                      1 |                      0 |
|                     1 | 2026-01-06 15:14:00+00:00 | large_trade_flow                       |                 30 |           48 |          10   |         41 |                0.313812 |                      0 |                      0 |
|                     1 | 2026-01-06 15:14:00+00:00 | return_corr_cross_sectional_rolling_5m |                 30 |          131 |          10   |         61 |                0.295982 |                      0 |                      0 |
|                     2 | 2026-01-06 14:44:00+00:00 | flow_return_alignment                  |                 30 |          153 |          11   |         37 |                0.302398 |                      5 |                      0 |
|                     2 | 2026-01-06 15:44:00+00:00 | block_activity                         |                 30 |           42 |          10.5 |         39 |                0.303702 |                      1 |                      0 |
|                     2 | 2026-01-06 14:44:00+00:00 | absorption                             |                 30 |          195 |          11   |         36 |                0.299196 |                      1 |                      0 |
|                     2 | 2026-01-06 15:14:00+00:00 | return_corr_raw_1m                     |                 30 |          251 |          12   |         44 |                0.296216 |                      1 |                      0 |
|                     2 | 2026-01-06 15:44:00+00:00 | flow_return_alignment                  |                 30 |          227 |          11   |         42 |                0.293956 |                      1 |                      0 |
|                     2 | 2026-01-06 15:14:00+00:00 | large_trade_flow                       |                 30 |           43 |          11   |         41 |                0.30581  |                      0 |                      0 |
|                     2 | 2026-01-06 15:44:00+00:00 | absorption                             |                 30 |          167 |          11   |         44 |                0.289031 |                      0 |                      0 |
|                     2 | 2026-01-06 15:14:00+00:00 | absorption                             |                 30 |          155 |          12   |         42 |                0.288994 |                      0 |                      0 |
|                     3 | 2026-01-06 14:44:00+00:00 | flow_return_alignment                  |                 30 |          153 |          11   |         37 |                0.302398 |                      5 |                      0 |
|                     3 | 2026-01-06 15:44:00+00:00 | block_activity                         |                 30 |           42 |          10.5 |         39 |                0.303702 |                      1 |                      0 |
|                     3 | 2026-01-06 14:44:00+00:00 | absorption                             |                 30 |          195 |          11   |         36 |                0.299196 |                      1 |                      0 |
|                     3 | 2026-01-06 15:14:00+00:00 | return_corr_raw_1m                     |                 30 |          251 |          12   |         44 |                0.296216 |                      1 |                      0 |
|                     3 | 2026-01-06 15:44:00+00:00 | flow_return_alignment                  |                 30 |          227 |          11   |         42 |                0.293956 |                      1 |                      0 |
|                     3 | 2026-01-06 15:14:00+00:00 | large_trade_flow                       |                 30 |           43 |          11   |         41 |                0.30581  |                      0 |                      0 |
|                     3 | 2026-01-06 15:44:00+00:00 | absorption                             |                 30 |          167 |          11   |         44 |                0.289031 |                      0 |                      0 |
|                     3 | 2026-01-06 15:14:00+00:00 | absorption                             |                 30 |          155 |          12   |         42 |                0.288994 |                      0 |                      0 |

## Recommendation

D2 is the preferred compromise here: D1 fragments earlier and produces many pure but tiny/less structured leaves; D3 delays penalty and retains larger mixed leaves. D2 keeps one coarse level before refinement and then starts controlling giant children.
