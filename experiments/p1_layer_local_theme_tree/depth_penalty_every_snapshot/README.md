# Full-day every-snapshot D1/D2/D3 penalty-start comparison

Data: 2026-01-06 all available snapshots for selected core 30m layer-scales. No snapshot sampling. `top_sector_share` is post-hoc evaluation only; sector metadata is not used in graph construction, split scoring, or tree training.

Selected layer-scales: return_corr_raw_1m@30m, return_corr_cross_sectional_rolling_5m@30m, block_activity@30m, large_trade_flow@30m, absorption@30m, flow_return_alignment@30m.

Implementation: fast graph-only connected-component validation. Coarse level keeps top-5 neighbors per node. Penalty/refinement level keeps top-2 neighbors per node. D1 applies penalty immediately; D2 after one coarse level; D3 after two coarse levels.

## Aggregate

|   penalty_start_depth |   groups |   avg_leaf_count |   median_leaf_size |   p90_leaf_size |   max_leaf_size |   mean_top_sector_share |   median_top_sector_share |   p90_top_sector_share |   mean_top_industry_share |   sector_pure_50 |   sector_pure_60 |   sector_pure_80 |
|----------------------:|---------:|-----------------:|-------------------:|----------------:|----------------:|------------------------:|--------------------------:|-----------------------:|--------------------------:|-----------------:|-----------------:|-----------------:|
|                     1 |     2254 |          63.6606 |                 10 |            18   |            3878 |                0.295623 |                  0.278282 |               0.398158 |                  0.207226 |             5805 |              752 |                7 |
|                     2 |     2254 |          67.1051 |                 10 |            18.6 |            3878 |                0.294465 |                  0.276856 |               0.397066 |                  0.206526 |             5881 |              766 |                6 |
|                     3 |     2254 |          67.1051 |                 10 |            18.6 |            3878 |                0.294465 |                  0.276856 |               0.397066 |                  0.206526 |             5881 |              766 |                6 |

## By-layer top rows

|   penalty_start_depth | layer_name                             |   lookback_minutes |   snapshots |   avg_leaf_count |   median_leaf_size |   max_leaf_size |   mean_top_sector_share |   sector_pure_60 |   sector_pure_80 |
|----------------------:|:---------------------------------------|-------------------:|------------:|-----------------:|-------------------:|----------------:|------------------------:|-----------------:|-----------------:|
|                     1 | flow_return_alignment                  |                 30 |         379 |         156.317  |                 10 |            1273 |                0.294269 |              238 |                0 |
|                     1 | absorption                             |                 30 |         379 |         112.842  |                 10 |            1091 |                0.294136 |              237 |                1 |
|                     1 | large_trade_flow                       |                 30 |         379 |          33.6675 |                  9 |             694 |                0.298757 |               99 |                0 |
|                     1 | return_corr_raw_1m                     |                 30 |         371 |          29.6253 |                  9 |            3878 |                0.296965 |               89 |                5 |
|                     1 | block_activity                         |                 30 |         379 |          32.0871 |                 10 |             343 |                0.297867 |               62 |                1 |
|                     1 | return_corr_cross_sectional_rolling_5m |                 30 |         367 |          15.1717 |                 10 |            2668 |                0.291649 |               27 |                0 |
|                     2 | flow_return_alignment                  |                 30 |         379 |         163.319  |                 10 |            1273 |                0.293207 |              243 |                1 |
|                     2 | absorption                             |                 30 |         379 |         121.66   |                 10 |            1091 |                0.292409 |              242 |                0 |
|                     2 | return_corr_raw_1m                     |                 30 |         371 |          32.3854 |                  9 |            3878 |                0.297372 |              103 |                5 |
|                     2 | large_trade_flow                       |                 30 |         379 |          33.4644 |                  9 |             694 |                0.297071 |               89 |                0 |
|                     2 | block_activity                         |                 30 |         379 |          33.628  |                 10 |             343 |                0.296558 |               64 |                0 |
|                     2 | return_corr_cross_sectional_rolling_5m |                 30 |         367 |          15.8174 |                 10 |            2668 |                0.290098 |               25 |                0 |
|                     3 | flow_return_alignment                  |                 30 |         379 |         163.319  |                 10 |            1273 |                0.293207 |              243 |                1 |
|                     3 | absorption                             |                 30 |         379 |         121.66   |                 10 |            1091 |                0.292409 |              242 |                0 |
|                     3 | return_corr_raw_1m                     |                 30 |         371 |          32.3854 |                  9 |            3878 |                0.297372 |              103 |                5 |
|                     3 | large_trade_flow                       |                 30 |         379 |          33.4644 |                  9 |             694 |                0.297071 |               89 |                0 |
|                     3 | block_activity                         |                 30 |         379 |          33.628  |                 10 |             343 |                0.296558 |               64 |                0 |
|                     3 | return_corr_cross_sectional_rolling_5m |                 30 |         367 |          15.8174 |                 10 |            2668 |                0.290098 |               25 |                0 |

## Interpretation

- D1 is most aggressive: smaller leaves and higher sector-pure counts, but higher fragmentation risk.

- D2 is the preferred architecture compromise: one coarse trading-structure layer first, then giant-child control.

- D3 preserves coarse trading behavior longer but leaves larger mixed themes and does not improve sector purity.

- Sector purity remains a post-hoc metric only. These graph themes are still primarily trading-behavior / risk-basket / microstructure structures, not supervised industry labels.
