# Every-snapshot P1 persistence validation — no sampling

Data: 2026-01-06, all available 388 snapshot times from 14:33 to 21:00 UTC. Matching: greedy one-step path tracking with Jaccard >= 0.25. Minimum community size = 8. Metadata/sector is post-hoc only.

## Main ranking

| layer_name                                 |   snapshots |   communities |   paths | continuation_rate   |   stable_paths_ge15min |   stable_paths_ge30min |   p50_duration_min |   p90_duration_min |   p99_duration_min |   max_duration_min |   p50_size |   p90_size |   max_size |
|:-------------------------------------------|------------:|--------------:|--------:|:--------------------|-----------------------:|-----------------------:|-------------------:|-------------------:|-------------------:|-------------------:|-----------:|-----------:|-----------:|
| block_activity@30m                         |         379 |         13760 |    6587 | 52.3%               |                    129 |                      0 |                  1 |                  5 |              16    |                 29 |         11 |         25 |         77 |
| return_corr_cross_sectional_rolling_5m@30m |         367 |         15663 |    8337 | 46.9%               |                     29 |                      0 |                  1 |                  4 |              10    |                 25 |         47 |         87 |        172 |
| return_corr_raw_1m@15m                     |         381 |         28073 |   20035 | 28.7%               |                      0 |                      0 |                  1 |                  2 |              11.66 |                 14 |         57 |        104 |        219 |
| return_corr_raw_1m@30m                     |         371 |         23159 |   17164 | 26.0%               |                     31 |                      2 |                  1 |                  2 |               7    |                 30 |         56 |         94 |        212 |
| large_trade_flow@30m                       |         379 |         13240 |   10010 | 24.5%               |                      7 |                      0 |                  1 |                  2 |               7    |                 28 |         10 |         33 |        113 |
| return_corr_cross_sectional_1m@15m         |         381 |         25965 |   20247 | 22.1%               |                      0 |                      0 |                  1 |                  2 |               7    |                 14 |         62 |        103 |        225 |
| return_corr_cross_sectional_1m@30m         |         371 |         20327 |   16088 | 20.9%               |                     16 |                      1 |                  1 |                  2 |               6    |                 30 |         62 |         98 |        207 |
| flow_return_alignment@5m                   |         388 |          5220 |    4495 | 13.9%               |                      0 |                      0 |                  1 |                  2 |               4    |                  7 |         10 |         13 |         26 |
| volume_expansion@15m                       |         383 |         65477 |   56714 | 13.4%               |                      1 |                      0 |                  1 |                  1 |               6    |                 17 |         13 |         33 |        224 |
| absorption@5m                              |         388 |         53972 |   48028 | 11.0%               |                      0 |                      0 |                  1 |                  1 |               4    |                 13 |         14 |         45 |        135 |
| report_latency@5m                          |         388 |         56261 |   50573 | 10.1%               |                      0 |                      0 |                  1 |                  1 |               4    |                 11 |         16 |         46 |        151 |
| price_impact@15m                           |         383 |         63311 |   57093 | 9.8%                |                      1 |                      0 |                  1 |                  1 |               4    |                 26 |         11 |         20 |        128 |
| volume_expansion@5m                        |         388 |         35631 |   32450 | 8.9%                |                      0 |                      0 |                  1 |                  1 |               4    |                 10 |         36 |         85 |        179 |
| price_impact@5m                            |         388 |         49674 |   45482 | 8.5%                |                      1 |                      0 |                  1 |                  1 |               3    |                 16 |         15 |         45 |        123 |
| absorption@15m                             |         383 |         69515 |   63742 | 8.3%                |                      1 |                      0 |                  1 |                  1 |               4    |                 15 |         11 |         22 |        162 |

## Main conclusion

Every-snapshot tracking is stricter than the earlier sampled version. The structural layers remain the most persistent: `block_activity@30m`, `return_corr_cross_sectional_rolling_5m@30m`, `return_corr_raw_1m@30m`, and `large_trade_flow@30m`.

Most event/microstructure layers produce many communities, but most paths last only 1–2 minutes. This supports layer-local P1, with cross-layer composition deferred to P2.

## Important correction vs sampled run

The previous sampled full-day run overestimated long survival for some event layers. With every snapshot included:

- `block_activity@30m` still has the best continuation rate and 129 paths surviving at least 15 minutes, but none survives 30 minutes under strict Jaccard tracking.
- `return_corr_raw_1m@30m` has 31 paths surviving at least 15 minutes and 2 paths surviving 30 minutes.
- `return_corr_cross_sectional_1m@30m` has 16 paths surviving at least 15 minutes and 1 path surviving 30 minutes.
- Most venue, flow, intensity, odd-lot, report-latency, and absorption layers are very short-lived in strict minute-by-minute tracking.
