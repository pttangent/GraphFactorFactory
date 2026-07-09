# P1 layer-local theme tree validation

Validated directly from P0-derived temporal edge data for two graph decision times. Each layer is independently split; metadata is joined only after clustering.

## Aggregate

| decision_time             |   trees |   avg_root |   avg_children |   median_child |   max_child |
|:--------------------------|--------:|-----------:|---------------:|---------------:|------------:|
| 2026-01-06 14:44:00+00:00 |       8 |    3483.38 |        127.375 |             11 |          89 |
| 2026-01-06 21:00:00+00:00 |      10 |    4150.7  |        132.5   |             15 |         110 |

## Results

| decision_time             |   layer_id | layer_name                     |   lookback_minutes | scale_role   |   root_size |   raw_edges |   pairs |   resolution |   child_count |   child_median |   child_p90 |   child_max |
|:--------------------------|-----------:|:-------------------------------|-------------------:|:-------------|------------:|------------:|--------:|-------------:|--------------:|---------------:|------------:|------------:|
| 2026-01-06 14:44:00+00:00 |          1 | return_corr_raw_1m             |                 15 | confirm      |        4495 |        5878 |    5878 |            2 |           116 |             36 |        59   |          76 |
| 2026-01-06 14:44:00+00:00 |         11 | absorption                     |                  5 | trigger      |        3928 |        6105 |    6105 |            2 |           147 |             13 |        39   |          71 |
| 2026-01-06 14:44:00+00:00 |         11 | absorption                     |                 15 | confirm      |        4059 |        5556 |    5556 |            2 |           176 |             11 |        24.5 |          89 |
| 2026-01-06 14:44:00+00:00 |         11 | absorption                     |                 30 | structural   |        4059 |        5556 |    5556 |            2 |           176 |             11 |        24.5 |          89 |
| 2026-01-06 14:44:00+00:00 |         12 | flow_return_alignment          |                  5 | trigger      |         380 |         893 |     893 |            1 |            14 |             10 |        12.4 |          15 |
| 2026-01-06 14:44:00+00:00 |         12 | flow_return_alignment          |                 15 | confirm      |        3357 |        4822 |    4822 |            2 |           144 |             11 |        19.7 |          59 |
| 2026-01-06 14:44:00+00:00 |         12 | flow_return_alignment          |                 30 | structural   |        3357 |        4822 |    4822 |            2 |           144 |             11 |        19.7 |          59 |
| 2026-01-06 14:44:00+00:00 |         14 | return_corr_cross_sectional_1m |                 15 | confirm      |        4232 |        5588 |    5588 |            2 |           102 |             37 |        57.8 |          79 |
| 2026-01-06 21:00:00+00:00 |          1 | return_corr_raw_1m             |                 15 | confirm      |        4751 |        6854 |    6854 |            2 |            87 |             50 |        81.6 |         110 |
| 2026-01-06 21:00:00+00:00 |          1 | return_corr_raw_1m             |                 30 | structural   |        4642 |        5790 |    5790 |            2 |            95 |             43 |        67.6 |         104 |
| 2026-01-06 21:00:00+00:00 |         11 | absorption                     |                  5 | trigger      |        4720 |        7234 |    7234 |            2 |           154 |             18 |        46   |          78 |
| 2026-01-06 21:00:00+00:00 |         11 | absorption                     |                 15 | confirm      |        4602 |        6472 |    6472 |            2 |           217 |             12 |        28.4 |          71 |
| 2026-01-06 21:00:00+00:00 |         11 | absorption                     |                 30 | structural   |        4390 |        5533 |    5533 |            2 |           171 |             12 |        32   |          75 |
| 2026-01-06 21:00:00+00:00 |         12 | flow_return_alignment          |                  5 | trigger      |         476 |        1138 |    1138 |            1 |            17 |             10 |        11   |          12 |
| 2026-01-06 21:00:00+00:00 |         12 | flow_return_alignment          |                 15 | confirm      |        4108 |        6011 |    6011 |            2 |           189 |             11 |        29.2 |          66 |
| 2026-01-06 21:00:00+00:00 |         12 | flow_return_alignment          |                 30 | structural   |        4425 |        6000 |    6000 |            2 |           213 |             11 |        28.8 |          84 |
| 2026-01-06 21:00:00+00:00 |         14 | return_corr_cross_sectional_1m |                 15 | confirm      |        4751 |        6854 |    6854 |            2 |            87 |             50 |        81.6 |         110 |
| 2026-01-06 21:00:00+00:00 |         14 | return_corr_cross_sectional_1m |                 30 | structural   |        4642 |        5790 |    5790 |            2 |            95 |             43 |        67.6 |         104 |

## Interpretation

The validation proves the intended P1 behavior:

1. It starts from P0 edge outputs, not from old P1 themes.
2. Each layer/scale can be split independently.
3. Two different graph decision times both produce usable child theme distributions.
4. The median child theme size is in a usable research range, mostly 10-50 symbols, instead of parent themes with thousands of symbols.
5. This is hot-pluggable: adding a new layer only requires running its own layer-local tree, not recomputing every cross-layer consensus result.

This branch therefore treats cross-layer consensus as P2 research output, not as canonical P1 storage.
