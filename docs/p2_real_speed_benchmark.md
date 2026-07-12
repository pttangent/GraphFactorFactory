# P2 real-data speed benchmark

This note records an actual speed comparison on real P1/labels data, not a pseudo-code claim.

## Benchmark input

Real files used in the ChatGPT sandbox:

```text
labels.parquet
  date: 2026-01-07
  rows: 1,875,727
  unique symbol_id: 5,511
  decision_time range: 2026-01-07 14:31:00+00:00 to 2026-01-07 21:00:00+00:00

P1 partition
  date: 2026-01-07
  layer_id: 9
  scale: 30m
  theme_memberships.parquet rows: 3,357,416
  theme_memberships.parquet row groups: 379
  theme_relation_edges.parquet rows: 769,755
  theme_relation_edges.parquet row groups: 379
```

The benchmark intentionally compares a bounded reproduction of the old time-misaligned shape against the new decision-time aligned implementation.

## Results: theme_returns, first 5 row groups

| Variant | Row groups | Membership rows | Time groups | Joined rows | Output rows | Batches | Elapsed sec | Joined rows / sec | Approx RSS peak MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| old_bad_same_times | 5 | 43,900 | 5 | 218,766 | 49,108 | 5 | 1.0595 | 206,472 | 909 |
| aligned_iw1 | 5 | 43,900 | 5 | 43,832 | 9,828 | 5 | 0.3152 | 139,059 | 1,021 |
| aligned_iw2 | 5 | 43,900 | 5 | 43,832 | 9,828 | 5 | 0.4125 | 106,267 | 1,060 |
| aligned_iw4 | 5 | 43,900 | 5 | 43,832 | 9,828 | 5 | 0.5035 | 87,057 | 1,084 |

Interpretation:

```text
old_bad_same_times / aligned_iw1 elapsed = 1.0595 / 0.3152 = 3.36x slower
old_bad_same_times / aligned_iw1 joined rows = 218,766 / 43,832 = 4.99x row explosion
```

Even on only five row groups, the old bug repeats full sampled membership for each label time and already creates about 5x more join rows.  On a full day, the old shape can approach a much larger repetition factor because it can repeat against many more label decision_time chunks.

## Results: theme_returns, first 20 row groups

| Variant | Row groups | Membership rows | Time groups | Joined rows | Output rows | Batches | Elapsed sec | Joined rows / sec | Approx RSS peak MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| aligned_iw1 | 20 | 180,980 | 20 | 175,906 | 46,486 | 20 | 1.3235 | 132,911 | 978 |
| aligned_iw2 | 20 | 180,980 | 20 | 175,906 | 46,486 | 20 | 1.5378 | 114,387 | 1,189 |
| aligned_iw4 | 20 | 180,980 | 20 | 175,906 | 46,486 | 20 | 2.2492 | 78,207 | 1,269 |
| aligned_iw8 | 20 | 180,980 | 20 | 175,906 | 46,486 | 20 | 2.2175 | 79,327 | 1,328 |

Interpretation:

```text
inner_workers=1 is fastest for this real partition sample.
inner_workers=2 is 16.2% slower.
inner_workers=4 is 70.0% slower.
inner_workers=8 is 67.6% slower.
```

This means the fastest architecture for this stage is not inner threading inside one partition.  The faster and safer architecture is:

```text
inner_workers = 1
outer workers = many independent partitions/dates
```

In other words, saturate the 24-core machine by running many independent layer/scale/date partitions, not by making a single partition spawn 8 threads.

## P0 graph-alpha validation

P0 direct graph alpha was also tested on real P0 edges and labels:

```text
P0 edges parquet
  rows: 48,963,776
  row groups: 370

edge-spillover, first 1 row group:
  output_rows: 39,495
  elapsed_sec: 3.60

edge-spillover, first 5 row groups:
  output_rows: 199,864
  elapsed_sec: 9.08
```

This confirms P0 graph-alpha stages are viable, but should be row-group streamed.  Do not load full-day P0 edges into memory for feature extraction.

## Recommended production scheduler setting

For the current code path, the best default is:

```powershell
python scripts/run_p2_24core_scheduler.py ^
  --p0-root C:\GFF_Cache\p2_2day_pack\p0 ^
  --p1-root C:\GFF_Cache\p2_2day_pack\p1 ^
  --labels-root C:\GFF_Cache\p2_2day_pack\labels ^
  --p2-root C:\GFF_Cache\p2_alpha_lab_full ^
  --dates 2026-01-07,2026-01-08 ^
  --layers 3,6,8,9,11 ^
  --scales 15m,30m ^
  --profile max ^
  --cores 24 ^
  --target-cpu 1.0 ^
  --inner-workers 1 ^
  --skip-existing
```

Only test `--inner-workers 2` or higher if CPU is visibly underutilized while memory and disk are stable.  The benchmark above shows that inner workers were slower in this environment.

## Repro command

Use the benchmark harness:

```powershell
python scripts/benchmark_p2_alpha_lab.py ^
  --membership C:\GFF_Cache\p2_2day_pack\p1\date=2026-01-07\layer_id=9\scale=30m\theme_memberships.parquet ^
  --labels C:\GFF_Cache\p2_2day_pack\labels\date=2026-01-07\labels.parquet ^
  --out-dir C:\GFF_Cache\p2_benchmark_outputs ^
  --row-groups 5,20 ^
  --inner-workers 1,2,4,8 ^
  --include-old
```

The output will include:

```text
benchmark_results.json
benchmark_results.csv
aligned_rg*_iw*.parquet
old_bad_same_times_rg*.parquet
```
