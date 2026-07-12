# P2 24-Core OOM-Safe Scheduling Plan

This plan targets a 24-core workstation and tries to keep CPU busy while avoiding the memory explosion caused by nested concurrency.

## Problem

The P2 pipeline has two different concurrency layers:

1. outer process workers from `p2_alpha_daily_features.py --workers`;
2. inner per-decision-time multithreading inside `build_returns_one` and `relation_one` introduced around commit `a32e169`.

At that commit the inner fanout is fixed at 8.

So this command is unsafe:

```powershell
python scripts/p2_alpha_daily_features.py build-theme-returns ... --workers 16
```

It is not truly 16-way. It can become roughly:

```text
16 outer processes * 8 inner threads = 128 execution slots
```

That can OOM and can also waste CPU on context switching.

## Target

For a 24-core CPU, target about 80% utilization:

```text
24 cores * 0.80 ~= 19 CPU slots
```

But for nested stages, CPU slots are approximate. Pandas merge/groupby, parquet reads, and memory bandwidth prevent perfect scaling. The scheduler therefore uses a small number of outer workers for nested stages and more workers only for single-level stages.

## Default balanced profile

Use:

```powershell
python scripts/run_p2_24core_scheduler.py ^
  --p1-root C:\GFF_Cache\p1_b50_b35_sharded ^
  --labels-root D:\DEV\US-Stock\GraphFactorFactory\data\graph_store_6m\canonical ^
  --p2-root C:\GFF_Cache\p2_alpha_lab ^
  --dates 2026-01-07,2026-01-08,2026-01-09 ^
  --layers 3,6,8,9,11 ^
  --scales 15m,30m ^
  --profile balanced ^
  --skip-existing
```

Balanced profile currently plans:

```text
build-theme-returns:      3 outer workers * 8 inner fanout ~= 24 slots
relation-spillover:       3 outer workers * 8 inner fanout ~= 24 slots
daily-relation-features: 19 outer workers * 1 inner fanout ~= 19 slots
```

This is intentionally not `--workers 16` for the nested stages. The nested stages are memory-heavy; the daily aggregation stage is much lighter and can use more outer workers.

## Safe profile

Start here if memory pressure is high:

```powershell
python scripts/run_p2_24core_scheduler.py ^
  --p1-root C:\GFF_Cache\p1_b50_b35_sharded ^
  --labels-root D:\DEV\US-Stock\GraphFactorFactory\data\graph_store_6m\canonical ^
  --p2-root C:\GFF_Cache\p2_alpha_lab ^
  --dates 2026-01-07,2026-01-08,2026-01-09 ^
  --layers 8,9 ^
  --scales 15m,30m ^
  --profile safe ^
  --skip-existing
```

Safe profile caps nested stages around 2 outer workers.

## Aggressive profile

Use this only after balanced is stable:

```powershell
python scripts/run_p2_24core_scheduler.py ^
  --p1-root C:\GFF_Cache\p1_b50_b35_sharded ^
  --labels-root D:\DEV\US-Stock\GraphFactorFactory\data\graph_store_6m\canonical ^
  --p2-root C:\GFF_Cache\p2_alpha_lab ^
  --layers 3,6,8,9,11 ^
  --scales 15m,30m ^
  --profile aggressive ^
  --skip-existing
```

Aggressive profile can use up to 4 nested outer workers. This can exceed the nominal 80% slot budget, but may be useful when parquet I/O or pandas internals underutilize CPU.

## Dry run

Always inspect the plan first:

```powershell
python scripts/run_p2_24core_scheduler.py ^
  --p1-root C:\GFF_Cache\p1_b50_b35_sharded ^
  --labels-root D:\DEV\US-Stock\GraphFactorFactory\data\graph_store_6m\canonical ^
  --p2-root C:\GFF_Cache\p2_alpha_lab ^
  --profile balanced ^
  --dry-run
```

The scheduler writes:

```text
<p2-root>/p2_24core_schedule_plan.json
```

This file records the chosen workers, estimated slots, filters, and profile.

## Stage-by-stage recovery

If one stage fails, rerun only that stage:

```powershell
python scripts/run_p2_24core_scheduler.py ... --stage theme    --skip-existing
python scripts/run_p2_24core_scheduler.py ... --stage relation --skip-existing
python scripts/run_p2_24core_scheduler.py ... --stage daily    --skip-existing
python scripts/run_p2_24core_scheduler.py ... --stage eval
```

## Rules

Do not do this on a32e169-style nested P2:

```powershell
--workers 16
```

for `build-theme-returns` or `relation-spillover`.

Use the scheduler instead. It keeps nested outer workers low and lets the daily aggregation stage use most of the 24-core budget.
