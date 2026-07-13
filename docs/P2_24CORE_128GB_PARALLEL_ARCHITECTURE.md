# P2 24-core / 128GB parallel architecture

## Resource contract

The workstation is treated as a single global resource domain:

- 24 CPU cores
- 128GB physical RAM
- 24GB reserved for Windows, filesystem cache and transient Arrow/Pandas peaks
- at most one heavy pipeline stage active at a time
- every heavy Python process handles one partition and then exits

## Stage plan (`balanced`)

| Stage | Outer processes | Threads per process | CPU slots | Estimated peak RAM |
|---|---:|---:|---:|---:|
| P0 node | 23 | 1 | 23 | 103.5GB |
| P0 edge spillover | 18 | 1 | 18 | 99GB |
| P0 graph state | 24 | 1 | 24 | 48GB |
| P0 evaluation | 17 | 1 | 17 | 102GB |
| Theme returns | 6 | 4 | 24 | 84GB |
| Relation spillover | 8 | 3 | 24 | 96GB |
| Intraday relation features | 20 | 1 | 20 | 100GB |
| Daily relation features | 14 | 1 | 14 | 98GB |

The scheduler writes the resolved plan to `p2_24core_schedule_plan.json` before execution.

## OOM and orphan-process fixes

1. Daily label injection uses eight bounded threads sharing one prepared label table. It no longer creates 32 independent Python processes.
2. The six-month runner is stage-serial: copy, label injection, P2, evaluation, cleanup. A later month cannot inject labels while the current month is computing P2.
3. `ProcessPoolExecutor` submissions are bounded and workers use `max_tasks_per_child=1`, forcing Windows to reclaim Pandas/Arrow high-water memory after each partition.
4. Theme and relation snapshot tasks are lazily grouped and have at most `2 × inner_workers` tasks in flight. `groups = list(...)` and all-snapshot future lists were removed.
5. Symmetric relation expansion is performed per snapshot rather than duplicating the full-day edge table.
6. P0 node and edge outputs are written row-group by row-group rather than concatenating every snapshot in memory.
7. Subprocess trees are attached to a Windows Job Object with `KILL_ON_JOB_CLOSE`; abnormal launcher exit removes multiprocessing descendants instead of leaving orphan Python workers.

## Recommended command

```bat
python scripts\run_p2_24core_scheduler.py ^
  --p0-root D:\GFF_Streaming_Workspace\p0 ^
  --p1-root D:\GFF_Streaming_Workspace\p1 ^
  --labels-root D:\GFF_Streaming_Workspace\p0 ^
  --p2-root D:\GFF_Streaming_Workspace\p2_out ^
  --cores 24 ^
  --ram-gb 128 ^
  --reserve-ram-gb 24 ^
  --profile balanced ^
  --inner-workers 0 ^
  --skip-existing
```

`--inner-workers 0` lets the scheduler choose 6×4 for theme returns and 8×3 for relation spillover. Do not restore 32-process label injection or `--cores 28` on this machine.
