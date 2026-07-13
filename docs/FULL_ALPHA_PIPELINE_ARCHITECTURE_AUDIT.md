# Full P0/P1/P2 alpha pipeline architecture audit

## Scope

This audit covers the complete production path rather than only P2 theme returns:

```text
P0 canonical edges
  -> strict date/layer/scale edge shards
  -> P0 node alpha / edge spillover / graph state / P0 evaluation
  -> P1 B50/B35 themes / relation edges / temporal edges
  -> theme returns
  -> relation spillover
  -> intraday and end-of-day relation factors
  -> monthly and global evaluation
```

The point-in-time formulas are unchanged. The changes in this audit concern partition identity, bounded memory, deterministic time ordering, process lifecycle, atomic outputs, and evaluation scope.

## Problems found and corrections

### 1. P0 scale identity could silently collapse to `default`

The old edge sharder accepted input without `scale` or `lookback_minutes` and silently assigned `scale=default`. It also opened a new writer pool for each source file, so multiple files for the same date/layer/scale could replace an earlier shard.

The v2 contract now:

- requires `scale` or `lookback_minutes` by default;
- validates that supplied scale agrees with derived lookback scale;
- permits `default` only with the explicit legacy flag `--allow-default-scale`;
- groups all source files for a date into one writer pool;
- validates chronological ordering;
- writes `.tmp` files and atomically promotes them only after success.

The six-month runner now creates strict local P0 alpha shards before P0 alpha extraction. Labels remain in the canonical date directories; edge computation uses the physical date/layer/scale shards.

### 2. P0 alpha and spillover

P0 node features and edge spillover now process Parquet row groups and write snapshot results incrementally. They no longer concatenate an entire day of snapshot outputs in memory.

P0 graph state remains a compact stage: it emits one aggregate row per decision-time/layer/scale. Its in-memory result is small compared with node and edge outputs.

P0 evaluation was still a global collection point. It now uses:

```text
one input partition
  -> snapshot metrics
  -> temporary metric shard
  -> streamed reducer
  -> final Parquet + CSV + online summary
```

No parent process receives every metric row as Python objects, and the reducer reads metric shards in bounded batches.

### 3. P1 theme construction was output-streaming but input-full-memory

The previous P1 builder read the full edge shard into Pandas, sorted it, and then iterated snapshots. The new `p1-streaming-v2` builder reads bounded Parquet batches and yields complete `decision_time` groups while retaining only one carry group across batch boundaries.

The stream requires globally non-decreasing `decision_time`. An unsorted shard now fails explicitly instead of falling back to a full-file sort.

P1 outputs are atomic. Interrupted runs leave `.tmp` files, not valid final Parquet files. `--skip-existing` accepts a partition only when the manifest contract and all non-empty declared outputs agree.

### 4. P1 temporal matching complexity

The previous temporal continuation calculation tested every previous leaf against every current leaf and repeatedly computed set intersections.

The new implementation builds a current member-to-theme index and counts overlaps for each previous theme. It preserves the same best-successor concept while reducing unnecessary Cartesian set operations.

### 5. P1 scheduler lifecycle

The P1 scheduler previously submitted every shard future immediately and used ordinary subprocesses. It now:

- derives workers from CPU and RAM budgets;
- keeps at most `workers` shard processes in flight;
- launches each builder through the process-tree runtime;
- uses Windows Job Objects so abnormal parent termination removes descendants;
- validates the P1 contract and physical outputs before marking a shard complete.

### 6. Theme returns

Theme returns previously streamed output but still loaded the full membership partition. They now consume `theme_memberships.parquet` as complete timestamp groups from bounded Parquet batches.

Labels are loaded once inside each outer partition process and indexed by `decision_time`. Snapshot computations can use bounded inner threads, but output is emitted in original timestamp order. This ordering contract is required by downstream dual-stream joins.

### 7. Relation spillover

The old relation stage loaded full theme returns and full relation edges, then created source and target copies. The new stage performs a constant-memory merge of two sorted streams:

```text
theme return group at t
  <-> relation edge group at t
  -> snapshot-local symmetric expansion
  -> spillover output at t
```

Only matching timestamps are processed. Symmetric expansion remains snapshot-local, and bounded inner parallelism is ordered so the output remains sorted.

### 8. Intraday and daily factor stages

Intraday relation factors now read and transform one snapshot at a time. Cross-sectional normalization remains inside that snapshot, and each output batch is PIT-audited before writing.

Daily factors intentionally retain the complete session because temporal episode construction and end-of-day aggregation require all session observations. This is a semantic requirement rather than an accidental full-file read. Daily workers remain more RAM-constrained than intraday workers.

A valid P1 partition with zero temporal edges is now distinguished from a missing/failed P1 partition. The former creates singleton episodes; the latter still fails.

### 9. Evaluation architecture and scope

P2 evaluation no longer executes:

```python
frames = [pd.read_parquet(path) for path in all_files]
pd.concat(frames)
```

Each feature partition produces a temporary metric shard. The final reducer streams those shards, writes final metrics incrementally, and maintains exact day/snapshot sets plus running IC/spread totals.

Evaluation CLI now accepts date, layer, and scale filters. Monthly scheduler runs therefore read only the current month. The six-month runner no longer evaluates the same month twice or rescan all prior months after every monthly run. One global evaluation is performed after all months complete.

## 24-core / 128GB resource plan

The scheduler reserves 24GB for Windows, filesystem cache, Arrow/Pandas transients and parent processes, then schedules at most 90% of the remaining 104GB: a 93.6GB planning ceiling.

Balanced defaults after the streaming changes:

| Stage | Outer processes | Inner threads | CPU slots | Estimated peak RAM |
|---|---:|---:|---:|---:|
| P0 node | 20 | 1 | 20 | 90GB |
| P0 edge spillover | 17 | 1 | 17 | 93.5GB |
| P0 graph state | 24 | 1 | 24 | 48GB |
| P0 evaluation | 24 | 1 | 24 | 72GB |
| Theme returns | 6 | 4 | 24 | 60GB |
| Relation spillover | 6 | 4 | 24 | 54GB |
| Intraday relation features | 24 | 1 | 24 | 84GB |
| Daily relation features | 13 | 1 | 13 | 91GB |
| P2 evaluation | 24 | 1 | 24 | 72GB |

These values are conservative planning estimates, not measured guarantees. The generated `p2_24core_schedule_plan.json` remains the runtime source of truth.

## Contracts that now fail closed

The pipeline refuses to continue when:

- P0 scale identity is absent, unless legacy fallback is explicitly requested;
- a streamed Parquet input reverses `decision_time` order;
- a P1 manifest claims data that is not physically present;
- a P2 past return is unavailable at feature time;
- an intraday target begins at or before feature time;
- a daily target is not next-open executable;
- a daily temporal file is absent without a completed zero-edge P1 manifest;
- any feature output contains failed PIT rows.

## Remaining limits

1. Daily feature construction still requires one full session partition in memory by design. If daily partitions later become much larger, the next step is a two-pass episode reducer, not snapshot-independent processing.
2. Labels are loaded once per outer P0/theme partition process. A future date-oriented worker can reuse one label frame across multiple layers/scales, but it would reduce partition-level scheduling flexibility.
3. The resource estimates must still be calibrated with measured peak RSS from a complete month on the 24-core/128GB workstation.
4. The architecture tests require the project environment with Pandas and PyArrow. They verify batch-boundary grouping, ordering, temporal matching, RAM plans, and month scoping.
5. This audit changes execution architecture, not the financial interpretation or statistical significance of the factors. P0, P1 and P2 alpha validation remains a separate research layer.
