# Phase 1 temporal round 4: continuous 60-minute A/B

## Data

- Trade date: 2026-01-07.
- Window: 15:10-16:09 UTC.
- 60 consecutive one-minute snapshots.
- All 13 layers.
- Observed-edge sparse consensus rebuilt from the real canonical edges.
- Typical final themes per minute: 186-243.

## Critical finding: minute timestamps are not 60 independent graph states

Of the 59 adjacent minute transitions:

- 31 have exactly identical edge hashes.
- The same 31 transitions also have exactly identical theme sets.
- Only 29 unique graph states exist in the 60-minute window.

Therefore lifecycle age, birth confirmation and path survival must not advance on every timestamp. They must advance on graph-state changes, while wall-clock duration is tracked separately.

## Naive minute-clock results

| Method | Continuation | Paths >=3 | Paths >=6 | Paths >=12 | Revivals |
|---|---:|---:|---:|---:|---:|
| global 0.50 | 80.44% | 93.17% | 5.59% | 1.20% | 0 |
| stable core 0.45 | 80.80% | 93.51% | 7.36% | 1.58% | 0 |
| stable core 0.50 | 80.42% | 93.48% | 5.66% | 1.20% | 0 |
| stable core 0.50 + grace/confirm | 80.55% | 93.60% | 6.09% | 1.37% | 19 |

These values are inflated by duplicate graph states.

## Effective-state-clock results

The A/B was repeated on the 29 unique graph states only.

| Method | Continuation | Paths >=3 | Paths >=6 | Paths >=12 | Revivals |
|---|---:|---:|---:|---:|---:|
| global 0.50 | 59.49% | 38.80% | 1.55% | 0.66% | 0 |
| stable core 0.45 | **60.20%** | **39.76%** | **1.94%** | 0.55% | 0 |
| stable core 0.50 | 59.38% | 38.92% | 1.47% | 0.54% | 0 |
| stable core 0.50 + grace/confirm | 59.89% | 39.33% | 1.57% | 0.55% | 38 |
| plus sector similarity | 59.71% | 39.13% | 1.48% | 0.59% | 34 |
| threshold 0.55 plus sector | 58.97% | 38.50% | 1.04% | 0.58% | 27 |

## Decision

1. Adopt stable-core matching threshold 0.45 as the current experimental matcher.
2. Do not claim an 80% continuation rate; the defensible effective-state estimate is about 60%.
3. Do not enable fixed temporal Leiden coupling globally; prior real-data tests showed negative impact.
4. Add graph-state hashing to the lifecycle clock.
5. Duplicate minute snapshots become `carry_forward` observations. They increase wall-clock duration but not effective age or confirmation count.
6. Grace/revival remains experimental. It creates valid-looking revivals but has not improved long-path survival enough to justify production use.
7. Sector similarity is a diagnostic, not a matching reward, until the canonical symbol_id-to-symbol dimension is restored.

## Required output fields

- `graph_state_hash`.
- `graph_state_index`.
- `graph_state_changed`.
- `wall_clock_age_minutes`.
- `effective_age_frames`.
- `carry_forward_count`.
- `birth_confirmation_state_count`.

## Reliability warning

The current canonical pack has one-minute timestamps but repeated graph states. Any lifecycle analysis that counts every timestamp as an independent transition materially overstates persistence.
