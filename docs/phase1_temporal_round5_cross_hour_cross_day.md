# Phase 1 temporal round 5: two-day cross-hour and cross-day validation

## Sampling design

Two real canonical trading dates were used:

- 2026-01-05
- 2026-01-07

The dates are separated by one missing trading date, so the cross-day result is a deliberately hard persistence test rather than an adjacent-session estimate.

For each day, seven ten-minute windows were sampled:

- 15:30-15:39 UTC
- 16:30-16:39 UTC
- 17:30-17:39 UTC
- 18:30-18:39 UTC
- 19:30-19:39 UTC
- 20:30-20:39 UTC
- 21:10-21:19 UTC

Total:

- 140 timestamps
- 32,075 rebuilt theme instances
- all 13 layers
- observed-edge sparse consensus

## Repeated graph states

Each day still contains repeated minute graphs:

| Date | Timestamps | Unique graph states | Duplicate ratio |
|---|---:|---:|---:|
| 2026-01-05 | 70 | 35 | 50.0% |
| 2026-01-07 | 70 | 34 | 51.4% |

All reported effective-state metrics remove transitions where the graph hash is unchanged.

## Effective-state results

### Within each ten-minute window

| Matcher | Mean continuation | Median continuation | Median overlap | Median Jaccard | Median family Jaccard |
|---|---:|---:|---:|---:|---:|
| member overlap 0.50 | 73.23% | 94.42% | 1.00 | 1.00 | 1.00 |
| mechanism score 0.45 | 73.57% | 93.91% | 1.00 | 1.00 | 1.00 |
| mechanism score 0.50 | 72.82% | 93.91% | 1.00 | 1.00 | 1.00 |

Short-horizon theme continuity is therefore real after duplicate-state correction.

### Cross-hour transitions

Twelve within-day transitions were tested from the end of one ten-minute window to the start of the next.

| Matcher | Mean continuation | Median continuation | Median Jaccard | Median family Jaccard |
|---|---:|---:|---:|---:|
| member overlap 0.50 | 0.80% | 0.45% | 0.203 | 0.643 |
| mechanism score 0.45 | 1.93% | 1.66% | 0.195 | 0.875 |
| mechanism score 0.50 | 0.78% | 0.65% | 0.220 | 1.000 |

The largest cross-hour continuation occurs in the final transition into the close window, but it is still only about 4%-5% under the permissive matcher.

### Sequential cross-day transition

The close of 2026-01-05 was compared with the first sampled window of 2026-01-07.

- continuation: 0.50%
- matched themes: 1 of 199 current themes
- member Jaccard: 0.20 under the mechanism-aware matcher

### Same-clock cross-day comparison

Each of the 70 sampled clock times on 2026-01-05 was compared with the same clock time on 2026-01-07.

| Matcher | Mean continuation | Median continuation | Median Jaccard | Median family Jaccard |
|---|---:|---:|---:|---:|
| member overlap 0.50 | 0.30% | 0.00% | 0.037 | 0.50 |
| mechanism score 0.45 | 1.29% | 1.30% | 0.20 | 0.75 |
| mechanism score 0.50 | 0.22% | 0.00% | 0.20 | 1.00 |

## Stable-core path result across all effective states

| Threshold | Median states | Paths >=3 | Paths >=6 | Paths >=12 | Maximum states |
|---|---:|---:|---:|---:|---:|
| 0.45 | 2 | 43.88% | 1.57% | 0.03% | 16 |
| 0.50 | 2 | 42.60% | 1.17% | 0.02% | 13 |
| 0.55 | 2 | 42.06% | 0.88% | 0.01% | 13 |

## Interpretation

One lifecycle definition cannot serve all horizons.

The final sparse themes are reliable as fast local market structures inside a ten-minute window, but member-set identity is almost completely lost over hourly and cross-day gaps. This does not necessarily mean the economic topic disappeared. It means that exact or near-exact member-community identity is the wrong object for slower horizons.

The current output should be separated into two levels:

1. `theme_instance`: fast structural community, minute-to-minute, member-sensitive.
2. `macro_theme_lineage`: slower topic state, hour-to-day, based on stable semantic and mechanism prototypes rather than direct member overlap.

## Required multi-timescale architecture

### Fast path

- state-clock lifecycle
- stable-core threshold 0.45
- intended horizon: adjacent effective states and short intrawindow persistence
- output: member-level structural alerts

### Slow path

Aggregate each ten-minute window before matching across hours or days.

A window prototype should include:

- member frequency and core-member frequency
- source-family distribution
- source-layer distribution
- sector and industry distribution
- market-cap profile
- graph-factor centroid
- alpha behavior profile

Slow-path matching must not reward raw member overlap as the dominant term. Candidate score:

- 0.25 semantic distribution similarity
- 0.20 mechanism-family similarity
- 0.20 graph-factor centroid similarity
- 0.15 stable-core member overlap
- 0.10 market-cap profile similarity
- 0.10 source-layer similarity

The weights are a starting grid, not validated defaults.

## Decision

- Keep stable-core 0.45 for the fast lifecycle.
- Do not claim that final theme member communities persist across hours or days.
- Do not lower the member threshold merely to manufacture cross-hour continuation.
- Build a separate window-level macro-theme lineage and validate it with metadata, null controls and forward alpha.
- Cross-day semantic claims remain blocked until canonical `symbol_id -> symbol` is persisted and verified.
