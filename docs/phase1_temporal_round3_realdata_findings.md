# Phase 1 round 3 real-data findings

Data: 2026-01-07, first 60 consecutive one-minute snapshots.

## Single-layer continuity is already high

Communities were matched with global one-to-one overlap-small >= 0.50.

| Layer | omega | continuation | median overlap | median Jaccard | mean communities |
|---|---:|---:|---:|---:|---:|
| absorption | 0.00 | 0.8053 | 1.00 | 1.00 | 231.7 |
| return_corr | 0.00 | 0.8055 | 1.00 | 1.00 | 247.3 |
| return_corr | 0.10 | 0.7750 | 1.00 | 1.00 | 237.8 |
| return_corr | 0.20 | 0.7469 | 1.00 | 1.00 | 228.7 |
| flow_return_alignment | 0.00 | 0.7930 | 1.00 | 1.00 | 231.2 |
| flow_return_alignment | 0.10 | 0.7522 | 1.00 | 1.00 | 219.6 |
| venue_fragmentation | 0.00 | 0.8612 | 1.00 | 1.00 | 308.7 |
| venue_fragmentation | 0.10 | 0.8329 | 1.00 | 1.00 | 290.9 |

## Interpretation

Fixed two-slice identity coupling does not improve these layers. It reduces continuation and compresses the number of communities, consistent with over-smoothing. The prior five-minute analysis understated layer continuity because it measured final cross-layer themes and used wider sampling intervals.

The dominant temporal failure is downstream:

1. leaf-community selection changes the evidence supplied to consensus;
2. observed-edge cross-family intersections are sensitive to small layer changes;
3. final consensus Leiden creates new member sets;
4. lifecycle matching compares only the latest full member set;
5. unconfirmed births are immediately published.

## Decision

- Do not enable temporal Leiden coupling globally.
- Keep `omega=0` as production default.
- Retain the two-slice detector only as an experimental, layer-specific option.
- Focus the next candidate on stable-core theme prototypes, birth confirmation, dormant/revival state and mechanism-aware global matching.
- Evaluate temporal coupling only where independent one-minute continuation is demonstrably low.

## Next A/B

A. Current final-theme lifecycle.
B. Global mechanism-aware matching.
C. B plus three-frame stable core.
D. C plus two-observation birth confirmation and one-frame dormant state.
E. D plus semantic sector and industry stability after symbol ID mapping is restored.

Selection must improve path survival without reducing member Jaccard, mechanism-family consistency, semantic purity or alpha calibration.
