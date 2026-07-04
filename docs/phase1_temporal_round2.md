# Phase 1 temporal round 2

Input: 2026-01-07, 15:10-16:10 UTC, 13 snapshots at five-minute spacing, all 13 layers.

## A/B result

| Method | Continuation | Paths >=3 frames | Median overlap | Median family Jaccard |
|---|---:|---:|---:|---:|
| Global member-overlap baseline | 10.96% | 2.37% | 0.667 | 0.633 |
| Mechanism-aware + one-frame grace | 15.06% | 2.87% | 0.667 | 0.750 |
| Loose prototype threshold | 23.06% | 4.43% | 0.500 | 0.750 |

The loose threshold is rejected because median member Jaccard falls to 0.20. The conservative candidate uses score = 0.55 member containment + 0.15 member Jaccard + 0.20 family Jaccard + 0.10 size similarity, threshold 0.45, global one-to-one matching and one grace frame.

## Metadata

Uploaded metadata has 5,002 unique symbols. Company coverage is 94.36%, sector 93.46%, industry 93.44%, market cap 94.28%; 282 rows have fetch errors. A canonical symbol_id-to-symbol dimension is still required before semantic purity is trusted.

## Required implementation

1. Global one-to-one matching instead of current candidate-order greedy assignment.
2. Mechanism-family compatibility in the matching score.
3. One-frame grace and reappearance events.
4. Births remain tentative until a second observation.
5. Persist symbols.parquet with Phase 0 canonical output.
6. Add layer-specific temporal thresholds after multi-day validation.
