# Dedicated cross-day identity A/B

The prior round emphasized cross-hour lineage and treated cross-day metrics as secondary. This round makes cross-day identity the primary target.

Data:

- 2026-01-05 and 2026-01-07
- seven 15-minute windows per day
- quality-filtered macro themes only
- daily recurrent archetypes require appearance in at least two intraday windows

Cross-day hypotheses:

1. same-clock window lineage
2. daily recurrent archetypes
3. trajectory archetypes using intraday clock profile and support trajectory

Null controls:

- shifted-clock cross-day matching
- member-profile permutation preserving non-member structural fields

Results:

- same-clock structural macro continuation: about 6.4%
- shifted-clock null: about 5.7%
- daily recurrent archetype continuation: 7.69%
- member-profile permutation null: about 8.1%-9.1%
- trajectory-aware archetype continuation: 7.69%
- trajectory permutation null: about 9.1%

Conclusion:

No tested cross-day identity method beats its null control. The apparent 4%-8% cross-day continuation in earlier rounds must not be interpreted as reliable lineage.

Accepted findings:

- minute-level fast themes are supported
- hour-level macro lineage is supported
- cross-day theme identity is not yet supported

Rejected approaches:

- lowering thresholds to manufacture cross-day continuation
- family/layer-only cross-day matching
- EWMA identity memory
- daily archetype recurrence without permutation separation

Next cross-day research object:

Cross-day work should move from one-to-one identity matching to latent recurrence testing over at least 10 adjacent sessions. A claim should require:

- actual match rate above shifted-clock and member-permutation nulls
- recurrence across at least three sessions
- stable unsupervised structural fingerprint
- out-of-sample forward behavior consistency
- metadata used only after discovery for naming and external validation
