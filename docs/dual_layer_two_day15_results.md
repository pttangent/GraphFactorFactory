# Dual-layer validation

Data: 2026-01-05 and 2026-01-07, seven 15-minute windows per day, 210 timestamps, 48,044 fast theme instances.

Macro promotion rules:

- support in at least 3 effective graph states
- persistence at least 50% of effective states
- core overlap plus member-frequency similarity at least 0.10
- global one-to-one matching

Results:

| System | Cross-hour | Sequential cross-day | Same-clock cross-day |
|---|---:|---:|---:|
| Fast theme | 4.09% | 1.01% | 1.76% |
| Balanced macro | 18.49% | 8.16% | 8.02% |
| Conservative macro | 14.65% | 4.08% | 4.00% |

Shifted-window null controls:

- balanced macro about 5.95%
- conservative macro about 3.56%

Interpretation:

- hour-level macro lineage is supported
- balanced cross-hour continuation is about 3.1 times null
- conservative cross-hour continuation is about 4.1 times null
- cross-day results remain too close to null for a reliable identity claim
- family/layer-only matching is rejected because it creates high false continuity

Recommended hourly candidate:

- threshold 0.50
- member evidence gate 0.10
- weights: core 0.15, member frequency 0.15, family 0.25, layer 0.20, persistence 0.10, consensus 0.15

Keep fast themes for minute structure and macro themes for hour-level lineage. Cross-day lineage remains experimental until canonical symbol metadata and more dates are available.
