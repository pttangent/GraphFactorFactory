# Intraday 7-arm 39-day Layer Analysis

## Scope
- Period: 2026-01-02 to 2026-02-27, 39 trading days
- Input: Phase 1 `layer_communities.parquet`
- Sampling: every 5th one-minute snapshot (5-minute evaluation grid)
- Minimum community size: 20
- Arms: A, B, C, D9, D11, D13, D15
- Metrics: confirmed paths, median life, S5/S15/S30/S60, revival rate, confirmed revival rate

## Main result
The current composite ranking selects D9 for 12 of 13 layers; `off_exchange` is tied across D variants and is reported as D11 by stable sort. This does **not** prove D9 is economically optimal: D9 mechanically allows dormant recovery and therefore raises persistence. The next analysis must compare each arm against matched null/control paths and penalize false revival, path identity drift, and member turnover.

## Recommended arm by current structural score

| Layer | Arm | S15 | S30 | S60 | Revival | Confirmed revival | Confirmed paths |
|---|---:|---:|---:|---:|---:|---:|---:|
| absorption | D9 | 64.65% | 9.52% | 0.004% | 1.74% | 1.74% | 24,579 |
| block_activity | D9 | 73.47% | 13.28% | 0.243% | 2.02% | 2.02% | 11,532 |
| flow_return_alignment | D9 | 55.88% | 2.26% | 0.000% | 0.05% | 0.05% | 17,360 |
| large_trade_flow | D9 | 68.93% | 6.68% | 0.000% | 0.03% | 0.03% | 6,031 |
| odd_lot_activity | D9 | 57.17% | 6.47% | 0.087% | 0.07% | 0.07% | 8,007 |
| off_exchange | D11 | 56.82% | 6.42% | 0.094% | 0.00% | 0.00% | 6,383 |
| price_impact | D9 | 63.19% | 7.63% | 0.009% | 0.47% | 0.47% | 22,750 |
| report_latency | D9 | 70.61% | 6.98% | 0.049% | 2.56% | 2.56% | 10,150 |
| return_corr | D9 | 63.85% | 6.23% | 0.035% | 0.09% | 0.09% | 20,146 |
| signed_flow | D9 | 63.26% | 3.89% | 0.000% | 0.05% | 0.05% | 5,599 |
| trade_intensity | D9 | 66.43% | 5.37% | 0.000% | 0.24% | 0.24% | 7,913 |
| venue_fragmentation | D9 | 55.68% | 4.08% | 0.013% | 0.08% | 0.08% | 15,570 |
| volume_expansion | D9 | 67.21% | 13.14% | 0.062% | 1.86% | 1.86% | 35,606 |

## Key interpretation
- `block_activity` and `volume_expansion` show the strongest S30 under D9 (~13.3% and ~13.1%).
- `absorption` reaches ~9.5% S30 under D9.
- `return_corr` reaches ~63.8% S15 but only ~6.2% S30: most themes survive 15 minutes but few persist to 30 minutes.
- `report_latency` has high S15 (~70.6%) and the highest revival rate (~2.56%), suggesting a mechanism layer where dormant recovery matters more.
- `signed_flow`, `flow_return_alignment`, and `venue_fragmentation` remain relatively short-lived even after D9; looser tracking does not create strong long-horizon persistence.
- S60 is nearly zero for all layers. The current 5-minute grid and community definitions describe short intraday episodes, not hour-long themes.

## Critical caveats
1. This is an intraday **structural persistence** comparison, not alpha validation.
2. D arms have a mechanical advantage because they permit dormant gaps and revival.
3. There is no matched open-birth/null control in this run.
4. There is no penalty yet for identity drift, split/merge, or false revival.
5. `return_corr` is still raw and has not been residualized against SPY/QQQ/IWM.
6. Sampling every fifth minute is a computational approximation; a final production run should repeat at 1-minute resolution for shortlisted arms.

## Next required work
1. Add matched controls per day/layer/arm using size, density, liquidity, and birth-time matching.
2. Add path identity metrics: core retention, member turnover, split/merge, fingerprint drift.
3. Add false-revival definition: revival followed by death within 2-3 states or severe fingerprint decay.
4. Rebuild `return_corr_market_residual` using SPY/QQQ/IWM residual returns; compare sector/industry purity against raw ReturnCorr.
5. Add network metrics from Phase 0 edges: density, clustering, assortativity, giant-component share, conductance, degree distribution, and proper power-law model comparison.
6. Run forward returns at 5/15/30/60m, close, and T+1 for each path and arm.
