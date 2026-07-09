# Full-day P1 layer-local theme persistence — 2026-01-06

This validation uses the full trading day from the graph/theme pack. For in-session speed, it tracks every 5th snapshot across the day, covering 78 checkpoints from 14:33 to 20:58 UTC. The tracking target is layer-local P1 communities. Symbol metadata is joined only after graph-only community tracking.

## Key result

The new architecture is validated: P1 can be interpreted as independent layer-local theme trees. The stability profile differs strongly by graph layer, which is exactly why consensus should be moved to P2 rather than hard-coded into P1.

## Stability ranking

Top layers by continuation rate:

| Rank | Layer-scale | Continuation rate | Stable paths >=5 checkpoints | Max duration checkpoints | Median size | Financial interpretation |
|---:|---|---:|---:|---:|---:|---|
| 1 | block_activity@30m | 38.0% | 99 | 6 | 11 | Most stable institutional/block-trading structure. |
| 2 | large_trade_flow@30m | 18.6% | 26 | 13 | 10 | Large-order flow has real persistence; suitable as structural factor. |
| 3 | return_corr_raw_1m@15m | 14.0% | 0 | 3 | 56 | Large price-correlation clusters persist short-term but do not form long stable micro-themes under strict Jaccard. |
| 4 | return_corr_cross_sectional_rolling_5m@30m | 13.2% | 12 | 9 | 47 | Best price layer for stable theme tree; longer window helps. |
| 5 | return_corr_raw_1m@30m | 9.8% | 21 | 8 | 56 | Stable enough for structural price-resonance themes. |
| 6 | return_corr_cross_sectional_1m@30m | 7.8% | 15 | 7 | 62 | Similar to raw return corr, but slightly less persistent. |
| 7 | volume_expansion@15m | 6.6% | 4 | 7 | 13 | Event heat layer; useful but short-lived. |
| 8 | volume_expansion@30m | 5.0% | 25 | 10 | 10 | More stable than 15m; better structural volume theme. |
| 9 | price_impact@15m | 5.0% | 4 | 13 | 11 | Liquidity stress occasionally persists strongly. |
| 10 | flow_return_alignment@30m | 4.2% | 24 | 6 | 11 | Confirmation-type flow/return themes, more stable at 30m. |

## Layer-by-layer conclusion

### Price correlation layers

- `return_corr_raw_1m@30m`, `return_corr_cross_sectional_1m@30m`, and `return_corr_cross_sectional_rolling_5m@30m` are meaningful P1 theme-tree layers.
- Their communities are larger than microstructure layers, with median theme sizes around 47–62 symbols.
- Financial meaning: these layers capture price-resonance baskets. They are suitable for sector rotation, relative strength, diffusion, and market-regime studies.
- The 30m price windows are more useful than 15m for stable theme tree construction.

### Block / large-trade layers

- `block_activity@30m` is the most stable layer by continuation rate.
- `large_trade_flow@30m` has the longest observed path, up to 13 sampled checkpoints.
- Financial meaning: these layers are not just noise. They likely capture institutional execution, portfolio rebalancing, or persistent large-order activity.
- These should be treated as structural graph factors, not merely event layers.

### Volume / trade-intensity layers

- `volume_expansion@30m` is more stable than `volume_expansion@15m` and `volume_expansion@5m`.
- `trade_intensity` is more event-like: many communities appear, but most do not survive long.
- Financial meaning: these layers capture market attention and heat. They are better for trigger/confirmation logic than for slow structural themes.

### Flow layers

- `signed_flow` is unstable in strict Jaccard tracking, especially at 5m/30m.
- `flow_return_alignment@30m` is more stable than 5m/15m.
- Financial meaning: directional flow is more like a transient alpha trigger, while flow-return alignment is a better confirmation layer.

### Liquidity layers

- `price_impact@15m` and `price_impact@30m` have occasional long paths.
- `absorption@30m` has more stable paths than 5m/15m, but still mostly short-lived.
- Financial meaning: liquidity stress and absorption are valuable, but they should be studied as event/pressure themes rather than stable industry themes.

### Venue / data-quality layers

- `venue_fragmentation@30m` has one of the longest observed paths, but most paths are short.
- `off_exchange` has occasional stable clusters, especially at 30m.
- `report_latency@5m` is mostly a data-quality structure, not a direct financial theme.
- Financial meaning: useful for execution/microstructure diagnostics and as a P2 risk-control view.

### Odd-lot layer

- `odd_lot_activity@30m` and `odd_lot_activity@5m` have some persistence, but most are short-lived.
- Financial meaning: likely captures fragmented retail-like activity and small-order attention.

## Architecture implication

The result supports the new design:

```text
P0 graph store
  -> P1 layer-local theme trees
  -> P2 cross-layer research / consensus / ensemble factor selection
```

Do not make P1 a fixed consensus theme system. Each graph layer behaves like an independent factor, and its stability profile is different. Cross-layer consensus should be a P2 research decision.
