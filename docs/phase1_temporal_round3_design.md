# Phase 1 temporal optimization round 3

## Conclusion from StockNetV2 review

StockNetV2 has a stronger temporal evaluation contract than current GFF: member Jaccard, overlap-small, core retention, weighted overlap, churn, birth, continuation, split, merge, death and revival are explicit outputs. This is useful and should be ported. It is not yet evidence that StockNetV2 applies temporal regularization inside community detection.

## Failure in current GFF temporal edges

The current replay uses one absolute enter/exit threshold for every layer. Layer weight scales differ, so 0.75/0.65 cannot have consistent meaning. Missing edges are also emitted as grace edges with zero support fields and then passed into Leiden. This mixes observed evidence with continuity priors.

## Literature-guided architecture

Dynamic community methods balance snapshot fit and temporal smoothness. Multislice modularity couples a node to itself across adjacent time slices. Evolutionary clustering combines current snapshot quality with history. Benchmark work warns of three distinct failure modes: glitches, over-smoothing and identity loss. Therefore GFF must measure responsiveness as well as stability.

## Round 3 candidate

### 1. Separate evidence graph and temporal prior

- `observed_edges`: current Phase 0 edges only.
- `prior_edges`: previous active edges with decay and age.
- Leiden input weight: `observed_weight + lambda_layer * prior_weight`.
- Persist both components separately.
- Grace-only edges must never pretend to have current support.

### 2. Layer-relative hysteresis

Thresholds are rolling within-layer quantiles, not global absolute values.

- enter: layer weight quantile 0.70 to 0.85.
- exit: layer weight quantile 0.50 to 0.70.
- maximum grace depends on layer event character.

Initial classes:

- state layers: return_corr, venue_fragmentation, odd_lot_activity: 2 grace frames.
- flow layers: signed_flow, flow_return_alignment, trade_intensity: 1 grace frame.
- event layers: large_trade_flow, block_activity, absorption: 0 or 1 grace frame.

### 3. Temporal partition coupling

For each layer and snapshot, create identity couplings from each active node at t-1 to itself at t. Optimize a two-slice graph with inter-slice coupling `omega_layer`. Use only the t partition as output. This is preferable to merely relabeling independently detected communities.

Grid:

- omega: 0.0, 0.05, 0.10, 0.20, 0.35.
- evaluate by layer.

### 4. Stable core rather than full-member union

A path prototype contains nodes present in at least 2 of the previous 3 observations, weighted by member core score. Matching uses core retention, full-member Jaccard, family similarity and size similarity.

### 5. Birth and reappearance rules

- first observation: tentative.
- second compatible observation within 2 frames: active.
- one missing frame: dormant, not death.
- compatible return: revival.
- event layers may activate immediately only when edge significance is high.

### 6. Split and merge

Use a bipartite overlap matrix and maximum-weight matching. Record split or merge only when secondary links exceed explicit overlap and family-consistency gates. Do not infer split/merge from candidate processing order.

## Evaluation gates

Run continuous 60 one-minute snapshots, not sparse representatives.

For every layer and final theme output report:

- continuation rate.
- path survival at 3, 6, 12, 30 and 60 frames.
- member Jaccard and overlap-small.
- core-member retention.
- partition NMI and adjusted Rand index.
- detection delay for newly emerging structures.
- false revival rate.
- member churn.
- semantic sector and industry stability.
- 5, 15, 30 and 60 minute forward alpha.

A candidate is rejected if stability improves by merging unrelated communities, shown by lower semantic purity, lower member Jaccard, worse null-model separation or slower detection of genuine births.

## Required A/B

A. Current independent Leiden plus post-hoc matching.
B. Layer-relative edge hysteresis only.
C. Temporal partition coupling only.
D. Hysteresis plus temporal coupling.
E. D plus stable-core lifecycle and birth confirmation.

The selected version must dominate A on persistence while preserving or improving semantic purity and financial calibration.
