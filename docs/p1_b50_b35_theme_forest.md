# P1 B50/B35 theme forest

This document describes the production-oriented P1 variant implemented in
`scripts/build_b50_b35_theme_forest.py`.

## Core decision

P1 should not choose between B50 and B35 as mutually exclusive boundaries.

The production structure is:

```text
B50 stable theme forest
  -> B35 local refinement leaves
  -> same-layer fuzzy relation graph
  -> fuzzy temporal continuation
```

B50 is the stable theme skeleton.  B35 is a local semantic magnifier under each
B50 leaf.

## Boundary rules

### B50 stable layer

- protect leaves with size `<= 10`
- recursively split until every B50 leaf has size `<= 50`
- if graph top-k splitting cannot reduce a giant leaf, use deterministic
  graph-aware forced chunks
- B50 is the default production-level theme forest for stable P2 research

### B35 local refinement layer

- every B50 leaf receives a B35 child view
- B50 leaves with size `<= 35` are passed through as a single B35 child
- B50 leaves with size `36..50` are locally refined into B35 children
- B35 is not recomputed globally from the root; it is a leaf-local refinement
  of the B50 tree

This preserves the B50 hierarchy while enabling finer relation analysis.

## Relation graph

`theme_relation_edges` rolls the original P0 stock-stock edges up to
theme-theme edges.

Same-layer relation semantics are inherited from the P0 graph layer:

- return-correlation layer -> return-correlation relation between leaves
- absorption layer -> absorption relation between leaves
- large-trade-flow layer -> large-trade-flow relation between leaves
- block-activity layer -> block-activity relation between leaves
- flow-return-alignment layer -> flow/return alignment relation between leaves

Hard and fuzzy relation are both stored:

- `hard_keep`: strict high-confidence relation above the hard threshold
- `relation_strength`: fuzzy 0..1 relation strength
- `relation_tier`: `weak`, `medium`, `strong`

Weak fuzzy relations should be treated as context unless they also show
temporal persistence or cross-layer confirmation.

## Temporal graph

`temporal_theme_edges` stores minute-to-minute continuation candidates for both
B50 and B35 levels.

Fields include:

- `jaccard`
- `containment`
- `continuation_strength`
- `hard_continue`
- `fuzzy_continue`

Fuzzy temporal continuation is meant to preserve gradual theme evolution,
splits, merges, and reappearances.  It should be validated in P2 because it can
raise continuation recall while lowering core-member retention.

## Output tables

The script writes:

```text
theme_nodes.parquet
theme_tree_edges.parquet
theme_memberships.parquet
theme_relation_edges.parquet
temporal_theme_edges.parquet
p1_b50_b35_summary.parquet
manifest.json
```

Use `--output-format csv` for small smoke tests.

## Example

```bash
python scripts/build_b50_b35_theme_forest.py \
  --p0-edges data/graph_store_6m/canonical/date=2026-01-07/edges.parquet \
  --out-dir artifacts/p1_b50_b35/date=2026-01-07 \
  --output-format parquet
```

Smoke test:

```bash
python scripts/build_b50_b35_theme_forest.py \
  --p0-edges /path/to/edges.parquet \
  --out-dir /tmp/p1_b50_b35_smoke \
  --max-groups 10 \
  --output-format csv
```

## Research interpretation

The boundary experiment on 2026-01-07 suggested:

- B50 is a safer production baseline
- B35 is more useful for discovery of finer semiconductor / AI-infrastructure
  relation chains
- the better architecture is therefore B50 first, then B35 local refinement,
  not global B35 replacing B50

This script implements that architecture.
