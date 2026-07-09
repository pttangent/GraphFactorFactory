# P1 layer-local theme tree architecture

This branch changes the research architecture from a fixed cross-layer consensus theme pipeline to a hot-pluggable layer-local theme tree pipeline.

## Why

In the previous consensus-oriented design, adding a new graph layer could force P1 to be recomputed because the theme definition depended on a fixed graph-layer combination. That is bad for research: every new graph is a candidate factor, and factor research must support hot-plugging and combinatorial studies.

The new rule is:

- **P0** builds and stores canonical graph layers. Each graph layer is a graph factor.
- **P1** builds one independent theme tree per graph layer from P0 outputs only.
- **P2** performs cross-layer alignment, consensus views, ensemble scoring, and strategy research.

Consensus themes are therefore not the canonical P1 output. They are a P2 derived view.

## Intended data flow

```text
P0 canonical graph store
    -> P1 layer-local theme tree builder
        -> return_corr tree
        -> flow_alignment tree
        -> absorption tree
        -> lead_lag tree
        -> volume_expansion tree
        -> any new hot-plug graph tree
    -> P2 research layer
        -> layer selection
        -> cross-layer theme alignment
        -> soft/strict consensus views
        -> ensemble factor scoring
        -> backtest / Qlib integration
```

## Design principles

1. P1 must be runnable from P0 graph outputs without rebuilding P0.
2. Adding a new graph layer should require only that layer's P1 tree to be computed.
3. Metadata is not used to build the tree. Metadata is used only after clustering for semantic labeling and validation.
4. Cross-layer consensus belongs to P2, because it is a multi-factor research choice.
5. P1 output should preserve hierarchy: root -> community -> child community -> leaf theme.

## Validation performed

Using the Drive pack `[pt tangent]/US-Stock/Smoke_Test_Output/3_days_graphs_thems_pack`, P0 graph package parts were downloaded and the P1 approach was validated on canonical temporal edge outputs from `graph_0106` / 2026-01-06.

The validated universe-level recursive multiplex test produced:

| Tree | Root symbols | Tree nodes | Leaf themes | Leaf median | Leaf max |
|---|---:|---:|---:|---:|---:|
| multiplex_support2 | 5,280 | 408 | 357 | 11 | 80 |
| multiplex_support3 | 1,579 | 1 | 1 | 1,579 | 1,579 |

The result shows that strict cross-layer support >= 3 is too sparse, while support >= 2 can build a usable tree. But this branch promotes **layer-local** P1 trees as the canonical architecture, because the graph layers should remain independently researchable factors.
