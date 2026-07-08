# Theme-tree refinement POC

This experiment records the first working proof-of-concept for splitting oversized GraphFactorFactory theme communities into smaller child themes.

## Goal

The goal is **unsupervised theme-tree construction**:

1. Use graph structure to split parent themes.
2. Do **not** use sector, industry, market cap, or company metadata during clustering.
3. Use metadata only after clustering for semantic naming and sanity checks.

## Data used

- Drive pack: `[pt tangent]/US-Stock/Smoke_Test_Output/3_days_graphs_thems_pack`
- Validated date: `2026-01-06`
- Reassembled package: `theme_0106.tar.part001` ... `theme_0106.tar.part015`
- Main snapshot analyzed: `2026-01-06 14:44:00+00:00`

## Method

For every large parent theme:

1. Read `themes.parquet`.
2. Build an induced graph from `temporal_edges.parquet` using only member nodes already inside that parent theme.
3. Aggregate repeated edges by undirected pair.
4. Optionally require edge support from multiple graph layers.
5. Run Leiden when available, otherwise NetworkX Louvain.
6. Export child communities.
7. Map node IDs back to `symbols.parquet` and `symbol_metadata.parquet` only after clustering.

## Main result from 2026-01-06 14:44 UTC

| Metric | Result |
|---|---:|
| Large parent themes refined | 18 |
| Parent average size | ~293 symbols |
| Refined child average size | ~13.7 symbols |
| Average child count per parent | ~17.8 |
| Average coverage | ~97.0% |
| Largest parent size | 1,592 symbols |
| Largest parent after split | 44 children, 99.6% coverage |

The result supports the thesis that the current parent themes are too large, but that they contain recoverable smaller graph communities.

## Financial interpretation

The strongest repeated signal was Health Care / Biotechnology. It appeared across multiple microstructure layers including absorption, trade intensity, signed flow, venue fragmentation, and return correlation. Because this label was assigned only after unsupervised splitting, it suggests the graph is capturing a real market structure rather than simply reproducing metadata.

Important weakness: some communities still include SPACs or abnormal tickers. The next production step should add a common-stock / tradability universe filter before running the theme-tree pipeline.

## Files

- `theme_tree_refinement.py`: unsupervised second-pass refinement script.
- `results/refinement_summary_0106_1444.csv`: summary metrics for refined parent themes.
- `results/layer_symbol_interpretation_0106_1444.md`: direct symbol-level interpretation by layer.
