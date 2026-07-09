# P1 experiment and legacy archive

This clean production branch keeps the runnable P1 production entrypoint in:

```text
scripts/build_b50_b35_theme_forest.py
```

Legacy research scripts and one-off validation reports were intentionally moved out of the production surface.  Their historical source remains available in the research branch:

```text
codex/p1-layer-local-theme-tree
```

Legacy scripts removed from the production `scripts/` directory on this branch:

```text
scripts/build_layer_local_theme_trees.py
scripts/analyze_p1_full_day_persistence.py
scripts/analyze_p1_every_snapshot_persistence.py
scripts/compare_penalty_start_depths.py
scripts/compare_penalty_start_depths_every_snapshot.py
```

Historical experiment result folders are research artifacts, not production pipeline inputs.  They should not be imported by production code.

Production P1 architecture:

```text
P0 canonical graph edges
  -> B50 stable layer-local theme forest
  -> B35 local refinement under each B50 leaf
  -> same-layer theme relation graph
  -> temporal theme continuation graph
```

Use this branch for production pipeline work. Use the legacy research branch only for reproducing old D1/D2/D3, persistence, or boundary experiments.
