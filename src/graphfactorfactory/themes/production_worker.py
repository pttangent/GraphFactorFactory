from __future__ import annotations


def detect_snapshot(args):
    t, edges, detector, consensus, names, universe_count, run_id = args
    parents, children = [], []
    group_cols = ["layer_id"] + (["lookback_minutes"] if "lookback_minutes" in edges.columns else [])
    for key, group in edges.groupby(group_cols, sort=False):
        if isinstance(key, tuple):
            layer_id, lookback = map(int, key)
        else:
            layer_id, lookback = int(key), 0
        if layer_id == 0:
            continue
        base_name = names.get(layer_id, str(layer_id))
        scale_name = f"{base_name}@{lookback}m" if lookback else base_name
        p, c = detector.detect_hierarchy(
            group,
            layer_id=layer_id,
            layer_name=scale_name,
            snapshot_time=t,
            universe_count=universe_count,
        )
        parents.extend(p)
        children.extend(c)
    themes = consensus.build(
        parents + children,
        snapshot_time=t,
        run_id=run_id,
        universe_count=universe_count,
    )
    return t, parents, children, themes
