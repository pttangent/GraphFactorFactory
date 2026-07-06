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
        import time
        t0 = time.perf_counter()
        p, c = detector.detect_hierarchy(
            group,
            layer_id=layer_id,
            layer_name=scale_name,
            snapshot_time=t,
            universe_count=universe_count,
        )
        leiden_seconds = float(time.perf_counter() - t0)
        from dataclasses import replace
        p = [replace(item, leiden_seconds=leiden_seconds) for item in p]
        c = [replace(item, leiden_seconds=leiden_seconds) for item in c]
        parents.extend(p)
        children.extend(c)
    
    t1 = time.perf_counter()
    themes = consensus.build(
        parents + children,
        snapshot_time=t,
        run_id=run_id,
        universe_count=universe_count,
    )
    consensus_seconds = float(time.perf_counter() - t1)
    from dataclasses import replace
    themes = [replace(item, consensus_seconds=consensus_seconds) for item in themes]
    return t, parents, children, themes
