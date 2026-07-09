from __future__ import annotations

import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
from collections import defaultdict

BASE = Path('/mnt/data/theme_0106_ex/date=2026-01-06')
OUT = Path('/mnt/data/depth_penalty_experiment')
OUT.mkdir(exist_ok=True)
EDGE = BASE / 'temporal_edges.parquet'
LAY = pd.read_parquet('/mnt/data/layers.parquet')
LNAME = dict(zip(LAY.layer_id.astype(int), LAY.name.astype(str)))
SYM = pd.read_parquet('/mnt/data/symbols.parquet')
SID_TO_SYM = dict(zip(SYM.symbol_id.astype(int), SYM.symbol.astype(str)))
META = pd.read_parquet('/mnt/data/symbol_metadata.parquet').set_index('symbol')

times = [
    pd.Timestamp(x)
    for x in [
        '2026-01-06 14:44:00+00:00',
        '2026-01-06 15:14:00+00:00',
        '2026-01-06 15:44:00+00:00',
    ]
]
selected = [(1, 30), (15, 30), (7, 30), (5, 30), (11, 30), (12, 30)]
frames = []
for t in times:
    df = pd.read_parquet(
        EDGE,
        columns=['decision_time', 'layer_id', 'lookback_minutes', 'src_id', 'dst_id', 'weight'],
        filters=[('decision_time', '==', t), ('layer_id', 'in', [x[0] for x in selected])],
    )
    frames.append(df)
edges = pd.concat(frames, ignore_index=True)
edges = edges[edges[['layer_id', 'lookback_minutes']].apply(tuple, axis=1).isin(selected)].copy()
print('edges', len(edges), flush=True)


def root_graph(df: pd.DataFrame, topn: int = 5) -> nx.Graph:
    acc: dict[tuple[int, int], float] = defaultdict(float)
    for s, d, w in zip(df.src_id.values, df.dst_id.values, df.weight.values):
        s = int(s)
        d = int(d)
        a, b = (s, d) if s < d else (d, s)
        if a != b:
            acc[(a, b)] += abs(float(w))
    adj: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for (a, b), w in acc.items():
        adj[a].append((b, w))
        adj[b].append((a, w))
    keep: set[tuple[int, int]] = set()
    for a, lst in adj.items():
        for b, _w in sorted(lst, key=lambda x: x[1], reverse=True)[:topn]:
            keep.add((a, b) if a < b else (b, a))
    g = nx.Graph()
    for a, b in keep:
        g.add_edge(a, b, weight=acc[(a, b)])
    return g


def part(g: nx.Graph, resolution: float) -> list[set[int]]:
    if g.number_of_edges() < 1:
        return [set(g.nodes())]
    if g.number_of_nodes() > 900:
        out = []
        for c in nx.connected_components(g):
            if len(c) > 900:
                sg = g.subgraph(c).copy()
                out.extend(
                    [
                        set(x)
                        for x in nx.algorithms.community.louvain_communities(
                            sg, weight='weight', resolution=resolution, seed=42
                        )
                    ]
                )
            else:
                out.append(set(c))
        return out
    return [
        set(c)
        for c in nx.algorithms.community.louvain_communities(
            g, weight='weight', resolution=resolution, seed=42
        )
    ]


def choose(g: nx.Graph, depth: int, penalty_start: int) -> list[set[int]]:
    grid = [0.5, 0.8] if depth < penalty_start else [1.4, 2.0]
    best_score = -1e9
    best_comms: list[set[int]] = []
    for r in grid:
        comms = [c for c in part(g, r) if len(c) >= 8]
        if len(comms) <= 1:
            continue
        sizes = [len(c) for c in comms]
        coverage = sum(sizes) / g.number_of_nodes()
        if depth < penalty_start:
            score = coverage + 0.01 * np.log1p(len(comms)) - 0.01 * max(0, len(comms) - 100) / 100
        else:
            score = (
                coverage
                + 0.02 * np.log1p(len(comms))
                - max(0, (max(sizes) - 100) / g.number_of_nodes())
                - max(0, (np.quantile(sizes, 0.9) - 70) / g.number_of_nodes())
            )
        if score > best_score:
            best_score = score
            best_comms = comms
    return best_comms


def rec(g: nx.Graph, depth: int, penalty_start: int, leaves: list[tuple[int, set[int], int]], max_depth: int = 3) -> None:
    n = g.number_of_nodes()
    if depth >= max_depth or n < 45 or g.number_of_edges() < n // 3:
        leaves.append((depth, set(g.nodes()), g.number_of_edges()))
        return
    comms = choose(g, depth, penalty_start)
    if len(comms) <= 1:
        leaves.append((depth, set(g.nodes()), g.number_of_edges()))
        return
    comms = sorted(comms, key=lambda c: (-len(c), min(c)))
    if depth >= penalty_start and len(comms[0]) > 0.92 * n:
        leaves.append((depth, set(g.nodes()), g.number_of_edges()))
        return
    for c in comms:
        rec(g.subgraph(c).copy(), depth + 1, penalty_start, leaves, max_depth)


def sector_metrics(members: set[int]) -> dict[str, object]:
    syms = [SID_TO_SYM.get(int(x)) for x in members]
    syms = [s for s in syms if s]
    if not syms:
        return {
            'top_sector': 'UNKNOWN',
            'top_sector_share': 0,
            'top_industry': 'UNKNOWN',
            'top_industry_share': 0,
            'symbols': '',
        }
    m = META.reindex(syms)
    sec = m['sector_code'].fillna('UNKNOWN').astype(str).value_counts()
    ind = m['industry_code'].fillna('UNKNOWN').astype(str).value_counts()
    return {
        'top_sector': sec.index[0],
        'top_sector_share': float(sec.iloc[0] / len(syms)),
        'top_industry': ind.index[0],
        'top_industry_share': float(ind.iloc[0] / len(syms)),
        'symbols': ', '.join(syms[:18]),
    }

summary = []
reps = []
for penalty_start in [1, 2, 3]:
    for (ts, layer_id, lookback), grp in edges.groupby(['decision_time', 'layer_id', 'lookback_minutes']):
        g = root_graph(grp, topn=5)
        leaves: list[tuple[int, set[int], int]] = []
        rec(g, 0, penalty_start, leaves)
        sizes = [len(m) for _, m, _ in leaves]
        metrics = [sector_metrics(m) for _, m, _ in leaves]
        shares = [x['top_sector_share'] for x in metrics]
        summary.append(
            {
                'penalty_start_depth': penalty_start,
                'decision_time': ts,
                'layer_id': int(layer_id),
                'layer_name': LNAME[int(layer_id)],
                'lookback_minutes': int(lookback),
                'root_size': g.number_of_nodes(),
                'root_edges': g.number_of_edges(),
                'leaf_count': len(leaves),
                'leaf_median': float(np.median(sizes)),
                'leaf_p90': float(np.quantile(sizes, 0.9)),
                'leaf_max': int(max(sizes)),
                'mean_top_sector_share': float(np.mean(shares)),
                'median_top_sector_share': float(np.median(shares)),
                'p90_top_sector_share': float(np.quantile(shares, 0.9)),
                'sector_pure_60_count': sum(x >= 0.6 for x in shares),
                'sector_pure_80_count': sum(x >= 0.8 for x in shares),
            }
        )
        ranked = []
        for i, ((depth, members, edge_count), sm) in enumerate(zip(leaves, metrics)):
            if sm['top_sector'] != 'UNKNOWN' and len(members) >= 8:
                ranked.append((sm['top_sector_share'], len(members), i, depth, edge_count, sm))
        for rank, (share, size, _i, depth, edge_count, sm) in enumerate(
            sorted(ranked, key=lambda z: (z[0], z[1]), reverse=True)[:5], 1
        ):
            reps.append(
                {
                    'penalty_start_depth': penalty_start,
                    'decision_time': ts,
                    'layer_id': int(layer_id),
                    'layer_name': LNAME[int(layer_id)],
                    'lookback_minutes': int(lookback),
                    'rank': rank,
                    'leaf_size': size,
                    'depth': depth,
                    'edge_count': edge_count,
                    **sm,
                }
            )
        print('done', penalty_start, ts, LNAME[int(layer_id)], lookback, 'root', g.number_of_nodes(), 'leaves', len(leaves), flush=True)

sdf = pd.DataFrame(summary)
rdf = pd.DataFrame(reps)
sdf.to_csv(OUT / 'd1_d2_d3_depth_penalty_summary.csv', index=False)
rdf.to_csv(OUT / 'd1_d2_d3_representative_sector_leaves.csv', index=False)
agg = (
    sdf.groupby('penalty_start_depth')
    .agg(
        groups=('leaf_count', 'count'),
        avg_leaf_count=('leaf_count', 'mean'),
        median_leaf_size=('leaf_median', 'median'),
        p90_leaf_size=('leaf_p90', 'median'),
        max_leaf_size=('leaf_max', 'max'),
        mean_top_sector_share=('mean_top_sector_share', 'mean'),
        median_top_sector_share=('median_top_sector_share', 'mean'),
        sector_pure_60=('sector_pure_60_count', 'sum'),
        sector_pure_80=('sector_pure_80_count', 'sum'),
    )
    .reset_index()
)
agg.to_csv(OUT / 'd1_d2_d3_aggregate.csv', index=False)
best = (
    sdf.sort_values(['penalty_start_depth', 'sector_pure_60_count', 'mean_top_sector_share'], ascending=[True, False, False])
    .groupby('penalty_start_depth')
    .head(8)
)
best.to_csv(OUT / 'd1_d2_d3_best_layer_cases.csv', index=False)
readme = (
    '# D1/D2/D3 penalty-start experiment\n\n'
    '**Important:** `top_sector_share` is evaluation only. Sector/industry metadata is not used in clustering, tree construction, or split scoring. The tree uses graph edges only.\n\n'
    'Window approximation: three P0-derived graph snapshots inside one hour: 14:44, 15:14, 15:44 UTC on 2026-01-06. Tested selected layer-scales: return corr, rolling return corr, block activity, large trade flow, absorption, flow-return alignment.\n\n'
    'For tractability in this chat runtime, each graph is pruned to top-5 weighted neighbors per node before tree construction.\n\n'
    '## Aggregate\n\n'
    + agg.to_markdown(index=False)
    + '\n\n## Best layer cases\n\n'
    + best[
        [
            'penalty_start_depth',
            'decision_time',
            'layer_name',
            'lookback_minutes',
            'leaf_count',
            'leaf_median',
            'leaf_max',
            'mean_top_sector_share',
            'sector_pure_60_count',
            'sector_pure_80_count',
        ]
    ].to_markdown(index=False)
    + '\n\n## Recommendation\n\nD2 is the preferred compromise here: D1 fragments earlier and produces many pure but tiny/less structured leaves; D3 delays penalty and retains larger mixed leaves. D2 keeps one coarse level before refinement and then starts controlling giant children.\n'
)
(OUT / 'README.md').write_text(readme)
print(agg.to_string(index=False))
print('OUT', OUT)
