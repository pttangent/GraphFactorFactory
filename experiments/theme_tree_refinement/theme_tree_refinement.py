#!/usr/bin/env python3
"""Unsupervised second-pass refinement for oversized theme communities.

This script splits large GraphFactorFactory parent themes into smaller child
communities. It does not use sector, industry, market cap, or any other metadata
for clustering. Metadata should only be used later for semantic labeling.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import networkx as nx
import pandas as pd


def as_set(xs: Iterable[int]) -> set[int]:
    return {int(x) for x in xs}


def detect_communities(g: nx.Graph, resolution: float, seed: int) -> list[set[int]]:
    """Prefer Leiden if installed; otherwise fall back to NetworkX Louvain."""
    try:
        import igraph as ig  # type: ignore
        import leidenalg  # type: ignore

        nodes = list(g.nodes())
        pos = {n: i for i, n in enumerate(nodes)}
        ig_edges = [(pos[u], pos[v]) for u, v in g.edges()]
        weights = [float(g[u][v].get("weight", 1.0)) for u, v in g.edges()]
        graph = ig.Graph(n=len(nodes), edges=ig_edges, directed=False)
        part = leidenalg.find_partition(
            graph,
            leidenalg.RBConfigurationVertexPartition,
            weights=weights,
            resolution_parameter=resolution,
            seed=seed,
        )
        return [{nodes[i] for i in comm} for comm in part]
    except Exception:
        return list(nx.algorithms.community.louvain_communities(g, weight="weight", resolution=resolution, seed=seed))


def build_induced_graph(edges: pd.DataFrame, members: set[int], min_layer_support: int) -> nx.Graph:
    """Build parent-theme induced graph from canonical temporal edges."""
    acc: dict[tuple[int, int], dict[str, object]] = {}
    for r in edges.itertuples(index=False):
        s = int(r.src_id)
        d = int(r.dst_id)
        if s == d or s not in members or d not in members:
            continue
        a, b = (s, d) if s < d else (d, s)
        item = acc.setdefault((a, b), {"weight": 0.0, "layers": set(), "count": 0})
        item["weight"] = float(item["weight"]) + abs(float(r.weight))
        item["count"] = int(item["count"]) + 1
        item["layers"].add(int(r.layer_id))  # type: ignore[index]

    g = nx.Graph()
    g.add_nodes_from(members)
    for (a, b), item in acc.items():
        support = len(item["layers"])  # type: ignore[arg-type]
        if support < min_layer_support:
            continue
        # Repeated evidence matters, but duplicate edges should be damped.
        weight = float(item["weight"]) * math.log1p(int(item["count"]))
        g.add_edge(a, b, weight=weight, support=support, count=int(item["count"]))
    return g


def choose_refinement(
    edges: pd.DataFrame,
    members: set[int],
    resolution_grid: list[float],
    support_grid: list[int],
    min_child_size: int,
    target_max_child: int,
    seed: int,
):
    best = None
    parent_size = len(members)
    for min_support in support_grid:
        g = build_induced_graph(edges, members, min_support)
        if g.number_of_edges() == 0:
            continue
        for resolution in resolution_grid:
            comms = [set(c) for c in detect_communities(g, resolution, seed) if len(c) >= min_child_size]
            if not comms:
                continue
            sizes = [len(c) for c in comms]
            covered = sum(sizes)
            coverage = covered / max(1, parent_size)
            median_size = float(pd.Series(sizes).median())
            max_size = max(sizes)
            giant_penalty = max(0.0, (max_size - target_max_child) / max(1, parent_size))
            size_penalty = abs(median_size - min(target_max_child / 2, 35)) / max(1, parent_size)
            score = coverage - giant_penalty - size_penalty + 0.01 * math.log1p(len(comms))
            candidate = (score, min_support, resolution, g, comms, coverage, sizes)
            if best is None or candidate[0] > best[0]:
                best = candidate
    return best


def refine(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    theme_dir = Path(args.theme_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    snap = pd.Timestamp(args.snapshot_time)
    if snap.tzinfo is None:
        snap = snap.tz_localize("UTC")

    themes = pd.read_parquet(theme_dir / "themes.parquet")
    themes = themes[themes["snapshot_time"] == snap]
    edges = pd.read_parquet(
        theme_dir / "temporal_edges.parquet",
        columns=["decision_time", "layer_id", "src_id", "dst_id", "weight"],
        filters=[("decision_time", "==", snap)],
    )

    resolution_grid = [float(x) for x in args.resolution_grid.split(",")]
    support_grid = [int(x) for x in args.min_layer_support_grid.split(",")]
    summary_rows = []
    child_rows = []

    for tr in themes.itertuples(index=False):
        parent_id = str(tr.theme_instance_id)
        members = as_set(tr.members)
        if len(members) < args.min_parent_size:
            continue
        chosen = choose_refinement(
            edges=edges,
            members=members,
            resolution_grid=resolution_grid,
            support_grid=support_grid,
            min_child_size=args.min_child_size,
            target_max_child=args.target_max_child,
            seed=args.seed,
        )
        if chosen is None:
            continue
        score, min_support, resolution, g, comms, coverage, sizes = chosen
        comms = sorted(comms, key=lambda c: (-len(c), min(c)))
        for child_id, comm in enumerate(comms):
            child_rows.append({
                "theme_instance_id": parent_id,
                "snapshot_time": str(snap),
                "child_id": child_id,
                "child_size": len(comm),
                "members": " ".join(map(str, sorted(comm))),
            })
        summary_rows.append({
            "theme_instance_id": parent_id,
            "snapshot_time": str(snap),
            "parent_size": len(members),
            "raw_edges": int(g.number_of_edges()),
            "children": len(comms),
            "covered": int(sum(sizes)),
            "coverage": coverage,
            "child_median": float(pd.Series(sizes).median()),
            "child_mean": float(pd.Series(sizes).mean()),
            "child_max": int(max(sizes)),
            "resolution": resolution,
            "min_layer_support": min_support,
            "is_market_mode": bool(getattr(tr, "is_market_mode", False)),
            "score": float(score),
        })

    summary = pd.DataFrame(summary_rows).sort_values("score", ascending=False)
    children = pd.DataFrame(child_rows)
    summary.to_csv(out_dir / "refinement_summary.csv", index=False)
    children.to_csv(out_dir / "refined_child_communities.csv", index=False)
    return summary, children


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--theme-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--snapshot-time", required=True)
    ap.add_argument("--min-parent-size", type=int, default=120)
    ap.add_argument("--target-max-child", type=int, default=80)
    ap.add_argument("--min-child-size", type=int, default=8)
    ap.add_argument("--resolution-grid", default="1.0,1.2,1.4,1.6,1.8,2.0")
    ap.add_argument("--min-layer-support-grid", default="1,2")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    summary, children = refine(args)
    print(json.dumps({"parents_refined": len(summary), "children": len(children)}, indent=2))


if __name__ == "__main__":
    main()
