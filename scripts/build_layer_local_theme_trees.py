#!/usr/bin/env python3
"""Build P1 layer-local theme trees directly from P0 graph outputs.

This script is intentionally independent of previous P1 theme products. It reads
canonical graph edges produced by P0 and builds one recursive theme tree per
layer/scale/snapshot. A new graph layer can be hot-plugged by running this script
only for that layer.

Metadata is not used for clustering. Symbol metadata should be joined later for
semantic labeling and research reports.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import networkx as nx
import pandas as pd


@dataclass(frozen=True)
class TreeConfig:
    min_node_size: int = 40
    min_child_size: int = 8
    max_depth: int = 4
    target_leaf_size: int = 40
    resolution_grid: tuple[float, ...] = (0.8, 1.0, 1.2, 1.5, 2.0)
    seed: int = 42


def _norm_col(df: pd.DataFrame, names: list[str]) -> str:
    for name in names:
        if name in df.columns:
            return name
    raise KeyError(f"none of columns found: {names}; available={list(df.columns)}")


def read_edges(path: Path) -> pd.DataFrame:
    """Read a P0 edge parquet file or a directory of parquet files."""
    if path.is_dir():
        files = sorted(path.rglob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"no parquet files under {path}")
        frames = [pd.read_parquet(f) for f in files]
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.read_parquet(path)

    src = _norm_col(df, ["src_id", "source_id", "src", "source"])
    dst = _norm_col(df, ["dst_id", "target_id", "dst", "target"])
    weight = _norm_col(df, ["weight", "edge_weight", "score"])
    layer = _norm_col(df, ["layer_id", "layer", "layer_name"])

    out = df.rename(columns={src: "src_id", dst: "dst_id", weight: "weight", layer: "layer_id"}).copy()
    if "decision_time" not in out.columns:
        if "snapshot_time" in out.columns:
            out = out.rename(columns={"snapshot_time": "decision_time"})
        elif "ts" in out.columns:
            out = out.rename(columns={"ts": "decision_time"})
        else:
            out["decision_time"] = "unknown"
    if "scale" not in out.columns:
        out["scale"] = "default"
    out = out[["decision_time", "layer_id", "scale", "src_id", "dst_id", "weight"]]
    out["src_id"] = out["src_id"].astype(int)
    out["dst_id"] = out["dst_id"].astype(int)
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(0.0)
    out = out[out["src_id"] != out["dst_id"]]
    return out


def detect_communities(g: nx.Graph, resolution: float, seed: int) -> list[set[int]]:
    try:
        import igraph as ig  # type: ignore
        import leidenalg  # type: ignore

        nodes = list(g.nodes())
        pos = {n: i for i, n in enumerate(nodes)}
        edges = [(pos[u], pos[v]) for u, v in g.edges()]
        weights = [float(g[u][v].get("weight", 1.0)) for u, v in g.edges()]
        ig_g = ig.Graph(n=len(nodes), edges=edges, directed=False)
        part = leidenalg.find_partition(
            ig_g,
            leidenalg.RBConfigurationVertexPartition,
            weights=weights,
            resolution_parameter=resolution,
            seed=seed,
        )
        return [{nodes[i] for i in comm} for comm in part]
    except Exception:
        return list(nx.algorithms.community.louvain_communities(g, weight="weight", resolution=resolution, seed=seed))


def build_graph(edges: pd.DataFrame, members: set[int] | None = None) -> nx.Graph:
    acc: dict[tuple[int, int], float] = {}
    counts: dict[tuple[int, int], int] = {}
    for r in edges.itertuples(index=False):
        s = int(r.src_id)
        d = int(r.dst_id)
        if members is not None and (s not in members or d not in members):
            continue
        a, b = (s, d) if s < d else (d, s)
        acc[(a, b)] = acc.get((a, b), 0.0) + abs(float(r.weight))
        counts[(a, b)] = counts.get((a, b), 0) + 1
    g = nx.Graph()
    if members is not None:
        g.add_nodes_from(members)
    for (a, b), w in acc.items():
        g.add_edge(a, b, weight=w * math.log1p(counts[(a, b)]), count=counts[(a, b)])
    return g


def score_partition(parent_size: int, comms: list[set[int]], cfg: TreeConfig) -> float:
    sizes = [len(c) for c in comms]
    if not sizes:
        return -1e9
    coverage = sum(sizes) / max(1, parent_size)
    max_child = max(sizes)
    median_child = float(pd.Series(sizes).median())
    giant_penalty = max(0.0, (max_child - cfg.target_leaf_size * 2) / max(1, parent_size))
    too_small_penalty = max(0.0, (cfg.min_child_size - median_child) / max(1, cfg.min_child_size))
    no_split_penalty = 0.5 if len(comms) <= 1 else 0.0
    return coverage - giant_penalty - too_small_penalty - no_split_penalty + 0.01 * math.log1p(len(comms))


def best_split(g: nx.Graph, cfg: TreeConfig) -> tuple[float, float, list[set[int]]]:
    best: tuple[float, float, list[set[int]]] = (-1e9, cfg.resolution_grid[0], [])
    parent_size = g.number_of_nodes()
    for res in cfg.resolution_grid:
        comms = [set(c) for c in detect_communities(g, res, cfg.seed) if len(c) >= cfg.min_child_size]
        score = score_partition(parent_size, comms, cfg)
        if score > best[0]:
            best = (score, res, comms)
    return best


def recursive_tree(
    edges: pd.DataFrame,
    members: set[int],
    node_id: str,
    depth: int,
    cfg: TreeConfig,
    node_rows: list[dict],
    edge_rows: list[dict],
    member_rows: list[dict],
) -> None:
    g = build_graph(edges, members)
    node_rows.append({"node_id": node_id, "parent_id": node_id.rsplit(".", 1)[0] if "." in node_id else "", "depth": depth, "size": len(members), "edges": g.number_of_edges()})

    if depth >= cfg.max_depth or len(members) < cfg.min_node_size or g.number_of_edges() < len(members):
        for m in sorted(members):
            member_rows.append({"node_id": node_id, "member_id": m})
        return

    score, res, comms = best_split(g, cfg)
    if len(comms) <= 1:
        for m in sorted(members):
            member_rows.append({"node_id": node_id, "member_id": m})
        return

    comms = sorted(comms, key=lambda c: (-len(c), min(c)))
    for i, child in enumerate(comms):
        child_id = f"{node_id}.{i}"
        edge_rows.append({"parent_id": node_id, "child_id": child_id, "depth": depth + 1, "child_size": len(child), "resolution": res, "split_score": score})
        recursive_tree(edges, child, child_id, depth + 1, cfg, node_rows, edge_rows, member_rows)


def build_layer_trees(edges: pd.DataFrame, out_dir: Path, cfg: TreeConfig, max_groups: int | None = None) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    groups = list(edges.groupby(["decision_time", "layer_id", "scale"], sort=True))
    if max_groups is not None:
        groups = groups[:max_groups]
    for (ts, layer, scale), grp in groups:
        members = set(map(int, pd.concat([grp["src_id"], grp["dst_id"]]).unique()))
        tree_name = f"ts={str(ts).replace(':','').replace(' ','T')}_layer={layer}_scale={scale}"
        safe = "".join(ch if ch.isalnum() or ch in "=-_.T" else "_" for ch in tree_name)
        node_rows: list[dict] = []
        edge_rows: list[dict] = []
        member_rows: list[dict] = []
        recursive_tree(grp, members, "root", 0, cfg, node_rows, edge_rows, member_rows)
        tree_dir = out_dir / safe
        tree_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(node_rows).to_csv(tree_dir / "theme_tree_nodes.csv", index=False)
        pd.DataFrame(edge_rows).to_csv(tree_dir / "theme_tree_edges.csv", index=False)
        pd.DataFrame(member_rows).to_csv(tree_dir / "theme_tree_members.csv", index=False)
        leaves = pd.DataFrame(member_rows).groupby("node_id").size() if member_rows else pd.Series(dtype=int)
        summaries.append({
            "tree": safe,
            "decision_time": ts,
            "layer_id": layer,
            "scale": scale,
            "root_size": len(members),
            "tree_nodes": len(node_rows),
            "leaf_count": int(len(leaves)),
            "leaf_median": float(leaves.median()) if len(leaves) else 0.0,
            "leaf_max": int(leaves.max()) if len(leaves) else 0,
        })
    summary = pd.DataFrame(summaries)
    summary.to_csv(out_dir / "layer_tree_summary.csv", index=False)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p0-edges", required=True, help="P0 canonical edge parquet file or directory")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--decision-time", default=None)
    ap.add_argument("--layer-id", default=None)
    ap.add_argument("--max-groups", type=int, default=None)
    ap.add_argument("--min-node-size", type=int, default=40)
    ap.add_argument("--min-child-size", type=int, default=8)
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--target-leaf-size", type=int, default=40)
    args = ap.parse_args()

    df = read_edges(Path(args.p0_edges))
    if args.decision_time is not None:
        df = df[df["decision_time"].astype(str) == args.decision_time]
    if args.layer_id is not None:
        df = df[df["layer_id"].astype(str) == str(args.layer_id)]

    cfg = TreeConfig(
        min_node_size=args.min_node_size,
        min_child_size=args.min_child_size,
        max_depth=args.max_depth,
        target_leaf_size=args.target_leaf_size,
    )
    summary = build_layer_trees(df, Path(args.out_dir), cfg, args.max_groups)
    print(json.dumps({"trees": len(summary), "out_dir": args.out_dir}, indent=2, default=str))


if __name__ == "__main__":
    main()
