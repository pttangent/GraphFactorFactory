#!/usr/bin/env python3
"""Shard-local B50/B35 P1 builder with fuzzy relation and fuzzy temporal outputs.

Input is a physical P0 edge shard, ideally one date + one layer + one scale:

    data/p0_edge_shards/date=YYYY-MM-DD/layer_id=X/scale=60m/edges.parquet

The script processes one decision_time at a time, writes output incrementally,
and only keeps the previous snapshot's leaves for temporal continuation. This
is designed to be launched by a shard-level multi-worker scheduler.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


@dataclass(frozen=True)
class BoundaryConfig:
    name: str
    small_leaf_size: int
    max_leaf_size: int
    topk_coarse: int = 5
    topk_refine: int = 3
    topk_strong: int = 2


@dataclass(frozen=True)
class RelationConfig:
    hard_threshold: float = 0.25
    fuzzy_min_strength: float = 0.15
    fuzzy_scale: float = 0.25
    strong_threshold: float = 0.60
    medium_threshold: float = 0.35


@dataclass(frozen=True)
class TemporalConfig:
    hard_jaccard: float = 0.25
    fuzzy_min_strength: float = 0.15


B50 = BoundaryConfig("B50", small_leaf_size=10, max_leaf_size=50)
B35 = BoundaryConfig("B35", small_leaf_size=8, max_leaf_size=35)


class TableSink:
    def __init__(self, path: Path, fmt: str = "parquet") -> None:
        self.path = path
        self.fmt = fmt
        self.writer: pq.ParquetWriter | None = None
        self.csv_header_written = False
        self.rows = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, rows: list[dict]) -> None:
        if not rows:
            return
        df = pd.DataFrame(rows)
        self.rows += len(df)
        if self.fmt == "csv":
            df.to_csv(self.path.with_suffix(".csv"), mode="a", header=not self.csv_header_written, index=False)
            self.csv_header_written = True
            return
        table = pa.Table.from_pandas(df, preserve_index=False)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.path.with_suffix(".parquet"), table.schema, compression="zstd")
        else:
            table = table.cast(self.writer.schema)
        self.writer.write_table(table)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None


def safe_token(x: object) -> str:
    s = str(x)
    return "".join(ch if ch.isalnum() or ch in "=-_.T" else "_" for ch in s)


def read_shard(path: Path) -> pd.DataFrame:
    cols = ["decision_time", "layer_id", "scale", "src_id", "dst_id", "weight"]
    available = set(pq.ParquetFile(path).schema.names)
    df = pd.read_parquet(path, columns=[c for c in cols if c in available])
    if "scale" not in df.columns:
        df["scale"] = "default"
    df = df[["decision_time", "layer_id", "scale", "src_id", "dst_id", "weight"]].copy()
    df = df[df["src_id"] != df["dst_id"]]
    df["src_id"] = df["src_id"].astype("int64")
    df["dst_id"] = df["dst_id"].astype("int64")
    df["layer_id"] = df["layer_id"].astype("int64")
    df["scale"] = df["scale"].astype(str)
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0).astype("float64")
    df["abs_weight"] = df["weight"].abs()
    return df.sort_values("decision_time", kind="mergesort")


def compact_pairs(edges: pd.DataFrame) -> pd.DataFrame:
    a = np.minimum(edges["src_id"].to_numpy(), edges["dst_id"].to_numpy())
    b = np.maximum(edges["src_id"].to_numpy(), edges["dst_id"].to_numpy())
    tmp = pd.DataFrame({"a": a, "b": b, "weight": edges["abs_weight"].to_numpy()})
    tmp = tmp[tmp["a"] != tmp["b"]]
    if tmp.empty:
        return tmp
    return tmp.groupby(["a", "b"], sort=False, as_index=False).agg(weight=("weight", "max"))


def build_adj_from_pairs(pairs: pd.DataFrame, members: set[int] | None = None) -> dict[int, dict[int, float]]:
    adj: dict[int, dict[int, float]] = defaultdict(dict)
    for a, b, w in pairs[["a", "b", "weight"]].itertuples(index=False, name=None):
        a = int(a)
        b = int(b)
        if members is not None and (a not in members or b not in members):
            continue
        adj[a][b] = float(w)
        adj[b][a] = float(w)
    return adj


def topk_components(members: set[int], adj: dict[int, dict[int, float]], topk: int, min_size: int = 1) -> list[set[int]]:
    base = set(members)
    graph: dict[int, list[int]] = defaultdict(list)
    for a in base:
        nbrs = [(b, w) for b, w in adj.get(a, {}).items() if b in base]
        nbrs.sort(key=lambda x: (-x[1], x[0]))
        for b, _ in nbrs[:topk]:
            graph[a].append(b)
            graph[b].append(a)
    seen: set[int] = set()
    comps: list[set[int]] = []
    for n in sorted(base):
        if n in seen:
            continue
        stack = [n]
        seen.add(n)
        comp: list[int] = []
        while stack:
            x = stack.pop()
            comp.append(x)
            for y in graph.get(x, []):
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        if len(comp) >= min_size:
            comps.append(set(comp))
    comps.sort(key=lambda c: (-len(c), min(c)))
    return comps


def forced_chunks(members: set[int], adj: dict[int, dict[int, float]], max_size: int) -> list[set[int]]:
    degree = {n: sum(w for b, w in adj.get(n, {}).items() if b in members) for n in members}
    remaining = set(members)
    chunks: list[set[int]] = []
    while remaining:
        seed = max(remaining, key=lambda n: (degree.get(n, 0.0), -n))
        chunk = {seed}
        remaining.remove(seed)
        frontier = [seed]
        while frontier and len(chunk) < max_size:
            x = frontier.pop(0)
            nbrs = [(b, adj.get(x, {}).get(b, 0.0), degree.get(b, 0.0)) for b in remaining if b in adj.get(x, {})]
            nbrs.sort(key=lambda z: (-z[1], -z[2], z[0]))
            for b, _, _ in nbrs:
                if len(chunk) >= max_size:
                    break
                if b in remaining:
                    remaining.remove(b)
                    chunk.add(b)
                    frontier.append(b)
        if len(chunk) < max_size and remaining:
            fillers = sorted(remaining, key=lambda n: (-degree.get(n, 0.0), n))
            for b in fillers[: max_size - len(chunk)]:
                remaining.remove(b)
                chunk.add(b)
        chunks.append(chunk)
    return chunks


def choose_split(members: set[int], adj: dict[int, dict[int, float]], cfg: BoundaryConfig) -> tuple[str, int, list[set[int]]]:
    if len(members) <= cfg.max_leaf_size:
        return "stop", 0, [members]
    best: tuple[float, str, int, list[set[int]]] | None = None
    for topk in [cfg.topk_coarse, cfg.topk_refine, cfg.topk_strong, 1]:
        comps = topk_components(members, adj, topk=topk, min_size=1)
        if len(comps) <= 1:
            continue
        max_child = max(len(c) for c in comps)
        tiny_ratio = sum(1 for c in comps if len(c) < cfg.small_leaf_size) / len(comps)
        reduction = 1.0 - max_child / max(1, len(members))
        score = 2.0 * reduction - 0.5 * tiny_ratio + 0.02 * math.log1p(len(comps))
        if best is None or score > best[0]:
            best = (score, f"topk_{topk}", topk, comps)
    if best is not None and max(len(c) for c in best[3]) < len(members):
        return best[1], best[2], best[3]
    return "forced_chunk", 0, forced_chunks(members, adj, cfg.max_leaf_size)


def weighted_degree(edges: pd.DataFrame) -> dict[int, float]:
    src = edges.groupby("src_id", sort=False)["abs_weight"].sum()
    dst = edges.groupby("dst_id", sort=False)["abs_weight"].sum()
    deg = src.add(dst, fill_value=0.0)
    return {int(k): float(v) for k, v in deg.items()}


def add_membership_rows(ts: str, layer: str, scale: str, theme_id: str, members: set[int], degree: dict[int, float], level: str, root_b50: str) -> list[dict]:
    max_deg = max([degree.get(m, 0.0) for m in members], default=0.0)
    rows: list[dict] = []
    for rank, m in enumerate(sorted(members, key=lambda x: (-degree.get(x, 0.0), x)), 1):
        rows.append({"decision_time": ts, "layer_id": layer, "scale": scale, "theme_id": theme_id, "member_id": int(m), "level": level, "root_b50_theme_id": root_b50, "rank_in_theme": rank, "core_score": float(degree.get(m, 0.0) / max_deg) if max_deg > 0 else 0.0})
    return rows


def split_to_b50(ts: str, layer: str, scale: str, members: set[int], adj: dict[int, dict[int, float]], degree: dict[int, float], prefix: str, depth: int) -> tuple[list[dict], list[dict], list[dict], list[tuple[str, set[int]]]]:
    node_rows: list[dict] = []
    tree_rows: list[dict] = []
    member_rows: list[dict] = []
    leaves: list[tuple[str, set[int]]] = []

    def rec(node_id: str, node_members: set[int], d: int) -> None:
        is_leaf = len(node_members) <= B50.max_leaf_size
        node_rows.append({"decision_time": ts, "layer_id": layer, "scale": scale, "theme_id": node_id, "parent_theme_id": node_id.rsplit(".", 1)[0] if "." in node_id else "", "level": "B50", "depth": d, "size": len(node_members), "is_leaf": is_leaf, "boundary_config": "B50", "root_b50_theme_id": node_id if is_leaf else ""})
        if is_leaf:
            leaves.append((node_id, node_members))
            member_rows.extend(add_membership_rows(ts, layer, scale, node_id, node_members, degree, "B50", node_id))
            return
        mode, topk, children = choose_split(node_members, adj, B50)
        for i, child in enumerate(children):
            child_id = f"{node_id}.b50_{i:03d}"
            tree_rows.append({"decision_time": ts, "layer_id": layer, "scale": scale, "parent_theme_id": node_id, "child_theme_id": child_id, "parent_level": "B50", "child_level": "B50", "depth": d + 1, "split_mode": mode, "topk": topk, "parent_size": len(node_members), "child_size": len(child), "child_share": len(child) / max(1, len(node_members))})
            rec(child_id, child, d + 1)

    rec(prefix, members, depth)
    return node_rows, tree_rows, member_rows, leaves


def refine_to_b35(ts: str, layer: str, scale: str, b50_leaves: list[tuple[str, set[int]]], adj: dict[int, dict[int, float]], degree: dict[int, float]) -> tuple[list[dict], list[dict], list[dict], list[tuple[str, set[int]]]]:
    node_rows: list[dict] = []
    tree_rows: list[dict] = []
    member_rows: list[dict] = []
    refined: list[tuple[str, set[int]]] = []
    for b50_id, members in b50_leaves:
        if len(members) <= B35.max_leaf_size:
            children = [members]
            mode = "passthrough"
            topk = 0
        else:
            mode, topk, children = choose_split(members, adj, B35)
        for i, child in enumerate(children):
            child_id = f"{b50_id}.b35_{i:03d}"
            node_rows.append({"decision_time": ts, "layer_id": layer, "scale": scale, "theme_id": child_id, "parent_theme_id": b50_id, "level": "B35", "depth": b50_id.count(".") + 1, "size": len(child), "is_leaf": True, "boundary_config": "B35", "root_b50_theme_id": b50_id})
            tree_rows.append({"decision_time": ts, "layer_id": layer, "scale": scale, "parent_theme_id": b50_id, "child_theme_id": child_id, "parent_level": "B50", "child_level": "B35", "depth": b50_id.count(".") + 1, "split_mode": mode, "topk": topk, "parent_size": len(members), "child_size": len(child), "child_share": len(child) / max(1, len(members))})
            member_rows.extend(add_membership_rows(ts, layer, scale, child_id, child, degree, "B35", b50_id))
            refined.append((child_id, child))
    return node_rows, tree_rows, member_rows, refined


def relation_tier(strength: float, cfg: RelationConfig) -> str:
    if strength >= cfg.strong_threshold:
        return "strong"
    if strength >= cfg.medium_threshold:
        return "medium"
    if strength >= cfg.fuzzy_min_strength:
        return "weak"
    return "discard"


def build_relation_edges(ts: str, layer: str, scale: str, edges: pd.DataFrame, leaves: list[tuple[str, set[int]]], level: str, cfg: RelationConfig) -> list[dict]:
    if not leaves or edges.empty:
        return []
    mapping = {m: leaf_id for leaf_id, members in leaves for m in members}
    leaf_size = {leaf_id: len(members) for leaf_id, members in leaves}
    tmp = edges[["src_id", "dst_id", "abs_weight"]].copy()
    tmp["src_theme_id"] = tmp["src_id"].map(mapping)
    tmp["dst_theme_id"] = tmp["dst_id"].map(mapping)
    tmp = tmp.dropna(subset=["src_theme_id", "dst_theme_id"])
    if tmp.empty:
        return []
    internal = tmp[tmp["src_theme_id"] == tmp["dst_theme_id"]].groupby("src_theme_id", sort=False)["abs_weight"].sum().to_dict()
    inter = tmp[tmp["src_theme_id"] != tmp["dst_theme_id"]].copy()
    if inter.empty:
        return []
    a = np.where(inter["src_theme_id"].to_numpy() < inter["dst_theme_id"].to_numpy(), inter["src_theme_id"], inter["dst_theme_id"])
    b = np.where(inter["src_theme_id"].to_numpy() < inter["dst_theme_id"].to_numpy(), inter["dst_theme_id"], inter["src_theme_id"])
    inter["a"] = a
    inter["b"] = b
    grouped = inter.groupby(["a", "b"], sort=False).agg(inter_leaf_weight=("abs_weight", "sum"), edge_count=("abs_weight", "size")).reset_index()
    rows: list[dict] = []
    for a_id, b_id, w, cnt in grouped[["a", "b", "inter_leaf_weight", "edge_count"]].itertuples(index=False, name=None):
        denom = math.sqrt(max(float(internal.get(a_id, 0.0)), 1e-12) * max(float(internal.get(b_id, 0.0)), 1e-12))
        norm = float(w) / denom if denom > 0 else 0.0
        strength = min(1.0, norm / cfg.fuzzy_scale)
        hard_keep = norm >= cfg.hard_threshold
        tier = relation_tier(strength, cfg)
        if not hard_keep and tier == "discard":
            continue
        rows.append({"decision_time": ts, "layer_id": layer, "scale": scale, "level": level, "src_theme_id": str(a_id), "dst_theme_id": str(b_id), "relation_strength": float(strength), "relation_tier": tier, "hard_keep": bool(hard_keep), "normalized_weight": float(norm), "inter_leaf_weight": float(w), "edge_count": int(cnt), "src_size": int(leaf_size.get(a_id, 0)), "dst_size": int(leaf_size.get(b_id, 0))})
    return rows


def temporal_edges(ts_prev: str, ts_cur: str, layer: str, scale: str, prev: list[tuple[str, set[int]]] | None, cur: list[tuple[str, set[int]]], level: str, cfg: TemporalConfig) -> list[dict]:
    if not prev:
        return []
    rows: list[dict] = []
    cur_sets = [(tid, set(m)) for tid, m in cur]
    for a_id, a_members in prev:
        a = set(a_members)
        best: tuple[float, str, float, float, int] | None = None
        for b_id, b in cur_sets:
            inter = len(a & b)
            if inter == 0:
                continue
            union = len(a | b)
            j = inter / union
            c = inter / min(len(a), len(b))
            strength = max(j, 0.7 * j + 0.3 * c)
            if best is None or strength > best[0]:
                best = (strength, b_id, j, c, inter)
        if best is None:
            continue
        strength, b_id, j, c, overlap = best
        if strength < cfg.fuzzy_min_strength and j < cfg.hard_jaccard:
            continue
        rows.append({"layer_id": layer, "scale": scale, "level": level, "src_theme_id": a_id, "dst_theme_id": b_id, "src_time": ts_prev, "dst_time": ts_cur, "continuation_strength": float(strength), "jaccard": float(j), "containment": float(c), "overlap": int(overlap), "hard_continue": bool(j >= cfg.hard_jaccard), "fuzzy_continue": bool(strength >= cfg.fuzzy_min_strength)})
    return rows


def build_shard(path: Path, out_dir: Path, output_format: str, max_snapshots: int | None, relation_cfg: RelationConfig, temporal_cfg: TemporalConfig) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = read_shard(path)
    node_sink = TableSink(out_dir / "theme_nodes", output_format)
    tree_sink = TableSink(out_dir / "theme_tree_edges", output_format)
    member_sink = TableSink(out_dir / "theme_memberships", output_format)
    relation_sink = TableSink(out_dir / "theme_relation_edges", output_format)
    temporal_sink = TableSink(out_dir / "temporal_theme_edges", output_format)
    summary_sink = TableSink(out_dir / "summary", output_format)
    prev_b50 = None
    prev_b35 = None
    prev_ts = None
    groups_done = 0
    try:
        for ts, grp in df.groupby("decision_time", sort=True):
            if max_snapshots is not None and groups_done >= max_snapshots:
                break
            ts_s = str(ts)
            layer = str(grp["layer_id"].iloc[0])
            scale = str(grp["scale"].iloc[0])
            members = set(map(int, pd.concat([grp["src_id"], grp["dst_id"]]).unique()))
            pairs = compact_pairs(grp)
            adj = build_adj_from_pairs(pairs, members)
            degree = weighted_degree(grp)
            root_id = f"ts={safe_token(ts_s)}|layer={safe_token(layer)}|scale={safe_token(scale)}|root"
            root_node = {"decision_time": ts_s, "layer_id": layer, "scale": scale, "theme_id": root_id, "parent_theme_id": "", "level": "ROOT", "depth": 0, "size": len(members), "is_leaf": False, "boundary_config": "root", "root_b50_theme_id": ""}
            n50, e50, m50, b50 = split_to_b50(ts_s, layer, scale, members, adj, degree, root_id, 1)
            n35, e35, m35, b35 = refine_to_b35(ts_s, layer, scale, b50, adj, degree)
            node_sink.write([root_node] + n50 + n35)
            tree_sink.write(e50 + e35)
            member_sink.write(m50 + m35)
            rel50 = build_relation_edges(ts_s, layer, scale, grp, b50, "B50", relation_cfg)
            rel35 = build_relation_edges(ts_s, layer, scale, grp, b35, "B35", relation_cfg)
            relation_sink.write(rel50 + rel35)
            if prev_ts is not None:
                temporal_sink.write(temporal_edges(str(prev_ts), ts_s, layer, scale, prev_b50, b50, "B50", temporal_cfg))
                temporal_sink.write(temporal_edges(str(prev_ts), ts_s, layer, scale, prev_b35, b35, "B35", temporal_cfg))

            def stats(leaves: list[tuple[str, set[int]]]) -> tuple[int, int, float]:
                sizes = [len(x[1]) for x in leaves]
                return len(sizes), max(sizes) if sizes else 0, float(np.median(sizes)) if sizes else 0.0

            b50_count, b50_max, b50_med = stats(b50)
            b35_count, b35_max, b35_med = stats(b35)
            summary_sink.write([{"decision_time": ts_s, "layer_id": layer, "scale": scale, "root_size": len(members), "raw_edges": int(len(grp)), "pair_edges": int(len(pairs)), "b50_leaf_count": b50_count, "b50_leaf_max": b50_max, "b50_leaf_median": b50_med, "b35_leaf_count": b35_count, "b35_leaf_max": b35_max, "b35_leaf_median": b35_med, "b50_relation_edges": len(rel50), "b35_relation_edges": len(rel35)}])
            prev_b50, prev_b35, prev_ts = b50, b35, ts_s
            groups_done += 1
            if groups_done % 25 == 0:
                print(json.dumps({"snapshots": groups_done, "out_dir": str(out_dir)}, default=str), flush=True)
    finally:
        for sink in [node_sink, tree_sink, member_sink, relation_sink, temporal_sink, summary_sink]:
            sink.close()
    manifest = {"builder": "build_b50_b35_theme_forest_streaming.py", "input": str(path), "out_dir": str(out_dir), "snapshots": groups_done, "output_format": output_format, "rows": {"theme_nodes": node_sink.rows, "theme_tree_edges": tree_sink.rows, "theme_memberships": member_sink.rows, "theme_relation_edges": relation_sink.rows, "temporal_theme_edges": temporal_sink.rows, "summary": summary_sink.rows}, "b50": B50.__dict__, "b35": B35.__dict__, "relation": relation_cfg.__dict__, "temporal": temporal_cfg.__dict__}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Build B50/B35 P1 for one physical date/layer/scale shard.")
    ap.add_argument("--p0-edges-shard", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--output-format", choices=["parquet", "csv"], default="parquet")
    ap.add_argument("--max-snapshots", type=int, default=None)
    ap.add_argument("--hard-relation-threshold", type=float, default=0.25)
    ap.add_argument("--fuzzy-relation-min", type=float, default=0.15)
    ap.add_argument("--fuzzy-relation-scale", type=float, default=0.25)
    ap.add_argument("--hard-temporal-jaccard", type=float, default=0.25)
    ap.add_argument("--fuzzy-temporal-min", type=float, default=0.15)
    args = ap.parse_args()
    rel_cfg = RelationConfig(hard_threshold=args.hard_relation_threshold, fuzzy_min_strength=args.fuzzy_relation_min, fuzzy_scale=args.fuzzy_relation_scale)
    temp_cfg = TemporalConfig(hard_jaccard=args.hard_temporal_jaccard, fuzzy_min_strength=args.fuzzy_temporal_min)
    manifest = build_shard(Path(args.p0_edges_shard), Path(args.out_dir), args.output_format, args.max_snapshots, rel_cfg, temp_cfg)
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()
