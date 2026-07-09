#!/usr/bin/env python3
"""Build P1 B50/B35 layer-local theme forests from P0 graph edges.

Design
------
P1 is intentionally layer-local.  Each P0 graph layer is treated as a graph
factor, and this builder creates a theme forest for each
(decision_time, layer_id, scale/lookback) group.

The production layout is two-level:

* B50 stable theme layer
    - protect small leaves <= 10
    - force recursive split until every B50 leaf is <= 50
* B35 refinement layer
    - every B50 leaf gets a B35 child view
    - B50 leaves <= 35 are passed through as one refined child
    - B50 leaves 36..50 are locally refined until refined leaves are <= 35

The relation graph is not a taxonomy.  It rolls original P0 stock-stock edges
up to theme-theme / leaf-leaf relation edges while preserving the original
layer semantics.  For example, relation edges in a return-corr layer still mean
return-correlation relations; relation edges in an absorption layer still mean
absorption relations.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


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


def _norm_col(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"none of {list(candidates)} found in columns={list(df.columns)}")


def read_edges(path: Path) -> pd.DataFrame:
    """Read P0 edge parquet and normalize column names."""
    if path.is_dir():
        files = sorted(path.rglob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"no parquet files under {path}")
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
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

    if "lookback_minutes" in out.columns:
        out["scale"] = out["lookback_minutes"].astype(str) + "m"
    elif "scale" not in out.columns:
        out["scale"] = "default"

    cols = ["decision_time", "layer_id", "scale", "src_id", "dst_id", "weight"]
    out = out[cols]
    out["src_id"] = out["src_id"].astype(int)
    out["dst_id"] = out["dst_id"].astype(int)
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(0.0).astype(float)
    out = out[out["src_id"] != out["dst_id"]].copy()
    out["abs_weight"] = out["weight"].abs()
    return out


def build_adj(edges: pd.DataFrame, members: set[int] | None = None) -> dict[int, dict[int, float]]:
    """Build max-abs weighted undirected adjacency."""
    adj: dict[int, dict[int, float]] = defaultdict(dict)
    if members is not None:
        mset = members
    else:
        mset = None
    for r in edges.itertuples(index=False):
        s = int(r.src_id)
        d = int(r.dst_id)
        if mset is not None and (s not in mset or d not in mset):
            continue
        w = abs(float(r.weight))
        if w <= 0:
            continue
        if w > adj[s].get(d, 0.0):
            adj[s][d] = w
        if w > adj[d].get(s, 0.0):
            adj[d][s] = w
    return adj


def topk_components(members: set[int], adj: dict[int, dict[int, float]], topk: int, min_size: int = 1) -> list[set[int]]:
    """Connected components after keeping top-k neighbors per node."""
    base = set(members)
    g: dict[int, list[int]] = defaultdict(list)
    for a in base:
        nbrs = [(b, w) for b, w in adj.get(a, {}).items() if b in base]
        nbrs.sort(key=lambda x: (-x[1], x[0]))
        for b, _ in nbrs[:topk]:
            g[a].append(b)
            g[b].append(a)

    seen: set[int] = set()
    comps: list[set[int]] = []
    for n in sorted(base):
        if n in seen:
            continue
        q = [n]
        seen.add(n)
        comp: list[int] = []
        while q:
            x = q.pop()
            comp.append(x)
            for y in g.get(x, []):
                if y not in seen:
                    seen.add(y)
                    q.append(y)
        if len(comp) >= min_size:
            comps.append(set(comp))
    comps.sort(key=lambda c: (-len(c), min(c)))
    return comps


def forced_chunks(members: set[int], adj: dict[int, dict[int, float]], max_size: int) -> list[set[int]]:
    """Deterministic graph-aware chunks used only when top-k splitting cannot reduce a large leaf.

    Seeds are chosen by weighted degree.  A chunk grows from a seed through
    strongest available neighbors, then fills by remaining high-degree nodes.
    This is a fallback that guarantees the leaf-size cap.
    """
    degree = {
        n: sum(w for b, w in adj.get(n, {}).items() if b in members)
        for n in members
    }
    remaining = set(members)
    chunks: list[set[int]] = []

    while remaining:
        seed = max(remaining, key=lambda n: (degree.get(n, 0.0), -n))
        chunk = {seed}
        remaining.remove(seed)
        frontier = [seed]

        while frontier and len(chunk) < max_size:
            x = frontier.pop(0)
            nbrs = [
                (b, adj.get(x, {}).get(b, 0.0), degree.get(b, 0.0))
                for b in remaining
                if b in adj.get(x, {})
            ]
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
    """Choose a split mode and child components for a large parent."""
    if len(members) <= cfg.max_leaf_size:
        return "stop", 0, [members]

    topks = [cfg.topk_coarse, cfg.topk_refine, cfg.topk_strong, 1]
    best: tuple[float, str, int, list[set[int]]] | None = None
    for topk in topks:
        comps = topk_components(members, adj, topk=topk, min_size=1)
        if len(comps) <= 1:
            continue

        max_child = max(len(c) for c in comps)
        tiny_ratio = sum(1 for c in comps if len(c) < cfg.small_leaf_size) / len(comps)
        reduction = 1.0 - max_child / max(1, len(members))
        # Prefer reductions that reduce the largest child without exploding tiny fragments.
        score = 2.0 * reduction - 0.5 * tiny_ratio + 0.02 * math.log1p(len(comps))
        if best is None or score > best[0]:
            best = (score, f"topk_{topk}", topk, comps)

    if best is not None and max(len(c) for c in best[3]) < len(members):
        return best[1], best[2], best[3]
    return "forced_chunk", 0, forced_chunks(members, adj, cfg.max_leaf_size)


def weighted_degree(edges: pd.DataFrame) -> dict[int, float]:
    deg: dict[int, float] = defaultdict(float)
    for r in edges.itertuples(index=False):
        w = abs(float(r.weight))
        deg[int(r.src_id)] += w
        deg[int(r.dst_id)] += w
    return deg


def add_membership_rows(
    theme_id: str,
    members: set[int],
    degree: dict[int, float],
    level: str,
    root_b50_theme_id: str,
    rows: list[dict],
) -> None:
    vals = [degree.get(m, 0.0) for m in members]
    max_deg = max(vals) if vals else 0.0
    ranked = sorted(members, key=lambda m: (-degree.get(m, 0.0), m))
    for rank, m in enumerate(ranked, 1):
        rows.append({
            "theme_id": theme_id,
            "member_id": int(m),
            "level": level,
            "root_b50_theme_id": root_b50_theme_id,
            "rank_in_theme": rank,
            "core_score": float(degree.get(m, 0.0) / max_deg) if max_deg > 0 else 0.0,
        })


def split_to_b50(
    members: set[int],
    adj: dict[int, dict[int, float]],
    degree: dict[int, float],
    prefix: str,
    depth: int,
    node_rows: list[dict],
    tree_rows: list[dict],
    member_rows: list[dict],
) -> list[tuple[str, set[int]]]:
    """Build B50 stable leaves and return leaf ids/members."""
    theme_id = prefix
    node_rows.append({
        "theme_id": theme_id,
        "parent_theme_id": theme_id.rsplit(".", 1)[0] if "." in theme_id else "",
        "level": "B50",
        "depth": depth,
        "size": len(members),
        "is_leaf": len(members) <= B50.max_leaf_size,
        "boundary_config": "B50",
        "root_b50_theme_id": theme_id if depth > 0 and len(members) <= B50.max_leaf_size else "",
    })

    if len(members) <= B50.max_leaf_size:
        add_membership_rows(theme_id, members, degree, "B50", theme_id, member_rows)
        return [(theme_id, members)]

    mode, topk, children = choose_split(members, adj, B50)
    out: list[tuple[str, set[int]]] = []
    for i, child in enumerate(children):
        child_id = f"{theme_id}.b50_{i:03d}"
        tree_rows.append({
            "parent_theme_id": theme_id,
            "child_theme_id": child_id,
            "parent_level": "B50",
            "child_level": "B50",
            "depth": depth + 1,
            "split_mode": mode,
            "topk": topk,
            "parent_size": len(members),
            "child_size": len(child),
            "child_share": len(child) / max(1, len(members)),
        })
        out.extend(split_to_b50(child, adj, degree, child_id, depth + 1, node_rows, tree_rows, member_rows))
    return out


def refine_b50_to_b35(
    b50_id: str,
    members: set[int],
    adj: dict[int, dict[int, float]],
    degree: dict[int, float],
    node_rows: list[dict],
    tree_rows: list[dict],
    member_rows: list[dict],
) -> list[tuple[str, set[int]]]:
    """Create local B35 refined leaves under one B50 leaf."""
    if len(members) <= B35.max_leaf_size:
        child_id = f"{b50_id}.b35_000"
        node_rows.append({
            "theme_id": child_id,
            "parent_theme_id": b50_id,
            "level": "B35",
            "depth": b50_id.count(".") + 1,
            "size": len(members),
            "is_leaf": True,
            "boundary_config": "B35",
            "root_b50_theme_id": b50_id,
        })
        tree_rows.append({
            "parent_theme_id": b50_id,
            "child_theme_id": child_id,
            "parent_level": "B50",
            "child_level": "B35",
            "depth": b50_id.count(".") + 1,
            "split_mode": "passthrough",
            "topk": 0,
            "parent_size": len(members),
            "child_size": len(members),
            "child_share": 1.0,
        })
        add_membership_rows(child_id, members, degree, "B35", b50_id, member_rows)
        return [(child_id, members)]

    mode, topk, children = choose_split(members, adj, B35)
    refined: list[tuple[str, set[int]]] = []
    for i, child in enumerate(children):
        child_id = f"{b50_id}.b35_{i:03d}"
        node_rows.append({
            "theme_id": child_id,
            "parent_theme_id": b50_id,
            "level": "B35",
            "depth": b50_id.count(".") + 1,
            "size": len(child),
            "is_leaf": True,
            "boundary_config": "B35",
            "root_b50_theme_id": b50_id,
        })
        tree_rows.append({
            "parent_theme_id": b50_id,
            "child_theme_id": child_id,
            "parent_level": "B50",
            "child_level": "B35",
            "depth": b50_id.count(".") + 1,
            "split_mode": mode,
            "topk": topk,
            "parent_size": len(members),
            "child_size": len(child),
            "child_share": len(child) / max(1, len(members)),
        })
        add_membership_rows(child_id, child, degree, "B35", b50_id, member_rows)
        refined.append((child_id, child))
    return refined


def relation_tier(strength: float, cfg: RelationConfig) -> str:
    if strength >= cfg.strong_threshold:
        return "strong"
    if strength >= cfg.medium_threshold:
        return "medium"
    if strength >= cfg.fuzzy_min_strength:
        return "weak"
    return "discard"


def build_relation_edges(
    edges: pd.DataFrame,
    leaves: list[tuple[str, set[int]]],
    level: str,
    rel_cfg: RelationConfig,
) -> list[dict]:
    member_to_leaf: dict[int, str] = {}
    leaf_size: dict[str, int] = {}
    for leaf_id, members in leaves:
        leaf_size[leaf_id] = len(members)
        for m in members:
            member_to_leaf[m] = leaf_id

    internal: dict[str, float] = defaultdict(float)
    inter: dict[tuple[str, str], float] = defaultdict(float)
    counts: dict[tuple[str, str], int] = defaultdict(int)

    for r in edges.itertuples(index=False):
        a = member_to_leaf.get(int(r.src_id))
        b = member_to_leaf.get(int(r.dst_id))
        if a is None or b is None:
            continue
        w = abs(float(r.weight))
        if a == b:
            internal[a] += w
        else:
            x, y = (a, b) if a < b else (b, a)
            inter[(x, y)] += w
            counts[(x, y)] += 1

    rows: list[dict] = []
    for (a, b), w in inter.items():
        denom = math.sqrt(max(internal.get(a, 0.0), 1e-12) * max(internal.get(b, 0.0), 1e-12))
        norm = w / denom if denom > 0 else 0.0
        strength = min(1.0, norm / rel_cfg.fuzzy_scale)
        hard_keep = norm >= rel_cfg.hard_threshold
        tier = relation_tier(strength, rel_cfg)
        if not hard_keep and tier == "discard":
            continue
        rows.append({
            "level": level,
            "src_theme_id": a,
            "dst_theme_id": b,
            "relation_strength": float(strength),
            "relation_tier": tier,
            "hard_keep": bool(hard_keep),
            "normalized_weight": float(norm),
            "inter_leaf_weight": float(w),
            "edge_count": int(counts[(a, b)]),
            "src_size": int(leaf_size[a]),
            "dst_size": int(leaf_size[b]),
        })
    return rows


def temporal_edges_for_level(
    snapshots: dict[tuple[str, str, str], list[tuple[str, set[int]]]],
    temp_cfg: TemporalConfig,
) -> list[dict]:
    """Build fuzzy and hard temporal continuation edges for each layer/scale/level."""
    rows: list[dict] = []
    by_key: dict[tuple[str, str], list[tuple[str, list[tuple[str, set[int]]]]]] = defaultdict(list)
    for (ts, layer, scale), leaves in snapshots.items():
        by_key[(str(layer), str(scale))].append((str(ts), leaves))

    for (layer, scale), seq in by_key.items():
        seq.sort(key=lambda x: x[0])
        for (ts0, leaves0), (ts1, leaves1) in zip(seq, seq[1:]):
            leaf1 = [(tid, set(m)) for tid, m in leaves1]
            for a_id, a_members in leaves0:
                best: tuple[float, str, float, float, int] | None = None
                a = set(a_members)
                if not a:
                    continue
                for b_id, b_members in leaf1:
                    b = set(b_members)
                    inter = len(a & b)
                    if inter == 0:
                        continue
                    union = len(a | b)
                    jaccard = inter / union
                    containment = inter / min(len(a), len(b))
                    strength = max(jaccard, 0.7 * jaccard + 0.3 * containment)
                    if best is None or strength > best[0]:
                        best = (strength, b_id, jaccard, containment, inter)
                if best is None:
                    continue
                strength, b_id, jaccard, containment, overlap = best
                if strength < temp_cfg.fuzzy_min_strength and jaccard < temp_cfg.hard_jaccard:
                    continue
                rows.append({
                    "layer_id": layer,
                    "scale": scale,
                    "src_theme_id": a_id,
                    "dst_theme_id": b_id,
                    "src_time": ts0,
                    "dst_time": ts1,
                    "continuation_strength": float(strength),
                    "jaccard": float(jaccard),
                    "containment": float(containment),
                    "overlap": int(overlap),
                    "hard_continue": bool(jaccard >= temp_cfg.hard_jaccard),
                    "fuzzy_continue": bool(strength >= temp_cfg.fuzzy_min_strength),
                })
    return rows


def safe_token(x: object) -> str:
    s = str(x).replace(":", "").replace(" ", "T")
    return "".join(ch if ch.isalnum() or ch in "=-_.T" else "_" for ch in s)


def write_table(df: pd.DataFrame, path: Path, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        df.to_csv(path.with_suffix(".csv"), index=False)
    elif fmt == "parquet":
        df.to_parquet(path.with_suffix(".parquet"), index=False)
    else:
        raise ValueError(f"unsupported output format: {fmt}")


def build_forest(
    edges: pd.DataFrame,
    out_dir: Path,
    output_format: str = "parquet",
    max_groups: int | None = None,
    relation_cfg: RelationConfig = RelationConfig(),
    temporal_cfg: TemporalConfig = TemporalConfig(),
) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    node_rows: list[dict] = []
    tree_rows: list[dict] = []
    member_rows: list[dict] = []
    relation_rows: list[dict] = []
    summary_rows: list[dict] = []
    snapshots_b50: dict[tuple[str, str, str], list[tuple[str, set[int]]]] = {}
    snapshots_b35: dict[tuple[str, str, str], list[tuple[str, set[int]]]] = {}

    groups = list(edges.groupby(["decision_time", "layer_id", "scale"], sort=True))
    if max_groups is not None:
        groups = groups[:max_groups]

    for group_idx, ((ts, layer, scale), grp) in enumerate(groups):
        members = set(map(int, pd.concat([grp["src_id"], grp["dst_id"]]).unique()))
        adj = build_adj(grp, members)
        degree = weighted_degree(grp)
        prefix = f"ts={safe_token(ts)}|layer={safe_token(layer)}|scale={safe_token(scale)}|root"
        root_id = prefix

        node_rows.append({
            "theme_id": root_id,
            "parent_theme_id": "",
            "level": "ROOT",
            "depth": 0,
            "size": len(members),
            "is_leaf": False,
            "boundary_config": "root",
            "root_b50_theme_id": "",
        })

        b50_leaves = split_to_b50(members, adj, degree, root_id, 1, node_rows, tree_rows, member_rows)
        b35_leaves: list[tuple[str, set[int]]] = []
        for b50_id, b50_members in b50_leaves:
            b35_leaves.extend(refine_b50_to_b35(b50_id, b50_members, adj, degree, node_rows, tree_rows, member_rows))

        key = (str(ts), str(layer), str(scale))
        snapshots_b50[key] = b50_leaves
        snapshots_b35[key] = b35_leaves

        rel_b50 = build_relation_edges(grp, b50_leaves, "B50", relation_cfg)
        rel_b35 = build_relation_edges(grp, b35_leaves, "B35", relation_cfg)
        for row in rel_b50 + rel_b35:
            row.update({"decision_time": ts, "layer_id": layer, "scale": scale})
        relation_rows.extend(rel_b50)
        relation_rows.extend(rel_b35)

        def leaf_stats(leaves: list[tuple[str, set[int]]]) -> tuple[int, int, float, float]:
            sizes = np.array([len(m) for _, m in leaves], dtype=float)
            if len(sizes) == 0:
                return 0, 0, 0.0, 0.0
            return int(len(sizes)), int(sizes.max()), float(np.median(sizes)), float(np.quantile(sizes, 0.9))

        b50_count, b50_max, b50_med, b50_p90 = leaf_stats(b50_leaves)
        b35_count, b35_max, b35_med, b35_p90 = leaf_stats(b35_leaves)
        summary_rows.append({
            "decision_time": ts,
            "layer_id": layer,
            "scale": scale,
            "root_size": len(members),
            "b50_leaf_count": b50_count,
            "b50_leaf_max": b50_max,
            "b50_leaf_median": b50_med,
            "b50_leaf_p90": b50_p90,
            "b35_leaf_count": b35_count,
            "b35_leaf_max": b35_max,
            "b35_leaf_median": b35_med,
            "b35_leaf_p90": b35_p90,
            "b50_relation_edges": len(rel_b50),
            "b35_relation_edges": len(rel_b35),
        })

        if (group_idx + 1) % 100 == 0:
            print(json.dumps({"processed_groups": group_idx + 1, "total_groups": len(groups)}, default=str), flush=True)

    temporal_b50 = temporal_edges_for_level(snapshots_b50, temporal_cfg)
    for row in temporal_b50:
        row["level"] = "B50"
    temporal_b35 = temporal_edges_for_level(snapshots_b35, temporal_cfg)
    for row in temporal_b35:
        row["level"] = "B35"

    node_df = pd.DataFrame(node_rows)
    tree_df = pd.DataFrame(tree_rows)
    member_df = pd.DataFrame(member_rows)
    relation_df = pd.DataFrame(relation_rows)
    temporal_df = pd.DataFrame(temporal_b50 + temporal_b35)
    summary_df = pd.DataFrame(summary_rows)

    write_table(node_df, out_dir / "theme_nodes", output_format)
    write_table(tree_df, out_dir / "theme_tree_edges", output_format)
    write_table(member_df, out_dir / "theme_memberships", output_format)
    write_table(relation_df, out_dir / "theme_relation_edges", output_format)
    write_table(temporal_df, out_dir / "temporal_theme_edges", output_format)
    write_table(summary_df, out_dir / "p1_b50_b35_summary", output_format)

    manifest = {
        "builder": "build_b50_b35_theme_forest.py",
        "groups": len(groups),
        "theme_nodes": len(node_df),
        "tree_edges": len(tree_df),
        "memberships": len(member_df),
        "relation_edges": len(relation_df),
        "temporal_edges": len(temporal_df),
        "output_format": output_format,
        "b50": B50.__dict__,
        "b35": B35.__dict__,
        "relation": relation_cfg.__dict__,
        "temporal": temporal_cfg.__dict__,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Build B50 stable theme forest plus B35 local refinement from P0 edges.")
    ap.add_argument("--p0-edges", required=True, help="P0 edges parquet file or directory")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--decision-time", default=None, help="Optional exact decision_time filter")
    ap.add_argument("--layer-id", default=None, help="Optional layer_id filter")
    ap.add_argument("--scale", default=None, help="Optional scale/lookback filter, e.g. 60m")
    ap.add_argument("--max-groups", type=int, default=None, help="Optional smoke-test group cap")
    ap.add_argument("--output-format", choices=["parquet", "csv"], default="parquet")
    ap.add_argument("--hard-relation-threshold", type=float, default=0.25)
    ap.add_argument("--fuzzy-relation-min", type=float, default=0.15)
    ap.add_argument("--fuzzy-relation-scale", type=float, default=0.25)
    ap.add_argument("--hard-temporal-jaccard", type=float, default=0.25)
    ap.add_argument("--fuzzy-temporal-min", type=float, default=0.15)
    args = ap.parse_args()

    df = read_edges(Path(args.p0_edges))
    if args.decision_time is not None:
        df = df[df["decision_time"].astype(str) == str(args.decision_time)]
    if args.layer_id is not None:
        df = df[df["layer_id"].astype(str) == str(args.layer_id)]
    if args.scale is not None:
        df = df[df["scale"].astype(str) == str(args.scale)]

    rel_cfg = RelationConfig(
        hard_threshold=args.hard_relation_threshold,
        fuzzy_min_strength=args.fuzzy_relation_min,
        fuzzy_scale=args.fuzzy_relation_scale,
    )
    temp_cfg = TemporalConfig(
        hard_jaccard=args.hard_temporal_jaccard,
        fuzzy_min_strength=args.fuzzy_temporal_min,
    )
    manifest = build_forest(
        df,
        Path(args.out_dir),
        output_format=args.output_format,
        max_groups=args.max_groups,
        relation_cfg=rel_cfg,
        temporal_cfg=temp_cfg,
    )
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()
