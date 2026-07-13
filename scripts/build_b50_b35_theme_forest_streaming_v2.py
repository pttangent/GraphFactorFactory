#!/usr/bin/env python3
"""Atomic, true-input-streaming B50/B35 P1 builder."""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from build_b50_b35_theme_forest_streaming import (
    B35,
    B50,
    RelationConfig,
    TemporalConfig,
    build_adj_from_pairs,
    build_relation_edges,
    compact_pairs,
    refine_to_b35,
    safe_token,
    split_to_b50,
    weighted_degree,
)
from p2_pit_core import iter_time_groups

P1_CONTRACT_VERSION = "p1-streaming-v2"


class AtomicTableSink:
    def __init__(self, path: Path, fmt: str = "parquet", max_rows_per_write: int = 100_000) -> None:
        self.final_path = path.with_suffix(".csv" if fmt == "csv" else ".parquet")
        self.temp_path = Path(str(self.final_path) + ".tmp")
        self.fmt = fmt
        self.max_rows_per_write = max(1, int(max_rows_per_write))
        self.writer: pq.ParquetWriter | None = None
        self.csv_header_written = False
        self.rows = 0
        self.write_batches = 0
        self.final_path.parent.mkdir(parents=True, exist_ok=True)
        self.temp_path.unlink(missing_ok=True)

    def write(self, rows: list[dict]) -> None:
        if not rows:
            return
        for start in range(0, len(rows), self.max_rows_per_write):
            chunk = rows[start : start + self.max_rows_per_write]
            frame = pd.DataFrame(chunk)
            self.rows += len(frame)
            self.write_batches += 1
            if self.fmt == "csv":
                frame.to_csv(
                    self.temp_path,
                    mode="a",
                    header=not self.csv_header_written,
                    index=False,
                )
                self.csv_header_written = True
                continue
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if self.writer is None:
                self.writer = pq.ParquetWriter(
                    self.temp_path,
                    table.schema,
                    compression="zstd",
                    use_dictionary=True,
                )
            elif table.schema != self.writer.schema:
                table = table.cast(self.writer.schema)
            self.writer.write_table(table)

    def close(self, commit: bool) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        if commit and self.rows:
            os.replace(self.temp_path, self.final_path)
        else:
            self.temp_path.unlink(missing_ok=True)
            if commit:
                self.final_path.unlink(missing_ok=True)


def _normalise_edges(group: pd.DataFrame) -> pd.DataFrame:
    required = {"decision_time", "layer_id", "scale", "src_id", "dst_id", "weight"}
    if missing := required - set(group):
        raise ValueError(f"P1 physical shard missing required columns {sorted(missing)}")
    group = group[["decision_time", "layer_id", "scale", "src_id", "dst_id", "weight"]].copy()
    group = group[group["src_id"] != group["dst_id"]]
    group["src_id"] = pd.to_numeric(group["src_id"], errors="coerce").astype("Int64")
    group["dst_id"] = pd.to_numeric(group["dst_id"], errors="coerce").astype("Int64")
    group = group.dropna(subset=["src_id", "dst_id"])
    group["src_id"] = group["src_id"].astype("int64")
    group["dst_id"] = group["dst_id"].astype("int64")
    group["layer_id"] = group["layer_id"].astype(str)
    group["scale"] = group["scale"].astype(str)
    if group["layer_id"].nunique(dropna=False) != 1:
        raise ValueError("P1 shard contains mixed layer_id values")
    if group["scale"].nunique(dropna=False) != 1:
        raise ValueError("P1 shard contains mixed scale values")
    group["weight"] = pd.to_numeric(group["weight"], errors="coerce").fillna(0.0).astype("float32")
    group["abs_weight"] = group["weight"].abs()
    return group


def temporal_edges_fast(
    ts_prev: str,
    ts_cur: str,
    layer: str,
    scale: str,
    prev: list[tuple[str, set[int]]] | None,
    cur: list[tuple[str, set[int]]],
    level: str,
    cfg: TemporalConfig,
) -> list[dict]:
    if not prev or not cur:
        return []
    current_for_member = {member: theme_id for theme_id, members in cur for member in members}
    current_sizes = {theme_id: len(members) for theme_id, members in cur}
    rows: list[dict] = []
    for source_id, source_members in prev:
        counts = Counter(current_for_member[member] for member in source_members if member in current_for_member)
        if not counts:
            continue
        best = None
        for target_id, overlap in counts.items():
            target_size = current_sizes[target_id]
            union = len(source_members) + target_size - overlap
            jaccard = overlap / max(union, 1)
            containment = overlap / max(min(len(source_members), target_size), 1)
            strength = max(jaccard, 0.7 * jaccard + 0.3 * containment)
            candidate = (strength, overlap, target_id, jaccard, containment)
            if best is None or candidate > best:
                best = candidate
        assert best is not None
        strength, overlap, target_id, jaccard, containment = best
        if strength < cfg.fuzzy_min_strength and jaccard < cfg.hard_jaccard:
            continue
        rows.append({
            "layer_id": layer,
            "scale": scale,
            "level": level,
            "src_theme_id": source_id,
            "dst_theme_id": target_id,
            "src_time": ts_prev,
            "dst_time": ts_cur,
            "continuation_strength": float(strength),
            "jaccard": float(jaccard),
            "containment": float(containment),
            "overlap": int(overlap),
            "hard_continue": bool(jaccard >= cfg.hard_jaccard),
            "fuzzy_continue": bool(strength >= cfg.fuzzy_min_strength),
        })
    return rows


def build_shard(
    path: Path,
    out_dir: Path,
    output_format: str,
    max_snapshots: int | None,
    relation_cfg: RelationConfig,
    temporal_cfg: TemporalConfig,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    sinks = {
        "theme_nodes": AtomicTableSink(out_dir / "theme_nodes", output_format),
        "theme_tree_edges": AtomicTableSink(out_dir / "theme_tree_edges", output_format),
        "theme_memberships": AtomicTableSink(out_dir / "theme_memberships", output_format),
        "theme_relation_edges": AtomicTableSink(out_dir / "theme_relation_edges", output_format),
        "temporal_theme_edges": AtomicTableSink(out_dir / "temporal_theme_edges", output_format),
        "summary": AtomicTableSink(out_dir / "summary", output_format),
    }
    columns = ["decision_time", "layer_id", "scale", "src_id", "dst_id", "weight"]
    prev_b50 = prev_b35 = None
    prev_ts = None
    snapshots = 0
    success = False
    try:
        for decision_time, raw_group in iter_time_groups(path, columns):
            if max_snapshots is not None and snapshots >= max_snapshots:
                break
            group = _normalise_edges(raw_group)
            if group.empty:
                continue
            ts = str(decision_time)
            layer = str(group["layer_id"].iloc[0])
            scale = str(group["scale"].iloc[0])
            member_array = np.unique(
                np.concatenate([group["src_id"].to_numpy(), group["dst_id"].to_numpy()])
            )
            members = set(map(int, member_array))
            pairs = compact_pairs(group)
            adjacency = build_adj_from_pairs(pairs, members)
            degree = weighted_degree(group)
            root_id = f"ts={safe_token(ts)}|layer={safe_token(layer)}|scale={safe_token(scale)}|root"
            root_node = {
                "decision_time": ts,
                "layer_id": layer,
                "scale": scale,
                "theme_id": root_id,
                "parent_theme_id": "",
                "level": "ROOT",
                "depth": 0,
                "size": len(members),
                "is_leaf": False,
                "boundary_config": "root",
                "root_b50_theme_id": "",
            }
            nodes50, tree50, memberships50, leaves50 = split_to_b50(
                ts, layer, scale, members, adjacency, degree, root_id, 1
            )
            nodes35, tree35, memberships35, leaves35 = refine_to_b35(
                ts, layer, scale, leaves50, adjacency, degree
            )
            sinks["theme_nodes"].write([root_node] + nodes50 + nodes35)
            sinks["theme_tree_edges"].write(tree50 + tree35)
            sinks["theme_memberships"].write(memberships50 + memberships35)
            relations50 = build_relation_edges(ts, layer, scale, group, leaves50, "B50", relation_cfg)
            relations35 = build_relation_edges(ts, layer, scale, group, leaves35, "B35", relation_cfg)
            sinks["theme_relation_edges"].write(relations50 + relations35)
            if prev_ts is not None:
                sinks["temporal_theme_edges"].write(
                    temporal_edges_fast(str(prev_ts), ts, layer, scale, prev_b50, leaves50, "B50", temporal_cfg)
                )
                sinks["temporal_theme_edges"].write(
                    temporal_edges_fast(str(prev_ts), ts, layer, scale, prev_b35, leaves35, "B35", temporal_cfg)
                )

            def stats(leaves: list[tuple[str, set[int]]]) -> tuple[int, int, float]:
                sizes = [len(item[1]) for item in leaves]
                return len(sizes), max(sizes) if sizes else 0, float(np.median(sizes)) if sizes else 0.0

            count50, max50, median50 = stats(leaves50)
            count35, max35, median35 = stats(leaves35)
            sinks["summary"].write([{
                "decision_time": ts,
                "layer_id": layer,
                "scale": scale,
                "root_size": len(members),
                "raw_edges": int(len(group)),
                "pair_edges": int(len(pairs)),
                "b50_leaf_count": count50,
                "b50_leaf_max": max50,
                "b50_leaf_median": median50,
                "b35_leaf_count": count35,
                "b35_leaf_max": max35,
                "b35_leaf_median": median35,
                "b50_relation_edges": len(relations50),
                "b35_relation_edges": len(relations35),
            }])
            prev_b50, prev_b35, prev_ts = leaves50, leaves35, ts
            snapshots += 1
            if snapshots % 25 == 0:
                print(json.dumps({"snapshots": snapshots, "out_dir": str(out_dir)}), flush=True)
        success = True
    finally:
        for sink in sinks.values():
            sink.close(commit=success)

    total_rows = sum(sink.rows for sink in sinks.values())
    manifest = {
        "builder": "build_b50_b35_theme_forest_streaming_v2.py",
        "p1_contract_version": P1_CONTRACT_VERSION,
        "status": "complete" if snapshots else "empty",
        "input": str(path),
        "out_dir": str(out_dir),
        "snapshots": snapshots,
        "output_rows": total_rows,
        "output_format": output_format,
        "input_mode": "parquet_batch_time_group_stream",
        "temporal_matching": "inverted_member_index_best_successor",
        "atomic_outputs": True,
        "maximum_rows_per_sink_write": 100_000,
        "rows": {name: sink.rows for name, sink in sinks.items()},
        "write_batches": {name: sink.write_batches for name, sink in sinks.items()},
        "b50": B50.__dict__,
        "b35": B35.__dict__,
        "relation": relation_cfg.__dict__,
        "temporal": temporal_cfg.__dict__,
    }
    temporary = out_dir / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, out_dir / "manifest.json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build B50/B35 P1 from one sorted date/layer/scale shard.")
    parser.add_argument("--p0-edges-shard", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--output-format", choices=["parquet", "csv"], default="parquet")
    parser.add_argument("--max-snapshots", type=int)
    parser.add_argument("--hard-relation-threshold", type=float, default=0.25)
    parser.add_argument("--fuzzy-relation-min", type=float, default=0.15)
    parser.add_argument("--fuzzy-relation-scale", type=float, default=0.25)
    parser.add_argument("--hard-temporal-jaccard", type=float, default=0.25)
    parser.add_argument("--fuzzy-temporal-min", type=float, default=0.15)
    args = parser.parse_args()
    relation = RelationConfig(
        hard_threshold=args.hard_relation_threshold,
        fuzzy_min_strength=args.fuzzy_relation_min,
        fuzzy_scale=args.fuzzy_relation_scale,
    )
    temporal = TemporalConfig(
        hard_jaccard=args.hard_temporal_jaccard,
        fuzzy_min_strength=args.fuzzy_temporal_min,
    )
    manifest = build_shard(
        Path(args.p0_edges_shard),
        Path(args.out_dir),
        args.output_format,
        args.max_snapshots,
        relation,
        temporal,
    )
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()
