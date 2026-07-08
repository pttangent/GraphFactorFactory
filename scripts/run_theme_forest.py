from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from graphfactorfactory.themes.forest import ThemeForestBuilder, ThemeForestConfig


def _load_metadata(graph_root: Path, metadata_path: str | None) -> pd.DataFrame | None:
    if not metadata_path:
        return None
    symbols = pd.read_parquet(graph_root / "dimensions" / "symbols.parquet")
    meta = pd.read_parquet(metadata_path).rename(
        columns={
            "company_name": "company",
            "sector_code": "sector",
            "industry_code": "industry",
        }
    )
    return symbols.merge(meta, on="symbol", how="left")


def run_day(theme_day: Path, output_day: Path, builder: ThemeForestBuilder, max_frames: int | None = None) -> dict:
    themes = pd.read_parquet(theme_day / "themes.parquet")
    layer_communities = pd.read_parquet(theme_day / "layer_communities.parquet", columns=["snapshot_time", "layer_name", "members"])
    subcommunities = pd.read_parquet(theme_day / "subcommunities.parquet", columns=["snapshot_time", "layer_name", "members"])

    times = sorted(themes["snapshot_time"].unique())
    if max_frames is not None:
        times = times[:max_frames]
        themes = themes[themes["snapshot_time"].isin(times)].copy()
        layer_communities = layer_communities[layer_communities["snapshot_time"].isin(times)].copy()
        subcommunities = subcommunities[subcommunities["snapshot_time"].isin(times)].copy()

    node_parts = []
    edge_parts = []
    member_parts = []
    for index, snapshot_time in enumerate(times, start=1):
        snapshot_themes = themes[themes["snapshot_time"] == snapshot_time]
        snapshot_parents = layer_communities[layer_communities["snapshot_time"] == snapshot_time]
        snapshot_children = subcommunities[subcommunities["snapshot_time"] == snapshot_time]
        result = builder.build_snapshot_forest(
            snapshot_time=snapshot_time,
            themes=snapshot_themes,
            layer_communities=snapshot_parents,
            subcommunities=snapshot_children,
        )
        node_parts.append(result.nodes)
        edge_parts.append(result.edges)
        member_parts.append(result.members)
        if index % 50 == 0 or index == len(times):
            print(f"ThemeForest progress: {index}/{len(times)} frames")

    nodes = pd.concat(node_parts, ignore_index=True) if node_parts else pd.DataFrame()
    edges = pd.concat(edge_parts, ignore_index=True) if edge_parts else pd.DataFrame()
    members = pd.concat(member_parts, ignore_index=True) if member_parts else pd.DataFrame()
    output_day.mkdir(parents=True, exist_ok=True)
    nodes.to_parquet(output_day / "theme_forest_nodes.parquet", index=False)
    edges.to_parquet(output_day / "theme_forest_edges.parquet", index=False)
    members.to_parquet(output_day / "theme_forest_members.parquet", index=False)

    leaves = nodes[nodes["level"] == 1].copy() if not nodes.empty else pd.DataFrame()
    if not leaves.empty:
        leaves["semantic_concentration_diagnostic"] = leaves[["top_sector_share", "top_industry_share"]].max(axis=1)
        leaves["forest_quality_score"] = (
            np.minimum(1.0, leaves["member_count"] / 40.0) * 0.30
            + leaves["evidence_layers"].clip(upper=6) / 6.0 * 0.35
            + leaves["metadata_coverage"] * 0.15
            + leaves["semantic_concentration_diagnostic"] * 0.20
        )
        leaves.sort_values(["forest_quality_score", "evidence_layers", "member_count"], ascending=[False, False, False]).head(1000).to_csv(output_day / "theme_forest_tearsheet_top1000.csv", index=False)

    roots = nodes[nodes["level"] == 0] if not nodes.empty else pd.DataFrame()
    summary = {
        "date": theme_day.name.split("=", 1)[-1],
        "frames": len(times),
        "broad_themes": int(len(themes)),
        "root_nodes": int(len(roots)),
        "split_roots": int((roots["split_status"] == "split").sum()) if not roots.empty else 0,
        "leaf_nodes": int(len(leaves)),
        "median_root_members": float(themes["members"].map(len).median()) if not themes.empty else 0.0,
        "median_leaf_members": float(leaves["member_count"].median()) if not leaves.empty else 0.0,
        "median_leaf_top_sector_share": float(leaves["top_sector_share"].median()) if not leaves.empty else 0.0,
        "median_leaf_top_industry_share": float(leaves["top_industry_share"].median()) if not leaves.empty else 0.0,
        "semantic_used_for_split": False,
    }
    (output_day / "theme_forest_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--theme-root", required=True)
    parser.add_argument("--graph-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--metadata")
    parser.add_argument("--date-start")
    parser.add_argument("--date-end")
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()

    theme_root = Path(args.theme_root).expanduser().resolve()
    graph_root = Path(args.graph_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    metadata = _load_metadata(graph_root, args.metadata)
    builder = ThemeForestBuilder(config=ThemeForestConfig(), metadata=metadata)
    summaries = []
    for day in sorted(theme_root.glob("date=*")):
        date = day.name.split("=", 1)[-1]
        if args.date_start and date < args.date_start:
            continue
        if args.date_end and date > args.date_end:
            continue
        summaries.append(run_day(day, output_root / day.name, builder, max_frames=args.max_frames))
    pd.DataFrame(summaries).to_csv(output_root / "theme_forest_summary_all.csv", index=False)
    print(json.dumps(summaries, indent=2, default=str))


if __name__ == "__main__":
    main()
