from __future__ import annotations

"""Every-snapshot P1 persistence analysis.

This analyzer is the no-sampling version of the full-day P1 layer-local theme
persistence check. It tracks every available snapshot in layer_communities.parquet.

Important: sector / industry metadata is used only after tracking for
interpretation. It never enters clustering, split scoring, or path matching.
"""

import argparse
import gc
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def interpret_members(member_ids, sid_to_sym, meta_by_sym, max_syms: int = 20):
    syms = [sid_to_sym.get(int(x), str(int(x))) for x in member_ids]
    m = meta_by_sym.reindex(syms)
    sectors = m["sector_code"].fillna("UNKNOWN").astype(str)
    industries = m["industry_code"].fillna("UNKNOWN").astype(str)
    quote_types = m["quote_type"].fillna("UNKNOWN").astype(str)
    sec_counts = sectors.value_counts()
    ind_counts = industries.value_counts()
    q_counts = quote_types.value_counts()
    return {
        "symbols": ", ".join(syms[:max_syms]),
        "top_sector": sec_counts.index[0] if len(sec_counts) else "UNKNOWN",
        "top_sector_count": int(sec_counts.iloc[0]) if len(sec_counts) else 0,
        "top_sector_share": float(sec_counts.iloc[0] / max(1, len(syms))) if len(sec_counts) else 0.0,
        "top_industry": ind_counts.index[0] if len(ind_counts) else "UNKNOWN",
        "top_industry_count": int(ind_counts.iloc[0]) if len(ind_counts) else 0,
        "top_industry_share": float(ind_counts.iloc[0] / max(1, len(syms))) if len(ind_counts) else 0.0,
        "quote_type_mix": "; ".join([f"{k}:{v}" for k, v in q_counts.head(4).items()]),
    }


def layer_meaning(layer_name: str) -> str:
    if "return_corr" in layer_name:
        return "價格共振/收益同步，較可能出現交易風格、板塊共振或行業共同波動。"
    if "volume_expansion" in layer_name:
        return "成交量擴張/市場注意力，偏事件熱度與短線活躍。"
    if "trade_intensity" in layer_name:
        return "交易筆數/成交活躍同步，偏交易熱度與微結構活躍簇。"
    if "signed_flow" in layer_name:
        return "方向性資金流同步，偏短線買賣壓與 order-flow trigger。"
    if "large_trade_flow" in layer_name:
        return "大單流同步，可能對應機構/大資金交易。"
    if "odd_lot" in layer_name:
        return "odd-lot 小單活躍，可能反映散戶/碎單/高頻交易注意力。"
    if "block_activity" in layer_name:
        return "大宗/區塊交易活動同步，偏機構調倉或 basket execution。"
    if "off_exchange" in layer_name:
        return "場外/暗池交易比例同步，偏執行路由與市場微結構。"
    if "venue_fragmentation" in layer_name:
        return "交易場所碎片化同步，偏流動性分散/執行結構。"
    if "price_impact" in layer_name:
        return "價格衝擊/流動性壓力相似，偏 liquidity risk。"
    if "absorption" in layer_name:
        return "資金流被價格吸收的相似結構，偏承接/壓力吸收。"
    if "flow_return_alignment" in layer_name:
        return "資金流與收益反應一致，偏確認型資金推動。"
    if "report_latency" in layer_name:
        return "資料延遲/報告品質相似，偏 data quality/QA。"
    return "一般交易圖層主題。"


def track_one_layer(df: pd.DataFrame, match_jaccard: float, min_size: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    layer_name = str(df["layer_name"].iloc[0])
    df = df[(df["size"] >= min_size) & (~df["is_market_mode"].fillna(False))].copy()
    df = df.sort_values(["snapshot_time", "community_id"])
    times = list(df["snapshot_time"].drop_duplicates())
    active: dict[str, set[int]] = {}
    path_stats: dict[str, dict] = {}
    next_path = 0
    first_snapshot_communities = int((df["snapshot_time"] == times[0]).sum()) if times else 0

    for ts, snap in df.groupby("snapshot_time", sort=True):
        current = []
        for row in snap.itertuples(index=False):
            members = row.members.tolist() if hasattr(row.members, "tolist") else list(row.members)
            current.append((set(map(int, members)), int(row.community_id), float(row.modularity), int(row.size)))

        inverted: dict[int, list[str]] = defaultdict(list)
        for path_id, members in active.items():
            for member in members:
                inverted[member].append(path_id)

        matched_previous: set[str] = set()
        new_active: dict[str, set[int]] = {}
        for members, _community_id, modularity, _size in current:
            counts: Counter[str] = Counter()
            for member in members:
                for path_id in inverted.get(member, []):
                    if path_id not in matched_previous:
                        counts[path_id] += 1

            best_path = None
            best_jaccard = 0.0
            for path_id, intersection in counts.items():
                union = len(members) + len(active[path_id]) - intersection
                jaccard = intersection / union if union else 0.0
                if jaccard > best_jaccard:
                    best_jaccard = jaccard
                    best_path = path_id

            if best_path is not None and best_jaccard >= match_jaccard:
                path_id = best_path
                matched_previous.add(path_id)
            else:
                path_id = f"{layer_name}|p{next_path:07d}"
                next_path += 1
                path_stats[path_id] = {
                    "layer_name": layer_name,
                    "layer_id": int(df["layer_id"].iloc[0]),
                    "start": ts,
                    "end": ts,
                    "frames": 0,
                    "sizes": [],
                    "mods": [],
                    "jaccards": [],
                    "best_members": members,
                    "best_size": 0,
                }

            stats = path_stats[path_id]
            stats["end"] = ts
            stats["frames"] += 1
            stats["sizes"].append(len(members))
            stats["mods"].append(modularity)
            if best_path is not None:
                stats["jaccards"].append(best_jaccard)
            if len(members) > stats["best_size"]:
                stats["best_members"] = members
                stats["best_size"] = len(members)
            new_active[path_id] = members

        active = new_active

    path_rows = []
    for path_id, stats in path_stats.items():
        sizes = stats["sizes"]
        path_rows.append({
            "path_id": path_id,
            "layer_id": stats["layer_id"],
            "layer_name": layer_name,
            "start": stats["start"],
            "end": stats["end"],
            "frames": stats["frames"],
            "duration_minutes_approx": stats["frames"],
            "avg_size": float(np.mean(sizes)),
            "median_size": float(np.median(sizes)),
            "max_size": int(np.max(sizes)),
            "avg_modularity": float(np.mean(stats["mods"])) if stats["mods"] else None,
            "avg_match_jaccard": float(np.mean(stats["jaccards"])) if stats["jaccards"] else None,
            "best_members": stats["best_members"],
        })

    paths = pd.DataFrame(path_rows)
    communities = int(len(df))
    path_count = int(len(paths))
    continuations = max(0, communities - path_count)
    possible = max(1, communities - first_snapshot_communities)
    summary = pd.DataFrame([{ 
        "layer_name": layer_name,
        "layer_id": int(df["layer_id"].iloc[0]),
        "snapshots": len(times),
        "communities": communities,
        "paths": path_count,
        "first_snapshot_communities": first_snapshot_communities,
        "continuations": continuations,
        "continuation_rate": continuations / possible,
        "stable_paths_ge15min": int((paths["frames"] >= 15).sum()),
        "stable_ratio_ge15min": float((paths["frames"] >= 15).sum() / max(1, path_count)),
        "stable_paths_ge30min": int((paths["frames"] >= 30).sum()),
        "stable_ratio_ge30min": float((paths["frames"] >= 30).sum() / max(1, path_count)),
        "p50_duration_min": float(paths["frames"].median()),
        "p90_duration_min": float(paths["frames"].quantile(0.9)),
        "p99_duration_min": float(paths["frames"].quantile(0.99)),
        "max_duration_min": int(paths["frames"].max()),
        "p50_size": float(df["size"].median()),
        "p90_size": float(df["size"].quantile(0.9)),
        "max_size": int(df["size"].max()),
        "avg_modularity": float(df["modularity"].mean()),
    }])
    return summary, paths, paths.sort_values(["frames", "avg_match_jaccard", "avg_size"], ascending=[False, False, False]).head(8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--theme-day-dir", required=True, help="Directory containing layer_communities.parquet")
    parser.add_argument("--symbols", required=True)
    parser.add_argument("--symbol-metadata", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--min-size", type=int, default=8)
    parser.add_argument("--match-jaccard", type=float, default=0.25)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    communities = pd.read_parquet(Path(args.theme_day_dir) / "layer_communities.parquet")
    communities["size"] = communities["members"].map(len).astype("int32")
    symbols = pd.read_parquet(args.symbols)
    metadata = pd.read_parquet(args.symbol_metadata)
    sid_to_sym = dict(zip(symbols.symbol_id.astype(int), symbols.symbol.astype(str)))
    meta_by_sym = metadata.set_index("symbol")

    summary_rows = []
    representative_rows = []
    compact_path_rows = []
    for layer_name in sorted(communities["layer_name"].unique()):
        layer_df = communities[communities["layer_name"] == layer_name]
        summary, paths, representative = track_one_layer(layer_df, args.match_jaccard, args.min_size)
        summary_rows.append(summary)
        compact_path_rows.append(paths.drop(columns=["best_members"]).sort_values("frames", ascending=False).head(100))
        for rank, row in enumerate(representative.itertuples(index=False), 1):
            info = interpret_members(row.best_members, sid_to_sym, meta_by_sym)
            representative_rows.append({
                "layer_name": layer_name,
                "layer_id": row.layer_id,
                "rank": rank,
                "path_id": row.path_id,
                "start": row.start,
                "end": row.end,
                "frames": row.frames,
                "duration_minutes_approx": row.duration_minutes_approx,
                "avg_size": row.avg_size,
                "max_size": row.max_size,
                "avg_match_jaccard": row.avg_match_jaccard,
                **info,
                "financial_meaning": layer_meaning(layer_name),
            })
        gc.collect()

    summary = pd.concat(summary_rows, ignore_index=True).sort_values(
        ["continuation_rate", "stable_paths_ge30min", "stable_paths_ge15min", "max_duration_min"],
        ascending=False,
    )
    representatives = pd.DataFrame(representative_rows).sort_values(["layer_id", "layer_name", "rank"])
    paths = pd.concat(compact_path_rows, ignore_index=True)
    summary.to_csv(out_dir / "every_snapshot_layer_stability_ranked.csv", index=False)
    representatives.to_csv(out_dir / "every_snapshot_representative_stable_themes.csv", index=False)
    paths.to_csv(out_dir / "every_snapshot_top_paths_compact.csv", index=False)


if __name__ == "__main__":
    main()
