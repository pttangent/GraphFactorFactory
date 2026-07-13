#!/usr/bin/env python3
"""Point-in-time-safe P0 graph alpha extraction and evaluation."""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from p2_alpha_pit_features import (
    DEFAULT_INTRADAY_HORIZONS,
    PIT_CONTRACT_VERSION,
    csvlist,
    csvset,
    ext,
    label_path,
    load_labels,
    write_manifest,
    write_parquet_atomic,
)


@dataclass(frozen=True)
class Part:
    date: str
    layer_id: str
    scale: str
    base: Path


def discover(root: str | Path, dates=None, layers=None, scales=None) -> list[Part]:
    parts = []
    for path in Path(root).rglob("edges.parquet"):
        date, layer, scale = ext(path, "date"), ext(path, "layer_id"), ext(path, "scale")
        if not date:
            continue
        layer = layer or "all"
        scale = scale or "default"
        if dates and date not in dates:
            continue
        if layers and layer not in layers and layer != "all":
            continue
        if scales and scale not in scales and scale != "default":
            continue
        parts.append(Part(date, layer, scale, path))
    return sorted(parts, key=lambda part: part.base.stat().st_size, reverse=True)


def normalize_edges(frame: pd.DataFrame, part: Part) -> pd.DataFrame:
    frame = frame.copy()
    frame["decision_time"] = pd.to_datetime(frame["decision_time"], utc=True, errors="coerce")
    frame["src_id"] = pd.to_numeric(frame["src_id"], errors="coerce").astype("Int64")
    frame["dst_id"] = pd.to_numeric(frame["dst_id"], errors="coerce").astype("Int64")
    frame = frame.dropna(subset=["decision_time", "src_id", "dst_id"]).copy()
    frame["src_id"] = frame["src_id"].astype("int64")
    frame["dst_id"] = frame["dst_id"].astype("int64")
    if "layer_id" not in frame:
        frame["layer_id"] = part.layer_id
    if "scale" not in frame:
        frame["scale"] = part.scale
    frame["weight"] = pd.to_numeric(frame.get("weight", 0), errors="coerce").fillna(0.0)
    frame["abs_weight"] = frame["weight"].abs()
    if "window_end" in frame:
        frame["window_end"] = pd.to_datetime(frame["window_end"], utc=True, errors="coerce")
        if not (frame["window_end"].isna() | (frame["window_end"] <= frame["decision_time"])).all():
            raise AssertionError("P0 edge window_end exceeds decision_time")
    return frame


def read_row_group(parquet: pq.ParquetFile, index: int) -> pd.DataFrame:
    wanted = ["decision_time", "window_start", "window_end", "layer_id", "scale", "src_id", "dst_id", "weight"]
    columns = [column for column in wanted if column in parquet.schema.names]
    return parquet.read_row_group(index, columns=columns).to_pandas()


def node_features_one(part: Part, labels_root: str, output_root: str, horizons: list[str], max_row_groups: int | None, skip_existing: bool = False) -> dict:
    started = time.time()
    output_dir = Path(output_root) / f"date={part.date}" / f"layer_id={part.layer_id}" / f"scale={part.scale}"
    if skip_existing and output_dir.exists():
        return {"date": part.date, "layer": part.layer_id, "scale": part.scale, "status": "skipped", "duration": 0}
    labels = load_labels(label_path(labels_root, part.date), horizons)
    labels_by_time = {key: value for key, value in labels.groupby("decision_time", sort=False)}
    parquet = pq.ParquetFile(part.base)
    count = parquet.metadata.num_row_groups if max_row_groups is None else min(max_row_groups, parquet.metadata.num_row_groups)
    frames = []
    for index in range(count):
        edges = normalize_edges(read_row_group(parquet, index), part)
        if edges.empty:
            continue
        source = edges.groupby(["decision_time", "layer_id", "scale", "src_id"], sort=False).agg(src_edge_count=("dst_id", "size"), src_weight_sum=("abs_weight", "sum"), src_weight_mean=("abs_weight", "mean"), src_weight_max=("abs_weight", "max")).reset_index().rename(columns={"src_id": "symbol_id"})
        destination = edges.groupby(["decision_time", "layer_id", "scale", "dst_id"], sort=False).agg(dst_edge_count=("src_id", "size"), dst_weight_sum=("abs_weight", "sum"), dst_weight_mean=("abs_weight", "mean"), dst_weight_max=("abs_weight", "max")).reset_index().rename(columns={"dst_id": "symbol_id"})
        features = source.merge(destination, on=["decision_time", "layer_id", "scale", "symbol_id"], how="outer").fillna(0)
        features["p0_total_edge_count"] = features.src_edge_count + features.dst_edge_count
        features["p0_total_weight_sum"] = features.src_weight_sum + features.dst_weight_sum
        decision_time = features["decision_time"].iloc[0]
        target = labels_by_time.get(decision_time)
        if target is not None:
            features = features.merge(target, on=["decision_time", "symbol_id"], how="inner")
        if not features.empty:
            features["pit_audit_pass"] = True
            frames.append(features)
    output = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not output.empty:
        write_parquet_atomic(output, output_dir / "p0_node_features.parquet")
    meta = {"stage": "p0_node_features", "status": "complete" if len(output) else "empty", "output_rows": len(output), "row_groups": count, "elapsed_sec": round(time.time() - started, 3)}
    write_manifest(output_dir, meta)
    return meta


def edge_spillover_one(part: Part, labels_root: str, output_root: str, horizons: list[str], past_horizon: str, max_row_groups: int | None, skip_existing: bool = False) -> dict:
    started = time.time()
    output_dir = Path(output_root) / f"date={part.date}" / f"layer_id={part.layer_id}" / f"scale={part.scale}"
    if skip_existing and output_dir.exists():
        return {"date": part.date, "layer": part.layer_id, "scale": part.scale, "status": "skipped", "duration": 0}
    labels = load_labels(label_path(labels_root, part.date), horizons)
    past = f"past_label_{past_horizon}"
    if past not in labels:
        raise ValueError(f"missing {past}")
    source_by_time = {}
    target_by_time = {}
    import re
    target_columns = [column for column in labels if re.fullmatch(r"label_\d+m", column)]
    metadata_columns = [column for column in labels if column == "label_entry_time" or re.fullmatch(r"label_(?:entry|exit)_time_\d+m", column)]
    for decision_time, subset in labels.groupby("decision_time", sort=False):
        source_columns = ["decision_time", "symbol_id", past, f"past_exit_time_{past_horizon}"]
        source_by_time[decision_time] = subset[source_columns].rename(columns={"symbol_id": "src_id", past: "src_past_return", f"past_exit_time_{past_horizon}": "src_past_available_time"})
        rename = {"symbol_id": "dst_id", **{column: "target_" + column.replace("label_", "") for column in target_columns}}
        for column in metadata_columns:
            match = re.fullmatch(r"label_(entry|exit)_time_(\d+m)", column)
            if match:
                rename[column] = f"target_{match.group(1)}_time_{match.group(2)}"
        selected = list(dict.fromkeys(["decision_time", "symbol_id"] + target_columns + metadata_columns))
        target_by_time[decision_time] = subset[selected].rename(columns=rename)
    parquet = pq.ParquetFile(part.base)
    count = parquet.metadata.num_row_groups if max_row_groups is None else min(max_row_groups, parquet.metadata.num_row_groups)
    frames = []
    for index in range(count):
        edges = normalize_edges(read_row_group(parquet, index), part)
        if edges.empty:
            continue
        decision_time = edges["decision_time"].iloc[0]
        source, target = source_by_time.get(decision_time), target_by_time.get(decision_time)
        if source is None or target is None:
            continue
        merged = edges.merge(source, on=["decision_time", "src_id"], how="inner")
        if merged.empty:
            continue
        valid = merged.src_past_available_time.notna()
        if valid.any() and not (merged.loc[valid, "src_past_available_time"] <= merged.loc[valid, "decision_time"]).all():
            raise AssertionError("P0 source past return unavailable at decision_time")
        merged["edge_signal"] = merged.weight * pd.to_numeric(merged.src_past_return, errors="coerce").fillna(0.0)
        aggregate = merged.groupby(["decision_time", "layer_id", "scale", "dst_id"], sort=False).agg(p0_edge_spillover_signal=("edge_signal", "mean"), p0_edge_spillover_sum=("edge_signal", "sum"), p0_edge_count=("src_id", "size"), p0_edge_abs_weight=("abs_weight", "sum"), p0_edge_mean_abs_weight=("abs_weight", "mean")).reset_index()
        output = aggregate.merge(target, on=["decision_time", "dst_id"], how="inner")
        if not output.empty:
            output["pit_audit_pass"] = True
            frames.append(output)
    output = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not output.empty:
        write_parquet_atomic(output, output_dir / "p0_edge_spillover_features.parquet")
    meta = {"stage": "p0_edge_spillover", "status": "complete" if len(output) else "empty", "output_rows": len(output), "row_groups": count, "past_horizon": past_horizon, "elapsed_sec": round(time.time() - started, 3)}
    write_manifest(output_dir, meta)
    return meta


def graph_state_one(part: Part, output_root: str, max_row_groups: int | None, skip_existing: bool = False) -> dict:
    started = time.time()
    output_dir = Path(output_root) / f"date={part.date}" / f"layer_id={part.layer_id}" / f"scale={part.scale}"
    if skip_existing and output_dir.exists():
        return {"date": part.date, "layer": part.layer_id, "scale": part.scale, "status": "skipped", "duration": 0}
    parquet = pq.ParquetFile(part.base)
    count = parquet.metadata.num_row_groups if max_row_groups is None else min(max_row_groups, parquet.metadata.num_row_groups)
    rows = []
    for index in range(count):
        edges = normalize_edges(read_row_group(parquet, index), part)
        for keys, group in edges.groupby(["decision_time", "layer_id", "scale"], sort=False):
            nodes = pd.unique(pd.concat([group.src_id, group.dst_id], ignore_index=True))
            rows.append({"decision_time": keys[0], "layer_id": keys[1], "scale": keys[2], "edge_count": len(group), "active_node_count": len(nodes), "avg_abs_weight": group.abs_weight.mean(), "sum_abs_weight": group.abs_weight.sum(), "max_abs_weight": group.abs_weight.max(), "density_proxy": len(group) / max(len(nodes), 1), "pit_audit_pass": True})
    output = pd.DataFrame(rows)
    if not output.empty:
        write_parquet_atomic(output, output_dir / "p0_graph_state_features.parquet")
    meta = {"stage": "p0_graph_state", "status": "complete" if len(output) else "empty", "output_rows": len(output)}
    write_manifest(output_dir, meta)
    return meta


def re_full_intraday_target(column: str) -> bool:
    import re
    return bool(re.fullmatch(r"(?:label_|target_)\d+m", column))


def evaluate_p0_one(path: Path) -> list[dict]:
    rows = []
    frame = pd.read_parquet(path)
    if "pit_audit_pass" in frame and not frame.pit_audit_pass.all():
        raise AssertionError(f"failed PIT rows in {path}")
    date = ext(path, "date") or "unknown"
    kind = "edge" if "edge_spillover" in path.name else "node"
    targets = [column for column in frame if (column.startswith("label_") or column.startswith("target_")) and re_full_intraday_target(column)]
    features = [column for column in frame if column.startswith("p0_") and pd.api.types.is_numeric_dtype(frame[column])]
    for keys, subset in frame.groupby(["decision_time", "layer_id", "scale"], dropna=False, sort=False):
        for feature in features:
            for target in targets:
                values = subset[[feature, target]].replace([np.inf, -np.inf], np.nan).dropna()
                if len(values) < 30:
                    continue
                q80, q20 = values[feature].quantile(0.8), values[feature].quantile(0.2)
                rank_ic = np.nan if values[feature].nunique() < 2 or values[target].nunique() < 2 else values[feature].rank().corr(values[target].rank())
                rows.append({"date": date, "decision_time": keys[0], "kind": kind, "layer_id": keys[1], "scale": keys[2], "feature": feature, "target": target, "sample_count": len(values), "rank_ic": rank_ic, "top_minus_bottom": values.loc[values[feature] >= q80, target].mean() - values.loc[values[feature] <= q20, target].mean()})
    return rows


def evaluate_p0(root: str | Path, output_dir: str | Path, workers: int = 12, month: str = None) -> dict:
    files = list(Path(root).rglob("p0_node_features.parquet")) + list(Path(root).rglob("p0_edge_spillover_features.parquet"))
    if month:
        files = [f for f in files if f"date={month}" in str(f)]
    
    list_of_rows = pool(files, workers, evaluate_p0_one)
    rows = [row for sublist in list_of_rows for row in sublist]
    
    metrics = pd.DataFrame(rows)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "p0_alpha_metrics.csv", index=False)
    summary = metrics.groupby(["kind", "feature", "target", "layer_id", "scale"], sort=False).agg(days=("date", "nunique"), snapshots=("decision_time", "nunique"), sample_count=("sample_count", "sum"), mean_rank_ic=("rank_ic", "mean"), mean_spread=("top_minus_bottom", "mean"), positive_period_rate=("top_minus_bottom", lambda values: float((values > 0).mean()))).reset_index() if not metrics.empty else pd.DataFrame()
    summary.to_csv(output_dir / "p0_alpha_summary.csv", index=False)
    meta = {"stage": "p0_alpha_eval", "status": "complete" if len(metrics) else "empty", "input_files": len(files), "metric_rows": len(metrics), "summary_rows": len(summary), "output_rows": len(metrics), "evaluation_scope": "per_decision_time_cross_section"}
    write_manifest(output_dir, meta)
    return meta


def pool(parts, workers, function, *args):
    if not parts:
        return []
    with cf.ProcessPoolExecutor(max_workers=workers) as executor:
        return [future.result() for future in cf.as_completed([executor.submit(function, part, *args) for part in parts])]


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("node-features", "edge-spillover", "graph-state"):
        sub = commands.add_parser(name)
        sub.add_argument("--p0-root", required=True)
        sub.add_argument("--labels-root")
        sub.add_argument("--out-root", required=True)
        sub.add_argument("--dates")
        sub.add_argument("--layers")
        sub.add_argument("--scales")
        sub.add_argument("--horizons", default=",".join(DEFAULT_INTRADAY_HORIZONS))
        sub.add_argument("--past-horizon", default="15m")
        sub.add_argument("--workers", type=int, default=16)
        sub.add_argument("--max-row-groups", type=int)
        sub.add_argument("--skip-existing", action="store_true")
    p0_eval = commands.add_parser("eval-p0")
    p0_eval.add_argument("--p0-alpha-root", type=str, required=True)
    p0_eval.add_argument("--out-dir", type=str, required=True)
    p0_eval.add_argument("--workers", type=int, default=12)
    p0_eval.add_argument("--month", type=str, default=None, help="Filter files by month (YYYY-MM)")
    args = parser.parse_args()
    if args.command == "eval-p0":
        result = evaluate_p0(args.p0_alpha_root, args.out_dir, args.workers, args.month)
    else:
        dates, layers, scales = csvset(args.dates), csvset(args.layers), csvset(args.scales)
        horizons = [h for h in (csvlist(args.horizons) or DEFAULT_INTRADAY_HORIZONS) if h.endswith("m")]
        parts = discover(args.p0_root, dates, layers, scales)
        if args.command == "node-features":
            result = pool(parts, args.workers, node_features_one, args.labels_root, args.out_root, horizons, args.max_row_groups, args.skip_existing)
        elif args.command == "edge-spillover":
            result = pool(parts, args.workers, edge_spillover_one, args.labels_root, args.out_root, horizons, args.past_horizon, args.max_row_groups, args.skip_existing)
        else:
            result = pool(parts, args.workers, graph_state_one, args.out_root, args.max_row_groups, args.skip_existing)
    print(json.dumps({"pit_contract_version": PIT_CONTRACT_VERSION, "result": result}, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
