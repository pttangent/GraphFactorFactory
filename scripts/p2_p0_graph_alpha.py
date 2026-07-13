#!/usr/bin/env python3
"""Point-in-time-safe, snapshot-streamed P0 graph alpha extraction."""
from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from p2_parallel_runtime import collect_process_map
from p2_alpha_pit_features import (
    DEFAULT_INTRADAY_HORIZONS,
    PIT_CONTRACT_VERSION,
    csvlist,
    csvset,
    ext,
    is_complete,
    iter_time_groups,
    label_path,
    load_labels,
    stream_frames,
    write_manifest,
)
from p2_p0_eval_streaming import evaluate_p0_streaming


@dataclass(frozen=True)
class Part:
    date: str
    layer_id: str
    scale: str
    base: Path


EDGE_COLUMNS = [
    "decision_time",
    "window_start",
    "window_end",
    "layer_id",
    "scale",
    "src_id",
    "dst_id",
    "weight",
]


def discover(root: str | Path, dates=None, layers=None, scales=None) -> list[Part]:
    parts: list[Part] = []
    unpartitioned: list[Path] = []
    for path in Path(root).rglob("edges.parquet"):
        date, layer, scale = ext(path, "date"), ext(path, "layer_id"), ext(path, "scale")
        if not date:
            continue
        if dates and date not in dates:
            continue
        if not layer or not scale:
            unpartitioned.append(path)
            continue
        if layers and layer not in layers:
            continue
        if scales and scale not in scales:
            continue
        parts.append(Part(date, layer, scale, path))
    if not parts and unpartitioned:
        sample = ", ".join(str(path) for path in unpartitioned[:3])
        raise ValueError(
            "P0 alpha requires physical date/layer/scale shards; "
            f"found only unpartitioned edges such as {sample}. "
            "Run shard_p0_edges_by_layer_scale.py first."
        )
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
    frame["layer_id"] = frame["layer_id"].astype(str)
    frame["scale"] = frame["scale"].astype(str)
    if not frame["layer_id"].eq(str(part.layer_id)).all():
        raise ValueError(f"mixed layer_id inside physical shard {part.base}")
    if not frame["scale"].eq(str(part.scale)).all():
        raise ValueError(f"mixed scale inside physical shard {part.base}")
    frame["weight"] = pd.to_numeric(frame.get("weight", 0), errors="coerce").fillna(0.0).astype("float32")
    frame["abs_weight"] = frame["weight"].abs()
    if "window_end" in frame:
        frame["window_end"] = pd.to_datetime(frame["window_end"], utc=True, errors="coerce")
        if not (frame["window_end"].isna() | (frame["window_end"] <= frame["decision_time"])).all():
            raise AssertionError("P0 edge window_end exceeds decision_time")
    return frame


def _edge_snapshots(part: Part, max_row_groups: int | None):
    for decision_time, raw in iter_time_groups(
        part.base,
        EDGE_COLUMNS,
        max_row_groups,
        time_column="decision_time",
        batch_size=250_000,
    ):
        edges = normalize_edges(raw, part)
        if not edges.empty:
            yield decision_time, edges


def _labels_at(indexed: pd.DataFrame, decision_time: pd.Timestamp) -> pd.DataFrame:
    try:
        return indexed.loc[[decision_time]].reset_index(drop=True)
    except KeyError:
        return pd.DataFrame(columns=indexed.columns)


def node_features_one(
    part: Part,
    labels_root: str,
    output_root: str,
    horizons: list[str],
    max_row_groups: int | None,
    skip_existing: bool = False,
) -> dict:
    started = time.time()
    output_dir = Path(output_root) / f"date={part.date}" / f"layer_id={part.layer_id}" / f"scale={part.scale}"
    output_path = output_dir / "p0_node_features.parquet"
    if skip_existing and is_complete(output_dir / "manifest.json") and output_path.exists():
        return {"date": part.date, "layer": part.layer_id, "scale": part.scale, "status": "skipped", "duration": 0}

    labels = load_labels(label_path(labels_root, part.date), horizons)
    labels_indexed = labels.set_index("decision_time", drop=False).sort_index()

    def frames():
        for decision_time, edges in _edge_snapshots(part, max_row_groups):
            source = (
                edges.groupby(["decision_time", "layer_id", "scale", "src_id"], sort=False)
                .agg(
                    src_edge_count=("dst_id", "size"),
                    src_weight_sum=("abs_weight", "sum"),
                    src_weight_mean=("abs_weight", "mean"),
                    src_weight_max=("abs_weight", "max"),
                )
                .reset_index()
                .rename(columns={"src_id": "symbol_id"})
            )
            destination = (
                edges.groupby(["decision_time", "layer_id", "scale", "dst_id"], sort=False)
                .agg(
                    dst_edge_count=("src_id", "size"),
                    dst_weight_sum=("abs_weight", "sum"),
                    dst_weight_mean=("abs_weight", "mean"),
                    dst_weight_max=("abs_weight", "max"),
                )
                .reset_index()
                .rename(columns={"dst_id": "symbol_id"})
            )
            features = source.merge(
                destination,
                on=["decision_time", "layer_id", "scale", "symbol_id"],
                how="outer",
                copy=False,
            )
            numeric = [column for column in features if column.endswith(("_count", "_sum", "_mean", "_max"))]
            features[numeric] = features[numeric].fillna(0)
            features["p0_total_edge_count"] = features["src_edge_count"] + features["dst_edge_count"]
            features["p0_total_weight_sum"] = features["src_weight_sum"] + features["dst_weight_sum"]
            target = _labels_at(labels_indexed, decision_time)
            if target.empty:
                continue
            output = features.merge(target, on=["decision_time", "symbol_id"], how="inner", copy=False)
            if not output.empty:
                output["pit_audit_pass"] = True
                yield output

    rows, batches = stream_frames(output_path, frames())
    meta = {
        "stage": "p0_node_features",
        "status": "complete" if rows else "empty",
        "output_rows": rows,
        "write_batches": batches,
        "input_mode": "complete_decision_time_stream",
        "elapsed_sec": round(time.time() - started, 3),
    }
    write_manifest(output_dir, meta)
    return meta


def edge_spillover_one(
    part: Part,
    labels_root: str,
    output_root: str,
    horizons: list[str],
    past_horizon: str,
    max_row_groups: int | None,
    skip_existing: bool = False,
) -> dict:
    started = time.time()
    output_dir = Path(output_root) / f"date={part.date}" / f"layer_id={part.layer_id}" / f"scale={part.scale}"
    output_path = output_dir / "p0_edge_spillover_features.parquet"
    if skip_existing and is_complete(output_dir / "manifest.json") and output_path.exists():
        return {"date": part.date, "layer": part.layer_id, "scale": part.scale, "status": "skipped", "duration": 0}

    labels = load_labels(label_path(labels_root, part.date), horizons)
    past = f"past_label_{past_horizon}"
    past_exit = f"past_exit_time_{past_horizon}"
    if past not in labels or past_exit not in labels:
        raise ValueError(f"missing {past} or {past_exit}")
    labels_indexed = labels.set_index("decision_time", drop=False).sort_index()
    target_columns = [column for column in labels if re.fullmatch(r"label_\d+m", column)]
    metadata_columns = [
        column
        for column in labels
        if column == "label_entry_time" or re.fullmatch(r"label_(?:entry|exit)_time_\d+m", column)
    ]

    def frames():
        for decision_time, edge_snapshot in _edge_snapshots(part, max_row_groups):
            subset = _labels_at(labels_indexed, decision_time)
            if subset.empty:
                continue
            source = subset[["decision_time", "symbol_id", past, past_exit]].rename(
                columns={
                    "symbol_id": "src_id",
                    past: "src_past_return",
                    past_exit: "src_past_available_time",
                }
            )
            rename = {
                "symbol_id": "dst_id",
                **{column: "target_" + column.replace("label_", "") for column in target_columns},
            }
            for column in metadata_columns:
                match = re.fullmatch(r"label_(entry|exit)_time_(\d+m)", column)
                if match:
                    rename[column] = f"target_{match.group(1)}_time_{match.group(2)}"
            selected = list(dict.fromkeys(["decision_time", "symbol_id"] + target_columns + metadata_columns))
            target = subset[selected].rename(columns=rename)
            merged = edge_snapshot.merge(source, on=["decision_time", "src_id"], how="inner", copy=False)
            if merged.empty:
                continue
            valid = merged["src_past_available_time"].notna()
            if valid.any() and not (
                merged.loc[valid, "src_past_available_time"] <= merged.loc[valid, "decision_time"]
            ).all():
                raise AssertionError("P0 source past return unavailable at decision_time")
            merged["edge_signal"] = merged["weight"] * pd.to_numeric(
                merged["src_past_return"], errors="coerce"
            ).fillna(0.0)
            aggregate = (
                merged.groupby(["decision_time", "layer_id", "scale", "dst_id"], sort=False)
                .agg(
                    p0_edge_spillover_signal=("edge_signal", "mean"),
                    p0_edge_spillover_sum=("edge_signal", "sum"),
                    p0_edge_count=("src_id", "size"),
                    p0_edge_abs_weight=("abs_weight", "sum"),
                    p0_edge_mean_abs_weight=("abs_weight", "mean"),
                )
                .reset_index()
            )
            output = aggregate.merge(target, on=["decision_time", "dst_id"], how="inner", copy=False)
            if not output.empty:
                output["pit_audit_pass"] = True
                yield output

    rows, batches = stream_frames(output_path, frames())
    meta = {
        "stage": "p0_edge_spillover",
        "status": "complete" if rows else "empty",
        "output_rows": rows,
        "write_batches": batches,
        "past_horizon": past_horizon,
        "input_mode": "complete_decision_time_stream",
        "elapsed_sec": round(time.time() - started, 3),
    }
    write_manifest(output_dir, meta)
    return meta


def graph_state_one(
    part: Part,
    output_root: str,
    max_row_groups: int | None,
    skip_existing: bool = False,
) -> dict:
    started = time.time()
    output_dir = Path(output_root) / f"date={part.date}" / f"layer_id={part.layer_id}" / f"scale={part.scale}"
    output_path = output_dir / "p0_graph_state_features.parquet"
    if skip_existing and is_complete(output_dir / "manifest.json") and output_path.exists():
        return {"date": part.date, "layer": part.layer_id, "scale": part.scale, "status": "skipped", "duration": 0}

    def frames():
        for _, edges in _edge_snapshots(part, max_row_groups):
            if edges.empty:
                continue
            nodes = np.unique(np.concatenate([edges["src_id"].to_numpy(), edges["dst_id"].to_numpy()]))
            row = {
                "decision_time": edges["decision_time"].iloc[0],
                "layer_id": edges["layer_id"].iloc[0],
                "scale": edges["scale"].iloc[0],
                "edge_count": len(edges),
                "active_node_count": len(nodes),
                "avg_abs_weight": float(edges["abs_weight"].mean()),
                "sum_abs_weight": float(edges["abs_weight"].sum()),
                "max_abs_weight": float(edges["abs_weight"].max()),
                "density_proxy": len(edges) / max(len(nodes), 1),
                "pit_audit_pass": True,
            }
            yield pd.DataFrame([row])

    rows, batches = stream_frames(output_path, frames())
    meta = {
        "stage": "p0_graph_state",
        "status": "complete" if rows else "empty",
        "output_rows": rows,
        "write_batches": batches,
        "input_mode": "complete_decision_time_stream",
        "elapsed_sec": round(time.time() - started, 3),
    }
    write_manifest(output_dir, meta)
    return meta


def evaluate_p0(root: str | Path, output_dir: str | Path, workers: int = 12, month: str | None = None) -> dict:
    return evaluate_p0_streaming(root, output_dir, workers, month)


def pool(parts, workers, function, *args):
    if not parts:
        return []
    worker_count = max(1, min(int(workers), len(parts)))
    return collect_process_map(
        parts,
        worker_count,
        function,
        *args,
        max_in_flight=worker_count * 2,
        max_tasks_per_child=1,
    )


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
    p0_eval.add_argument("--p0-alpha-root", required=True)
    p0_eval.add_argument("--out-dir", required=True)
    p0_eval.add_argument("--workers", type=int, default=12)
    p0_eval.add_argument("--month")
    args = parser.parse_args()

    if args.command == "eval-p0":
        result = evaluate_p0(args.p0_alpha_root, args.out_dir, args.workers, args.month)
    else:
        dates, layers, scales = csvset(args.dates), csvset(args.layers), csvset(args.scales)
        horizons = [
            horizon
            for horizon in (csvlist(args.horizons) or DEFAULT_INTRADAY_HORIZONS)
            if horizon.endswith("m")
        ]
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
