#!/usr/bin/env python3
"""Streaming P0 alpha evaluator without global DataFrame collection."""
from __future__ import annotations

import hashlib
import re
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from p2_parallel_runtime import collect_process_map
from p2_pit_core import ext, iter_partition_batches, iter_time_groups, stream_frames, write_manifest


def _shard_name(path: Path) -> str:
    return hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:16] + ".parquet"


def _target_column(column: str) -> bool:
    return bool(re.fullmatch(r"(?:label_|target_)\d+m", column))


def _evaluate_snapshot(frame: pd.DataFrame, date: str, kind: str) -> pd.DataFrame:
    if "pit_audit_pass" in frame and not bool(frame["pit_audit_pass"].fillna(False).all()):
        raise AssertionError("refusing P0 evaluation with failed PIT rows")
    targets = [column for column in frame if _target_column(column)]
    features = [column for column in frame if column.startswith("p0_") and pd.api.types.is_numeric_dtype(frame[column])]
    rows: list[dict] = []
    for keys, subset in frame.groupby(["decision_time", "layer_id", "scale"], dropna=False, sort=False):
        for feature in features:
            for target in targets:
                values = subset[[feature, target]].replace([np.inf, -np.inf], np.nan).dropna()
                if len(values) < 30:
                    continue
                q80, q20 = values[feature].quantile(0.8), values[feature].quantile(0.2)
                rank_ic = np.nan if values[feature].nunique() < 2 or values[target].nunique() < 2 else values[feature].rank().corr(values[target].rank())
                rows.append({
                    "date": date,
                    "decision_time": keys[0],
                    "kind": kind,
                    "layer_id": keys[1],
                    "scale": keys[2],
                    "feature": feature,
                    "target": target,
                    "sample_count": len(values),
                    "rank_ic": rank_ic,
                    "top_minus_bottom": values.loc[values[feature] >= q80, target].mean() - values.loc[values[feature] <= q20, target].mean(),
                })
    return pd.DataFrame(rows)


def _evaluate_file_to_shard(path: Path, shard_dir: str) -> dict:
    date = ext(path, "date") or "unknown"
    kind = "edge" if "edge_spillover" in path.name else "node"
    shard_path = Path(shard_dir) / _shard_name(path)

    def metric_frames():
        for _, snapshot in iter_time_groups(path):
            metrics = _evaluate_snapshot(snapshot, date, kind)
            if not metrics.empty:
                yield metrics

    rows, batches = stream_frames(shard_path, metric_frames())
    return {"input": str(path), "shard": str(shard_path), "rows": rows, "batches": batches, "status": "complete" if rows else "empty"}


def _update_summary(accumulator: dict, metrics: pd.DataFrame) -> None:
    keys = ["kind", "feature", "target", "layer_id", "scale"]
    for group_key, subset in metrics.groupby(keys, sort=False, dropna=False):
        state = accumulator.setdefault(group_key, {
            "dates": set(),
            "snapshots": set(),
            "sample_count": 0,
            "rank_ic_sum": 0.0,
            "rank_ic_count": 0,
            "spread_sum": 0.0,
            "spread_count": 0,
            "positive_count": 0,
        })
        state["dates"].update(subset["date"].dropna().astype(str).tolist())
        state["snapshots"].update(pd.to_datetime(subset["decision_time"], utc=True, errors="coerce").dropna().astype(str).tolist())
        state["sample_count"] += int(subset["sample_count"].sum())
        rank = pd.to_numeric(subset["rank_ic"], errors="coerce")
        spread = pd.to_numeric(subset["top_minus_bottom"], errors="coerce")
        state["rank_ic_sum"] += float(rank.sum(skipna=True))
        state["rank_ic_count"] += int(rank.notna().sum())
        state["spread_sum"] += float(spread.sum(skipna=True))
        state["spread_count"] += int(spread.notna().sum())
        state["positive_count"] += int((spread > 0).sum())


def _summary_frame(accumulator: dict) -> pd.DataFrame:
    rows = []
    for key, state in accumulator.items():
        kind, feature, target, layer_id, scale = key
        rows.append({
            "kind": kind,
            "feature": feature,
            "target": target,
            "layer_id": layer_id,
            "scale": scale,
            "days": len(state["dates"]),
            "snapshots": len(state["snapshots"]),
            "sample_count": state["sample_count"],
            "mean_rank_ic": state["rank_ic_sum"] / state["rank_ic_count"] if state["rank_ic_count"] else np.nan,
            "mean_spread": state["spread_sum"] / state["spread_count"] if state["spread_count"] else np.nan,
            "positive_period_rate": state["positive_count"] / state["spread_count"] if state["spread_count"] else np.nan,
        })
    return pd.DataFrame(rows)


def evaluate_p0_streaming(root: str | Path, output_dir: str | Path, workers: int = 12, month: str | None = None) -> dict:
    started = time.time()
    files = list(Path(root).rglob("p0_node_features.parquet")) + list(Path(root).rglob("p0_edge_spillover_features.parquet"))
    if month:
        files = [path for path in files if f"date={month}" in str(path)]
    files.sort(key=lambda path: path.stat().st_size, reverse=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = output_dir / ".p0_metric_shards"
    shutil.rmtree(shard_dir, ignore_errors=True)
    shard_dir.mkdir(parents=True, exist_ok=True)

    if not files:
        metadata = {"stage": "p0_alpha_eval", "status": "empty", "input_files": 0, "output_rows": 0}
        write_manifest(output_dir, metadata)
        return metadata

    worker_count = max(1, min(int(workers), len(files)))
    results = collect_process_map(files, worker_count, _evaluate_file_to_shard, str(shard_dir), max_in_flight=worker_count * 2, max_tasks_per_child=1)
    shard_paths = [Path(result["shard"]) for result in results if result["status"] == "complete" and Path(result["shard"]).exists()]

    metrics_parquet = output_dir / "p0_alpha_metrics.parquet"
    metrics_csv = output_dir / "p0_alpha_metrics.csv"
    if metrics_csv.exists():
        metrics_csv.unlink()
    accumulator: dict = {}
    header = True

    def metric_frames():
        nonlocal header
        for shard in sorted(shard_paths):
            for metrics in iter_partition_batches(shard, batch_size=100_000):
                if metrics.empty:
                    continue
                _update_summary(accumulator, metrics)
                metrics.to_csv(metrics_csv, mode="a", header=header, index=False)
                header = False
                yield metrics

    metric_rows, _ = stream_frames(metrics_parquet, metric_frames())
    if header:
        pd.DataFrame().to_csv(metrics_csv, index=False)
    summary = _summary_frame(accumulator)
    summary.to_csv(output_dir / "p0_alpha_summary.csv", index=False)
    shutil.rmtree(shard_dir, ignore_errors=True)
    metadata = {
        "stage": "p0_alpha_eval",
        "status": "complete" if metric_rows else "empty",
        "input_files": len(files),
        "metric_rows": metric_rows,
        "summary_rows": len(summary),
        "output_rows": metric_rows,
        "evaluation_scope": "per_decision_time_cross_section",
        "evaluation_input_mode": "partition_shards_streamed_reducer",
        "month": month,
        "workers": worker_count,
        "elapsed_sec": round(time.time() - started, 3),
    }
    write_manifest(output_dir, metadata)
    return metadata
