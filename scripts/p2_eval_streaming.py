#!/usr/bin/env python3
"""Partition-safe, streaming P2 alpha evaluation."""
from __future__ import annotations

import hashlib
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from p2_parallel_runtime import collect_process_map
from p2_pit_core import ext, iter_time_groups, read_partition, stream_frames, write_manifest, write_parquet_atomic
from p2_pit_features import evaluate_daily_frame, evaluate_intraday_frame


def _shard_name(path: Path) -> str:
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:16]
    return f"{digest}.parquet"


def _evaluate_file_to_shard(path: Path, mode: str, shard_dir: str) -> dict:
    shard_path = Path(shard_dir) / _shard_name(path)
    frames: list[pd.DataFrame] = []
    if mode == "intraday":
        for _, snapshot in iter_time_groups(path):
            metrics = evaluate_intraday_frame(snapshot)
            if not metrics.empty:
                frames.append(metrics)
    else:
        frame = read_partition(path)
        metrics = evaluate_daily_frame(frame) if not frame.empty else pd.DataFrame()
        if not metrics.empty:
            frames.append(metrics)
    output = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not output.empty:
        write_parquet_atomic(output, shard_path)
    return {"input": str(path), "shard": str(shard_path), "rows": len(output), "status": "complete" if len(output) else "empty"}


def _discover_feature_files(
    root: str | Path,
    filename: str,
    dates: set[str] | None,
    layers: set[str] | None,
    scales: set[str] | None,
) -> list[Path]:
    files: list[Path] = []
    for path in Path(root).rglob(filename):
        date, layer, scale = ext(path, "date"), ext(path, "layer_id"), ext(path, "scale")
        if dates and date not in dates:
            continue
        if layers and layer not in layers:
            continue
        if scales and scale not in scales:
            continue
        files.append(path)
    files.sort(key=lambda path: path.stat().st_size, reverse=True)
    return files


def _update_summary(accumulator: dict, metrics: pd.DataFrame, mode: str) -> None:
    if metrics.empty:
        return
    keys = ["score", "target", "layer_id", "scale", "level"]
    grouped = metrics.groupby(keys, sort=False, dropna=False)
    for group_key, subset in grouped:
        state = accumulator.setdefault(group_key, {
            "days": 0,
            "snapshots": 0,
            "sample_count": 0,
            "rank_ic_sum": 0.0,
            "rank_ic_count": 0,
            "spread_sum": 0.0,
            "spread_count": 0,
            "positive_count": 0,
        })
        state["days"] += int(subset["date"].nunique())
        state["snapshots"] += int(subset["decision_time"].nunique()) if mode == "intraday" else int(subset["date"].nunique())
        state["sample_count"] += int(subset["sample_count"].sum())
        rank = pd.to_numeric(subset["rank_ic"], errors="coerce")
        spread = pd.to_numeric(subset["top_minus_bottom"], errors="coerce")
        state["rank_ic_sum"] += float(rank.sum(skipna=True))
        state["rank_ic_count"] += int(rank.notna().sum())
        state["spread_sum"] += float(spread.sum(skipna=True))
        state["spread_count"] += int(spread.notna().sum())
        state["positive_count"] += int((spread > 0).sum())


def _summary_frame(accumulator: dict) -> pd.DataFrame:
    rows: list[dict] = []
    for key, state in accumulator.items():
        score, target, layer_id, scale, level = key
        rows.append({
            "score": score,
            "target": target,
            "layer_id": layer_id,
            "scale": scale,
            "level": level,
            "days": state["days"],
            "snapshots": state["snapshots"],
            "sample_count": state["sample_count"],
            "mean_rank_ic": state["rank_ic_sum"] / state["rank_ic_count"] if state["rank_ic_count"] else np.nan,
            "mean_spread": state["spread_sum"] / state["spread_count"] if state["spread_count"] else np.nan,
            "positive_period_rate": state["positive_count"] / state["spread_count"] if state["spread_count"] else np.nan,
        })
    return pd.DataFrame(rows)


def evaluate_feature_root(
    root: str | Path,
    output_dir: str | Path,
    mode: str,
    workers: int = 8,
    dates: set[str] | None = None,
    layers: set[str] | None = None,
    scales: set[str] | None = None,
) -> dict:
    started = time.time()
    filename = "intraday_relation_features.parquet" if mode == "intraday" else "daily_relation_features.parquet"
    files = _discover_feature_files(root, filename, dates, layers, scales)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = output_dir / f".{mode}_metric_shards"
    shutil.rmtree(shard_dir, ignore_errors=True)
    shard_dir.mkdir(parents=True, exist_ok=True)

    if not files:
        metadata = {"stage": f"{mode}_feature_eval", "status": "empty", "input_count": 0, "output_rows": 0}
        write_manifest(output_dir, metadata)
        return metadata

    worker_count = max(1, min(int(workers), len(files)))
    results = collect_process_map(files, worker_count, _evaluate_file_to_shard, mode, str(shard_dir), max_in_flight=worker_count * 2, max_tasks_per_child=1)
    shard_paths = [Path(result["shard"]) for result in results if result["status"] == "complete" and Path(result["shard"]).exists()]

    metrics_parquet = output_dir / f"{mode}_alpha_metrics.parquet"
    metrics_csv = output_dir / f"{mode}_alpha_metrics.csv"
    if metrics_csv.exists():
        metrics_csv.unlink()
    accumulator: dict = {}
    header = True

    def metric_frames():
        nonlocal header
        for shard in sorted(shard_paths):
            metrics = pd.read_parquet(shard)
            if metrics.empty:
                continue
            _update_summary(accumulator, metrics, mode)
            metrics.to_csv(metrics_csv, mode="a", header=header, index=False)
            header = False
            yield metrics

    metric_rows, _ = stream_frames(metrics_parquet, metric_frames())
    if header:
        pd.DataFrame().to_csv(metrics_csv, index=False)
    summary = _summary_frame(accumulator)
    summary_path = output_dir / f"{mode}_alpha_summary.csv"
    summary.to_csv(summary_path, index=False)
    shutil.rmtree(shard_dir, ignore_errors=True)

    metadata = {
        "stage": f"{mode}_feature_eval",
        "status": "complete" if metric_rows else "empty",
        "input_count": len(files),
        "metric_rows": metric_rows,
        "summary_rows": len(summary),
        "output_rows": metric_rows,
        "evaluation_scope": "per_decision_time_cross_section" if mode == "intraday" else "per_date_eod_cross_section",
        "evaluation_input_mode": "partition_shards_no_global_concat",
        "workers": worker_count,
        "dates_filter": sorted(dates) if dates else None,
        "metrics_csv": str(metrics_csv),
        "metrics_parquet": str(metrics_parquet),
        "summary": str(summary_path),
        "elapsed_sec": round(time.time() - started, 3),
    }
    write_manifest(output_dir, metadata)
    return metadata
