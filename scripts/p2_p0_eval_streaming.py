#!/usr/bin/env python3
"""Parallel P0 alpha evaluator with partitioned metrics and bounded memory."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from p2_parallel_runtime import collect_process_map
from p2_pit_core import ext, iter_time_groups, stream_frames, write_manifest

SUMMARY_KEYS = ["kind", "feature", "target", "layer_id", "scale"]
SUMMARY_COLUMNS = [
    *SUMMARY_KEYS,
    "days",
    "snapshots",
    "sample_count",
    "mean_rank_ic",
    "mean_spread",
    "positive_period_rate",
]


def _shard_name(path: Path) -> str:
    return hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:16]


def _target_column(column: str) -> bool:
    return bool(re.fullmatch(r"(?:label_|target_)\d+m", column))


def _evaluate_snapshot(frame: pd.DataFrame, date: str, kind: str) -> pd.DataFrame:
    if "pit_audit_pass" in frame and not bool(frame["pit_audit_pass"].fillna(False).all()):
        raise AssertionError("refusing P0 evaluation with failed PIT rows")
    targets = [column for column in frame if _target_column(column)]
    features = [
        column
        for column in frame
        if column.startswith("p0_") and pd.api.types.is_numeric_dtype(frame[column])
    ]
    rows: list[dict] = []
    for keys, subset in frame.groupby(
        ["decision_time", "layer_id", "scale"],
        dropna=False,
        sort=False,
    ):
        for feature in features:
            for target in targets:
                values = (
                    subset[[feature, target]]
                    .replace([np.inf, -np.inf], np.nan)
                    .dropna()
                )
                if len(values) < 30:
                    continue
                q80 = values[feature].quantile(0.8)
                q20 = values[feature].quantile(0.2)
                if values[feature].nunique() < 2 or values[target].nunique() < 2:
                    rank_ic = np.nan
                else:
                    rank_ic = values[feature].rank().corr(values[target].rank())
                rows.append(
                    {
                        "date": date,
                        "decision_time": keys[0],
                        "kind": kind,
                        "layer_id": keys[1],
                        "scale": keys[2],
                        "feature": feature,
                        "target": target,
                        "sample_count": len(values),
                        "rank_ic": rank_ic,
                        "top_minus_bottom": (
                            values.loc[values[feature] >= q80, target].mean()
                            - values.loc[values[feature] <= q20, target].mean()
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _update_summary(accumulator: dict, metrics: pd.DataFrame) -> None:
    for group_key, subset in metrics.groupby(
        SUMMARY_KEYS,
        sort=False,
        dropna=False,
    ):
        state = accumulator.setdefault(
            group_key,
            {
                "snapshots": set(),
                "sample_count": 0,
                "rank_ic_sum": 0.0,
                "rank_ic_count": 0,
                "spread_sum": 0.0,
                "spread_count": 0,
                "positive_count": 0,
            },
        )
        timestamps = (
            pd.to_datetime(subset["decision_time"], utc=True, errors="coerce")
            .dropna()
            .astype(str)
            .tolist()
        )
        state["snapshots"].update(timestamps)
        state["sample_count"] += int(subset["sample_count"].sum())
        rank = pd.to_numeric(subset["rank_ic"], errors="coerce")
        spread = pd.to_numeric(subset["top_minus_bottom"], errors="coerce")
        state["rank_ic_sum"] += float(rank.sum(skipna=True))
        state["rank_ic_count"] += int(rank.notna().sum())
        state["spread_sum"] += float(spread.sum(skipna=True))
        state["spread_count"] += int(spread.notna().sum())
        state["positive_count"] += int((spread > 0).sum())


def _partial_summary_records(accumulator: dict, date: str) -> list[dict]:
    records: list[dict] = []
    for key, state in accumulator.items():
        kind, feature, target, layer_id, scale = key
        records.append(
            {
                "date": date,
                "kind": kind,
                "feature": feature,
                "target": target,
                "layer_id": layer_id,
                "scale": scale,
                "snapshots": len(state["snapshots"]),
                "sample_count": state["sample_count"],
                "rank_ic_sum": state["rank_ic_sum"],
                "rank_ic_count": state["rank_ic_count"],
                "spread_sum": state["spread_sum"],
                "spread_count": state["spread_count"],
                "positive_count": state["positive_count"],
            }
        )
    return records


def _finalize_summary(records: list[dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    partial = pd.DataFrame.from_records(records)
    rows: list[dict] = []
    for key, subset in partial.groupby(
        SUMMARY_KEYS,
        sort=False,
        dropna=False,
    ):
        kind, feature, target, layer_id, scale = key
        rank_count = int(subset["rank_ic_count"].sum())
        spread_count = int(subset["spread_count"].sum())
        rows.append(
            {
                "kind": kind,
                "feature": feature,
                "target": target,
                "layer_id": layer_id,
                "scale": scale,
                "days": int(subset["date"].nunique()),
                "snapshots": int(subset["snapshots"].sum()),
                "sample_count": int(subset["sample_count"].sum()),
                "mean_rank_ic": (
                    float(subset["rank_ic_sum"].sum()) / rank_count
                    if rank_count
                    else np.nan
                ),
                "mean_spread": (
                    float(subset["spread_sum"].sum()) / spread_count
                    if spread_count
                    else np.nan
                ),
                "positive_period_rate": (
                    int(subset["positive_count"].sum()) / spread_count
                    if spread_count
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def _evaluate_file_to_shard(
    path: Path,
    shard_dir: str,
    csv_dir: str | None,
) -> dict:
    date = ext(path, "date") or "unknown"
    kind = "edge" if "edge_spillover" in path.name else "node"
    token = _shard_name(path)
    shard_path = Path(shard_dir) / f"part-{token}.parquet"
    csv_path = Path(csv_dir) / f"part-{token}.csv" if csv_dir else None
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.unlink(missing_ok=True)

    accumulator: dict = {}
    csv_header = True

    def metric_frames():
        nonlocal csv_header
        for _, snapshot in iter_time_groups(path):
            metrics = _evaluate_snapshot(snapshot, date, kind)
            if metrics.empty:
                continue
            _update_summary(accumulator, metrics)
            if csv_path is not None:
                metrics.to_csv(
                    csv_path,
                    mode="a",
                    header=csv_header,
                    index=False,
                )
                csv_header = False
            yield metrics

    rows, batches = stream_frames(shard_path, metric_frames())
    if not rows and csv_path is not None:
        csv_path.unlink(missing_ok=True)
    return {
        "input": str(path),
        "shard": str(shard_path),
        "csv": str(csv_path) if csv_path is not None and csv_path.exists() else None,
        "rows": rows,
        "batches": batches,
        "status": "complete" if rows else "empty",
        "summary_records": _partial_summary_records(accumulator, date),
    }


def _replace_directory(source: Path, destination: Path) -> None:
    if destination.is_dir():
        shutil.rmtree(destination)
    elif destination.exists():
        destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, destination)


def _concatenate_csv_shards(paths: list[Path], destination: Path) -> None:
    temporary = Path(str(destination) + ".tmp")
    temporary.unlink(missing_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("wb") as output:
        first = True
        for path in sorted(paths):
            with path.open("rb") as source:
                if not first:
                    source.readline()
                shutil.copyfileobj(source, output, length=1024 * 1024)
            first = False
    os.replace(temporary, destination)


def p0_eval_complete(output_dir: str | Path) -> bool:
    output_dir = Path(output_dir)
    try:
        payload = json.loads(
            (output_dir / "manifest.json").read_text(encoding="utf-8")
        )
        return (
            payload.get("status") == "complete"
            and (output_dir / "p0_alpha_metrics.parquet").is_dir()
            and (output_dir / "p0_alpha_summary.csv").exists()
        )
    except Exception:
        return False


def evaluate_p0_streaming(
    root: str | Path,
    output_dir: str | Path,
    workers: int = 12,
    month: str | None = None,
    csv_mode: str = "none",
    skip_existing: bool = False,
) -> dict:
    """Evaluate P0 partitions in parallel.

    The canonical metrics artifact is a partitioned Parquet dataset at
    ``p0_alpha_metrics.parquet/``. Full CSV output is optional because serial
    CSV serialization was the dominant wall-clock bottleneck.
    """
    if csv_mode not in {"none", "sharded", "single"}:
        raise ValueError("csv_mode must be one of: none, sharded, single")

    started = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if skip_existing and p0_eval_complete(output_dir):
        return {
            "stage": "p0_alpha_eval",
            "status": "skipped",
            "month": month,
            "elapsed_sec": 0.0,
        }

    files = list(Path(root).rglob("p0_node_features.parquet"))
    files += list(Path(root).rglob("p0_edge_spillover_features.parquet"))
    if month:
        files = [path for path in files if f"date={month}" in str(path)]
    files.sort(key=lambda path: path.stat().st_size, reverse=True)

    if not files:
        metadata = {
            "stage": "p0_alpha_eval",
            "status": "empty",
            "input_files": 0,
            "output_rows": 0,
            "month": month,
        }
        write_manifest(output_dir, metadata)
        return metadata

    work_root = output_dir / ".p0_eval_work"
    shutil.rmtree(work_root, ignore_errors=True)
    metric_work = work_root / "metrics"
    csv_work = work_root / "csv"
    metric_work.mkdir(parents=True, exist_ok=True)
    csv_dir_arg = None
    if csv_mode in {"sharded", "single"}:
        csv_work.mkdir(parents=True, exist_ok=True)
        csv_dir_arg = str(csv_work)

    worker_count = max(1, min(int(workers), len(files)))
    try:
        results = collect_process_map(
            files,
            worker_count,
            _evaluate_file_to_shard,
            str(metric_work),
            csv_dir_arg,
            max_in_flight=worker_count * 2,
            max_tasks_per_child=1,
        )
        metric_rows = int(sum(result["rows"] for result in results))
        metric_shards = [
            Path(result["shard"])
            for result in results
            if result["status"] == "complete"
            and Path(result["shard"]).exists()
        ]
        summary_records = [
            record
            for result in results
            for record in result["summary_records"]
        ]
        summary = _finalize_summary(summary_records)

        final_metrics = output_dir / "p0_alpha_metrics.parquet"
        _replace_directory(metric_work, final_metrics)

        full_csv = output_dir / "p0_alpha_metrics.csv"
        csv_dataset = output_dir / "p0_alpha_metrics_csv"
        if csv_mode == "none":
            full_csv.unlink(missing_ok=True)
            shutil.rmtree(csv_dataset, ignore_errors=True)
        elif csv_mode == "sharded":
            full_csv.unlink(missing_ok=True)
            _replace_directory(csv_work, csv_dataset)
        else:
            csv_paths = [
                Path(result["csv"])
                for result in results
                if result.get("csv") and Path(result["csv"]).exists()
            ]
            _concatenate_csv_shards(csv_paths, full_csv)
            shutil.rmtree(csv_dataset, ignore_errors=True)

        summary_tmp = output_dir / "p0_alpha_summary.csv.tmp"
        summary.to_csv(summary_tmp, index=False)
        os.replace(summary_tmp, output_dir / "p0_alpha_summary.csv")

        metadata = {
            "stage": "p0_alpha_eval",
            "status": "complete" if metric_rows else "empty",
            "input_files": len(files),
            "metric_rows": metric_rows,
            "metric_shards": len(metric_shards),
            "summary_rows": len(summary),
            "output_rows": metric_rows,
            "evaluation_scope": "per_decision_time_cross_section",
            "evaluation_input_mode": "parallel_partition_metric_shards",
            "metrics_layout": "partitioned_parquet_dataset",
            "metrics_path": str(final_metrics),
            "csv_mode": csv_mode,
            "month": month,
            "workers": worker_count,
            "parallel_summary_reduction": True,
            "serial_global_dataframe_concat": False,
            "elapsed_sec": round(time.time() - started, 3),
        }
        write_manifest(output_dir, metadata)
        return metadata
    finally:
        shutil.rmtree(work_root, ignore_errors=True)
