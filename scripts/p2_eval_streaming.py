#!/usr/bin/env python3
"""Partition-safe, resumable P2 alpha evaluation."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from p2_checkpoint import file_fingerprint, read_json, write_json_atomic
from p2_exact_eval import evaluate_daily_frame_exact, evaluate_intraday_frame_exact
from p2_parallel_runtime import collect_process_map
from p2_pit_core import ext, iter_time_groups, read_partition, write_manifest
from p2_streaming_io import stream_frames

EVAL_CONTRACT_VERSION = "p2-eval-resumable-partitioned-v3"
SUMMARY_KEYS = ["score", "target", "layer_id", "scale", "level"]
SUMMARY_COLUMNS = [
    *SUMMARY_KEYS,
    "days",
    "snapshots",
    "sample_count",
    "mean_rank_ic",
    "mean_spread",
    "positive_period_rate",
]
SUMMARY_STATE_COLUMNS = [
    "date",
    *SUMMARY_KEYS,
    "snapshots",
    "sample_count",
    "rank_ic_sum",
    "rank_ic_count",
    "spread_sum",
    "spread_count",
    "positive_count",
]


def _token(path: Path) -> str:
    return hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:16]


def _input_signature(files: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda value: str(value.resolve())):
        digest.update(json.dumps(file_fingerprint(path), sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


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
    for group_key, subset in metrics.groupby(SUMMARY_KEYS, sort=False, dropna=False):
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
        if mode == "intraday" and "decision_time" in subset:
            state["snapshots"].update(subset["decision_time"].dropna().astype(str).tolist())
        else:
            state["snapshots"].update(subset["date"].dropna().astype(str).tolist())
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
        score, target, layer_id, scale, level = key
        records.append(
            {
                "date": str(date),
                "score": str(score),
                "target": str(target),
                "layer_id": str(layer_id),
                "scale": str(scale),
                "level": str(level),
                "snapshots": len(state["snapshots"]),
                "sample_count": int(state["sample_count"]),
                "rank_ic_sum": float(state["rank_ic_sum"]),
                "rank_ic_count": int(state["rank_ic_count"]),
                "spread_sum": float(state["spread_sum"]),
                "spread_count": int(state["spread_count"]),
                "positive_count": int(state["positive_count"]),
            }
        )
    return records


def _summary_frame(records: list[dict] | pd.DataFrame) -> pd.DataFrame:
    partial = records if isinstance(records, pd.DataFrame) else pd.DataFrame.from_records(records, columns=SUMMARY_STATE_COLUMNS)
    if partial.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    rows: list[dict] = []
    for key, subset in partial.groupby(SUMMARY_KEYS, sort=False, dropna=False):
        score, target, layer_id, scale, level = key
        rank_count = int(subset["rank_ic_count"].sum())
        spread_count = int(subset["spread_count"].sum())
        rows.append(
            {
                "score": score,
                "target": target,
                "layer_id": layer_id,
                "scale": scale,
                "level": level,
                "days": int(subset["date"].nunique()),
                "snapshots": int(subset["snapshots"].sum()),
                "sample_count": int(subset["sample_count"].sum()),
                "mean_rank_ic": float(subset["rank_ic_sum"].sum()) / rank_count if rank_count else np.nan,
                "mean_spread": float(subset["spread_sum"].sum()) / spread_count if spread_count else np.nan,
                "positive_period_rate": int(subset["positive_count"].sum()) / spread_count if spread_count else np.nan,
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def _state_valid(state_path: Path, input_path: Path, mode: str, metric_path: Path, csv_path: Path | None) -> dict | None:
    payload = read_json(state_path)
    if not payload:
        return None
    if payload.get("evaluation_contract_version") != EVAL_CONTRACT_VERSION:
        return None
    if payload.get("mode") != mode or payload.get("input_fingerprint") != file_fingerprint(input_path):
        return None
    rows = int(payload.get("rows", 0))
    if rows > 0 and not metric_path.exists():
        return None
    if rows == 0 and metric_path.exists():
        return None
    if csv_path is not None and rows > 0 and not csv_path.exists():
        return None
    return payload


def _evaluate_file_to_shard(
    path: Path,
    mode: str,
    metric_dir: str,
    state_dir: str,
    csv_dir: str | None,
) -> dict:
    token = _token(path)
    metric_path = Path(metric_dir) / f"part-{token}.parquet"
    state_path = Path(state_dir) / f"part-{token}.json"
    csv_path = Path(csv_dir) / f"part-{token}.csv" if csv_dir else None
    cached = _state_valid(state_path, path, mode, metric_path, csv_path)
    if cached is not None:
        return {**cached, "status": "reused", "shard": str(metric_path), "csv": str(csv_path) if csv_path else None}

    date = ext(path, "date") or "unknown"
    accumulator: dict = {}
    csv_header = True
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.unlink(missing_ok=True)

    def metric_frames():
        nonlocal csv_header
        if mode == "intraday":
            source_frames = (snapshot for _, snapshot in iter_time_groups(path))
        else:
            frame = read_partition(path)
            source_frames = iter([frame]) if not frame.empty else iter([])
        for source in source_frames:
            metrics = evaluate_intraday_frame_exact(source) if mode == "intraday" else evaluate_daily_frame_exact(source)
            if metrics.empty:
                continue
            _update_summary(accumulator, metrics, mode)
            if csv_path is not None:
                metrics.to_csv(csv_path, mode="a", header=csv_header, index=False)
                csv_header = False
            yield metrics

    rows, batches = stream_frames(metric_path, metric_frames())
    if rows == 0 and csv_path is not None:
        csv_path.unlink(missing_ok=True)
    payload = {
        "evaluation_contract_version": EVAL_CONTRACT_VERSION,
        "mode": mode,
        "input": str(path),
        "input_fingerprint": file_fingerprint(path),
        "rows": int(rows),
        "batches": int(batches),
        "summary_records": _partial_summary_records(accumulator, date),
    }
    write_json_atomic(state_path, payload)
    return {**payload, "status": "complete" if rows else "empty", "shard": str(metric_path), "csv": str(csv_path) if csv_path else None}


def _write_frame_atomic(frame: pd.DataFrame, path: Path, *, parquet: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    temporary.unlink(missing_ok=True)
    if parquet:
        frame.to_parquet(temporary, index=False)
    else:
        frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


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


def _evaluation_complete(output_dir: Path, mode: str, input_signature: str) -> bool:
    payload = read_json(output_dir / "manifest.json")
    return bool(
        payload
        and payload.get("status") == "complete"
        and payload.get("evaluation_contract_version") == EVAL_CONTRACT_VERSION
        and payload.get("mode") == mode
        and payload.get("input_signature") == input_signature
        and (output_dir / f"{mode}_alpha_metrics.parquet").is_dir()
        and (output_dir / f"{mode}_alpha_summary.csv").exists()
        and (output_dir / f"{mode}_alpha_summary_state.parquet").exists()
    )


def evaluate_feature_root(
    root: str | Path,
    output_dir: str | Path,
    mode: str,
    workers: int = 8,
    dates: set[str] | None = None,
    layers: set[str] | None = None,
    scales: set[str] | None = None,
    csv_mode: str = "none",
    skip_existing: bool = False,
) -> dict:
    if mode not in {"intraday", "daily"}:
        raise ValueError("mode must be intraday or daily")
    if csv_mode not in {"none", "sharded", "single"}:
        raise ValueError("csv_mode must be none, sharded, or single")

    started = time.time()
    filename = "intraday_relation_features.parquet" if mode == "intraday" else "daily_relation_features.parquet"
    files = _discover_feature_files(root, filename, dates, layers, scales)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    signature = _input_signature(files)
    if skip_existing and _evaluation_complete(output_dir, mode, signature):
        return {
            "stage": f"{mode}_feature_eval",
            "status": "skipped",
            "mode": mode,
            "evaluation_contract_version": EVAL_CONTRACT_VERSION,
            "elapsed_sec": 0.0,
        }

    if not files:
        shutil.rmtree(output_dir / f"{mode}_alpha_metrics.parquet", ignore_errors=True)
        shutil.rmtree(output_dir / f".{mode}_metric_state", ignore_errors=True)
        shutil.rmtree(output_dir / f"{mode}_alpha_metrics_csv", ignore_errors=True)
        for name in (f"{mode}_alpha_metrics.csv", f"{mode}_alpha_summary.csv", f"{mode}_alpha_summary_state.parquet"):
            (output_dir / name).unlink(missing_ok=True)
        metadata = {
            "stage": f"{mode}_feature_eval",
            "status": "empty",
            "mode": mode,
            "input_count": 0,
            "output_rows": 0,
            "evaluation_contract_version": EVAL_CONTRACT_VERSION,
            "input_signature": signature,
        }
        write_manifest(output_dir, metadata)
        return metadata

    metric_dir = output_dir / f"{mode}_alpha_metrics.parquet"
    state_dir = output_dir / f".{mode}_metric_state"
    csv_dir = output_dir / f"{mode}_alpha_metrics_csv"
    metric_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    csv_dir_arg = None
    if csv_mode in {"sharded", "single"}:
        csv_dir.mkdir(parents=True, exist_ok=True)
        csv_dir_arg = str(csv_dir)

    worker_count = max(1, min(int(workers), len(files)))
    results = collect_process_map(
        files,
        worker_count,
        _evaluate_file_to_shard,
        mode,
        str(metric_dir),
        str(state_dir),
        csv_dir_arg,
        max_in_flight=worker_count * 2,
        max_tasks_per_child=8,
    )

    expected_tokens = {_token(path) for path in files}
    for directory, suffix in ((metric_dir, ".parquet"), (state_dir, ".json")):
        for path in directory.glob(f"part-*{suffix}"):
            if path.stem.replace("part-", "") not in expected_tokens:
                path.unlink(missing_ok=True)
    if csv_dir.exists():
        for path in csv_dir.glob("part-*.csv"):
            if path.stem.replace("part-", "") not in expected_tokens:
                path.unlink(missing_ok=True)

    metric_rows = int(sum(int(result.get("rows", 0)) for result in results))
    summary_records = [record for result in results for record in result.get("summary_records", [])]
    summary_state = pd.DataFrame.from_records(summary_records, columns=SUMMARY_STATE_COLUMNS)
    summary = _summary_frame(summary_state)
    state_path = output_dir / f"{mode}_alpha_summary_state.parquet"
    summary_path = output_dir / f"{mode}_alpha_summary.csv"
    _write_frame_atomic(summary_state, state_path, parquet=True)
    _write_frame_atomic(summary, summary_path, parquet=False)

    full_csv = output_dir / f"{mode}_alpha_metrics.csv"
    if csv_mode == "none":
        full_csv.unlink(missing_ok=True)
        shutil.rmtree(csv_dir, ignore_errors=True)
    elif csv_mode == "sharded":
        full_csv.unlink(missing_ok=True)
    else:
        csv_paths = [Path(result["csv"]) for result in results if result.get("csv") and Path(result["csv"]).exists()]
        _concatenate_csv_shards(csv_paths, full_csv)

    metadata = {
        "stage": f"{mode}_feature_eval",
        "status": "complete" if metric_rows else "empty",
        "mode": mode,
        "evaluation_contract_version": EVAL_CONTRACT_VERSION,
        "missing_data_semantics": "exact_pairwise_complete_by_validity_mask",
        "input_count": len(files),
        "input_signature": signature,
        "metric_rows": metric_rows,
        "metric_shards": sum(1 for path in metric_dir.glob("part-*.parquet")),
        "reused_shards": sum(1 for result in results if result.get("status") == "reused"),
        "summary_rows": len(summary),
        "output_rows": metric_rows,
        "evaluation_scope": "per_decision_time_cross_section" if mode == "intraday" else "per_date_eod_cross_section",
        "evaluation_input_mode": "resumable_parallel_partition_shards",
        "metrics_layout": "partitioned_parquet_dataset",
        "csv_mode": csv_mode,
        "workers": worker_count,
        "dates_filter": sorted(dates) if dates else None,
        "layers_filter": sorted(layers) if layers else None,
        "scales_filter": sorted(scales) if scales else None,
        "parallel_summary_reduction": True,
        "serial_global_dataframe_concat": False,
        "elapsed_sec": round(time.time() - started, 3),
    }
    write_manifest(output_dir, metadata)
    return metadata


def merge_evaluation_states(
    evaluation_root: str | Path,
    output_dir: str | Path,
    mode: str,
) -> dict:
    """Build an exact global summary from small monthly reducer states."""
    started = time.time()
    filename = f"{mode}_alpha_summary_state.parquet"
    output_dir = Path(output_dir)
    files = [path for path in Path(evaluation_root).rglob(filename) if output_dir not in path.parents]
    files.sort()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not files:
        metadata = {
            "stage": f"{mode}_global_eval_merge",
            "status": "empty",
            "input_states": 0,
            "output_rows": 0,
            "evaluation_contract_version": EVAL_CONTRACT_VERSION,
        }
        write_manifest(output_dir, metadata)
        return metadata

    states = [pd.read_parquet(path) for path in files]
    combined = pd.concat(states, ignore_index=True) if states else pd.DataFrame(columns=SUMMARY_STATE_COLUMNS)
    combined = combined.reindex(columns=SUMMARY_STATE_COLUMNS)
    summary = _summary_frame(combined)
    _write_frame_atomic(combined, output_dir / filename, parquet=True)
    _write_frame_atomic(summary, output_dir / f"{mode}_alpha_summary.csv", parquet=False)
    metadata = {
        "stage": f"{mode}_global_eval_merge",
        "status": "complete" if len(summary) else "empty",
        "mode": mode,
        "evaluation_contract_version": EVAL_CONTRACT_VERSION,
        "input_states": len(files),
        "state_rows": len(combined),
        "summary_rows": len(summary),
        "output_rows": len(summary),
        "global_input_mode": "monthly_summary_state_merge_no_raw_feature_rescan",
        "elapsed_sec": round(time.time() - started, 3),
    }
    write_manifest(output_dir, metadata)
    return metadata
