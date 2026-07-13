#!/usr/bin/env python3
"""Parallel, resumable P0 alpha evaluation with exact pairwise semantics."""
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

from p2_checkpoint import file_fingerprint, read_json, write_json_atomic
from p2_parallel_runtime import collect_process_map
from p2_pit_core import ext, iter_time_groups, write_manifest
from p2_streaming_io import stream_frames

EVAL_CONTRACT_VERSION = "p0-eval-pairwise-resumable-v3"
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


def _target_column(column: str) -> bool:
    return bool(re.fullmatch(r"(?:label_|target_)\d+m", column))


def _input_signature(files: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda value: str(value.resolve())):
        digest.update(json.dumps(file_fingerprint(path), sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def _validity_groups(frame: pd.DataFrame, columns: list[str]) -> list[tuple[np.ndarray, list[str]]]:
    groups: dict[tuple[int, bytes], tuple[np.ndarray, list[str]]] = {}
    for column in columns:
        mask = frame[column].notna().to_numpy(dtype=np.bool_, copy=False)
        key = (len(mask), np.packbits(mask, bitorder="little").tobytes())
        if key not in groups:
            groups[key] = (mask, [column])
        else:
            groups[key][1].append(column)
    return list(groups.values())


def _rank_correlation_matrix(features: pd.DataFrame, targets: pd.DataFrame) -> np.ndarray:
    feature_ranks = features.rank(axis=0, method="average")
    target_ranks = targets.rank(axis=0, method="average")
    feature_values = (feature_ranks - feature_ranks.mean(axis=0)).to_numpy(dtype=np.float64, copy=False)
    target_values = (target_ranks - target_ranks.mean(axis=0)).to_numpy(dtype=np.float64, copy=False)
    numerator = feature_values.T @ target_values
    denominator = np.outer(
        np.sqrt(np.square(feature_values).sum(axis=0)),
        np.sqrt(np.square(target_values).sum(axis=0)),
    )
    return np.divide(
        numerator,
        denominator,
        out=np.full(numerator.shape, np.nan, dtype=np.float64),
        where=denominator > 0,
    )


def _spread_matrix(features: pd.DataFrame, targets: pd.DataFrame) -> np.ndarray:
    feature_values = features.to_numpy(dtype=np.float64, copy=False)
    target_values = targets.to_numpy(dtype=np.float64, copy=False)
    quantiles = features.quantile([0.8, 0.2], axis=0)
    q80 = quantiles.loc[0.8].to_numpy(dtype=np.float64, copy=False)
    q20 = quantiles.loc[0.2].to_numpy(dtype=np.float64, copy=False)
    high = feature_values >= q80.reshape(1, -1)
    low = feature_values <= q20.reshape(1, -1)
    high_count = high.sum(axis=0).astype(np.float64)
    low_count = low.sum(axis=0).astype(np.float64)
    high_sum = high.astype(np.float64).T @ target_values
    low_sum = low.astype(np.float64).T @ target_values
    high_mean = np.divide(
        high_sum,
        high_count.reshape(-1, 1),
        out=np.full(high_sum.shape, np.nan, dtype=np.float64),
        where=high_count.reshape(-1, 1) > 0,
    )
    low_mean = np.divide(
        low_sum,
        low_count.reshape(-1, 1),
        out=np.full(low_sum.shape, np.nan, dtype=np.float64),
        where=low_count.reshape(-1, 1) > 0,
    )
    return high_mean - low_mean


def _evaluate_snapshot(frame: pd.DataFrame, date: str, kind: str) -> pd.DataFrame:
    if "pit_audit_pass" in frame and not bool(frame["pit_audit_pass"].fillna(False).all()):
        raise AssertionError("refusing P0 evaluation with failed PIT rows")
    required = {"decision_time", "layer_id", "scale"}
    if missing := required - set(frame):
        raise ValueError(f"P0 evaluation input missing required columns {sorted(missing)}")
    if frame["layer_id"].isna().any() or frame["layer_id"].nunique(dropna=False) != 1:
        raise ValueError("P0 evaluation partition contains mixed or missing layer_id")
    if frame["scale"].isna().any() or frame["scale"].nunique(dropna=False) != 1:
        raise ValueError("P0 evaluation partition contains mixed or missing scale")

    targets = [column for column in frame if _target_column(column)]
    features = [
        column
        for column in frame
        if column.startswith("p0_") and pd.api.types.is_numeric_dtype(frame[column])
    ]
    if not targets or not features:
        return pd.DataFrame()

    frame = frame.replace([np.inf, -np.inf], np.nan)
    layer_id = frame["layer_id"].iloc[0]
    scale = frame["scale"].iloc[0]
    rows: list[dict] = []
    for decision_time, subset in frame.groupby("decision_time", sort=False, dropna=False):
        if pd.isna(decision_time) or len(subset) < 30:
            continue
        for feature_mask, feature_columns in _validity_groups(subset, features):
            for target_mask, target_columns in _validity_groups(subset, targets):
                pair_mask = feature_mask & target_mask
                sample_count = int(pair_mask.sum())
                if sample_count < 30:
                    continue
                pair_features = subset.loc[pair_mask, feature_columns]
                pair_targets = subset.loc[pair_mask, target_columns]
                correlations = _rank_correlation_matrix(pair_features, pair_targets)
                spreads = _spread_matrix(pair_features, pair_targets)
                for feature_index, feature in enumerate(feature_columns):
                    for target_index, target in enumerate(target_columns):
                        rows.append(
                            {
                                "date": date,
                                "decision_time": decision_time,
                                "kind": kind,
                                "layer_id": layer_id,
                                "scale": scale,
                                "feature": feature,
                                "target": target,
                                "sample_count": sample_count,
                                "rank_ic": correlations[feature_index, target_index],
                                "top_minus_bottom": spreads[feature_index, target_index],
                            }
                        )
    return pd.DataFrame(rows)


def _update_summary(accumulator: dict, metrics: pd.DataFrame) -> None:
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
        state["snapshots"].update(subset["decision_time"].dropna().astype(str).tolist())
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
                "date": str(date),
                "kind": str(kind),
                "feature": str(feature),
                "target": str(target),
                "layer_id": str(layer_id),
                "scale": str(scale),
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
                "mean_rank_ic": float(subset["rank_ic_sum"].sum()) / rank_count if rank_count else np.nan,
                "mean_spread": float(subset["spread_sum"].sum()) / spread_count if spread_count else np.nan,
                "positive_period_rate": int(subset["positive_count"].sum()) / spread_count if spread_count else np.nan,
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def _state_valid(state_path: Path, input_path: Path, metric_path: Path, csv_path: Path | None) -> dict | None:
    payload = read_json(state_path)
    if not payload:
        return None
    if payload.get("evaluation_contract_version") != EVAL_CONTRACT_VERSION:
        return None
    if payload.get("input_fingerprint") != file_fingerprint(input_path):
        return None
    rows = int(payload.get("rows", 0))
    if rows > 0 and not metric_path.exists():
        return None
    if rows == 0 and metric_path.exists():
        return None
    if csv_path is not None and rows > 0 and not csv_path.exists():
        return None
    return payload


def _evaluate_file_to_shard(path: Path, metric_dir: str, state_dir: str, csv_dir: str | None) -> dict:
    token = _token(path)
    metric_path = Path(metric_dir) / f"part-{token}.parquet"
    state_path = Path(state_dir) / f"part-{token}.json"
    csv_path = Path(csv_dir) / f"part-{token}.csv" if csv_dir else None
    cached = _state_valid(state_path, path, metric_path, csv_path)
    if cached is not None:
        return {**cached, "status": "reused", "shard": str(metric_path), "csv": str(csv_path) if csv_path else None}

    date = ext(path, "date") or "unknown"
    kind = "edge" if "edge_spillover" in path.name else "node"
    accumulator: dict = {}
    csv_header = True
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.unlink(missing_ok=True)

    def metric_frames():
        nonlocal csv_header
        for _, snapshot in iter_time_groups(path):
            metrics = _evaluate_snapshot(snapshot, date, kind)
            if metrics.empty:
                continue
            _update_summary(accumulator, metrics)
            if csv_path is not None:
                metrics.to_csv(csv_path, mode="a", header=csv_header, index=False)
                csv_header = False
            yield metrics

    rows, batches = stream_frames(metric_path, metric_frames())
    if rows == 0 and csv_path is not None:
        csv_path.unlink(missing_ok=True)
    payload = {
        "evaluation_contract_version": EVAL_CONTRACT_VERSION,
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
    with temporary.open("wb") as output:
        first = True
        for path in sorted(paths):
            with path.open("rb") as source:
                if not first:
                    source.readline()
                shutil.copyfileobj(source, output, length=1024 * 1024)
            first = False
    os.replace(temporary, destination)


def p0_eval_complete(output_dir: str | Path, input_signature: str | None = None) -> bool:
    output_dir = Path(output_dir)
    payload = read_json(output_dir / "manifest.json")
    return bool(
        payload
        and payload.get("status") == "complete"
        and payload.get("evaluation_contract_version") == EVAL_CONTRACT_VERSION
        and (input_signature is None or payload.get("input_signature") == input_signature)
        and (output_dir / "p0_alpha_metrics.parquet").is_dir()
        and (output_dir / "p0_alpha_summary.csv").exists()
        and (output_dir / "p0_alpha_summary_state.parquet").exists()
    )


def evaluate_p0_streaming(
    root: str | Path,
    output_dir: str | Path,
    workers: int = 12,
    month: str | None = None,
    csv_mode: str = "none",
    skip_existing: bool = False,
) -> dict:
    if csv_mode not in {"none", "sharded", "single"}:
        raise ValueError("csv_mode must be none, sharded, or single")
    started = time.time()
    files = list(Path(root).rglob("p0_node_features.parquet"))
    files += list(Path(root).rglob("p0_edge_spillover_features.parquet"))
    if month:
        files = [path for path in files if f"date={month}" in str(path)]
    files.sort(key=lambda path: path.stat().st_size, reverse=True)
    signature = _input_signature(files)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if skip_existing and p0_eval_complete(output_dir, signature):
        return {
            "stage": "p0_alpha_eval",
            "status": "skipped",
            "month": month,
            "evaluation_contract_version": EVAL_CONTRACT_VERSION,
            "elapsed_sec": 0.0,
        }
    if not files:
        shutil.rmtree(output_dir / "p0_alpha_metrics.parquet", ignore_errors=True)
        shutil.rmtree(output_dir / ".p0_metric_state", ignore_errors=True)
        shutil.rmtree(output_dir / "p0_alpha_metrics_csv", ignore_errors=True)
        for name in ("p0_alpha_metrics.csv", "p0_alpha_summary.csv", "p0_alpha_summary_state.parquet"):
            (output_dir / name).unlink(missing_ok=True)
        metadata = {
            "stage": "p0_alpha_eval",
            "status": "empty",
            "input_files": 0,
            "output_rows": 0,
            "month": month,
            "input_signature": signature,
            "evaluation_contract_version": EVAL_CONTRACT_VERSION,
        }
        write_manifest(output_dir, metadata)
        return metadata

    metric_dir = output_dir / "p0_alpha_metrics.parquet"
    state_dir = output_dir / ".p0_metric_state"
    csv_dir = output_dir / "p0_alpha_metrics_csv"
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
    _write_frame_atomic(summary_state, output_dir / "p0_alpha_summary_state.parquet", parquet=True)
    _write_frame_atomic(summary, output_dir / "p0_alpha_summary.csv", parquet=False)

    full_csv = output_dir / "p0_alpha_metrics.csv"
    if csv_mode == "none":
        full_csv.unlink(missing_ok=True)
        shutil.rmtree(csv_dir, ignore_errors=True)
    elif csv_mode == "sharded":
        full_csv.unlink(missing_ok=True)
    else:
        csv_paths = [Path(result["csv"]) for result in results if result.get("csv") and Path(result["csv"]).exists()]
        _concatenate_csv_shards(csv_paths, full_csv)

    metadata = {
        "stage": "p0_alpha_eval",
        "status": "complete" if metric_rows else "empty",
        "evaluation_contract_version": EVAL_CONTRACT_VERSION,
        "missing_data_semantics": "exact_pairwise_complete_by_validity_mask",
        "input_files": len(files),
        "input_signature": signature,
        "metric_rows": metric_rows,
        "metric_shards": sum(1 for path in metric_dir.glob("part-*.parquet")),
        "reused_shards": sum(1 for result in results if result.get("status") == "reused"),
        "summary_rows": len(summary),
        "output_rows": metric_rows,
        "evaluation_scope": "per_decision_time_cross_section",
        "evaluation_input_mode": "resumable_parallel_partition_shards",
        "metrics_layout": "partitioned_parquet_dataset",
        "csv_mode": csv_mode,
        "month": month,
        "workers": worker_count,
        "parallel_summary_reduction": True,
        "serial_global_dataframe_concat": False,
        "elapsed_sec": round(time.time() - started, 3),
    }
    write_manifest(output_dir, metadata)
    return metadata
