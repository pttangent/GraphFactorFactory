#!/usr/bin/env python3
"""Streaming partition runner for PIT relation features."""
from __future__ import annotations

import json
import time
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from p2_checkpoint import file_fingerprint, stage_checkpoint_valid
from p2_daily_streaming import build_daily_feature_frame_streaming
from p2_pit_core import Part, iter_time_groups, read_partition, write_manifest, write_parquet_atomic
from p2_pit_features import build_intraday_feature_frame
from p2_streaming_io import stream_frames

INTRADAY_FEATURE_CONTRACT = "intraday-relation-features-v3"
DAILY_FEATURE_CONTRACT = "daily-relation-features-stream-v4"


def _checked_intraday_frames(
    source_path: Path,
    underreaction_past_horizon: str,
    max_row_groups: int | None,
) -> Iterable[pd.DataFrame | None]:
    for _, snapshot in iter_time_groups(source_path, None, max_row_groups):
        output = build_intraday_feature_frame(snapshot, underreaction_past_horizon)
        if output.empty:
            continue
        if "pit_audit_pass" not in output or not bool(output["pit_audit_pass"].fillna(False).all()):
            failed = int((~output.get("pit_audit_pass", pd.Series(False, index=output.index)).fillna(False)).sum())
            raise AssertionError(f"{failed} intraday feature rows failed PIT audit")
        yield output


def _temporal_paths(temporal_root: str | Path, part: Part) -> tuple[Path, Path]:
    partition_dir = Path(temporal_root) / f"date={part.date}" / f"layer_id={part.layer_id}" / f"scale={part.scale}"
    return partition_dir / "temporal_theme_edges.parquet", partition_dir / "manifest.json"


def _temporal_fingerprint(temporal_root: str | Path, part: Part) -> dict:
    temporal_path, manifest_path = _temporal_paths(temporal_root, part)
    if temporal_path.exists():
        return {"mode": "file", "fingerprint": file_fingerprint(temporal_path)}
    if manifest_path.exists():
        return {"mode": "zero-edge-manifest", "fingerprint": file_fingerprint(manifest_path)}
    return {"mode": "missing", "fingerprint": None}


def _load_temporal_edges(
    temporal_root: str | Path,
    part: Part,
    max_row_groups: int | None,
) -> tuple[pd.DataFrame, str]:
    temporal_path, manifest_path = _temporal_paths(temporal_root, part)
    if temporal_path.exists():
        return read_partition(temporal_path, None, max_row_groups), "p1_temporal_edges"

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as error:
        raise FileNotFoundError(f"missing daily temporal identity file and readable manifest: {temporal_path}") from error
    temporal_rows = int(manifest.get("rows", {}).get("temporal_theme_edges", -1))
    if manifest.get("status") in {"complete", "empty"} and temporal_rows == 0:
        empty = pd.DataFrame(columns=["level", "src_theme_id", "dst_theme_id", "src_time", "dst_time"])
        return empty, "p1_complete_zero_temporal_edges_singleton_episodes"
    raise FileNotFoundError(f"missing daily temporal identity file: {temporal_path}")


def build_feature_one(
    part: Part,
    output_root: str | Path,
    mode: str,
    underreaction_past_horizon: str,
    late_minutes: int,
    skip_existing: bool,
    max_row_groups: int | None,
    temporal_root: str | Path | None = None,
) -> dict:
    started = time.time()
    output_dir = Path(output_root) / f"date={part.date}" / f"layer_id={part.layer_id}" / f"scale={part.scale}"
    filename = "intraday_relation_features.parquet" if mode == "intraday" else "daily_relation_features.parquet"
    output_path = output_dir / filename
    contract = INTRADAY_FEATURE_CONTRACT if mode == "intraday" else DAILY_FEATURE_CONTRACT

    inputs = {"signals": file_fingerprint(part.base)}
    if mode == "daily":
        if temporal_root is None:
            raise ValueError("daily mode requires --p1-root with temporal_theme_edges.parquet")
        inputs["temporal_identity"] = _temporal_fingerprint(temporal_root, part)
    config = {
        "mode": mode,
        "underreaction_past_horizon": underreaction_past_horizon,
        "late_minutes": late_minutes if mode == "daily" else None,
        "max_row_groups": max_row_groups,
    }
    if skip_existing and stage_checkpoint_valid(
        output_dir / "manifest.json",
        stage=f"{mode}_relation_features",
        contract_version=contract,
        inputs=inputs,
        config=config,
        output_path=output_path,
    ):
        return {"stage": f"{mode}_relation_features", "status": "skipped", "date": part.date, "layer_id": part.layer_id, "scale": part.scale}

    output_rows = write_batches = 0
    input_mode = "snapshot_time_stream"
    temporal_identity = "not_applicable"
    if mode == "intraday":
        output_rows, write_batches = stream_frames(
            output_path,
            _checked_intraday_frames(part.base, underreaction_past_horizon, max_row_groups),
        )
    else:
        temporal, temporal_identity = _load_temporal_edges(temporal_root, part, max_row_groups)
        output = build_daily_feature_frame_streaming(
            part.base,
            underreaction_past_horizon,
            late_minutes,
            temporal,
            max_row_groups,
        )
        if not output.empty:
            if "pit_audit_pass" not in output or not bool(output["pit_audit_pass"].fillna(False).all()):
                failed = int((~output.get("pit_audit_pass", pd.Series(False, index=output.index)).fillna(False)).sum())
                raise AssertionError(f"{failed} daily feature rows failed PIT audit")
            write_parquet_atomic(output, output_path)
            output_rows = len(output)
            write_batches = 1
        else:
            output_path.unlink(missing_ok=True)
        input_mode = "single_pass_episode_state_plus_rolling_late_window"

    metadata = {
        "stage": f"{mode}_relation_features",
        "stage_contract_version": contract,
        "status": "complete" if output_rows else "empty",
        "date": part.date,
        "layer_id": part.layer_id,
        "scale": part.scale,
        "output_rows": output_rows,
        "write_batches": write_batches,
        "mode": mode,
        "late_minutes": late_minutes if mode == "daily" else None,
        "underreaction_past_horizon": underreaction_past_horizon,
        "normalization_scope": "snapshot_cross_section" if mode == "intraday" else "eod_cross_section",
        "temporal_identity": temporal_identity,
        "input_mode": input_mode,
        "maximum_full_signal_partitions_in_memory": 0 if mode == "daily" else None,
        "inputs": inputs,
        "config": config,
        "output": str(output_path),
        "elapsed_sec": round(time.time() - started, 3),
    }
    write_manifest(output_dir, metadata)
    return metadata
