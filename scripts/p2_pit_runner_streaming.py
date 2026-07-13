#!/usr/bin/env python3
"""Streaming partition runner for PIT relation features."""
from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from p2_pit_core import (
    Part,
    is_complete,
    iter_time_groups,
    read_partition,
    stream_frames,
    write_manifest,
    write_parquet_atomic,
)
from p2_pit_features import build_daily_feature_frame, build_intraday_feature_frame


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
    if skip_existing and is_complete(output_dir / "manifest.json") and output_path.exists():
        return {"stage": f"{mode}_relation_features", "status": "skipped", "date": part.date, "layer_id": part.layer_id, "scale": part.scale}

    output_rows = write_batches = 0
    input_mode = "full_session_required"
    if mode == "intraday":
        output_rows, write_batches = stream_frames(
            output_path,
            _checked_intraday_frames(part.base, underreaction_past_horizon, max_row_groups),
        )
        input_mode = "snapshot_time_stream"
    else:
        source = read_partition(part.base, None, max_row_groups)
        if not source.empty:
            if temporal_root is None:
                raise ValueError("daily mode requires --p1-root with temporal_theme_edges.parquet")
            temporal_path = Path(temporal_root) / f"date={part.date}" / f"layer_id={part.layer_id}" / f"scale={part.scale}" / "temporal_theme_edges.parquet"
            if not temporal_path.exists():
                raise FileNotFoundError(f"missing daily temporal identity file: {temporal_path}")
            temporal = read_partition(temporal_path, None, max_row_groups)
            if "level" in temporal:
                temporal = temporal[temporal["level"].astype(str).isin(source["level"].astype(str).unique())]
            output = build_daily_feature_frame(source, underreaction_past_horizon, late_minutes, temporal)
            if not output.empty:
                if "pit_audit_pass" not in output or not bool(output["pit_audit_pass"].fillna(False).all()):
                    failed = int((~output.get("pit_audit_pass", pd.Series(False, index=output.index)).fillna(False)).sum())
                    raise AssertionError(f"{failed} daily feature rows failed PIT audit")
                write_parquet_atomic(output, output_path)
                output_rows = len(output)
                write_batches = 1

    metadata = {
        "stage": f"{mode}_relation_features",
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
        "temporal_identity": "not_applicable" if mode == "intraday" else "p1_temporal_episode",
        "input_mode": input_mode,
        "output": str(output_path),
        "elapsed_sec": round(time.time() - started, 3),
    }
    write_manifest(output_dir, metadata)
    return metadata
