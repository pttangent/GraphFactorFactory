#!/usr/bin/env python3
"""Runtime-safe partition runner for PIT relation features."""
from __future__ import annotations

import time
from pathlib import Path

from p2_pit_core import Part, is_complete, read_partition, write_manifest, write_parquet_atomic
from p2_pit_features import build_daily_feature_frame, build_intraday_feature_frame


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
    if skip_existing and is_complete(output_dir / "manifest.json"):
        return {"stage": f"{mode}_relation_features", "status": "skipped", "date": part.date, "layer_id": part.layer_id, "scale": part.scale}

    source = read_partition(part.base, None, max_row_groups)
    output_rows = 0
    if not source.empty:
        if mode == "intraday":
            output = build_intraday_feature_frame(source, underreaction_past_horizon)
        else:
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
            if "pit_audit_pass" not in output:
                raise AssertionError(f"{mode} output omitted pit_audit_pass")
            if not bool(output["pit_audit_pass"].all()):
                failed = int((~output["pit_audit_pass"]).sum())
                raise AssertionError(f"{failed} {mode} feature rows failed PIT audit")
            write_parquet_atomic(output, output_path)
            output_rows = len(output)

    metadata = {
        "stage": f"{mode}_relation_features",
        "status": "complete" if output_rows else "empty",
        "date": part.date,
        "layer_id": part.layer_id,
        "scale": part.scale,
        "output_rows": output_rows,
        "mode": mode,
        "late_minutes": late_minutes if mode == "daily" else None,
        "underreaction_past_horizon": underreaction_past_horizon,
        "normalization_scope": "snapshot_cross_section" if mode == "intraday" else "eod_cross_section",
        "temporal_identity": "not_applicable" if mode == "intraday" else "p1_temporal_episode",
        "output": str(output_path),
        "elapsed_sec": round(time.time() - started, 3),
    }
    write_manifest(output_dir, metadata)
    return metadata
