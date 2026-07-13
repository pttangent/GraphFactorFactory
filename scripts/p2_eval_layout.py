#!/usr/bin/env python3
"""Compatibility cleanup for old single-file and temporary evaluation layouts."""
from __future__ import annotations

import shutil
from pathlib import Path


def prepare_eval_output(output_dir: str | Path, mode: str, csv_mode: str = "none") -> dict:
    if mode not in {"p0", "intraday", "daily"}:
        raise ValueError("mode must be p0, intraday, or daily")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    prefix = "p0" if mode == "p0" else mode
    metrics = root / f"{prefix}_alpha_metrics.parquet"
    removed: list[str] = []

    if metrics.exists() and not metrics.is_dir():
        metrics.unlink()
        removed.append(str(metrics))
    for temporary in (
        Path(str(metrics) + ".tmp"),
        root / f"{prefix}_alpha_summary.csv.tmp",
        root / f"{prefix}_alpha_summary_state.parquet.tmp",
    ):
        if temporary.exists():
            temporary.unlink()
            removed.append(str(temporary))

    legacy_directories = [root / ".p0_eval_work", root / ".p0_metric_shards"] if mode == "p0" else [
        root / f".{mode}_metric_shards",
        root / f".{mode}_eval_work",
    ]
    for directory in legacy_directories:
        if directory.exists():
            shutil.rmtree(directory)
            removed.append(str(directory))

    if csv_mode == "none":
        legacy_csv = root / f"{prefix}_alpha_metrics.csv"
        if legacy_csv.exists():
            legacy_csv.unlink()
            removed.append(str(legacy_csv))
    return {"output_dir": str(root), "mode": mode, "removed_legacy_artifacts": removed}
