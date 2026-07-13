#!/usr/bin/env python3
"""Six-month local/NAS runner for the PIT-safe P0/P2 pipeline.

Copy, label injection, strict P0 edge sharding, compute, evaluation and cleanup
are serialized. Monthly evaluation is performed exactly once by the scheduler.
"""
from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

from daily_label_integration import inject_daily_labels
from p2_alpha_pit_features import DEFAULT_HORIZONS
from p2_parallel_runtime import run_process_tree

NAS_P0_ROOT = Path(r"P:\US-Stock\GFF_Full_Workspace\graph_store_6m\canonical")
NAS_P1_ROOT = Path(r"P:\US-Stock\GFF_Full_Workspace\p1_b50_b35_sharded")
LOCAL_WORKSPACE = Path(r"D:\GFF_Streaming_Workspace")
LOCAL_P0 = LOCAL_WORKSPACE / "p0"
LOCAL_P0_SHARDS = LOCAL_WORKSPACE / "p0_alpha_shards"
LOCAL_P1 = LOCAL_WORKSPACE / "p1"
LOCAL_P2_OUT = LOCAL_WORKSPACE / "p2_out"
NAS_P2_OUT = Path(r"P:\US-Stock\GFF_Full_Workspace\p2_alpha_full_run")
MAPPING_PATH = Path(r"D:\DEV\US-Stock\GraphFactorFactory\artifacts\global_symbol_mapping.parquet")
DAILY_LABELS_PATH = Path(r"D:\DEV\US-Stock\RAW_DATA\1d\daily_labels_2026.parquet")
MONTHS = ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06"]
CORES = 24
RAM_GB = 128
RESERVE_RAM_GB = 24
LABEL_WORKERS = 8


def run(command: list[str]) -> None:
    environment = os.environ.copy()
    environment.setdefault("GFF_MAX_TASKS_PER_CHILD", "1")
    return_code = run_process_tree(command, env=environment)
    if return_code != 0:
        raise RuntimeError(f"command failed with code {return_code}: {' '.join(command)}")


def robocopy_dir(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = ["robocopy", str(source), str(destination), "/E", "/MT:16", "/R:3", "/W:1", "/NFL", "/NDL", "/NJH", "/NJS", "/A-:R"]
    return_code = run_process_tree(command, env=os.environ.copy())
    if return_code >= 8:
        raise RuntimeError(f"robocopy failed for {source} with code {return_code}")


def prefetch_month(month: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] prefetching {month}", flush=True)
    LOCAL_P0.mkdir(parents=True, exist_ok=True)
    LOCAL_P1.mkdir(parents=True, exist_ok=True)
    for source in NAS_P0_ROOT.glob(f"date={month}-*"):
        robocopy_dir(source, LOCAL_P0 / source.name)
    for source in NAS_P1_ROOT.glob(f"date={month}-*"):
        robocopy_dir(source, LOCAL_P1 / source.name)


def inject_month_labels(month: str) -> None:
    if not MAPPING_PATH.exists() or not DAILY_LABELS_PATH.exists():
        raise FileNotFoundError("stable mapping or PIT-safe daily labels missing")
    print(f"[{time.strftime('%H:%M:%S')}] injecting daily labels for {month}", flush=True)
    report = inject_daily_labels(LOCAL_P0, DAILY_LABELS_PATH, MAPPING_PATH, month, workers=LABEL_WORKERS)
    if report["updated_files"] == 0:
        raise RuntimeError(f"no daily labels injected for {month}")
    print(f"[{time.strftime('%H:%M:%S')}] injected next-open labels: {report}", flush=True)


def build_p0_alpha_shards(dates: list[str]) -> None:
    """Create real date/layer/scale partitions before P0 alpha extraction."""
    shutil.rmtree(LOCAL_P0_SHARDS, ignore_errors=True)
    LOCAL_P0_SHARDS.mkdir(parents=True, exist_ok=True)
    print(f"[{time.strftime('%H:%M:%S')}] building strict P0 alpha shards", flush=True)
    run([
        sys.executable,
        "scripts/shard_p0_edges_by_layer_scale.py",
        "--p0-root", str(LOCAL_P0),
        "--out-root", str(LOCAL_P0_SHARDS),
        "--dates", ",".join(dates),
        "--batch-size", "250000",
    ])


def run_p2_month(month: str) -> None:
    dates = sorted(path.name.split("=", 1)[1] for path in LOCAL_P0.glob(f"date={month}-*"))
    if not dates:
        raise RuntimeError(f"no local P0 dates found for {month}")
    inject_month_labels(month)
    build_p0_alpha_shards(dates)
    run([
        sys.executable,
        "scripts/run_p2_24core_scheduler.py",
        "--p0-root", str(LOCAL_P0_SHARDS),
        "--labels-root", str(LOCAL_P0),
        "--p1-root", str(LOCAL_P1),
        "--p2-root", str(LOCAL_P2_OUT),
        "--dates", ",".join(dates),
        "--layers", "3,6,8,9,11",
        "--scales", "15m,30m",
        "--horizons", ",".join(DEFAULT_HORIZONS),
        "--profile", "balanced",
        "--cores", str(CORES),
        "--ram-gb", str(RAM_GB),
        "--reserve-ram-gb", str(RESERVE_RAM_GB),
        "--target-cpu", "1.0",
        "--inner-workers", "0",
        "--skip-existing",
    ])


def cleanup_month(month: str) -> None:
    for directory in LOCAL_P0.glob(f"date={month}-*"):
        shutil.rmtree(directory, ignore_errors=True)
    for directory in LOCAL_P1.glob(f"date={month}-*"):
        shutil.rmtree(directory, ignore_errors=True)
    shutil.rmtree(LOCAL_P0_SHARDS, ignore_errors=True)


def run_global_evaluation() -> None:
    eval_workers = 12
    run([sys.executable, "scripts/p2_alpha_daily_features.py", "evaluate-intraday", "--features-root", str(LOCAL_P2_OUT / "intraday_relation_features"), "--out-dir", str(LOCAL_P2_OUT / "intraday_relation_eval/global"), "--workers", str(eval_workers)])
    run([sys.executable, "scripts/p2_alpha_daily_features.py", "evaluate-daily", "--features-root", str(LOCAL_P2_OUT / "daily_relation_features"), "--out-dir", str(LOCAL_P2_OUT / "daily_relation_eval/global"), "--workers", str(eval_workers)])


def main() -> None:
    for month in MONTHS:
        prefetch_month(month)
        run_p2_month(month)
        cleanup_month(month)
    run_global_evaluation()


if __name__ == "__main__":
    main()
