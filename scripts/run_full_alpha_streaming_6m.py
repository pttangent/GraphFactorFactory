#!/usr/bin/env python3
"""Six-month local/NAS runner for the PIT-safe P0/P2 pipeline.

The production path scans each canonical trade date once and writes only final
P0 factors. Completed months are
archived to NAS, checkpointed, and removed from the local NVMe.
"""
from __future__ import annotations

import json
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
P0_DATE_WORKERS = 8
P0_BATCH_SIZE = 500_000
P0_MIN_FREE_GB = 50.0
MIN_FREE_GB_BEFORE_MONTH = 100.0

DATE_PARTITION_STAGES = [
    "p0_node_features",
    "p0_edge_spillover",
    "p0_graph_state",
    "p0_direct_status",
    "theme_returns",
    "relation_spillover",
    "intraday_relation_features",
    "daily_relation_features",
]
EVAL_STAGES = ["p0_alpha", "intraday_relation_eval", "daily_relation_eval"]


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
    command = [
        "robocopy",
        str(source),
        str(destination),
        "/E",
        "/MT:16",
        "/R:3",
        "/W:1",
        "/NFL",
        "/NDL",
        "/NJH",
        "/NJS",
        "/A-:R",
    ]
    return_code = run_process_tree(command, env=os.environ.copy())
    if return_code >= 8:
        raise RuntimeError(f"robocopy failed for {source} with code {return_code}")


def ensure_free_space(path: Path, minimum_free_gb: float) -> float:
    path.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(path).free / (1024**3)
    if free_gb < minimum_free_gb:
        raise RuntimeError(
            f"disk fuse: {path} has {free_gb:.2f}GB free; "
            f"requires at least {minimum_free_gb:.2f}GB"
        )
    return free_gb


def directory_stats(path: Path) -> tuple[int, int]:
    files = 0
    size = 0
    if not path.exists():
        return files, size
    for item in path.rglob("*"):
        if item.is_file():
            files += 1
            size += item.stat().st_size
    return files, size


def source_month_size_bytes(month: str) -> int:
    total = 0
    for root in (NAS_P0_ROOT, NAS_P1_ROOT):
        for directory in root.glob(f"date={month}-*"):
            total += directory_stats(directory)[1]
    return total


def month_status_path(month: str) -> Path:
    return NAS_P2_OUT / "_month_status" / f"month={month}" / "_SUCCESS.json"


def month_is_complete(month: str) -> bool:
    try:
        payload = json.loads(month_status_path(month).read_text(encoding="utf-8"))
        return payload.get("status") == "complete" and payload.get("month") == month
    except Exception:
        return False


def prefetch_month(month: str) -> None:
    source_gb = source_month_size_bytes(month) / (1024**3)
    required_gb = MIN_FREE_GB_BEFORE_MONTH + source_gb
    free_gb = ensure_free_space(LOCAL_WORKSPACE, required_gb)
    print(
        f"[{time.strftime('%H:%M:%S')}] prefetching {month}; "
        f"source={source_gb:.1f}GB local_free={free_gb:.1f}GB required={required_gb:.1f}GB",
        flush=True,
    )
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
    report = inject_daily_labels(
        LOCAL_P0,
        DAILY_LABELS_PATH,
        MAPPING_PATH,
        month,
        workers=LABEL_WORKERS,
    )
    if report["updated_files"] == 0:
        raise RuntimeError(f"no daily labels injected for {month}")
    print(f"[{time.strftime('%H:%M:%S')}] injected next-open labels: {report}", flush=True)


def run_p2_month(month: str) -> None:
    dates = sorted(path.name.split("=", 1)[1] for path in LOCAL_P0.glob(f"date={month}-*"))
    if not dates:
        raise RuntimeError(f"no local P0 dates found for {month}")
    inject_month_labels(month)
    run(
        [
            sys.executable,
            "scripts/run_p2_24core_scheduler.py",
            "--p0-root",
            str(LOCAL_P0),
            "--labels-root",
            str(LOCAL_P0),
            "--p1-root",
            str(LOCAL_P1),
            "--p2-root",
            str(LOCAL_P2_OUT),
            "--dates",
            ",".join(dates),
            "--horizons",
            ",".join(DEFAULT_HORIZONS),
            "--profile",
            "balanced",
            "--cores",
            str(CORES),
            "--ram-gb",
            str(RAM_GB),
            "--reserve-ram-gb",
            str(RESERVE_RAM_GB),
            "--target-cpu",
            "1.0",
            "--inner-workers",
            "0",
            "--p0-date-workers",
            str(P0_DATE_WORKERS),
            "--p0-batch-size",
            str(P0_BATCH_SIZE),
            "--p0-min-free-gb",
            str(P0_MIN_FREE_GB),
            "--skip-existing",
        ]
    )


def _month_sources(month: str) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for stage in DATE_PARTITION_STAGES:
        local_stage = LOCAL_P2_OUT / stage
        nas_stage = NAS_P2_OUT / stage
        for source in sorted(local_stage.glob(f"date={month}-*")):
            pairs.append((source, nas_stage / source.name))
    scope = month.replace("-", "")
    for stage in EVAL_STAGES:
        source = LOCAL_P2_OUT / stage / scope
        if source.exists():
            pairs.append((source, NAS_P2_OUT / stage / scope))
    return pairs


def archive_month_outputs(month: str) -> None:
    NAS_P2_OUT.mkdir(parents=True, exist_ok=True)
    pairs = _month_sources(month)
    if not pairs:
        raise RuntimeError(f"no P0/P2 outputs found to archive for {month}")

    copied: list[dict] = []
    for source, destination in pairs:
        robocopy_dir(source, destination)
        source_stats = directory_stats(source)
        destination_stats = directory_stats(destination)
        if source_stats != destination_stats:
            raise RuntimeError(
                f"archive verification failed for {source}: "
                f"source={source_stats}, destination={destination_stats}"
            )
        copied.append(
            {
                "source": str(source),
                "destination": str(destination),
                "files": source_stats[0],
                "bytes": source_stats[1],
            }
        )

    metadata_dir = NAS_P2_OUT / "_run_metadata" / f"month={month}"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    for name in ("p2_24core_schedule_plan.json", "p0_direct_run_summary.json"):
        source = LOCAL_P2_OUT / name
        if source.exists():
            shutil.copy2(source, metadata_dir / name)

    status_path = month_status_path(month)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = status_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {
                "status": "complete",
                "month": month,
                "archived_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "items": copied,
                "total_files": sum(item["files"] for item in copied),
                "total_bytes": sum(item["bytes"] for item in copied),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    os.replace(temporary, status_path)

    for source, _ in pairs:
        shutil.rmtree(source, ignore_errors=True)
    print(f"[{time.strftime('%H:%M:%S')}] archived and checkpointed {month}", flush=True)


def cleanup_month_inputs(month: str) -> None:
    for directory in LOCAL_P0.glob(f"date={month}-*"):
        shutil.rmtree(directory, ignore_errors=True)
    for directory in LOCAL_P1.glob(f"date={month}-*"):
        shutil.rmtree(directory, ignore_errors=True)


def purge_local_month_outputs(month: str) -> None:
    for source, _ in _month_sources(month):
        shutil.rmtree(source, ignore_errors=True)


def run_global_evaluation() -> None:
    eval_workers = 12
    run(
        [
            sys.executable,
            "scripts/p2_alpha_daily_features.py",
            "evaluate-intraday",
            "--features-root",
            str(NAS_P2_OUT / "intraday_relation_features"),
            "--out-dir",
            str(NAS_P2_OUT / "intraday_relation_eval" / "global"),
            "--workers",
            str(eval_workers),
        ]
    )
    run(
        [
            sys.executable,
            "scripts/p2_alpha_daily_features.py",
            "evaluate-daily",
            "--features-root",
            str(NAS_P2_OUT / "daily_relation_features"),
            "--out-dir",
            str(NAS_P2_OUT / "daily_relation_eval" / "global"),
            "--workers",
            str(eval_workers),
        ]
    )


def main() -> None:
    for month in MONTHS:
        if month_is_complete(month):
            print(f"[{time.strftime('%H:%M:%S')}] {month} already archived; skipping", flush=True)
            purge_local_month_outputs(month)
            cleanup_month_inputs(month)
            continue
        prefetch_month(month)
        run_p2_month(month)
        archive_month_outputs(month)
        cleanup_month_inputs(month)
    run_global_evaluation()


if __name__ == "__main__":
    main()
