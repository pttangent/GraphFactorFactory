#!/usr/bin/env python3
"""Six-month local/NAS runner for the PIT-safe P2 pipeline."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Queue

from daily_label_integration import inject_daily_labels
from p2_alpha_pit_features import DEFAULT_HORIZONS

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


def robocopy_dir(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(["robocopy", str(source), str(destination), "/E", "/MT:16", "/R:3", "/W:1", "/NFL", "/NDL", "/NJH", "/NJS", "/A-:R"], capture_output=True)
    if result.returncode >= 8:
        raise RuntimeError(f"robocopy failed for {source}: {result.stderr.decode(errors='ignore')}")


def pull_month(month: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] prefetching {month}")
    LOCAL_P0.mkdir(parents=True, exist_ok=True)
    LOCAL_P1.mkdir(parents=True, exist_ok=True)
    for source in NAS_P0_ROOT.glob(f"date={month}-*"):
        robocopy_dir(source, LOCAL_P0 / source.name)
    for source in NAS_P1_ROOT.glob(f"date={month}-*"):
        robocopy_dir(source, LOCAL_P1 / source.name)
    if not MAPPING_PATH.exists() or not DAILY_LABELS_PATH.exists():
        raise FileNotFoundError("stable mapping or PIT-safe daily labels missing")
    report = inject_daily_labels(LOCAL_P0, DAILY_LABELS_PATH, MAPPING_PATH, month)
    if report["updated_files"] == 0:
        raise RuntimeError(f"no daily labels injected for {month}")
    print(f"[{time.strftime('%H:%M:%S')}] injected next-open labels: {report}")


def run(command: list[str]) -> None:
    result = subprocess.run(command, env=os.environ.copy())
    if result.returncode != 0:
        print(f"ERROR: command failed with code {result.returncode}: {' '.join(command)}", flush=True)
        os._exit(result.returncode)


def run_p2_month(month: str) -> None:
    dates = sorted(path.name.split("=", 1)[1] for path in LOCAL_P0.glob(f"date={month}-*"))
    if not dates:
        return
    run([
        sys.executable,
        "scripts/run_p2_24core_scheduler.py",
        "--p0-root", str(LOCAL_P0),
        "--labels-root", str(LOCAL_P0),
        "--p1-root", str(LOCAL_P1),
        "--p2-root", str(LOCAL_P2_OUT),
        "--dates", ",".join(dates),
        "--layers", "3,6,8,9,11",
        "--scales", "15m,30m",
        "--horizons", ",".join(DEFAULT_HORIZONS),
        "--profile", "max",
        "--cores", "28",
        "--target-cpu", "1.0",
        "--inner-workers", "1",
        "--skip-existing",
    ])
    month_str = month.replace("-", "")
    run([sys.executable, "scripts/p2_alpha_daily_features.py", "evaluate-intraday", "--features-root", str(LOCAL_P2_OUT / "intraday_relation_features"), "--out-dir", str(LOCAL_P2_OUT / f"intraday_relation_eval/{month_str}")])
    run([sys.executable, "scripts/p2_alpha_daily_features.py", "evaluate-daily", "--features-root", str(LOCAL_P2_OUT / "daily_relation_features"), "--out-dir", str(LOCAL_P2_OUT / f"daily_relation_eval/{month_str}")])
    for directory in LOCAL_P0.glob(f"date={month}-*"):
        shutil.rmtree(directory, ignore_errors=True)
    for directory in LOCAL_P1.glob(f"date={month}-*"):
        shutil.rmtree(directory, ignore_errors=True)


def main() -> None:
    queue: Queue[str | None] = Queue(maxsize=1)

    def producer() -> None:
        for month in MONTHS:
            pull_month(month)
            queue.put(month)
        queue.put(None)

    def consumer() -> None:
        while True:
            month = queue.get()
            if month is None:
                return
            run_p2_month(month)
            queue.task_done()

    producer_thread = threading.Thread(target=producer)
    consumer_thread = threading.Thread(target=consumer)
    producer_thread.start()
    consumer_thread.start()
    producer_thread.join()
    consumer_thread.join()
    run([sys.executable, "scripts/p2_alpha_daily_features.py", "evaluate-intraday", "--features-root", str(LOCAL_P2_OUT / "intraday_relation_features"), "--out-dir", str(LOCAL_P2_OUT / "intraday_relation_eval_global")])
    run([sys.executable, "scripts/p2_alpha_daily_features.py", "evaluate-daily", "--features-root", str(LOCAL_P2_OUT / "daily_relation_features"), "--out-dir", str(LOCAL_P2_OUT / "daily_relation_eval_global")])


if __name__ == "__main__":
    main()
