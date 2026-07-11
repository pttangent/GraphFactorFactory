#!/usr/bin/env python3
"""Partition-safe P1 runner for B50/B35 theme forest builds.

Rule: one child process = one date/layer_id/scale partition.

The previous runner launched one child per date. Each child then let
build_b50_b35_theme_forest.py rglob/concat every parquet under the date,
group all snapshots, and keep all output rows in memory until final write.
With many workers this multiplies date-sized peak RSS and can OOM.

This runner discovers date=YYYY-MM-DD/layer_id=X/scale=Ym/edges.parquet
partitions and runs the builder once per partition. It also streams child logs
to files instead of capture_output=True, so parent memory stays small.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

LOG = logging.getLogger("run_p1_parallel")

THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "ARROW_NUM_THREADS",
    "POLARS_MAX_THREADS",
)

@dataclass(frozen=True)
class Task:
    input_path: Path
    out_dir: Path
    date: str
    layer_id: str | None = None
    scale: str | None = None

    @property
    def label(self) -> str:
        bits = [self.date]
        if self.layer_id is not None:
            bits.append(f"layer_id={self.layer_id}")
        if self.scale is not None:
            bits.append(f"scale={self.scale}")
        return "/".join(bits)

def csvset(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {x.strip() for x in value.split(",") if x.strip()}

def part_value(path: Path, key: str) -> str | None:
    prefix = key + "="
    for part in path.parts:
        if part.startswith(prefix):
            return part.split("=", 1)[1]
    return None

def complete(out_dir: Path) -> bool:
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if int(manifest.get("groups", 0) or 0) <= 0:
        return False
    required = [
        out_dir / "theme_memberships.parquet",
        out_dir / "theme_relation_edges.parquet",
        out_dir / "p1_b50_b35_summary.parquet",
    ]
    return all(p.exists() and p.stat().st_size > 0 for p in required)

def discover_date_dirs(p0_root: Path) -> list[Path]:
    if p0_root.is_dir() and p0_root.name.startswith("date="):
        return [p0_root]
    return sorted(d for d in p0_root.iterdir() if d.is_dir() and d.name.startswith("date="))

def discover_partition_tasks(
    p0_root: Path,
    out_root: Path,
    dates: set[str] | None = None,
    layers: set[str] | None = None,
    scales: set[str] | None = None,
) -> list[Task]:
    tasks: list[Task] = []
    for date_dir in discover_date_dirs(p0_root):
        date = part_value(date_dir, "date")
        if not date or (dates and date not in dates):
            continue
        for edge_file in sorted(date_dir.rglob("edges.parquet")):
            layer_id = part_value(edge_file, "layer_id")
            scale = part_value(edge_file, "scale")
            if not layer_id or not scale:
                continue
            if layers and layer_id not in layers:
                continue
            if scales and scale not in scales:
                continue
            out_dir = out_root / f"date={date}" / f"layer_id={layer_id}" / f"scale={scale}"
            tasks.append(Task(edge_file, out_dir, date, layer_id, scale))
    return tasks

def discover_date_tasks(p0_root: Path, out_root: Path, dates: set[str] | None = None) -> list[Task]:
    tasks: list[Task] = []
    for date_dir in discover_date_dirs(p0_root):
        date = part_value(date_dir, "date")
        if not date or (dates and date not in dates):
            continue
        tasks.append(Task(date_dir, out_root / f"date={date}", date))
    return tasks

def build_env(child_threads: int) -> dict[str, str]:
    env = os.environ.copy()
    for key in THREAD_ENV_KEYS:
        env[key] = str(child_threads)
    env["PYTHONUNBUFFERED"] = "1"
    return env

def run_task(
    task: Task,
    script_path: Path,
    output_format: str,
    child_threads: int,
    extra_args: Iterable[str],
) -> dict[str, object]:
    t0 = time.time()
    task.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = task.out_dir / "run_p1_parallel.log"
    cmd = [
        sys.executable,
        str(script_path),
        "--p0-edges",
        str(task.input_path),
        "--out-dir",
        str(task.out_dir),
        "--output-format",
        output_format,
    ]
    if task.layer_id is not None:
        cmd += ["--layer-id", str(task.layer_id)]
    if task.scale is not None:
        cmd += ["--scale", str(task.scale)]
    cmd += list(extra_args)

    with log_path.open("w", encoding="utf-8") as log:
        log.write(json.dumps({"task": task.label, "cmd": cmd, "input": str(task.input_path)}, ensure_ascii=False) + "\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=build_env(child_threads),
            check=False,
        )
    elapsed = time.time() - t0
    if proc.returncode != 0:
        return {
            "status": "failed",
            "task": task.label,
            "returncode": proc.returncode,
            "elapsed_sec": round(elapsed, 3),
            "log": str(log_path),
        }
    return {
        "status": "complete",
        "task": task.label,
        "elapsed_sec": round(elapsed, 3),
        "out_dir": str(task.out_dir),
        "log": str(log_path),
    }

def main() -> None:
    parser = argparse.ArgumentParser(description="Partition-safe parallel P1 B50/B35 runner.")
    parser.add_argument("--p0-root", required=True, help="Root containing date=YYYY-MM-DD P0 edge partitions.")
    parser.add_argument("--out-root", required=True, help="Output root for P1 artifacts.")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent child processes. Start 16-20 on 24-core/128GB before trying 24.")
    parser.add_argument("--child-threads", type=int, default=1, help="OMP/MKL/OpenBLAS/Arrow threads per child process.")
    parser.add_argument("--task-granularity", choices=["auto", "partition", "date"], default="auto")
    parser.add_argument("--dates", default=None, help="Comma-separated dates, e.g. 2026-01-07,2026-01-08")
    parser.add_argument("--layers", default=None, help="Comma-separated layer ids for partition mode, e.g. 3,8,9")
    parser.add_argument("--scales", default=None, help="Comma-separated scales for partition mode, e.g. 15m,30m")
    parser.add_argument("--output-format", choices=["parquet", "csv"], default="parquet")
    parser.add_argument("--script", default="scripts/build_b50_b35_theme_forest.py", help="P1 builder script path.")
    parser.add_argument("--no-skip", action="store_true", help="Rebuild partitions even when manifest and outputs exist.")
    parser.add_argument("--dry-run", action="store_true", help="Only print discovered tasks.")
    parser.add_argument("--builder-extra-args", default="", help="Extra args passed to build_b50_b35_theme_forest.py.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p0_root = Path(args.p0_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    dates = csvset(args.dates)
    layers = csvset(args.layers)
    scales = csvset(args.scales)
    script_path = Path(args.script)

    tasks: list[Task] = []
    if args.task_granularity in ("auto", "partition"):
        tasks = discover_partition_tasks(p0_root, out_root, dates=dates, layers=layers, scales=scales)
        if not tasks and args.task_granularity == "partition":
            raise SystemExit("No date/layer_id/scale edges.parquet partitions found.")
    if not tasks and args.task_granularity in ("auto", "date"):
        LOG.warning("Falling back to date-level tasks. This is more memory intensive.")
        tasks = discover_date_tasks(p0_root, out_root, dates=dates)

    skipped = 0
    if not args.no_skip:
        before = len(tasks)
        tasks = [t for t in tasks if not complete(t.out_dir)]
        skipped = before - len(tasks)

    tasks.sort(key=lambda t: (t.date, t.layer_id or "", t.scale or ""))
    LOG.info("Discovered %d runnable tasks (%d skipped).", len(tasks), skipped)
    if args.dry_run:
        for task in tasks:
            print(json.dumps({"task": task.label, "input": str(task.input_path), "out_dir": str(task.out_dir)}, ensure_ascii=False))
        return

    extra_args = args.builder_extra_args.split() if args.builder_extra_args else []
    start_t = time.time()
    complete_count = 0
    failed: list[dict[str, object]] = []

    # ThreadPool is intentional: each task runs an external Python process.
    # ProcessPool would add another parent-side process per task and double process count.
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_to_task = {
            pool.submit(run_task, task, script_path, args.output_format, args.child_threads, extra_args): task
            for task in tasks
        }
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"status": "failed", "task": task.label, "error": repr(exc)}
            if result.get("status") == "complete":
                complete_count += 1
                LOG.info("[%d/%d] Completed %s", complete_count, len(tasks), task.label)
            else:
                failed.append(result)
                LOG.error("Failed %s: %s", task.label, result)

    run_manifest = {
        "status": "complete" if not failed else "failed",
        "tasks_total": len(tasks),
        "tasks_complete": complete_count,
        "tasks_failed": len(failed),
        "tasks_skipped": skipped,
        "workers": args.workers,
        "child_threads": args.child_threads,
        "task_granularity": args.task_granularity,
        "elapsed_sec": round(time.time() - start_t, 3),
        "failed": failed[:50],
    }
    (out_root / "run_p1_parallel_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    LOG.info("Done: %s", json.dumps(run_manifest, ensure_ascii=False))
    if failed:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
