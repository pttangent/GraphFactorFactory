from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class BoundaryTask:
    task_id: str
    date_from: str
    date_to: str
    seed: int


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def discover_trade_dates(phase1_root: Path, start: str, end: str) -> list[str]:
    dates = sorted(
        path.name.split("=", 1)[1]
        for path in phase1_root.glob("date=*")
        if path.is_dir() and start <= path.name.split("=", 1)[1] <= end
    )
    if len(dates) < 2:
        raise RuntimeError(f"Need at least two Phase 1 dates in {phase1_root}; found {dates}")
    return dates


def read_trade_dates(path: Path, start: str, end: str) -> list[str]:
    dates = sorted({line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and start <= line.strip() <= end})
    if len(dates) < 2:
        raise RuntimeError("Trade-date file must contain at least two dates in range")
    return dates


def build_tasks(config: dict[str, Any], dates: list[str]) -> list[BoundaryTask]:
    seed0 = int(config.get("random_seed", 0))
    return [
        BoundaryTask(f"{left}__{right}", left, right, seed0 + index * 10_000)
        for index, (left, right) in enumerate(zip(dates[:-1], dates[1:]))
    ]


def generate_null_mapping(config: dict[str, Any], dates: list[str], output_root: Path) -> pd.DataFrame:
    count = int(config.get("controls", {}).get("day_order_nulls_per_boundary", 3))
    seed0 = int(config.get("random_seed", 0))
    rows: list[dict[str, Any]] = []
    for boundary_index, (date_from, date_to) in enumerate(zip(dates[:-1], dates[1:])):
        candidates = [date for date in dates if date not in {date_from, date_to}]
        rng = random.Random(seed0 + boundary_index * 10_000 + 991)
        rng.shuffle(candidates)
        if len(candidates) < count:
            raise RuntimeError(f"Not enough null dates for {date_from}->{date_to}")
        for replicate, null_date_to in enumerate(candidates[:count]):
            rows.append(
                {
                    "actual_date_from": date_from,
                    "actual_date_to": date_to,
                    "control_type": "day_order",
                    "replicate": replicate,
                    "null_date_from": date_from,
                    "null_date_to": null_date_to,
                    "seed": seed0 + boundary_index * 10_000 + replicate,
                }
            )
    frame = pd.DataFrame(rows)
    path = output_root / "null_mapping.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".parquet.tmp")
    frame.to_parquet(temp, index=False)
    os.replace(temp, path)
    return frame


def checkpoint_path(output_root: Path, task: BoundaryTask) -> Path:
    return output_root / "boundaries" / task.task_id / "boundary_checkpoint.json"


def load_checkpoint(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def expected_unit_dirs(output_root: Path, task: BoundaryTask, arms: list[str], null_count: int) -> list[Path]:
    base = output_root / "shards" / f"date_from={task.date_from}" / f"date_to={task.date_to}"
    units: list[Path] = []
    for arm in arms:
        units.append(base / f"arm={arm}" / "control=actual" / "replicate=0")
        for replicate in range(null_count):
            units.append(base / f"arm={arm}" / "control=day_order" / f"replicate={replicate}")
    return units


def boundary_complete(output_root: Path, task: BoundaryTask, arms: list[str], null_count: int) -> bool:
    return all((unit / "_SUCCESS").exists() and (unit / "task_manifest.json").exists() for unit in expected_unit_dirs(output_root, task, arms, null_count))


def run_worker(worker: Path, config_path: Path, task: BoundaryTask, output_root: Path, phase1_root: Path) -> None:
    command = [
        sys.executable,
        str(worker),
        "--config",
        str(config_path),
        "--date-from",
        task.date_from,
        "--date-to",
        task.date_to,
        "--seed",
        str(task.seed),
        "--output-dir",
        str(output_root),
        "--phase1-root",
        str(phase1_root),
    ]
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Boundary-level carry-over A/B orchestrator")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--worker", type=Path, default=Path("scripts/run_monthly_carryover_task.py"))
    parser.add_argument("--trade-dates-file", type=Path)
    parser.add_argument("--phase1-root", type=Path, default=Path("outputs/theme_discovery_phase1"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only-boundary")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--repo-commit", default=os.environ.get("GIT_COMMIT", "unknown"))
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    config_hash = json_hash(config)
    phase1_root = args.phase1_root.expanduser().resolve()
    output_root = Path(config["output_root"]) / config["run_name"] / config_hash[:12]
    output_root.mkdir(parents=True, exist_ok=True)

    dates = read_trade_dates(args.trade_dates_file, config["date_start"], config["date_end"]) if args.trade_dates_file else discover_trade_dates(phase1_root, config["date_start"], config["date_end"])
    tasks = build_tasks(config, dates)
    if args.only_boundary:
        left, right = args.only_boundary.split(":", 1)
        tasks = [task for task in tasks if task.date_from == left and task.date_to == right]

    null_mapping = generate_null_mapping(config, dates, output_root)
    arms = list(config.get("arms", {}))
    null_count = int(config.get("controls", {}).get("day_order_nulls_per_boundary", 3))
    run_manifest = {
        "run_name": config["run_name"],
        "config_hash": config_hash,
        "repo_commit": args.repo_commit,
        "phase1_root": str(phase1_root),
        "trade_dates": dates,
        "cross_month_boundaries": [task.task_id for task in tasks if task.date_from[:7] != task.date_to[:7]],
        "arms": arms,
        "null_mapping_rows": len(null_mapping),
        "task_count": len(tasks),
        "created_at": utc_now(),
    }
    atomic_json(output_root / "run_manifest.json", run_manifest)
    atomic_json(output_root / "task_index.json", {"tasks": [asdict(task) for task in tasks]})

    if args.dry_run:
        print(json.dumps(run_manifest, ensure_ascii=False, indent=2))
        return
    if not args.worker.exists():
        raise FileNotFoundError(args.worker)

    log_path = output_root / "logs" / "runner.jsonl"
    error_path = output_root / "logs" / "errors.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def process_task(task: BoundaryTask) -> None:
        checkpoint = checkpoint_path(output_root, task)
        previous = load_checkpoint(checkpoint)
        if args.resume and boundary_complete(output_root, task, arms, null_count):
            atomic_json(checkpoint, {"task_id": task.task_id, "status": "success", "validated_at": utc_now(), "config_hash": config_hash})
            print(f"SKIP {task.task_id}: all units valid")
            return
        if args.retry_failed and (not previous or previous.get("status") != "failed"):
            return
        running = {"task_id": task.task_id, "status": "running", "config_hash": config_hash, "started_at": utc_now(), "error": None}
        atomic_json(checkpoint, running)
        try:
            run_worker(args.worker, args.config, task, output_root, phase1_root)
            if not boundary_complete(output_root, task, arms, null_count):
                raise RuntimeError("worker exited but expected arm/control units are incomplete")
            finished = {**running, "status": "success", "finished_at": utc_now()}
            atomic_json(checkpoint, finished)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(finished) + "\n")
        except Exception as exc:
            failed = {**running, "status": "failed", "finished_at": utc_now(), "error": repr(exc)}
            atomic_json(checkpoint, failed)
            with error_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(failed) + "\n")
            raise

    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
        futures = [executor.submit(process_task, task) for task in tasks]
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
