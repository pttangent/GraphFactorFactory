from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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


def discover_trade_dates(input_root: Path, start: str, end: str) -> list[str]:
    candidates: set[str] = set()
    for path in input_root.rglob("date=*"):
        if path.is_dir() and path.name.startswith("date="):
            candidates.add(path.name.split("=", 1)[1])
    for path in input_root.glob("shards_????-??-??"):
        candidates.add(path.name.removeprefix("shards_"))
    dates = sorted(date for date in candidates if start <= date <= end)
    if len(dates) < 2:
        raise RuntimeError(
            f"Need at least two trade dates under {input_root}; discovered {dates}. "
            "Pass --trade-dates-file when dates cannot be inferred from folder names."
        )
    return dates


def read_trade_dates(path: Path, start: str, end: str) -> list[str]:
    dates = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    dates = sorted({date for date in dates if start <= date <= end})
    if len(dates) < 2:
        raise RuntimeError("Trade-date file must contain at least two dates in range.")
    return dates


def build_tasks(config: dict[str, Any], dates: list[str]) -> list[BoundaryTask]:
    seed0 = int(config.get("random_seed", 0))
    tasks: list[BoundaryTask] = []
    for boundary_index, (date_from, date_to) in enumerate(zip(dates[:-1], dates[1:])):
        task_seed = seed0 + boundary_index * 10_000
        tasks.append(
            BoundaryTask(
                task_id=f"{date_from}__{date_to}",
                date_from=date_from,
                date_to=date_to,
                seed=task_seed,
            )
        )
    return tasks


def checkpoint_path(output_root: Path, task: BoundaryTask) -> Path:
    return output_root / "boundaries" / task.task_id / "boundary_checkpoint.json"


def load_checkpoint(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def run_worker(
    worker: Path,
    config_path: Path,
    task: BoundaryTask,
    temp_dir: Path,
) -> None:
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
        str(temp_dir),
        "--phase1-root",
        str(args.phase1_root)
    ]
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Checkpointed monthly carry-over A/B orchestrator (Boundary Level)")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--worker", type=Path, default=Path("scripts/run_monthly_carryover_task.py"))
    parser.add_argument("--trade-dates-file", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only-boundary", help="YYYY-MM-DD:YYYY-MM-DD")
    parser.add_argument("--repo-commit", default=os.environ.get("GIT_COMMIT", "unknown"))
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--phase1-root", type=str, default="outputs/theme_discovery_phase1")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    config_hash = json_hash(config)
    input_root = Path(config["input_root"])
    output_root = Path(config["output_root"]) / config["run_name"] / config_hash[:12]
    output_root.mkdir(parents=True, exist_ok=True)

    if args.trade_dates_file:
        dates = read_trade_dates(args.trade_dates_file, config["date_start"], config["date_end"])
    else:
        dates = discover_trade_dates(input_root, config["date_start"], config["date_end"])

    tasks = build_tasks(config, dates)
    if args.only_boundary:
        left, right = args.only_boundary.split(":", 1)
        tasks = [task for task in tasks if task.date_from == left and task.date_to == right]

    run_manifest = {
        "run_name": config["run_name"],
        "config_path": str(args.config),
        "config_hash": config_hash,
        "repo_commit": args.repo_commit,
        "trade_dates": dates,
        "task_count": len(tasks),
        "created_at": utc_now(),
    }
    atomic_json(output_root / "run_manifest.json", run_manifest)
    atomic_json(output_root / "task_index.json", {"tasks": [asdict(task) for task in tasks]})

    # Pre-generate null mapping
    print("Generating null mapping...", flush=True)
    subprocess.run([sys.executable, "scripts/generate_null_mapping.py", "--config", str(args.config)], check=True)

    if args.dry_run:
        print(json.dumps(run_manifest, ensure_ascii=False, indent=2))
        for task in tasks:
            print(task.task_id)
        return

    if not args.worker.exists():
        raise FileNotFoundError(f"Worker not found: {args.worker}")

    log_path = output_root / "logs" / "runner.jsonl"
    error_path = output_root / "logs" / "errors.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def process_task(task: BoundaryTask) -> None:
        checkpoint = checkpoint_path(output_root, task)
        previous = load_checkpoint(checkpoint)
        
        # We don't check full `validate_completed` here, the worker checks arm-level _SUCCESS internally.
        # But if the worker successfully finishes, we mark boundary as success.
        if args.resume and previous and previous.get("status") == "success":
            print(f"SKIP {task.task_id}: valid completed boundary")
            return
            
        if args.retry_failed and (not previous or previous.get("status") != "failed"):
            return
            
        # The worker will output to output_root / "shards"
        temp_dir = output_root  # We just pass the output root, the worker writes to shards/...
        
        running = {
            "task_id": task.task_id,
            "status": "running",
            "repo_commit": args.repo_commit,
            "config_hash": config_hash,
            "started_at": utc_now(),
            "finished_at": None,
            "error": None,
        }
        atomic_json(checkpoint, running)
        
        try:
            run_worker(args.worker, args.config, task, temp_dir)
            finished = {**running, "status": "success", "finished_at": utc_now()}
            atomic_json(checkpoint, finished)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(finished, ensure_ascii=False) + "\n")
            print(f"DONE {task.task_id}")
        except Exception as exc:
            failed = {**running, "status": "failed", "finished_at": utc_now(), "error": repr(exc)}
            atomic_json(checkpoint, failed)
            with error_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(failed, ensure_ascii=False) + "\n")
            print(f"FAILED {task.task_id}: {exc}", file=sys.stderr)

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        list(executor.map(process_task, tasks))


if __name__ == "__main__":
    main()
