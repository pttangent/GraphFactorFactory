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
class Task:
    task_id: str
    date_from: str
    date_to: str
    arm: str
    control: str
    control_index: int
    seed: int


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def build_tasks(config: dict[str, Any], dates: list[str]) -> list[Task]:
    arms = list(config["arms"])
    controls = config.get("controls", {})
    null_count = int(controls.get("day_order_nulls_per_boundary", 3))
    seed0 = int(config.get("random_seed", 0))
    tasks: list[Task] = []
    for boundary_index, (date_from, date_to) in enumerate(zip(dates[:-1], dates[1:])):
        for arm_index, arm in enumerate(arms):
            task_seed = seed0 + boundary_index * 10_000 + arm_index * 100
            tasks.append(
                Task(
                    task_id=f"{date_from}__{date_to}__{arm}__actual__0",
                    date_from=date_from,
                    date_to=date_to,
                    arm=arm,
                    control="actual",
                    control_index=0,
                    seed=task_seed,
                )
            )
            for control_index in range(null_count):
                tasks.append(
                    Task(
                        task_id=f"{date_from}__{date_to}__{arm}__day_order__{control_index}",
                        date_from=date_from,
                        date_to=date_to,
                        arm=arm,
                        control="day_order",
                        control_index=control_index,
                        seed=task_seed + control_index + 1,
                    )
                )
    return tasks


def task_dir(output_root: Path, task: Task) -> Path:
    return (
        output_root
        / "shards"
        / f"date_from={task.date_from}"
        / f"date_to={task.date_to}"
        / f"arm={task.arm}"
        / f"control={task.control}"
        / f"replicate={task.control_index}"
    )


def checkpoint_path(output_root: Path, task: Task) -> Path:
    return output_root / "checkpoints" / f"{task.task_id}.json"


def load_checkpoint(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def validate_completed(output_root: Path, task: Task) -> tuple[bool, str]:
    directory = task_dir(output_root, task)
    success = directory / "_SUCCESS"
    manifest_path = directory / "task_manifest.json"
    if not success.exists() or not manifest_path.exists():
        return False, "missing success marker or manifest"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for relative, expected in manifest.get("sha256", {}).items():
            path = directory / relative
            if not path.exists() or sha256_file(path) != expected:
                return False, f"checksum mismatch: {relative}"
    except Exception as exc:  # defensive validation at resume boundary
        return False, f"manifest validation error: {exc}"
    return True, "ok"


def finalize_temp(temp_dir: Path, final_dir: Path, task: Task, config_hash: str, repo_commit: str) -> None:
    required = ["bridge_candidates.parquet", "path_states.parquet", "revival_events.parquet", "matched_controls.parquet", "outcomes.parquet"]
    missing = [name for name in required if not (temp_dir / name).exists()]
    if missing:
        raise RuntimeError(f"Worker did not produce required files: {missing}")
    hashes = {name: sha256_file(temp_dir / name) for name in required}
    manifest = {
        "task": asdict(task),
        "config_hash": config_hash,
        "repo_commit": repo_commit,
        "finished_at": utc_now(),
        "sha256": hashes,
    }
    atomic_json(temp_dir / "task_manifest.json", manifest)
    if final_dir.exists():
        shutil.rmtree(final_dir)
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temp_dir, final_dir)
    (final_dir / "_SUCCESS").write_text("success\n", encoding="utf-8")


def run_worker(
    worker: Path,
    config_path: Path,
    task: Task,
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
        "--arm",
        task.arm,
        "--control",
        task.control,
        "--control-index",
        str(task.control_index),
        "--seed",
        str(task.seed),
        "--output-dir",
        str(temp_dir),
    ]
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Checkpointed monthly carry-over A/B orchestrator")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--worker", type=Path, default=Path("scripts/run_monthly_carryover_task.py"))
    parser.add_argument("--trade-dates-file", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only-arm")
    parser.add_argument("--only-boundary", help="YYYY-MM-DD:YYYY-MM-DD")
    parser.add_argument("--from-task-id")
    parser.add_argument("--repo-commit", default=os.environ.get("GIT_COMMIT", "unknown"))
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
    if args.only_arm:
        tasks = [task for task in tasks if task.arm == args.only_arm]
    if args.only_boundary:
        left, right = args.only_boundary.split(":", 1)
        tasks = [task for task in tasks if task.date_from == left and task.date_to == right]
    if args.from_task_id:
        ids = [task.task_id for task in tasks]
        if args.from_task_id not in ids:
            raise ValueError(f"Unknown --from-task-id: {args.from_task_id}")
        tasks = tasks[ids.index(args.from_task_id) :]

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

    if args.dry_run:
        print(json.dumps(run_manifest, ensure_ascii=False, indent=2))
        for task in tasks:
            print(task.task_id)
        return

    if not args.worker.exists():
        raise FileNotFoundError(
            f"Worker not found: {args.worker}. The worker must implement the CLI contract documented in "
            "docs/monthly_validation_local_agent_guide_zh.md and produce the five required parquet files."
        )

    log_path = output_root / "logs" / "runner.jsonl"
    error_path = output_root / "logs" / "errors.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    for task in tasks:
        checkpoint = checkpoint_path(output_root, task)
        previous = load_checkpoint(checkpoint)
        valid, reason = validate_completed(output_root, task)
        if args.resume and valid:
            print(f"SKIP {task.task_id}: valid completed shard")
            continue
        if args.retry_failed and (not previous or previous.get("status") != "failed"):
            continue
        final_dir = task_dir(output_root, task)
        if final_dir.exists() and not valid:
            corrupt = output_root / "corrupt" / f"{task.task_id}__{datetime.now().strftime('%Y%m%dT%H%M%S')}"
            corrupt.parent.mkdir(parents=True, exist_ok=True)
            os.replace(final_dir, corrupt)
        temp_dir = output_root / ".tmp" / task.task_id
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
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
            finalize_temp(temp_dir, final_dir, task, config_hash, args.repo_commit)
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


if __name__ == "__main__":
    main()
