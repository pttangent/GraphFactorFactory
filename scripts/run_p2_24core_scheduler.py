#!/usr/bin/env python3
"""RAM-aware scheduler for the PIT-safe P0/P2 alpha pipeline."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from p2_alpha_pit_features import DEFAULT_HORIZONS, DEFAULT_INTRADAY_HORIZONS, PIT_CONTRACT_VERSION
from p2_parallel_runtime import run_process_tree

THREAD_CAPS = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "ARROW_NUM_THREADS": "1",
    "POLARS_MAX_THREADS": "1",
    "PYTHONUNBUFFERED": "1",
}
BASE_GB_PER_PROCESS = {
    "p0-direct-date": 6.0,
    "p0-node-features": 3.8,
    "p0-edge-spillover": 3.8,
    "p0-graph-state": 2.0,
    "p0-eval": 3.8,
    "build-theme-returns": 3.8,
    "relation-spillover": 3.8,
    "intraday-relation-features": 3.8,
    "daily-relation-features": 3.8,
    "p2-eval": 3.0,
}
PROFILE_MEMORY_MULTIPLIER = {"safe": 1.25, "balanced": 1.0, "aggressive": 0.90, "max": 0.85}


@dataclass(frozen=True)
class StagePlan:
    stage: str
    workers: int
    inner_workers: int
    estimated_slots: int
    memory_gb_per_worker: float
    estimated_peak_ram_gb: float
    reason: str


def csv_arg(name: str, value: str | None) -> list[str]:
    return [name, value] if value else []


def _worker_cap(budget_gb: float, gb_per_worker: float, target: int) -> int:
    return max(1, min(target, int(budget_gb // gb_per_worker)))


def _nested_shape(target: int, outer_cap: int, requested_inner: int, inner_cap: int = 4) -> tuple[int, int]:
    if requested_inner > 0:
        inner = min(inner_cap, requested_inner)
        return max(1, min(outer_cap, math.ceil(target / inner))), inner
    best_slots, best_outer, best_inner = 1, 1, 1
    for inner in range(1, inner_cap + 1):
        outer = max(1, min(outer_cap, target // inner))
        slots = outer * inner
        if slots > best_slots or (slots == best_slots and inner > best_inner):
            best_slots, best_outer, best_inner = slots, outer, inner
    return best_outer, best_inner


def build_plan(
    cores: int,
    target_cpu: float,
    profile: str,
    inner_workers: int,
    ram_gb: float = 128.0,
    reserve_ram_gb: float = 24.0,
) -> dict[str, StagePlan]:
    target = max(1, min(int(cores), int(math.ceil(cores * target_cpu))))
    usable_ram = max(8.0, (float(ram_gb) - float(reserve_ram_gb)) * 0.90)
    multiplier = PROFILE_MEMORY_MULTIPLIER[profile]

    def memory(stage: str) -> float:
        return BASE_GB_PER_PROCESS[stage] * multiplier

    def simple(stage: str, reason: str) -> StagePlan:
        per_worker = memory(stage)
        workers = _worker_cap(usable_ram, per_worker, target)
        return StagePlan(stage, workers, 1, workers, per_worker, workers * per_worker, reason)

    theme_memory = memory("build-theme-returns")
    theme_outer, theme_inner = _nested_shape(target, _worker_cap(usable_ram, theme_memory, target), inner_workers)
    relation_memory = memory("relation-spillover")
    relation_outer, relation_inner = _nested_shape(target, _worker_cap(usable_ram, relation_memory, target), inner_workers)
    return {
        "p0-direct-date": simple(
            "p0-direct-date",
            "one canonical date scan writes node, spillover, and graph outputs without physical alpha shards",
        ),
        "p0-node-features": simple("p0-node-features", "legacy physical-shard compatibility"),
        "p0-edge-spillover": simple("p0-edge-spillover", "legacy physical-shard compatibility"),
        "p0-graph-state": simple("p0-graph-state", "legacy physical-shard compatibility"),
        "p0-eval": simple("p0-eval", "resumable partition metrics plus small reducer states"),
        "build-theme-returns": StagePlan(
            "build-theme-returns",
            theme_outer,
            theme_inner,
            theme_outer * theme_inner,
            theme_memory,
            theme_outer * theme_memory,
            "streamed memberships, one-label-table worker cache, ordered bounded snapshot threads",
        ),
        "relation-spillover": StagePlan(
            "relation-spillover",
            relation_outer,
            relation_inner,
            relation_outer * relation_inner,
            relation_memory,
            relation_outer * relation_memory,
            "dual sorted time streams and snapshot-local symmetric expansion",
        ),
        "intraday-relation-features": simple(
            "intraday-relation-features",
            "snapshot-streamed feature transforms with source-aware checkpoints",
        ),
        "daily-relation-features": simple(
            "daily-relation-features",
            "date-layer-scale full-session aggregation with bounded outer processes",
        ),
        "p2-eval": simple(
            "p2-eval",
            "resumable partition metric shards and parallel local summary reduction",
        ),
    }


def run_command(command: list[str], environment: dict[str, str], dry_run: bool) -> None:
    print("\n$ " + " ".join(map(str, command)), flush=True)
    if dry_run:
        return
    return_code = run_process_tree(command, env=environment)
    if return_code != 0:
        raise SystemExit(return_code)


def common_filters(args: argparse.Namespace) -> list[str]:
    output: list[str] = []
    for name, value in (
        ("--dates", args.dates),
        ("--layers", args.layers),
        ("--scales", args.scales),
        ("--levels", args.levels),
        ("--horizons", args.horizons),
    ):
        output += csv_arg(name, value)
    if args.max_row_groups is not None:
        output += ["--max-row-groups", str(args.max_row_groups)]
    if args.skip_existing:
        output.append("--skip-existing")
    return output


def eval_filters(args: argparse.Namespace) -> list[str]:
    output: list[str] = []
    for name, value in (("--dates", args.dates), ("--layers", args.layers), ("--scales", args.scales)):
        output += csv_arg(name, value)
    return output


def p0_filters(args: argparse.Namespace) -> list[str]:
    output: list[str] = []
    for name, value in (("--dates", args.dates), ("--layers", args.layers), ("--scales", args.scales)):
        output += csv_arg(name, value)
    intraday = [horizon for horizon in args.horizons.split(",") if horizon.endswith("m")]
    output += ["--horizons", ",".join(intraday or DEFAULT_INTRADAY_HORIZONS)]
    if args.max_row_groups is not None:
        output += ["--max-row-groups", str(args.max_row_groups)]
    if args.skip_existing:
        output.append("--skip-existing")
    return output


def scope_name(dates: str | None) -> str:
    if not dates:
        return "global"
    months = {value.strip()[:7] for value in dates.split(",") if value.strip()}
    return next(iter(months)).replace("-", "") if len(months) == 1 else "selected"


def single_month(dates: str | None) -> str | None:
    if not dates:
        return None
    months = {value.strip()[:7] for value in dates.split(",") if value.strip()}
    return next(iter(months)) if len(months) == 1 else None


def main() -> None:
    parser = argparse.ArgumentParser(description="PIT-safe 24-core / 128GB alpha scheduler")
    parser.add_argument("--p0-root")
    parser.add_argument("--p1-root")
    parser.add_argument("--labels-root", required=True)
    parser.add_argument("--p2-root", required=True)
    parser.add_argument("--p2-script", default="scripts/p2_alpha_daily_features.py")
    parser.add_argument("--p0-script", default="scripts/p2_p0_graph_alpha_v2.py")
    parser.add_argument("--dates")
    parser.add_argument("--layers")
    parser.add_argument("--scales")
    parser.add_argument("--levels", default="B50,B35")
    parser.add_argument("--horizons", default=",".join(DEFAULT_HORIZONS))
    parser.add_argument("--past-horizon", default="15m")
    parser.add_argument("--underreaction-past-horizon", default="15m")
    parser.add_argument("--late-minutes", type=int, default=60)
    parser.add_argument("--tiers")
    parser.add_argument("--cores", type=int, default=24)
    parser.add_argument("--target-cpu", type=float, default=1.0)
    parser.add_argument("--inner-workers", type=int, default=0)
    parser.add_argument("--ram-gb", type=float, default=128.0)
    parser.add_argument("--reserve-ram-gb", type=float, default=24.0)
    parser.add_argument("--p0-date-workers", type=int, default=8)
    parser.add_argument("--p0-batch-size", type=int, default=500_000)
    parser.add_argument("--p0-min-free-gb", type=float, default=50.0)
    parser.add_argument("--p0-disk-check-every", type=int, default=25)
    parser.add_argument("--tasks-per-child", type=int, default=8)
    parser.add_argument("--parquet-target-rows", type=int, default=100_000)
    parser.add_argument("--eval-csv-mode", choices=["none", "sharded", "single"], default="none")
    parser.add_argument("--profile", choices=["safe", "balanced", "aggressive", "max"], default="balanced")
    parser.add_argument(
        "--stage",
        choices=[
            "all", "p0", "p0-node", "p0-edge", "p0-graph", "p0-eval",
            "theme", "relation", "intraday", "daily", "intraday-eval", "daily-eval", "eval",
        ],
        default="all",
    )
    parser.add_argument("--max-row-groups", type=int)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not (0 < args.target_cpu <= 1.0):
        raise SystemExit("--target-cpu must be in (0, 1.0]")
    if args.inner_workers < 0 or args.tasks_per_child < 0:
        raise SystemExit("--inner-workers and --tasks-per-child must be >=0")
    if args.ram_gb <= args.reserve_ram_gb:
        raise SystemExit("--ram-gb must exceed --reserve-ram-gb")
    if args.p0_date_workers < 1 or args.parquet_target_rows < 1:
        raise SystemExit("worker counts and parquet target rows must be positive")

    root = Path(args.p2_root)
    root.mkdir(parents=True, exist_ok=True)
    plan = build_plan(args.cores, args.target_cpu, args.profile, args.inner_workers, args.ram_gb, args.reserve_ram_gb)
    direct_cap = plan["p0-direct-date"].workers
    p0_date_workers = max(1, min(args.p0_date_workers, direct_cap, args.cores))
    payload = {
        "pit_contract_version": PIT_CONTRACT_VERSION,
        "profile": args.profile,
        "cores": args.cores,
        "target_cpu": args.target_cpu,
        "ram_gb": args.ram_gb,
        "reserve_ram_gb": args.reserve_ram_gb,
        "schedulable_ram_gb": (args.ram_gb - args.reserve_ram_gb) * 0.90,
        "worker_recycling": {"tasks_per_child": args.tasks_per_child, "zero_means_pool_lifetime": True},
        "parquet_target_rows": args.parquet_target_rows,
        "p0_execution": {
            "mode": "canonical_date_single_pass",
            "physical_alpha_shards": False,
            "date_workers": p0_date_workers,
            "batch_size": args.p0_batch_size,
            "min_free_gb": args.p0_min_free_gb,
        },
        "stage_plan": {key: asdict(value) for key, value in plan.items()},
        "filters": {
            "dates": args.dates,
            "layers": args.layers,
            "scales": args.scales,
            "levels": args.levels,
            "horizons": args.horizons,
        },
        "created_at_epoch": time.time(),
    }
    (root / "p2_24core_schedule_plan.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)

    environment = os.environ.copy()
    environment.update(THREAD_CAPS)
    environment["GFF_MAX_TASKS_PER_CHILD"] = str(args.tasks_per_child)
    environment["GFF_PARQUET_TARGET_ROWS"] = str(args.parquet_target_rows)
    python = sys.executable
    common = common_filters(args)

    if args.stage in {"all", "p0", "p0-node", "p0-edge", "p0-graph"}:
        if not args.p0_root:
            raise SystemExit("--p0-root required for P0 stages")
        feature_map = {"p0-node": "node", "p0-edge": "spillover", "p0-graph": "graph"}
        features = feature_map.get(args.stage, "node,spillover,graph")
        command = [
            python, args.p0_script, "direct", "--p0-root", args.p0_root,
            "--labels-root", args.labels_root, "--out-root", str(root),
            "--features", features, "--past-horizon", args.past_horizon,
            "--workers", str(p0_date_workers), "--batch-size", str(args.p0_batch_size),
            "--min-free-gb", str(args.p0_min_free_gb),
            "--disk-check-every", str(args.p0_disk_check_every),
        ] + p0_filters(args)
        run_command(command, environment, args.dry_run)

    if args.stage in {"all", "p0", "p0-eval"}:
        eval_dir = root / "p0_alpha" / scope_name(args.dates)
        stage = plan["p0-eval"]
        command = [
            python, args.p0_script, "eval-p0", "--p0-alpha-root", str(root),
            "--out-dir", str(eval_dir), "--workers", str(stage.workers),
            "--csv-mode", args.eval_csv_mode,
        ]
        month = single_month(args.dates)
        if month:
            command += ["--month", month]
        if args.skip_existing:
            command.append("--skip-existing")
        run_command(command, environment, args.dry_run)

    if args.stage in {"all", "theme"}:
        if not args.p1_root:
            raise SystemExit("--p1-root required for theme stage")
        stage = plan["build-theme-returns"]
        run_command(
            [python, args.p2_script, "build-theme-returns", "--p1-root", args.p1_root,
             "--labels-root", args.labels_root, "--out-root", str(root / "theme_returns"),
             "--workers", str(stage.workers), "--inner-workers", str(stage.inner_workers)] + common,
            environment,
            args.dry_run,
        )

    if args.stage in {"all", "relation"}:
        if not args.p1_root:
            raise SystemExit("--p1-root required for relation stage")
        stage = plan["relation-spillover"]
        command = [
            python, args.p2_script, "relation-spillover", "--p1-root", args.p1_root,
            "--theme-returns-root", str(root / "theme_returns"),
            "--out-root", str(root / "relation_spillover"),
            "--past-horizon", args.past_horizon,
            "--workers", str(stage.workers), "--inner-workers", str(stage.inner_workers),
        ] + common
        command += csv_arg("--tiers", args.tiers)
        run_command(command, environment, args.dry_run)

    if args.stage in {"all", "intraday"}:
        stage = plan["intraday-relation-features"]
        run_command(
            [python, args.p2_script, "intraday-relation-features",
             "--signals-root", str(root / "relation_spillover"),
             "--out-root", str(root / "intraday_relation_features"),
             "--workers", str(stage.workers),
             "--underreaction-past-horizon", args.underreaction_past_horizon] + common,
            environment,
            args.dry_run,
        )

    if args.stage in {"all", "daily"}:
        if not args.p1_root:
            raise SystemExit("--p1-root required for daily temporal episode identity")
        stage = plan["daily-relation-features"]
        run_command(
            [python, args.p2_script, "daily-relation-features",
             "--signals-root", str(root / "relation_spillover"),
             "--p1-root", args.p1_root,
             "--out-root", str(root / "daily_relation_features"),
             "--workers", str(stage.workers), "--late-minutes", str(args.late_minutes),
             "--underreaction-past-horizon", args.underreaction_past_horizon] + common,
            environment,
            args.dry_run,
        )

    eval_scope = scope_name(args.dates)
    if args.stage in {"all", "eval", "intraday-eval"}:
        stage = plan["p2-eval"]
        command = [
            python, args.p2_script, "evaluate-intraday",
            "--features-root", str(root / "intraday_relation_features"),
            "--out-dir", str(root / "intraday_relation_eval" / eval_scope),
            "--workers", str(stage.workers), "--csv-mode", args.eval_csv_mode,
        ] + eval_filters(args)
        if args.skip_existing:
            command.append("--skip-existing")
        run_command(command, environment, args.dry_run)

    if args.stage in {"all", "eval", "daily-eval"}:
        stage = plan["p2-eval"]
        command = [
            python, args.p2_script, "evaluate-daily",
            "--features-root", str(root / "daily_relation_features"),
            "--out-dir", str(root / "daily_relation_eval" / eval_scope),
            "--workers", str(stage.workers), "--csv-mode", args.eval_csv_mode,
        ] + eval_filters(args)
        if args.skip_existing:
            command.append("--skip-existing")
        run_command(command, environment, args.dry_run)


if __name__ == "__main__":
    main()
