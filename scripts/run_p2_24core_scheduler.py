#!/usr/bin/env python3
"""RAM-aware scheduler for the PIT-safe P2 pipeline."""
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
    "GFF_MAX_TASKS_PER_CHILD": "1",
}

BASE_GB_PER_PROCESS = {
    "p0-node-features": 4.5,
    "p0-edge-spillover": 5.5,
    "p0-graph-state": 2.0,
    "p0-eval": 6.0,
    "build-theme-returns": 14.0,
    "relation-spillover": 12.0,
    "intraday-relation-features": 5.0,
    "daily-relation-features": 7.0,
}
PROFILE_MEMORY_MULTIPLIER = {
    "safe": 1.25,
    "balanced": 1.0,
    "aggressive": 0.90,
    "max": 0.85,
}


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


def _worker_cap(usable_ram_gb: float, gb_per_worker: float, target: int) -> int:
    return max(1, min(target, int(usable_ram_gb // gb_per_worker)))


def _nested_shape(target: int, outer_cap: int, requested_inner: int, inner_cap: int = 4) -> tuple[int, int]:
    if requested_inner > 0:
        inner = min(inner_cap, requested_inner)
        outer = max(1, min(outer_cap, math.ceil(target / inner)))
        return outer, inner
    best = (1, 1, 1)
    for inner in range(1, inner_cap + 1):
        outer = max(1, min(outer_cap, target // inner))
        slots = outer * inner
        candidate = (slots, outer, inner)
        if slots <= target and candidate > best:
            best = candidate
    return best[1], best[2]


def build_plan(
    cores: int,
    target_cpu: float,
    profile: str,
    inner_workers: int,
    ram_gb: float = 128.0,
    reserve_ram_gb: float = 24.0,
) -> dict[str, StagePlan]:
    target = max(1, min(int(cores), int(math.ceil(cores * target_cpu))))
    usable_ram = max(8.0, float(ram_gb) - float(reserve_ram_gb))
    multiplier = PROFILE_MEMORY_MULTIPLIER[profile]

    def memory(stage: str) -> float:
        return BASE_GB_PER_PROCESS[stage] * multiplier

    def simple(stage: str, reason: str) -> StagePlan:
        per_worker = memory(stage)
        workers = _worker_cap(usable_ram, per_worker, target)
        return StagePlan(stage, workers, 1, workers, per_worker, workers * per_worker, reason)

    theme_memory = memory("build-theme-returns")
    theme_cap = _worker_cap(usable_ram, theme_memory, target)
    theme_outer, theme_inner = _nested_shape(target, theme_cap, inner_workers)
    relation_memory = memory("relation-spillover")
    relation_cap = _worker_cap(usable_ram, relation_memory, target)
    relation_outer, relation_inner = _nested_shape(target, relation_cap, inner_workers)

    return {
        "p0-node-features": simple("p0-node-features", "row-group streamed P0 nodes; RAM-capped outer processes"),
        "p0-edge-spillover": simple("p0-edge-spillover", "row-group streamed P0 spillover; RAM-capped outer processes"),
        "p0-graph-state": simple("p0-graph-state", "light row-group graph-state stage"),
        "p0-eval": simple("p0-eval", "monthly P0 evaluation"),
        "build-theme-returns": StagePlan(
            "build-theme-returns",
            theme_outer,
            theme_inner,
            theme_outer * theme_inner,
            theme_memory,
            theme_outer * theme_memory,
            "RAM-limited outer processes plus bounded snapshot threads",
        ),
        "relation-spillover": StagePlan(
            "relation-spillover",
            relation_outer,
            relation_inner,
            relation_outer * relation_inner,
            relation_memory,
            relation_outer * relation_memory,
            "RAM-limited outer processes; symmetric expansion is snapshot-local",
        ),
        "intraday-relation-features": simple("intraday-relation-features", "snapshot-local normalization"),
        "daily-relation-features": simple("daily-relation-features", "EOD temporal-episode aggregation"),
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
    for name, value in (("--dates", args.dates), ("--layers", args.layers), ("--scales", args.scales), ("--levels", args.levels), ("--horizons", args.horizons)):
        output += csv_arg(name, value)
    if args.max_row_groups is not None:
        output += ["--max-row-groups", str(args.max_row_groups)]
    if args.skip_existing:
        output.append("--skip-existing")
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


def main() -> None:
    parser = argparse.ArgumentParser(description="PIT-safe 24-core / 128GB alpha scheduler")
    parser.add_argument("--p0-root")
    parser.add_argument("--p1-root")
    parser.add_argument("--labels-root", required=True)
    parser.add_argument("--p2-root", required=True)
    parser.add_argument("--p2-script", default="scripts/p2_alpha_daily_features.py")
    parser.add_argument("--p0-script", default="scripts/p2_p0_graph_alpha.py")
    parser.add_argument("--dates")
    parser.add_argument("--layers", default="3,6,8,9,11")
    parser.add_argument("--scales", default="15m,30m")
    parser.add_argument("--levels", default="B50,B35")
    parser.add_argument("--horizons", default=",".join(DEFAULT_HORIZONS))
    parser.add_argument("--past-horizon", default="15m")
    parser.add_argument("--underreaction-past-horizon", default="15m")
    parser.add_argument("--late-minutes", type=int, default=60)
    parser.add_argument("--tiers")
    parser.add_argument("--cores", type=int, default=24)
    parser.add_argument("--target-cpu", type=float, default=1.0)
    parser.add_argument("--inner-workers", type=int, default=0, help="0=auto; threads inside each heavy outer process")
    parser.add_argument("--ram-gb", type=float, default=128.0)
    parser.add_argument("--reserve-ram-gb", type=float, default=24.0)
    parser.add_argument("--profile", choices=["safe", "balanced", "aggressive", "max"], default="balanced")
    parser.add_argument(
        "--stage",
        choices=["all", "p0", "p0-node", "p0-edge", "p0-graph", "p0-eval", "theme", "relation", "intraday", "daily", "intraday-eval", "daily-eval", "eval"],
        default="all",
    )
    parser.add_argument("--max-row-groups", type=int)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not (0 < args.target_cpu <= 1.0):
        raise SystemExit("--target-cpu must be in (0, 1.0]")
    if args.inner_workers < 0:
        raise SystemExit("--inner-workers must be >=0")
    if args.ram_gb <= args.reserve_ram_gb:
        raise SystemExit("--ram-gb must exceed --reserve-ram-gb")

    root = Path(args.p2_root)
    root.mkdir(parents=True, exist_ok=True)
    plan = build_plan(args.cores, args.target_cpu, args.profile, args.inner_workers, args.ram_gb, args.reserve_ram_gb)
    payload = {
        "pit_contract_version": PIT_CONTRACT_VERSION,
        "profile": args.profile,
        "cores": args.cores,
        "target_cpu": args.target_cpu,
        "ram_gb": args.ram_gb,
        "reserve_ram_gb": args.reserve_ram_gb,
        "usable_ram_gb": args.ram_gb - args.reserve_ram_gb,
        "worker_recycling": "one_partition_per_process",
        "maximum_outer_python_workers": max(stage.workers for stage in plan.values()),
        "stage_plan": {key: asdict(value) for key, value in plan.items()},
        "filters": {
            "dates": args.dates,
            "layers": args.layers,
            "scales": args.scales,
            "levels": args.levels,
            "horizons": args.horizons,
            "past_horizon": args.past_horizon,
            "underreaction_past_horizon": args.underreaction_past_horizon,
        },
        "created_at_epoch": time.time(),
    }
    (root / "p2_24core_schedule_plan.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)

    environment = os.environ.copy()
    environment.update(THREAD_CAPS)
    python = sys.executable
    common = common_filters(args)

    if args.stage in {"all", "p0", "p0-node"}:
        if not args.p0_root:
            raise SystemExit("--p0-root required for P0 stages")
        stage = plan["p0-node-features"]
        run_command([python, args.p0_script, "node-features", "--p0-root", args.p0_root, "--labels-root", args.labels_root, "--out-root", str(root / "p0_node_features"), "--workers", str(stage.workers)] + p0_filters(args), environment, args.dry_run)
    if args.stage in {"all", "p0", "p0-edge"}:
        if not args.p0_root:
            raise SystemExit("--p0-root required for P0 stages")
        stage = plan["p0-edge-spillover"]
        run_command([python, args.p0_script, "edge-spillover", "--p0-root", args.p0_root, "--labels-root", args.labels_root, "--out-root", str(root / "p0_edge_spillover"), "--past-horizon", args.past_horizon, "--workers", str(stage.workers)] + p0_filters(args), environment, args.dry_run)
    if args.stage in {"all", "p0", "p0-graph"}:
        if not args.p0_root:
            raise SystemExit("--p0-root required for P0 stages")
        stage = plan["p0-graph-state"]
        run_command([python, args.p0_script, "graph-state", "--p0-root", args.p0_root, "--out-root", str(root / "p0_graph_state"), "--workers", str(stage.workers)] + p0_filters(args), environment, args.dry_run)
    if args.stage in {"all", "p0", "p0-eval"}:
        month = args.dates.split(",")[0][:7] if args.dates else "unknown"
        eval_dir = f"p0_alpha/{month.replace('-', '')}"
        if args.skip_existing and (root / eval_dir / "p0_alpha_metrics.csv").exists():
            print(f"Skipping existing P0 eval at {root / eval_dir}", flush=True)
        else:
            stage = plan["p0-eval"]
            run_command([python, args.p0_script, "eval-p0", "--p0-alpha-root", str(root), "--out-dir", str(root / eval_dir), "--month", month, "--workers", str(stage.workers)], environment, args.dry_run)

    if args.stage in {"all", "theme"}:
        if not args.p1_root:
            raise SystemExit("--p1-root required for theme stage")
        stage = plan["build-theme-returns"]
        run_command([python, args.p2_script, "build-theme-returns", "--p1-root", args.p1_root, "--labels-root", args.labels_root, "--out-root", str(root / "theme_returns"), "--workers", str(stage.workers), "--inner-workers", str(stage.inner_workers)] + common, environment, args.dry_run)
    if args.stage in {"all", "relation"}:
        if not args.p1_root:
            raise SystemExit("--p1-root required for relation stage")
        stage = plan["relation-spillover"]
        command = [python, args.p2_script, "relation-spillover", "--p1-root", args.p1_root, "--theme-returns-root", str(root / "theme_returns"), "--out-root", str(root / "relation_spillover"), "--past-horizon", args.past_horizon, "--workers", str(stage.workers), "--inner-workers", str(stage.inner_workers)] + common
        command += csv_arg("--tiers", args.tiers)
        run_command(command, environment, args.dry_run)
    if args.stage in {"all", "intraday"}:
        stage = plan["intraday-relation-features"]
        run_command([python, args.p2_script, "intraday-relation-features", "--signals-root", str(root / "relation_spillover"), "--out-root", str(root / "intraday_relation_features"), "--workers", str(stage.workers), "--underreaction-past-horizon", args.underreaction_past_horizon] + common, environment, args.dry_run)
    if args.stage in {"all", "daily"}:
        if not args.p1_root:
            raise SystemExit("--p1-root required for daily temporal episode identity")
        stage = plan["daily-relation-features"]
        run_command([python, args.p2_script, "daily-relation-features", "--signals-root", str(root / "relation_spillover"), "--p1-root", args.p1_root, "--out-root", str(root / "daily_relation_features"), "--workers", str(stage.workers), "--late-minutes", str(args.late_minutes), "--underreaction-past-horizon", args.underreaction_past_horizon] + common, environment, args.dry_run)
    if args.stage in {"all", "eval", "intraday-eval"}:
        run_command([python, args.p2_script, "evaluate-intraday", "--features-root", str(root / "intraday_relation_features"), "--out-dir", str(root / "intraday_relation_eval")], environment, args.dry_run)
    if args.stage in {"all", "eval", "daily-eval"}:
        run_command([python, args.p2_script, "evaluate-daily", "--features-root", str(root / "daily_relation_features"), "--out-dir", str(root / "daily_relation_eval")], environment, args.dry_run)


if __name__ == "__main__":
    main()
