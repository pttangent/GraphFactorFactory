#!/usr/bin/env python3
"""Resource-aware scheduler for the PIT-safe P2 pipeline."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from p2_alpha_pit_features import DEFAULT_HORIZONS, DEFAULT_INTRADAY_HORIZONS, PIT_CONTRACT_VERSION

THREAD_CAPS = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "ARROW_NUM_THREADS": "1",
    "POLARS_MAX_THREADS": "1",
    "PYTHONUNBUFFERED": "1",
}


@dataclass(frozen=True)
class StagePlan:
    stage: str
    workers: int
    inner_workers: int
    estimated_slots: int
    reason: str


def csv_arg(name: str, value: str | None) -> list[str]:
    return [name, value] if value else []


def build_plan(cores: int, target_cpu: float, profile: str, inner_workers: int) -> dict[str, StagePlan]:
    target = max(1, int(math.ceil(cores * target_cpu)))
    inner_workers = max(1, inner_workers)
    if profile == "safe":
        p0, nested, feature = min(target, 12), max(1, min(4, target // inner_workers)), min(target, 16)
    elif profile == "balanced":
        p0, nested, feature = min(target, 18), max(1, min(8, math.ceil(target / inner_workers))), min(target, 20)
    elif profile == "aggressive":
        p0, nested, feature = min(cores, 22), max(1, min(12, math.ceil(cores / inner_workers))), min(cores, 24)
    else:
        p0, nested, feature = cores, max(1, math.ceil(cores / inner_workers)), cores
    return {
        "p0-node-features": StagePlan("p0-node-features", p0, 1, p0, "snapshot-local P0 node features"),
        "p0-edge-spillover": StagePlan("p0-edge-spillover", p0, 1, p0, "PIT-aligned P0 edge spillover"),
        "p0-graph-state": StagePlan("p0-graph-state", p0, 1, p0, "snapshot graph state"),
        "build-theme-returns": StagePlan("build-theme-returns", nested, inner_workers, nested * inner_workers, "actual label-exit-time alignment"),
        "relation-spillover": StagePlan("relation-spillover", nested, inner_workers, nested * inner_workers, "symmetric neighbor diffusion"),
        "intraday-relation-features": StagePlan("intraday-relation-features", feature, 1, feature, "snapshot-local normalization"),
        "daily-relation-features": StagePlan("daily-relation-features", feature, 1, feature, "EOD aggregation for next-open labels"),
    }


def run_command(command: list[str], environment: dict[str, str], dry_run: bool) -> None:
    print("\n$ " + " ".join(map(str, command)), flush=True)
    if dry_run:
        return
    completed = subprocess.run(command, env=environment)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


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
    intraday = [h for h in args.horizons.split(",") if h.endswith("m")]
    output += ["--horizons", ",".join(intraday or DEFAULT_INTRADAY_HORIZONS)]
    if args.max_row_groups is not None:
        output += ["--max-row-groups", str(args.max_row_groups)]
    if args.skip_existing:
        output.append("--skip-existing")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="PIT-safe full P2 alpha scheduler")
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
    parser.add_argument("--inner-workers", type=int, default=1)
    parser.add_argument("--profile", choices=["safe", "balanced", "aggressive", "max"], default="max")
    parser.add_argument(
        "--stage",
        choices=["all", "p0", "p0-node", "p0-edge", "p0-graph", "p0-eval", "theme", "relation", "intraday", "daily", "intraday-eval", "daily-eval", "eval"],
        default="all",
    )
    parser.add_argument("--max-row-groups", type=int)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not (0 < args.target_cpu <= 1.5):
        raise SystemExit("--target-cpu must be in (0, 1.5]")
    if args.inner_workers < 1:
        raise SystemExit("--inner-workers must be >=1")

    root = Path(args.p2_root)
    root.mkdir(parents=True, exist_ok=True)
    plan = build_plan(args.cores, args.target_cpu, args.profile, args.inner_workers)
    payload = {
        "pit_contract_version": PIT_CONTRACT_VERSION,
        "profile": args.profile,
        "cores": args.cores,
        "target_cpu": args.target_cpu,
        "inner_workers": args.inner_workers,
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
        run_command([python, args.p0_script, "eval-p0", "--p0-alpha-root", str(root), "--out-dir", str(root / "p0_alpha_eval")], environment, args.dry_run)

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
        stage = plan["daily-relation-features"]
        run_command([python, args.p2_script, "daily-relation-features", "--signals-root", str(root / "relation_spillover"), "--out-root", str(root / "daily_relation_features"), "--workers", str(stage.workers), "--late-minutes", str(args.late_minutes), "--underreaction-past-horizon", args.underreaction_past_horizon] + common, environment, args.dry_run)
    if args.stage in {"all", "eval", "intraday-eval"}:
        run_command([python, args.p2_script, "evaluate-intraday", "--features-root", str(root / "intraday_relation_features"), "--out-dir", str(root / "intraday_relation_eval")], environment, args.dry_run)
    if args.stage in {"all", "eval", "daily-eval"}:
        run_command([python, args.p2_script, "evaluate-daily", "--features-root", str(root / "daily_relation_features"), "--out-dir", str(root / "daily_relation_eval")], environment, args.dry_run)


if __name__ == "__main__":
    main()
