#!/usr/bin/env python3
"""OOM-safe 24-core scheduler for the P2 alpha lab.

Why this exists
---------------
The current P2 stage can use two levels of concurrency:

* outer process parallelism from ``p2_alpha_daily_features.py --workers``;
* inner per-decision-time fanout introduced in ``build_returns_one`` and
  ``relation_one``.  At commit a32e169 that inner fanout is fixed at 8.

Running ``--workers 16`` on those nested stages is therefore not 16-way; it can
behave like roughly 16 * 8 = 128 concurrent execution slots and can OOM or waste
CPU on context switching.  This scheduler treats concurrency as a slot budget
and gives nested stages only a small number of outer processes.

Default target for a 24-core workstation:

* theme returns:     3 outer processes * 8 inner fanout ~= 24 slots
* relation spillover:3 outer processes * 8 inner fanout ~= 24 slots
* daily features:    19 outer processes * 1 inner fanout ~= 19 slots

In practice the nested stages mix CPU + memory + parquet I/O, so the balanced
profile tends to keep CPU busy without launching dozens of memory-heavy child
processes.  Start with --profile safe if RAM is tight, then move to balanced.
"""
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
from typing import Iterable

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
    estimated_slots: int
    reason: str


def csv_arg(name: str, value: str | None) -> list[str]:
    return [name, value] if value else []


def build_plan(cores: int, target_cpu: float, profile: str, inner_fanout: int) -> dict[str, StagePlan]:
    target_slots = max(1, int(math.ceil(cores * target_cpu)))

    # a32e169 has fixed inner fanout of 8 for build-theme-returns and relation-spillover.
    # Use only a few outer workers there.  Daily aggregation has no inner fanout.
    if profile == "safe":
        nested_outer = max(1, target_slots // max(inner_fanout, 1))
        nested_outer = min(nested_outer, 2)
        daily_workers = min(target_slots, 16)
    elif profile == "aggressive":
        nested_outer = max(1, math.ceil(target_slots / max(inner_fanout, 1)))
        nested_outer = min(nested_outer, 4)
        daily_workers = min(max(target_slots, 20), cores)
    else:  # balanced
        nested_outer = max(1, math.ceil(target_slots / max(inner_fanout, 1)))
        nested_outer = min(nested_outer, 3)
        daily_workers = min(target_slots, 20)

    return {
        "build-theme-returns": StagePlan(
            "build-theme-returns",
            nested_outer,
            nested_outer * inner_fanout,
            "nested stage: outer workers are kept low because each child has inner decision-time fanout",
        ),
        "relation-spillover": StagePlan(
            "relation-spillover",
            nested_outer,
            nested_outer * inner_fanout,
            "nested stage: same memory pattern as build-theme-returns, so use the same cap",
        ),
        "daily-relation-features": StagePlan(
            "daily-relation-features",
            daily_workers,
            daily_workers,
            "single-level partition stage: can use most of the target CPU slot budget",
        ),
    }


def run_cmd(cmd: list[str], env: dict[str, str], dry_run: bool) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    if dry_run:
        return
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def base_filter_args(args: argparse.Namespace) -> list[str]:
    out: list[str] = []
    out += csv_arg("--dates", args.dates)
    out += csv_arg("--layers", args.layers)
    out += csv_arg("--scales", args.scales)
    out += csv_arg("--levels", args.levels)
    out += csv_arg("--horizons", args.horizons)
    if args.max_row_groups is not None:
        out += ["--max-row-groups", str(args.max_row_groups)]
    if args.skip_existing:
        out.append("--skip-existing")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="24-core OOM-safe P2 alpha lab scheduler")
    parser.add_argument("--p1-root", required=True)
    parser.add_argument("--labels-root", required=True)
    parser.add_argument("--p2-root", required=True)
    parser.add_argument("--p2-script", default="scripts/p2_alpha_daily_features.py")
    parser.add_argument("--dates")
    parser.add_argument("--layers", default="3,6,8,9,11")
    parser.add_argument("--scales", default="15m,30m")
    parser.add_argument("--levels", default="B50,B35")
    parser.add_argument("--horizons", default="5m,15m,30m,60m,120m")
    parser.add_argument("--past-horizon", default="15m")
    parser.add_argument("--tiers")
    parser.add_argument("--cores", type=int, default=24)
    parser.add_argument("--target-cpu", type=float, default=0.80)
    parser.add_argument("--inner-fanout", type=int, default=8, help="Current inner ThreadPool fanout in a32e169 nested stages")
    parser.add_argument("--profile", choices=["safe", "balanced", "aggressive"], default="balanced")
    parser.add_argument("--stage", choices=["all", "theme", "relation", "daily", "eval"], default="all")
    parser.add_argument("--max-row-groups", type=int)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not (0 < args.target_cpu <= 1.0):
        raise SystemExit("--target-cpu must be in (0, 1]")
    if args.inner_fanout < 1:
        raise SystemExit("--inner-fanout must be >= 1")

    p2_root = Path(args.p2_root)
    p2_root.mkdir(parents=True, exist_ok=True)
    plan = build_plan(args.cores, args.target_cpu, args.profile, args.inner_fanout)
    env = os.environ.copy()
    env.update(THREAD_CAPS)

    plan_payload = {
        "profile": args.profile,
        "cores": args.cores,
        "target_cpu": args.target_cpu,
        "target_slots": int(math.ceil(args.cores * args.target_cpu)),
        "inner_fanout_assumption": args.inner_fanout,
        "stage_plan": {k: asdict(v) for k, v in plan.items()},
        "filters": {
            "dates": args.dates,
            "layers": args.layers,
            "scales": args.scales,
            "levels": args.levels,
            "horizons": args.horizons,
        },
        "created_at_epoch": time.time(),
    }
    (p2_root / "p2_24core_schedule_plan.json").write_text(json.dumps(plan_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(plan_payload, indent=2, ensure_ascii=False), flush=True)

    common = base_filter_args(args)
    py = sys.executable
    p2 = args.p2_script

    if args.stage in ("all", "theme"):
        s = plan["build-theme-returns"]
        run_cmd([
            py, p2, "build-theme-returns",
            "--p1-root", args.p1_root,
            "--labels-root", args.labels_root,
            "--out-root", str(p2_root / "theme_returns"),
            "--workers", str(s.workers),
        ] + common, env, args.dry_run)

    if args.stage in ("all", "relation"):
        s = plan["relation-spillover"]
        cmd = [
            py, p2, "relation-spillover",
            "--p1-root", args.p1_root,
            "--theme-returns-root", str(p2_root / "theme_returns"),
            "--out-root", str(p2_root / "relation_spillover"),
            "--past-horizon", args.past_horizon,
            "--workers", str(s.workers),
        ] + common
        cmd += csv_arg("--tiers", args.tiers)
        run_cmd(cmd, env, args.dry_run)

    if args.stage in ("all", "daily"):
        s = plan["daily-relation-features"]
        run_cmd([
            py, p2, "daily-relation-features",
            "--signals-root", str(p2_root / "relation_spillover"),
            "--out-root", str(p2_root / "daily_relation_features"),
            "--workers", str(s.workers),
        ] + common, env, args.dry_run)

    if args.stage in ("all", "eval"):
        run_cmd([
            py, p2, "evaluate-daily",
            "--features-root", str(p2_root / "daily_relation_features"),
            "--out-dir", str(p2_root / "daily_relation_eval"),
        ], env, args.dry_run)


if __name__ == "__main__":
    main()
