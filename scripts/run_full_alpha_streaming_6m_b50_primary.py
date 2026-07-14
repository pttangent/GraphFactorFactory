#!/usr/bin/env python3
"""B50-primary policy launcher for the existing six-month PIT-safe pipeline.

This wrapper does not alter any label, feature, PIT, IC, spread, or checkpoint
logic. It only injects an explicit Theme level policy into the public scheduler:

    GFF_RESEARCH_LEVELS=B50          # default primary research
    GFF_RESEARCH_LEVELS=B50,B35      # explicit nested replication run

Monthly falsification audits use the public, resumable parallel report runner.
All resource parameters are supplied through environment/CLI policy rather than
hard-coded inside Feature or Eval functions.
"""
from __future__ import annotations

import os
from pathlib import Path

import run_full_alpha_streaming_6m as base


def _research_levels() -> str:
    value = os.environ.get("GFF_RESEARCH_LEVELS", "B50")
    levels = [item.strip() for item in value.split(",") if item.strip()]
    unknown = set(levels) - {"B50", "B35"}
    if not levels or unknown:
        raise SystemExit(f"GFF_RESEARCH_LEVELS must be B50 or B50,B35; got {value!r}")
    if "B35" in levels and "B50" not in levels:
        raise SystemExit("B35 is a nested replication and cannot run without B50")
    return ",".join(dict.fromkeys(levels))


def _append_env_option(command: list[str], env_name: str, option: str) -> None:
    value = os.environ.get(env_name)
    if value not in (None, ""):
        command.extend([option, value])


LEVELS = _research_levels()
_ORIGINAL_RUN = base.run
_ORIGINAL_MONTH_SOURCES = base._month_sources


def policy_run(command: list[str]) -> None:
    command = list(command)
    script = Path(command[1]).name if len(command) > 1 else ""
    if script == "run_p2_24core_scheduler.py" and "--levels" not in command:
        command.extend(["--levels", LEVELS])
    elif script == "generate_monthly_alpha_report.py":
        command[1] = "scripts/generate_monthly_alpha_report_with_risk_parallel_v2.py"
        command.extend(
            [
                "--labels-root", str(base.LOCAL_P0),
                "--p1-root", str(base.LOCAL_P1),
                "--primary-level", "B50",
                "--replication-level", "B35",
            ]
        )
        _append_env_option(command, "GFF_RISK_AUDIT_WORKERS", "--workers")
        _append_env_option(command, "GFF_RISK_AUDIT_TASKS_PER_CHILD", "--tasks-per-child")
        _append_env_option(command, "GFF_RISK_AUDIT_MEMORY_BUDGET_GB", "--memory-budget-gb")
        _append_env_option(command, "GFF_RISK_AUDIT_MAX_FULL_READ_GB", "--max-full-read-gb")
        _append_env_option(command, "GFF_RISK_AUDIT_READ_MODE", "--read-mode")
        _append_env_option(command, "GFF_RISK_AUDIT_MAX_IN_FLIGHT", "--max-in-flight")
    _ORIGINAL_RUN(command)


def month_sources_with_report(month: str):
    pairs = list(_ORIGINAL_MONTH_SOURCES(month))
    scope = month.replace("-", "")
    source = base.LOCAL_P2_OUT / "monthly_alpha_report" / scope
    if source.exists():
        pairs.append((source, base.NAS_P2_OUT / "monthly_alpha_report" / scope))
    return pairs


base.run = policy_run
base._month_sources = month_sources_with_report


def main() -> None:
    print(
        f"B50-primary research policy active: levels={LEVELS}; "
        "B35 is counted only as an explicit nested replication. "
        "Risk audits are partition-parallel and checkpoint-resumable.",
        flush=True,
    )
    base.main()


if __name__ == "__main__":
    main()
