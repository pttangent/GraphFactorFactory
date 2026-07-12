#!/usr/bin/env python3
"""Orchestrate P1 QC and P2 alpha-lab stages.

This is a coordinator, not an alpha proof script.  In particular, round 4 only
builds real daily_labels.parquet from a user-supplied stable-symbol daily OHLC
file.  It no longer generates mock labels by default and no longer calls itself
"true daily alpha validation".
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOG = logging.getLogger("run_p2_pipeline")

DEFAULT_ROUNDS: dict[str, dict[str, str]] = {
    "1": {
        "dates": "2026-01-07",
        "layers": "9",
        "scales": "15m,30m",
        "levels": "B50,B35",
    },
    "2": {
        "dates": "2026-01-07,2026-01-08,2026-01-09",
        "layers": "6,8,9",
        "scales": "15m,30m",
        "levels": "B50,B35",
    },
    "3": {
        "dates": "2026-01-07,2026-01-08,2026-01-09,2026-01-12,2026-01-13,2026-01-14,2026-01-15,2026-01-16,2026-01-20,2026-01-21",
        "layers": "3,6,8,9,11",
        "scales": "15m,30m",
        "levels": "B50,B35",
    },
}


def run_cmd(cmd: list[str], cwd: Path | None = None) -> None:
    LOG.info("Executing: %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def csv_override(default: str, override: str | None) -> str:
    return override if override else default


def round_params(args: argparse.Namespace) -> dict[str, str]:
    base = DEFAULT_ROUNDS[args.round].copy()
    base["dates"] = csv_override(base["dates"], args.dates)
    base["layers"] = csv_override(base["layers"], args.layers)
    base["scales"] = csv_override(base["scales"], args.scales)
    base["levels"] = csv_override(base["levels"], args.levels)
    return base


def require_path(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"{label} does not exist: {path}")


def run_qc(args: argparse.Namespace) -> None:
    LOG.info("=== Phase: P1 QC ===")
    require_path(args.p1_root, "--p1-root")
    cmd = [
        sys.executable,
        "scripts/p1_qc.py",
        "--p1-root",
        str(args.p1_root),
        "--workers",
        str(args.workers),
    ]
    if args.allow_empty_relations:
        cmd.append("--allow-empty-relations")
    run_cmd(cmd)


def run_p2_lab(args: argparse.Namespace) -> None:
    LOG.info("=== Phase: P2 Intraday Alpha Lab ===")
    require_path(args.p1_root, "--p1-root")
    require_path(args.labels_root, "--labels-root")
    args.p2_root.mkdir(parents=True, exist_ok=True)

    params = round_params(args)
    dates_arg = ["--dates", params["dates"]] if params.get("dates") else []
    common = [
        "--layers", params["layers"],
        "--scales", params["scales"],
        "--levels", params["levels"],
        "--workers", str(args.workers),
    ] + dates_arg

    run_cmd([
        sys.executable,
        "scripts/p2_alpha_daily_features.py",
        "build-theme-returns",
        "--p1-root", str(args.p1_root),
        "--labels-root", str(args.labels_root),
        "--out-root", str(args.p2_root / "theme_returns"),
    ] + common)

    run_cmd([
        sys.executable,
        "scripts/p2_alpha_daily_features.py",
        "relation-spillover",
        "--p1-root", str(args.p1_root),
        "--theme-returns-root", str(args.p2_root / "theme_returns"),
        "--out-root", str(args.p2_root / "relation_spillover"),
        "--past-horizon", args.past_horizon,
    ] + common)

    run_cmd([
        sys.executable,
        "scripts/p2_alpha_daily_features.py",
        "daily-relation-features",
        "--signals-root", str(args.p2_root / "relation_spillover"),
        "--out-root", str(args.p2_root / "daily_relation_features"),
    ] + common)

    run_cmd([
        sys.executable,
        "scripts/p2_alpha_daily_features.py",
        "evaluate-daily",
        "--features-root", str(args.p2_root / "daily_relation_features"),
        "--out-dir", str(args.p2_root / "daily_relation_eval"),
    ])


def build_daily_labels(args: argparse.Namespace) -> None:
    LOG.info("=== Phase: Build Real Daily Labels ===")
    args.p2_root.mkdir(parents=True, exist_ok=True)
    out_path = args.daily_labels_out or (args.p2_root / "daily_labels.parquet")

    cmd = [
        sys.executable,
        "scripts/build_daily_labels.py",
        "--out-path", str(out_path),
        "--date-col", args.daily_date_col,
        "--stable-id-col", args.daily_stable_id_col,
        "--symbol-col", args.daily_symbol_col,
        "--open-col", args.daily_open_col,
        "--close-col", args.daily_close_col,
        "--max-abs-return", str(args.max_abs_return),
    ]
    if args.daily_prices:
        require_path(args.daily_prices, "--daily-prices")
        cmd += ["--raw-daily-prices", str(args.daily_prices)]
    elif args.allow_mock_daily_labels:
        cmd.append("--allow-mock")
        if out_path.name == "daily_labels.parquet":
            LOG.warning("Mock labels requested; consider using --daily-labels-out daily_labels_mock.parquet")
    else:
        raise SystemExit(
            "Round 4/build-labels requires --daily-prices. "
            "Use --allow-mock-daily-labels only for smoke tests."
        )
    if args.allow_extreme_returns:
        cmd.append("--allow-extreme-returns")
    run_cmd(cmd)


def write_manifest(args: argparse.Namespace, start_t: float) -> None:
    args.p2_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "round": args.round,
        "p1_root": str(args.p1_root),
        "labels_root": str(args.labels_root) if args.labels_root else None,
        "p2_root": str(args.p2_root),
        "workers": args.workers,
        "elapsed_sec": round(time.time() - start_t, 3),
        "dates_override": args.dates,
        "layers_override": args.layers,
        "scales_override": args.scales,
        "levels_override": args.levels,
    }
    (args.p2_root / "run_p2_pipeline_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="P1 QC + P2 alpha-lab pipeline orchestrator.")
    parser.add_argument("--round", choices=["qc", "1", "2", "3", "4", "build-labels"], required=True)
    parser.add_argument("--p1-root", type=Path, default=Path(r"C:\GFF_Cache\p1_b50_b35_sharded"))
    parser.add_argument("--labels-root", type=Path, default=Path(r"D:\DEV\US-Stock\GraphFactorFactory\data\graph_store_6m\canonical"))
    parser.add_argument("--p2-root", type=Path, default=Path(r"C:\GFF_Cache\p2_alpha_lab"))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--dates", default=None, help="Override round dates, comma-separated.")
    parser.add_argument("--layers", default=None, help="Override round layers, comma-separated.")
    parser.add_argument("--scales", default=None, help="Override round scales, comma-separated.")
    parser.add_argument("--levels", default=None, help="Override levels, e.g. B50,B35.")
    parser.add_argument("--past-horizon", default="15m")
    parser.add_argument("--allow-empty-relations", action="store_true")

    # Daily-label stage parameters.
    parser.add_argument("--daily-prices", type=Path, default=None, help="Stable-symbol daily OHLC parquet file or directory.")
    parser.add_argument("--daily-labels-out", type=Path, default=None)
    parser.add_argument("--daily-date-col", default="date")
    parser.add_argument("--daily-stable-id-col", default="stable_symbol_id")
    parser.add_argument("--daily-symbol-col", default="symbol")
    parser.add_argument("--daily-open-col", default="open")
    parser.add_argument("--daily-close-col", default="close")
    parser.add_argument("--max-abs-return", type=float, default=5.0)
    parser.add_argument("--allow-extreme-returns", action="store_true")
    parser.add_argument("--allow-mock-daily-labels", action="store_true", help="Smoke tests only; never use for alpha research.")
    args = parser.parse_args()

    start_t = time.time()
    if args.round == "qc":
        run_qc(args)
    elif args.round in {"1", "2", "3"}:
        run_p2_lab(args)
    elif args.round in {"4", "build-labels"}:
        build_daily_labels(args)
    write_manifest(args, start_t)


if __name__ == "__main__":
    main()
