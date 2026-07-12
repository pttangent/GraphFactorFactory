#!/usr/bin/env python3
"""PIT-safe P2 public API and command-line entrypoint."""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
from pathlib import Path

from p2_pit_core import *
from p2_pit_theme import *
from p2_pit_features import *
from p2_pit_features import _build_feature_one


def pool(parts: list[Part], workers: int, function, *args) -> list[dict]:
    if not parts:
        return []
    results: list[dict] = []
    with cf.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(function, part, *args) for part in parts]
        for future in cf.as_completed(futures):
            results.append(future.result())
    return results


def save_run_summary(root: str | Path, results: list[dict]) -> None:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "run_summary.json").write_text(
        json.dumps({"pit_contract_version": PIT_CONTRACT_VERSION, "results": results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="PIT-safe P2 theme and relation alpha pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--dates")
        subparser.add_argument("--layers")
        subparser.add_argument("--scales")
        subparser.add_argument("--levels", default="B50,B35")
        subparser.add_argument("--horizons", default=",".join(DEFAULT_HORIZONS))
        subparser.add_argument("--workers", type=int, default=16)
        subparser.add_argument("--inner-workers", type=int, default=1)
        subparser.add_argument("--max-row-groups", type=int)
        subparser.add_argument("--skip-existing", action="store_true")

    build_returns = subparsers.add_parser("build-theme-returns")
    common(build_returns)
    build_returns.add_argument("--p1-root", required=True)
    build_returns.add_argument("--labels-root", required=True)
    build_returns.add_argument("--out-root", required=True)

    spillover = subparsers.add_parser("relation-spillover")
    common(spillover)
    spillover.add_argument("--p1-root", required=True)
    spillover.add_argument("--theme-returns-root", required=True)
    spillover.add_argument("--out-root", required=True)
    spillover.add_argument("--past-horizon", default="15m")
    spillover.add_argument("--tiers")

    for command in ("intraday-relation-features", "daily-relation-features"):
        feature_parser = subparsers.add_parser(command)
        common(feature_parser)
        feature_parser.add_argument("--signals-root", required=True)
        feature_parser.add_argument("--out-root", required=True)
        feature_parser.add_argument("--p1-root")
        feature_parser.add_argument("--late-minutes", type=int, default=60)
        feature_parser.add_argument("--underreaction-past-horizon", default="15m")

    for command in ("evaluate-intraday", "evaluate-daily"):
        evaluation = subparsers.add_parser(command)
        evaluation.add_argument("--features-root", required=True)
        evaluation.add_argument("--out-dir", required=True)

    args = parser.parse_args()
    dates = csvset(getattr(args, "dates", None))
    layers = csvset(getattr(args, "layers", None))
    scales = csvset(getattr(args, "scales", None))
    levels = csvset(getattr(args, "levels", None))
    horizons = csvlist(getattr(args, "horizons", None)) or DEFAULT_HORIZONS

    if args.command == "build-theme-returns":
        parts = discover(args.p1_root, "theme_memberships.parquet", dates, layers, scales)
        results = pool(parts, args.workers, build_theme_returns_one, args.labels_root, args.out_root, horizons, levels, args.skip_existing, args.max_row_groups, args.inner_workers)
        save_run_summary(args.out_root, results)
    elif args.command == "relation-spillover":
        parts = discover(args.p1_root, "theme_relation_edges.parquet", dates, layers, scales)
        results = pool(parts, args.workers, relation_spillover_one, args.theme_returns_root, args.out_root, horizons, args.past_horizon, levels, csvset(args.tiers), args.skip_existing, args.max_row_groups, args.inner_workers)
        save_run_summary(args.out_root, results)
    elif args.command in {"intraday-relation-features", "daily-relation-features"}:
        mode = "intraday" if args.command.startswith("intraday") else "daily"
        parts = discover(args.signals_root, "relation_spillover_signals.parquet", dates, layers, scales)
        if mode == "daily" and not args.p1_root:
            raise SystemExit("daily-relation-features requires --p1-root for temporal episode identity")
        results = pool(parts, args.workers, _build_feature_one, args.out_root, mode, args.underreaction_past_horizon, args.late_minutes, args.skip_existing, args.max_row_groups, args.p1_root)
        save_run_summary(args.out_root, results)
    elif args.command == "evaluate-intraday":
        results = evaluate_feature_root(args.features_root, args.out_dir, "intraday")
    else:
        results = evaluate_feature_root(args.features_root, args.out_dir, "daily")
    print(json.dumps({"command": args.command, "result": results}, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
