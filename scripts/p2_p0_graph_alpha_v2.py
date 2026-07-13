#!/usr/bin/env python3
"""Resource-safe P0 graph alpha CLI with streaming evaluation."""
from __future__ import annotations

import argparse
import json

from p2_alpha_pit_features import DEFAULT_INTRADAY_HORIZONS, PIT_CONTRACT_VERSION, csvlist, csvset
from p2_parallel_runtime import collect_process_map
from p2_p0_eval_streaming import evaluate_p0_streaming
from p2_p0_graph_alpha import discover, edge_spillover_one, graph_state_one, node_features_one


def pool(parts, workers, function, *args):
    if not parts:
        return []
    worker_count = max(1, min(int(workers), len(parts)))
    return collect_process_map(parts, worker_count, function, *args, max_in_flight=worker_count * 2, max_tasks_per_child=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("node-features", "edge-spillover", "graph-state"):
        sub = commands.add_parser(name)
        sub.add_argument("--p0-root", required=True)
        sub.add_argument("--labels-root")
        sub.add_argument("--out-root", required=True)
        sub.add_argument("--dates")
        sub.add_argument("--layers")
        sub.add_argument("--scales")
        sub.add_argument("--horizons", default=",".join(DEFAULT_INTRADAY_HORIZONS))
        sub.add_argument("--past-horizon", default="15m")
        sub.add_argument("--workers", type=int, default=16)
        sub.add_argument("--max-row-groups", type=int)
        sub.add_argument("--skip-existing", action="store_true")
    evaluate = commands.add_parser("eval-p0")
    evaluate.add_argument("--p0-alpha-root", required=True)
    evaluate.add_argument("--out-dir", required=True)
    evaluate.add_argument("--workers", type=int, default=12)
    evaluate.add_argument("--month")
    args = parser.parse_args()

    if args.command == "eval-p0":
        result = evaluate_p0_streaming(args.p0_alpha_root, args.out_dir, args.workers, args.month)
    else:
        dates, layers, scales = csvset(args.dates), csvset(args.layers), csvset(args.scales)
        horizons = [horizon for horizon in (csvlist(args.horizons) or DEFAULT_INTRADAY_HORIZONS) if horizon.endswith("m")]
        parts = discover(args.p0_root, dates, layers, scales)
        if args.command == "node-features":
            result = pool(parts, args.workers, node_features_one, args.labels_root, args.out_root, horizons, args.max_row_groups, args.skip_existing)
        elif args.command == "edge-spillover":
            result = pool(parts, args.workers, edge_spillover_one, args.labels_root, args.out_root, horizons, args.past_horizon, args.max_row_groups, args.skip_existing)
        else:
            result = pool(parts, args.workers, graph_state_one, args.out_root, args.max_row_groups, args.skip_existing)
    print(json.dumps({"pit_contract_version": PIT_CONTRACT_VERSION, "result": result}, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
