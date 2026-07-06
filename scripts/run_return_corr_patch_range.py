from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path

from graphfactorfactory.application.return_corr_patch import RETURN_CORR_LAYER_IDS, ReturnCorrPatchPipeline
from graphfactorfactory.domain.config import BuildConfig
from graphfactorfactory.infrastructure.nodefactorfactory import ParquetNodeFactorSource


def _dates_from_store(root: Path, start: str | None, end: str | None) -> list[str]:
    dates = sorted(path.name.split("=", 1)[1] for path in (root / "canonical").glob("date=*") if path.is_dir())
    if start:
        dates = [date for date in dates if date >= start]
    if end:
        dates = [date for date in dates if date <= end]
    return dates


def _success_marker(output_root: Path, trade_date: str) -> Path:
    return output_root / "canonical" / f"date={trade_date}" / "_SUCCESS_RETURNCORR_PATCH.json"


def _run_one(args_tuple):
    node_factors, source_store, output_store, config_path, trade_date, max_workers = args_tuple
    output_root = Path(output_store).expanduser().resolve()
    marker = _success_marker(output_root, trade_date)
    if marker.exists():
        payload = json.loads(marker.read_text())
        payload["status"] = "skipped_complete"
        return payload
    config = BuildConfig.from_yaml(config_path)
    source = ParquetNodeFactorSource(node_factors)
    payload = ReturnCorrPatchPipeline(
        source,
        source_store,
        output_store,
        config,
        max_workers=max_workers,
    ).build_date(trade_date)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(payload, indent=2, default=str))
    payload["status"] = "completed"
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume-safe ReturnCorr-only Phase 0 patch over a date range")
    parser.add_argument("--node-factors", required=True)
    parser.add_argument("--source-graph-store", required=True)
    parser.add_argument("--output-graph-store", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--start", default="2026-01-02")
    parser.add_argument("--end", default="2026-02-27")
    parser.add_argument("--max-workers", type=int, default=1, help="Snapshot workers inside each date")
    parser.add_argument("--date-workers", type=int, default=1, help="Parallel dates; keep max-workers=1 when >1")
    parser.add_argument("--force", action="store_true", help="Delete success markers and rerun selected dates")
    args = parser.parse_args()

    source_root = Path(args.source_graph_store).expanduser().resolve()
    output_root = Path(args.output_graph_store).expanduser().resolve()
    if source_root == output_root:
        raise SystemExit("source and output graph stores must differ")
    dates = _dates_from_store(source_root, args.start, args.end)
    if not dates:
        raise SystemExit("No baseline dates found in requested range")
    if args.date_workers > 1 and args.max_workers > 1:
        raise SystemExit("Avoid nested process pools: use either date-workers > 1 or max-workers > 1, not both")
    if args.force:
        for date in dates:
            _success_marker(output_root, date).unlink(missing_ok=True)

    tasks = [
        (args.node_factors, str(source_root), str(output_root), args.config, date, args.max_workers)
        for date in dates
    ]
    results = []
    if args.date_workers == 1:
        for task in tasks:
            result = _run_one(task)
            results.append(result)
            print(json.dumps({"date": result["date"], "status": result["status"]}), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.date_workers) as pool:
            futures = {pool.submit(_run_one, task): task[4] for task in tasks}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(json.dumps({"date": result["date"], "status": result["status"]}), flush=True)

    completed = sorted(result["date"] for result in results if result["status"] in {"completed", "skipped_complete"})
    summary = {
        "source_graph_store": str(source_root),
        "output_graph_store": str(output_root),
        "date_count": len(dates),
        "completed_dates": completed,
        "patched_layer_ids": list(RETURN_CORR_LAYER_IDS),
        "multiplex_layer_0": "stale_disabled_until_full_rebuild",
    }
    summary_path = output_root / "return_corr_patch_run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
