#!/usr/bin/env python3
"""RAM-aware shard-local P1 scheduler with bounded child processes."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from p2_parallel_runtime import bounded_thread_map, run_process_tree

P1_CONTRACT_VERSION = "p1-streaming-v2"
P2_MEASURED_GB_PER_PROCESS = 3.8


def parse_csv(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def discover_shards(root: Path, dates: set[str] | None, layers: set[str] | None, scales: set[str] | None) -> list[Path]:
    shards = sorted(root.rglob("edges.parquet"))
    output: list[Path] = []
    for path in shards:
        parts = {part.split("=", 1)[0]: part.split("=", 1)[1] for part in path.parts if "=" in part}
        if dates is not None and parts.get("date") not in dates:
            continue
        if layers is not None and parts.get("layer_id") not in layers:
            continue
        if scales is not None and parts.get("scale") not in scales:
            continue
        output.append(path)
    output.sort(key=lambda path: path.stat().st_size, reverse=True)
    return output


def output_dir_for(shard: Path, shard_root: Path, out_root: Path) -> Path:
    return out_root / shard.parent.relative_to(shard_root)


def manifest_complete(out_dir: Path, output_format: str) -> bool:
    manifest_path = out_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if manifest.get("p1_contract_version") != P1_CONTRACT_VERSION or manifest.get("status") not in {"complete", "empty"}:
        return False
    if manifest.get("status") == "empty":
        return True
    suffix = ".csv" if output_format == "csv" else ".parquet"
    rows = manifest.get("rows", {})
    required = ["theme_nodes", "theme_memberships", "summary"]
    optional = ["theme_tree_edges", "theme_relation_edges", "temporal_theme_edges"]
    if not all((out_dir / f"{name}{suffix}").exists() for name in required if int(rows.get(name, 0)) > 0):
        return False
    return all((out_dir / f"{name}{suffix}").exists() for name in optional if int(rows.get(name, 0)) > 0)


def run_one(
    shard: Path,
    builder: Path,
    shard_root: Path,
    out_root: Path,
    output_format: str,
    max_snapshots: int | None,
    skip_existing: bool,
) -> dict:
    out_dir = output_dir_for(shard, shard_root, out_root)
    if skip_existing and manifest_complete(out_dir, output_format):
        return {"status": "skipped", "shard": str(shard), "out_dir": str(out_dir)}
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(builder),
        "--p0-edges-shard", str(shard),
        "--out-dir", str(out_dir),
        "--output-format", output_format,
    ]
    if max_snapshots is not None:
        command += ["--max-snapshots", str(max_snapshots)]
    environment = os.environ.copy()
    environment.update({
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "ARROW_NUM_THREADS": "1",
        "POLARS_MAX_THREADS": "1",
    })
    return_code = run_process_tree(command, env=environment)
    result = {
        "status": "done" if return_code == 0 and manifest_complete(out_dir, output_format) else "failed",
        "returncode": return_code,
        "shard": str(shard),
        "out_dir": str(out_dir),
    }
    (out_dir / "run_status.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def resolved_workers(requested: int, cores: int, ram_gb: float, reserve_ram_gb: float, gb_per_worker: float) -> int:
    schedulable = max(1.0, (ram_gb - reserve_ram_gb) * 0.90)
    ram_cap = max(1, int(schedulable // gb_per_worker))
    target = requested if requested > 0 else cores
    return max(1, min(target, cores, ram_cap))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run B50/B35 P1 at date/layer/scale shard granularity.")
    parser.add_argument("--shard-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--builder", default="scripts/build_b50_b35_theme_forest_streaming_v2.py")
    parser.add_argument("--workers", type=int, default=0, help="0=derive from CPU and RAM budget")
    parser.add_argument("--cores", type=int, default=24)
    parser.add_argument("--ram-gb", type=float, default=128.0)
    parser.add_argument("--reserve-ram-gb", type=float, default=24.0)
    parser.add_argument(
        "--gb-per-worker",
        type=float,
        default=P2_MEASURED_GB_PER_PROCESS,
        help="Defaults to the ce6e2f3 24-process scheduler memory standard.",
    )
    parser.add_argument("--dates")
    parser.add_argument("--layers")
    parser.add_argument("--scales")
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--max-snapshots", type=int)
    parser.add_argument("--output-format", choices=["parquet", "csv"], default="parquet")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    shard_root = Path(args.shard_root)
    out_root = Path(args.out_root)
    builder = Path(args.builder)
    shards = discover_shards(shard_root, parse_csv(args.dates), parse_csv(args.layers), parse_csv(args.scales))
    if args.max_shards is not None:
        shards = shards[:args.max_shards]
    if not shards:
        raise FileNotFoundError(f"no shards found under {shard_root}")
    workers = resolved_workers(args.workers, args.cores, args.ram_gb, args.reserve_ram_gb, args.gb_per_worker)
    out_root.mkdir(parents=True, exist_ok=True)
    plan = {
        "p1_contract_version": P1_CONTRACT_VERSION,
        "memory_reference": "ce6e2f3_p2_scheduler_24_processes_healthy_60gb_observed",
        "shards": len(shards),
        "workers": workers,
        "cores": args.cores,
        "ram_gb": args.ram_gb,
        "reserve_ram_gb": args.reserve_ram_gb,
        "schedulable_ram_gb": (args.ram_gb - args.reserve_ram_gb) * 0.90,
        "gb_per_worker": args.gb_per_worker,
        "estimated_peak_ram_gb": workers * args.gb_per_worker,
        "largest_shard_mb": round(shards[0].stat().st_size / 1024 / 1024, 2),
        "input_mode": "streamed_snapshot_groups",
        "maximum_shards_in_flight": workers,
    }
    (out_root / "p1_schedule_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(json.dumps(plan, indent=2), flush=True)

    def execute(shard: Path) -> dict:
        return run_one(shard, builder, shard_root, out_root, args.output_format, args.max_snapshots, args.skip_existing)

    results: list[dict] = []
    for result in bounded_thread_map(shards, workers, execute, max_in_flight=workers):
        results.append(result)
        print(json.dumps(result, default=str), flush=True)
    summary = {
        "p1_contract_version": P1_CONTRACT_VERSION,
        "total": len(results),
        "done": sum(result["status"] == "done" for result in results),
        "skipped": sum(result["status"] == "skipped" for result in results),
        "failed": sum(result["status"] == "failed" for result in results),
        "results": results,
    }
    (out_root / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if summary["failed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
