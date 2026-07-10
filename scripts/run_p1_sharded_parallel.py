#!/usr/bin/env python3
"""Run shard-local P1 builders with multi-worker scheduling."""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
from pathlib import Path


def parse_csv(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {x.strip() for x in value.split(",") if x.strip()}


def discover_shards(root: Path, dates: set[str] | None, layers: set[str] | None, scales: set[str] | None) -> list[Path]:
    shards = sorted(root.rglob("edges.parquet"))
    out: list[Path] = []
    for p in shards:
        parts = {part.split("=", 1)[0]: part.split("=", 1)[1] for part in p.parts if "=" in part}
        if dates is not None and parts.get("date") not in dates:
            continue
        if layers is not None and parts.get("layer_id") not in layers:
            continue
        if scales is not None and parts.get("scale") not in scales:
            continue
        out.append(p)
    out.sort(key=lambda x: x.stat().st_size, reverse=True)
    return out


def output_dir_for(shard: Path, shard_root: Path, out_root: Path) -> Path:
    rel = shard.parent.relative_to(shard_root)
    return out_root / rel


def run_one(
    builder: Path,
    shard: Path,
    shard_root: Path,
    out_root: Path,
    output_format: str,
    max_groups: int | None,
    skip_existing: bool,
) -> dict:
    out_dir = output_dir_for(shard, shard_root, out_root)
    manifest = out_dir / "manifest.json"
    if skip_existing and manifest.exists():
        return {"status": "skipped", "shard": str(shard), "out_dir": str(out_dir)}
    out_dir.mkdir(parents=True, exist_ok=True)

    # The existing production builder accepts a single parquet path.  Once P0 is
    # physically sharded, the builder no longer reads a whole dense date.
    cmd = [
        sys.executable,
        str(builder),
        "--p0-edges",
        str(shard),
        "--out-dir",
        str(out_dir),
        "--output-format",
        output_format,
    ]
    if max_groups is not None:
        cmd += ["--max-groups", str(max_groups)]

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    proc = subprocess.run(cmd, text=True, capture_output=True, env=env)
    log = out_dir / "run.log"
    log.write_text((proc.stdout or "") + "\nSTDERR:\n" + (proc.stderr or ""), encoding="utf-8")
    if proc.returncode != 0:
        return {"status": "failed", "returncode": proc.returncode, "shard": str(shard), "out_dir": str(out_dir), "log": str(log)}
    return {"status": "done", "shard": str(shard), "out_dir": str(out_dir), "log": str(log)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Run B50/B35 P1 at date/layer/scale shard granularity.")
    ap.add_argument("--shard-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--builder", default="scripts/build_b50_b35_theme_forest.py")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dates", default=None)
    ap.add_argument("--layers", default=None)
    ap.add_argument("--scales", default=None)
    ap.add_argument("--max-shards", type=int, default=None)
    ap.add_argument("--max-groups", type=int, default=None, help="Smoke-test cap passed to each shard builder")
    ap.add_argument("--output-format", choices=["parquet", "csv"], default="parquet")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    shard_root = Path(args.shard_root)
    out_root = Path(args.out_root)
    builder = Path(args.builder)
    shards = discover_shards(shard_root, parse_csv(args.dates), parse_csv(args.layers), parse_csv(args.scales))
    if args.max_shards is not None:
        shards = shards[: args.max_shards]
    if not shards:
        raise FileNotFoundError(f"no shards found under {shard_root}")
    out_root.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"shards": len(shards), "workers": args.workers, "largest_shard_mb": round(shards[0].stat().st_size / 1024 / 1024, 2)}, indent=2), flush=True)

    results: list[dict] = []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [
            ex.submit(run_one, builder, s, shard_root, out_root, args.output_format, args.max_groups, args.skip_existing)
            for s in shards
        ]
        for fut in cf.as_completed(futs):
            res = fut.result()
            results.append(res)
            print(json.dumps(res, default=str), flush=True)

    summary = {
        "total": len(results),
        "done": sum(r["status"] == "done" for r in results),
        "skipped": sum(r["status"] == "skipped" for r in results),
        "failed": sum(r["status"] == "failed" for r in results),
        "results": results,
    }
    (out_root / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if summary["failed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
