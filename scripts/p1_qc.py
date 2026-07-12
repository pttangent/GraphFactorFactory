#!/usr/bin/env python3
"""QC P1 B50/B35 partition outputs.

This checker is intentionally schema-aware for the current
build_b50_b35_theme_forest.py output.  The key production invariant is:

    B50 leaf max <= 50 and B35 leaf max <= 35

The earlier draft looked for generic columns named level/leaf_count in the
summary table.  The real summary columns are b50_leaf_max and b35_leaf_max,
so this script checks those first and falls back to recomputing leaf sizes
from theme_memberships.parquet when needed.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOG = logging.getLogger("p1_qc")

REQUIRED_FILES = (
    "manifest.json",
    "theme_memberships.parquet",
    "theme_relation_edges.parquet",
    "p1_b50_b35_summary.parquet",
)


def _safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_manifest_read_error": repr(exc)}


def _check_summary_schema(summary_path: Path, errors: list[str], warnings: list[str]) -> None:
    try:
        summary = pd.read_parquet(summary_path)
    except Exception as exc:
        errors.append(f"Failed to read p1_b50_b35_summary.parquet: {exc}")
        return

    if summary.empty:
        errors.append("p1_b50_b35_summary.parquet is empty")
        return

    # Current builder schema: one row per decision_time/layer_id/scale group.
    if "b50_leaf_max" in summary.columns:
        b50_max = pd.to_numeric(summary["b50_leaf_max"], errors="coerce").max()
        if pd.notna(b50_max) and float(b50_max) > 50:
            errors.append(f"B50 leaf size exceeded 50 (max={b50_max})")
    else:
        warnings.append("summary missing b50_leaf_max; will rely on membership fallback")

    if "b35_leaf_max" in summary.columns:
        b35_max = pd.to_numeric(summary["b35_leaf_max"], errors="coerce").max()
        if pd.notna(b35_max) and float(b35_max) > 35:
            errors.append(f"B35 leaf size exceeded 35 (max={b35_max})")
    else:
        warnings.append("summary missing b35_leaf_max; will rely on membership fallback")

    for col in ("b50_leaf_count", "b35_leaf_count"):
        if col in summary.columns:
            total = pd.to_numeric(summary[col], errors="coerce").fillna(0).sum()
            if total <= 0:
                errors.append(f"summary has non-positive total {col}")
        else:
            warnings.append(f"summary missing {col}")


def _check_membership_fallback(mem_path: Path, errors: list[str], warnings: list[str]) -> None:
    """Recompute leaf sizes from memberships to avoid false QC passes."""
    try:
        mem = pd.read_parquet(mem_path, columns=["theme_id", "level", "member_id"])
    except Exception as exc:
        errors.append(f"Failed to read theme_memberships.parquet: {exc}")
        return

    if mem.empty:
        errors.append("theme_memberships.parquet is empty")
        return

    if not {"theme_id", "level", "member_id"}.issubset(mem.columns):
        errors.append(f"theme_memberships schema missing required columns; columns={list(mem.columns)}")
        return

    levels = set(mem["level"].astype(str).unique())
    if "B50" not in levels:
        errors.append("theme_memberships missing B50 rows")
    if "B35" not in levels:
        errors.append("theme_memberships missing B35 rows")

    grouped = mem.groupby(["level", "theme_id"], observed=True)["member_id"].nunique()
    if ("B50" in grouped.index.get_level_values(0)):
        b50_max = int(grouped.loc["B50"].max())
        if b50_max > 50:
            errors.append(f"B50 membership leaf size exceeded 50 (max={b50_max})")
    if ("B35" in grouped.index.get_level_values(0)):
        b35_max = int(grouped.loc["B35"].max())
        if b35_max > 35:
            errors.append(f"B35 membership leaf size exceeded 35 (max={b35_max})")

    duplicate_count = int(mem.duplicated(["level", "theme_id", "member_id"]).sum())
    if duplicate_count > 0:
        warnings.append(f"theme_memberships has {duplicate_count} duplicate level/theme/member rows")


def check_partition(part_dir: Path, root: Path, allow_empty_relations: bool = False) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    for name in REQUIRED_FILES:
        p = part_dir / name
        if not p.exists():
            errors.append(f"Missing {name}")
        elif name.endswith(".parquet") and p.stat().st_size <= 100:
            errors.append(f"Empty or tiny {name}")

    manifest_path = part_dir / "manifest.json"
    if manifest_path.exists():
        manifest = _read_manifest(manifest_path)
        if manifest.get("_manifest_read_error"):
            errors.append(f"Failed to read manifest.json: {manifest['_manifest_read_error']}")
        else:
            groups = int(manifest.get("groups", 0) or 0)
            if groups <= 0:
                errors.append("manifest groups <= 0")
            for field in ("memberships", "theme_nodes"):
                val = int(manifest.get(field, 0) or 0)
                if val <= 0:
                    errors.append(f"manifest {field} <= 0")
            relation_edges = int(manifest.get("relation_edges", 0) or 0)
            if relation_edges <= 0 and not allow_empty_relations:
                errors.append("manifest relation_edges <= 0")

    summary_path = part_dir / "p1_b50_b35_summary.parquet"
    if summary_path.exists():
        _check_summary_schema(summary_path, errors, warnings)

    mem_path = part_dir / "theme_memberships.parquet"
    if mem_path.exists() and mem_path.stat().st_size > 100:
        _check_membership_fallback(mem_path, errors, warnings)

    rel_path = part_dir / "theme_relation_edges.parquet"
    if rel_path.exists() and rel_path.stat().st_size > 100:
        try:
            rel = pd.read_parquet(rel_path, columns=["level", "src_theme_id", "dst_theme_id"])
            if rel.empty and not allow_empty_relations:
                errors.append("theme_relation_edges.parquet is empty")
        except Exception as exc:
            errors.append(f"Failed to read theme_relation_edges.parquet: {exc}")

    return {
        "partition": _safe_rel(part_dir, root),
        "status": "failed" if errors else "passed",
        "errors": errors,
        "warnings": warnings,
    }


def discover_partitions(p1_root: Path) -> list[Path]:
    return sorted(d for d in p1_root.rglob("scale=*") if d.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser(description="QC partitioned P1 B50/B35 outputs.")
    parser.add_argument("--p1-root", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--out-report", default=None, help="Optional JSON QC report path.")
    parser.add_argument("--require-run-manifest", action="store_true", help="Fail if run_p1_parallel_manifest.json is missing.")
    parser.add_argument("--allow-empty-relations", action="store_true", help="Allow sparse partitions with zero relation edges.")
    args = parser.parse_args()

    p1_root = Path(args.p1_root)
    if not p1_root.exists():
        raise SystemExit(f"p1-root does not exist: {p1_root}")

    run_manifest_path = p1_root / "run_p1_parallel_manifest.json"
    run_manifest: dict[str, Any] | None = None
    top_errors: list[str] = []
    if run_manifest_path.exists():
        run_manifest = _read_manifest(run_manifest_path)
        if run_manifest.get("_manifest_read_error"):
            top_errors.append(f"Failed to read run_p1_parallel_manifest.json: {run_manifest['_manifest_read_error']}")
        elif int(run_manifest.get("tasks_failed", 0) or 0) > 0 or run_manifest.get("failed"):
            top_errors.append("run_p1_parallel_manifest.json indicates failed tasks")
        else:
            LOG.info("Main runner manifest passed: 0 failed tasks.")
    elif args.require_run_manifest:
        top_errors.append(f"Missing {run_manifest_path}")
    else:
        LOG.warning("run_p1_parallel_manifest.json not found; continuing partition-level QC only.")

    partitions = discover_partitions(p1_root)
    LOG.info("Found %d partition directories to QC.", len(partitions))
    if not partitions:
        raise SystemExit("No date/layer_id/scale partitions found under p1-root.")

    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(check_partition, p, p1_root, args.allow_empty_relations): p
            for p in partitions
        }
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            results.append(fut.result())
            if i % 500 == 0:
                LOG.info("QC progress: %d / %d", i, len(partitions))

    failed = [r for r in results if r["errors"]]
    warning_count = sum(len(r["warnings"]) for r in results)
    report = {
        "status": "failed" if top_errors or failed else "passed",
        "top_errors": top_errors,
        "p1_root": str(p1_root),
        "partitions_total": len(partitions),
        "partitions_failed": len(failed),
        "warnings_total": warning_count,
        "run_manifest": run_manifest,
        "failed_examples": failed[:50],
    }

    out_report = Path(args.out_report) if args.out_report else p1_root / "p1_qc_report.json"
    out_report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    LOG.info("Wrote QC report to %s", out_report)

    if top_errors or failed:
        LOG.error("P1 QC FAILED: top_errors=%d, failed_partitions=%d", len(top_errors), len(failed))
        for item in failed[:10]:
            LOG.error("  [%s] %s", item["partition"], "; ".join(item["errors"]))
        raise SystemExit(1)

    LOG.info("P1 QC PASSED: %d partitions, warnings=%d", len(partitions), warning_count)


if __name__ == "__main__":
    main()
