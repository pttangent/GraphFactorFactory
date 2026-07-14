#!/usr/bin/env python3
"""Candidate-granular P0 extension for the resumable parallel risk audit.

P2 remains one task per partition. P0 uses one task per existing
``date × layer × scale × feature × target`` candidate, preventing duplicate
column requests and providing finer checkpoint reuse.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import generate_monthly_alpha_report_with_risk_parallel as base

risk = base.risk
configure = base.configure
_effective_workers = base._effective_workers


def parallel_scan_p0(root: Path, labels_root: Path | None, month: str, scope, config: dict):
    if not scope:
        return pd.DataFrame(), {"status": "no_candidates", "tasks": 0}
    if labels_root is None:
        return pd.DataFrame(), {"status": "labels_root_missing", "tasks": 0}

    tasks = []
    checkpoint_root = Path(config["checkpoint_root"])
    for date_dir in sorted((root / "p0_edge_spillover").glob(f"date={month}-*")):
        date = risk.part(date_dir, "date") or date_dir.name.split("=", 1)[-1]
        labels = labels_root / f"date={date}" / "labels.parquet"
        if not labels.exists():
            continue
        for layer, scale, feature, target in sorted(scope):
            path = date_dir / f"layer_id={layer}" / f"scale={scale}" / "p0_edge_spillover_features.parquet"
            if not path.exists():
                continue
            item = {"feature": feature, "target": target}
            scope_hash = base._scope_hash({"item": item, "past_horizon": "15m", "min_sample": risk.MIN_SAMPLE})
            kind = f"p0/{feature}/{target}"
            tasks.append({
                "path": str(path),
                "labels": str(labels),
                "checkpoint": str(base._checkpoint_path(checkpoint_root, kind, path)),
                "scope_hash": scope_hash,
                "date": date,
                "layer": layer,
                "scale": scale,
                "candidates": [item],
                "read_mode": config["read_mode"],
                "max_full_bytes": config["max_full_bytes"],
            })

    tasks.sort(key=lambda task: Path(task["path"]).stat().st_size, reverse=True)
    results, stats = base._run_tasks(tasks, base._process_p0, config, "p0")
    states = base._states_from_results(results, ["date", "layer_id", "scale", "feature", "target", "signal"])
    summary = risk.summarize(states, ["date", "layer_id", "scale", "feature", "target", "signal"])
    output = risk.pivot(summary, ["layer_id", "scale", "feature", "target"])
    if not output.empty:
        network = output.get("mean_rank_ic__network_spillover", pd.Series(np.nan, index=output.index))
        own = output.get("mean_rank_ic__own_past_return", pd.Series(np.nan, index=output.index))
        residual = output.get("mean_rank_ic__network_spillover_residual", pd.Series(np.nan, index=output.index))
        output["residual_ic_retention"] = residual.abs() / network.abs().replace(0, np.nan)
        output["risk_status"] = [risk.classify(x, y, z, True) for x, y, z in zip(network, own, residual)]
    return output, stats


def _scan_p0_entry(root: Path, labels_root: Path | None, month: str, scope):
    return parallel_scan_p0(Path(root), Path(labels_root) if labels_root else None, month, scope, base._CONFIG)


base.parallel_scan_p0 = parallel_scan_p0
base._scan_p0_entry = _scan_p0_entry


def main() -> None:
    base.main()


if __name__ == "__main__":
    main()
