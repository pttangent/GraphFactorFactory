#!/usr/bin/env python3
"""Parallel, resumable runner for monthly Alpha falsification audits.

The production pipeline and all financial definitions remain untouched. Work is
partitioned by existing Parquet file. Each worker writes one atomic JSON state
checkpoint; completed partitions are reused after interruption.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

for _name in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "ARROW_NUM_THREADS", "POLARS_MAX_THREADS",
):
    os.environ.setdefault(_name, "1")

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import generate_monthly_alpha_report_with_risk as risk
from p2_parallel_runtime import bounded_process_map

PARALLEL_CONTRACT = "monthly-alpha-falsification-parallel-v1"
_CONFIG: dict[str, Any] = {}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(risk.jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _scope_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _checkpoint_path(root: Path, kind: str, path: Path) -> Path:
    token = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:24]
    return root / kind / f"part-{token}.json"


def _valid_checkpoint(path: Path, fingerprint: dict[str, Any], scope: str) -> dict[str, Any] | None:
    payload = _read_json(path)
    if (
        payload.get("contract") == PARALLEL_CONTRACT
        and payload.get("status") == "complete"
        and payload.get("source_fingerprint") == fingerprint
        and payload.get("scope_hash") == scope
        and isinstance(payload.get("records"), list)
    ):
        return payload
    return None


def _records(states: dict[tuple, dict[str, Any]], names: list[str]) -> list[dict[str, Any]]:
    output = []
    for key, state in states.items():
        row = dict(zip(names, key))
        row.update(state)
        output.append(risk.jsonable(row))
    return output


def _full_read(path: Path, columns: list[str], mode: str, max_full_bytes: int) -> bool:
    if mode == "full":
        return True
    if mode == "stream":
        return False
    return path.stat().st_size <= max_full_bytes


def _p2_groups(path: Path, columns: list[str], use_full: bool):
    if use_full:
        frame = pd.read_parquet(path, columns=columns)
        frame["decision_time"] = pd.to_datetime(frame["decision_time"], utc=True, errors="coerce")
        frame = frame.dropna(subset=["decision_time"])
        if not frame.empty and not frame["pit_audit_pass"].fillna(False).all():
            raise AssertionError(f"PIT failure: {path}")
        for (timestamp, level), group in frame.groupby(["decision_time", "level"], sort=False, dropna=False):
            yield timestamp, str(level), group
        return
    for timestamp, snapshot in risk.iter_time_groups(path, columns, time_column="decision_time"):
        if not snapshot["pit_audit_pass"].fillna(False).all():
            raise AssertionError(f"PIT failure: {path} {timestamp}")
        for level, group in snapshot.groupby("level", sort=False, dropna=False):
            yield timestamp, str(level), group


def _process_p2(task: dict[str, Any]) -> dict[str, Any]:
    source, checkpoint = Path(task["path"]), Path(task["checkpoint"])
    fingerprint = _fingerprint(source)
    cached = _valid_checkpoint(checkpoint, fingerprint, task["scope_hash"])
    if cached is not None:
        return {"status": "reused", "checkpoint": str(checkpoint), "records": cached["records"],
                "rows": int(cached.get("rows", 0)), "groups": int(cached.get("groups", 0))}
    started = time.time()
    states = defaultdict(risk.new_state)
    parquet = pq.ParquetFile(source)
    try:
        available = set(parquet.schema.names)
    finally:
        parquet.close()
    targets = sorted(set(task["under_targets"]) | set(task["corr_targets"]))
    columns = [column for column in [
        "decision_time", "level", "expected_pressure_z", "target_pre_response_z",
        "daily_underreaction_score", "daily_consensus_score", "pit_audit_pass", *targets,
    ] if column in available]
    required = {"decision_time", "level", "expected_pressure_z", "target_pre_response_z", "pit_audit_pass"}
    rows = groups = 0
    if required.issubset(columns):
        use_full = _full_read(source, columns, task["read_mode"], int(task["max_full_bytes"]))
        for _timestamp, level, group in _p2_groups(source, columns, use_full):
            groups += 1
            rows += len(group)
            for target in targets:
                if target not in group:
                    continue
                if target in task["under_targets"] and "daily_underreaction_score" in group:
                    values = group[["expected_pressure_z", "target_pre_response_z", "daily_underreaction_score", target]].replace([np.inf, -np.inf], np.nan).dropna()
                    if len(values) >= risk.MIN_SAMPLE:
                        a = values["expected_pressure_z"].to_numpy(float)
                        b = -values["target_pre_response_z"].to_numpy(float)
                        c = values["daily_underreaction_score"].to_numpy(float)
                        d = risk.residual(c, b)
                        y = values[target].to_numpy(float)
                        for signal, score in (("A_expected_pressure", a), ("B_simple_reversal", b),
                                              ("C_full_underreaction", c), ("D_residualized_on_reversal", d)):
                            count, ic, spread = risk.metric(score, y)
                            risk.update(states[("risk1", task["date"], task["layer"], task["scale"], level, target, signal)], count, ic, spread)
                if target in task["corr_targets"]:
                    needed = ["expected_pressure_z", "target_pre_response_z", target]
                    if "daily_consensus_score" in group:
                        needed.append("daily_consensus_score")
                    values = group[needed].replace([np.inf, -np.inf], np.nan).dropna()
                    if len(values) >= risk.MIN_SAMPLE:
                        own = values["target_pre_response_z"].to_numpy(float)
                        pressure = values["expected_pressure_z"].to_numpy(float)
                        y = values[target].to_numpy(float)
                        signals = [("own_past_return", own), ("network_pressure", pressure),
                                   ("network_pressure_residual", risk.residual(pressure, own))]
                        if "daily_consensus_score" in values:
                            consensus = values["daily_consensus_score"].to_numpy(float)
                            signals.extend((("network_consensus", consensus),
                                            ("network_consensus_residual", risk.residual(consensus, own))))
                        for signal, score in signals:
                            count, ic, spread = risk.metric(score, y)
                            risk.update(states[("risk2_p2", task["date"], task["layer"], task["scale"], level, target, signal)], count, ic, spread)
    records = _records(states, ["audit", "date", "layer_id", "scale", "level", "target", "signal"])
    payload = {"contract": PARALLEL_CONTRACT, "status": "complete", "source_fingerprint": fingerprint,
               "scope_hash": task["scope_hash"], "records": records, "rows": rows, "groups": groups,
               "elapsed_sec": round(time.time() - started, 3)}
    _atomic_json(checkpoint, payload)
    return {"status": "computed", "checkpoint": str(checkpoint), "records": records, "rows": rows, "groups": groups}


def _p0_groups(path: Path, columns: list[str], use_full: bool):
    if use_full:
        frame = pd.read_parquet(path, columns=columns)
        frame["decision_time"] = pd.to_datetime(frame["decision_time"], utc=True, errors="coerce")
        frame = frame.dropna(subset=["decision_time"])
        if "pit_audit_pass" in frame and not frame["pit_audit_pass"].fillna(False).all():
            raise AssertionError(f"P0 PIT failure: {path}")
        for timestamp, group in frame.groupby("decision_time", sort=False, dropna=False):
            yield timestamp, group
        return
    for timestamp, group in risk.iter_time_groups(path, columns, time_column="decision_time"):
        if "pit_audit_pass" in group and not group["pit_audit_pass"].fillna(False).all():
            raise AssertionError(f"P0 PIT failure: {path} {timestamp}")
        yield timestamp, group


def _process_p0(task: dict[str, Any]) -> dict[str, Any]:
    source, labels_file, checkpoint = Path(task["path"]), Path(task["labels"]), Path(task["checkpoint"])
    fingerprint = {"source": _fingerprint(source), "labels": _fingerprint(labels_file)}
    cached = _valid_checkpoint(checkpoint, fingerprint, task["scope_hash"])
    if cached is not None:
        return {"status": "reused", "checkpoint": str(checkpoint), "records": cached["records"],
                "rows": int(cached.get("rows", 0)), "groups": int(cached.get("groups", 0))}
    started = time.time()
    states = defaultdict(risk.new_state)
    horizons = sorted({str(item["target"]).replace("target_", "") for item in task["candidates"]} | {"15m"})
    labels = risk.load_labels(labels_file, horizons)
    if "past_label_15m" not in labels:
        raise ValueError(f"past_label_15m missing from {labels_file}")
    own = labels[["decision_time", "symbol_id", "past_label_15m"]].rename(
        columns={"symbol_id": "dst_id", "past_label_15m": "own_past_return"})
    parquet = pq.ParquetFile(source)
    try:
        available = set(parquet.schema.names)
    finally:
        parquet.close()
    columns = [column for column in ["decision_time", "dst_id", "pit_audit_pass",
               *[item["feature"] for item in task["candidates"]],
               *[item["target"] for item in task["candidates"]]] if column in available]
    use_full = _full_read(source, columns, task["read_mode"], int(task["max_full_bytes"]))
    rows = groups = 0
    for timestamp, snapshot in _p0_groups(source, columns, use_full):
        groups += 1
        rows += len(snapshot)
        label_slice = own.loc[own["decision_time"].eq(timestamp)]
        if label_slice.empty:
            continue
        merged = snapshot.merge(label_slice, on=["decision_time", "dst_id"], how="inner", validate="many_to_one")
        for item in task["candidates"]:
            feature, target = item["feature"], item["target"]
            if feature not in merged or target not in merged:
                continue
            values = merged[[feature, "own_past_return", target]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(values) < risk.MIN_SAMPLE:
                continue
            network = values[feature].to_numpy(float)
            own_return = values["own_past_return"].to_numpy(float)
            y = values[target].to_numpy(float)
            for signal, score in (("network_spillover", network), ("own_past_return", own_return),
                                  ("network_spillover_residual", risk.residual(network, own_return))):
                count, ic, spread = risk.metric(score, y)
                risk.update(states[(task["date"], task["layer"], task["scale"], feature, target, signal)], count, ic, spread)
    records = _records(states, ["date", "layer_id", "scale", "feature", "target", "signal"])
    payload = {"contract": PARALLEL_CONTRACT, "status": "complete", "source_fingerprint": fingerprint,
               "scope_hash": task["scope_hash"], "records": records, "rows": rows, "groups": groups,
               "elapsed_sec": round(time.time() - started, 3)}
    _atomic_json(checkpoint, payload)
    return {"status": "computed", "checkpoint": str(checkpoint), "records": records, "rows": rows, "groups": groups}


def _requested_workers(value: int) -> int:
    return max(1, int(value)) if int(value) > 0 else max(1, int(os.cpu_count() or 1))


def _effective_workers(tasks: list[dict[str, Any]], requested: int, config: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    if not tasks:
        return 1, {"requested_workers": requested, "effective_workers": 0}
    effective = min(requested, len(tasks))
    budget = float(config.get("memory_budget_gb", 0.0))
    estimate = 0.0
    if budget > 0:
        full_sizes = []
        threshold = int(config["max_full_bytes"])
        for task in tasks:
            size = Path(task["path"]).stat().st_size
            if config["read_mode"] == "full" or (config["read_mode"] == "auto" and size <= threshold):
                full_sizes.append(size / (1024 ** 3) * float(config["memory_expansion_factor"]))
        estimate = max(full_sizes, default=float(config["stream_worker_gb"]))
        estimate = max(estimate, float(config["stream_worker_gb"]))
        effective = min(effective, max(1, int(math.floor(budget / estimate))))
    return effective, {"requested_workers": requested, "effective_workers": effective,
                       "memory_budget_gb": budget, "estimated_worker_gb": round(estimate, 3)}


def _run_tasks(tasks: list[dict[str, Any]], worker, config: dict[str, Any], label: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    requested = _requested_workers(int(config["workers"]))
    workers, plan = _effective_workers(tasks, requested, config)
    results: list[dict[str, Any]] = []
    counts = defaultdict(int)
    started = time.time()
    iterator = bounded_process_map(tasks, workers, worker,
                                   max_in_flight=int(config["max_in_flight"]) or workers,
                                   max_tasks_per_child=config["tasks_per_child"])
    for index, result in enumerate(iterator, start=1):
        results.append(result)
        counts[result["status"]] += 1
        every = int(config["progress_every"])
        if every and (index % every == 0 or index == len(tasks)):
            print(f"[risk-audit/{label}] {index}/{len(tasks)} computed={counts['computed']} reused={counts['reused']} "
                  f"rows={sum(int(item.get('rows', 0)) for item in results):,} elapsed={time.time()-started:.1f}s", flush=True)
    stats = {**plan, "tasks": len(tasks), "computed": counts["computed"], "reused": counts["reused"],
             "rows": sum(int(item.get("rows", 0)) for item in results),
             "groups": sum(int(item.get("groups", 0)) for item in results),
             "elapsed_sec": round(time.time() - started, 3), "checkpoint_root": str(config["checkpoint_root"])}
    return results, stats


def _states_from_results(results: Iterable[dict[str, Any]], names: list[str]):
    states = defaultdict(risk.new_state)
    state_columns = ["snapshots", "sample_count", "ic_sum", "ic_count", "spread_sum", "spread_count", "positive"]
    for result in results:
        for row in result.get("records", []):
            key = tuple(row[name] for name in names)
            state = states[key]
            for column in state_columns:
                state[column] += row.get(column, 0)
    return states


def parallel_scan_p2(root: Path, month: str, under_scope, corr_scope, config: dict[str, Any]):
    checkpoint_root = Path(config["checkpoint_root"])
    tasks = []
    for path in sorted((root / "intraday_relation_features").glob(f"date={month}-*/layer_id=*/scale=*/intraday_relation_features.parquet"), key=lambda p: p.stat().st_size, reverse=True):
        date, layer, scale = risk.part(path, "date") or "", risk.part(path, "layer_id") or "", risk.part(path, "scale") or ""
        under_targets = sorted({target for item_layer, item_scale, target in under_scope if item_layer == layer and item_scale == scale})
        corr_targets = sorted({target for item_layer, item_scale, target in corr_scope if item_layer == layer and item_scale == scale})
        if not under_targets and not corr_targets:
            continue
        scope = _scope_hash({"under": under_targets, "corr": corr_targets, "min_sample": risk.MIN_SAMPLE})
        tasks.append({"path": str(path), "checkpoint": str(_checkpoint_path(checkpoint_root, "p2", path)),
                      "scope_hash": scope, "date": date, "layer": layer, "scale": scale,
                      "under_targets": under_targets, "corr_targets": corr_targets,
                      "read_mode": config["read_mode"], "max_full_bytes": config["max_full_bytes"]})
    results, stats = _run_tasks(tasks, _process_p2, config, "p2")
    states = _states_from_results(results, ["audit", "date", "layer_id", "scale", "level", "target", "signal"])
    summary = risk.summarize(states, ["audit", "date", "layer_id", "scale", "level", "target", "signal"])
    r1 = risk.pivot(summary[summary.audit.eq("risk1")], ["layer_id", "scale", "level", "target"]) if not summary.empty else pd.DataFrame()
    r2 = risk.pivot(summary[summary.audit.eq("risk2_p2")], ["layer_id", "scale", "level", "target"]) if not summary.empty else pd.DataFrame()
    if not r1.empty:
        c = r1.get("mean_rank_ic__C_full_underreaction", pd.Series(np.nan, index=r1.index))
        b = r1.get("mean_rank_ic__B_simple_reversal", pd.Series(np.nan, index=r1.index))
        d = r1.get("mean_rank_ic__D_residualized_on_reversal", pd.Series(np.nan, index=r1.index))
        r1["c_minus_b_abs_ic"] = c.abs() - b.abs()
        r1["residual_ic_retention"] = d.abs() / c.abs().replace(0, np.nan)
        r1["risk_status"] = [risk.classify(x, y, z) for x, y, z in zip(c, b, d)]
    if not r2.empty:
        own = r2.get("mean_rank_ic__own_past_return", pd.Series(np.nan, index=r2.index))
        for family in ("pressure", "consensus"):
            network = r2.get(f"mean_rank_ic__network_{family}", pd.Series(np.nan, index=r2.index))
            residual = r2.get(f"mean_rank_ic__network_{family}_residual", pd.Series(np.nan, index=r2.index))
            r2[f"{family}_residual_ic_retention"] = residual.abs() / network.abs().replace(0, np.nan)
            r2[f"{family}_risk_status"] = [risk.classify(x, y, z, True) for x, y, z in zip(network, own, residual)]
    return r1, r2, stats


def parallel_scan_p0(root: Path, labels_root: Path | None, month: str, scope, config: dict[str, Any]):
    if not scope:
        return pd.DataFrame(), {"status": "no_candidates", "tasks": 0}
    if labels_root is None:
        return pd.DataFrame(), {"status": "labels_root_missing", "tasks": 0}
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for date_dir in sorted((root / "p0_edge_spillover").glob(f"date={month}-*")):
        date = risk.part(date_dir, "date") or date_dir.name.split("=", 1)[-1]
        labels = labels_root / f"date={date}" / "labels.parquet"
        if not labels.exists():
            continue
        for layer, scale, feature, target in scope:
            path = date_dir / f"layer_id={layer}" / f"scale={scale}" / "p0_edge_spillover_features.parquet"
            if path.exists():
                grouped[(str(path), str(labels), date, layer, scale)].append({"feature": feature, "target": target})
    tasks = []
    checkpoint_root = Path(config["checkpoint_root"])
    for (path_value, labels_value, date, layer, scale), items in grouped.items():
        path = Path(path_value)
        scope_hash = _scope_hash({"items": sorted(items, key=lambda item: (item["feature"], item["target"])),
                                  "past_horizon": "15m", "min_sample": risk.MIN_SAMPLE})
        tasks.append({"path": path_value, "labels": labels_value,
                      "checkpoint": str(_checkpoint_path(checkpoint_root, "p0", path)),
                      "scope_hash": scope_hash, "date": date, "layer": layer, "scale": scale,
                      "candidates": items, "read_mode": config["read_mode"],
                      "max_full_bytes": config["max_full_bytes"]})
    tasks.sort(key=lambda task: Path(task["path"]).stat().st_size, reverse=True)
    results, stats = _run_tasks(tasks, _process_p0, config, "p0")
    states = _states_from_results(results, ["date", "layer_id", "scale", "feature", "target", "signal"])
    summary = risk.summarize(states, ["date", "layer_id", "scale", "feature", "target", "signal"])
    output = risk.pivot(summary, ["layer_id", "scale", "feature", "target"])
    if not output.empty:
        network = output.get("mean_rank_ic__network_spillover", pd.Series(np.nan, index=output.index))
        own = output.get("mean_rank_ic__own_past_return", pd.Series(np.nan, index=output.index))
        residual = output.get("mean_rank_ic__network_spillover_residual", pd.Series(np.nan, index=output.index))
        output["residual_ic_retention"] = residual.abs() / network.abs().replace(0, np.nan)
        output["risk_status"] = [risk.classify(x, y, z, True) for x, y, z in zip(network, own, residual)]
    return output, stats


def _scan_p2_entry(root: Path, month: str, under_scope, corr_scope, progress: int = 25):
    return parallel_scan_p2(Path(root), month, under_scope, corr_scope, _CONFIG)


def _scan_p0_entry(root: Path, labels_root: Path | None, month: str, scope):
    return parallel_scan_p0(Path(root), Path(labels_root) if labels_root else None, month, scope, _CONFIG)


def configure(*, p2_root: str | Path, month: str, workers: int, tasks_per_child: int | None,
              checkpoint_root: str | Path | None, read_mode: str, max_full_read_gb: float,
              memory_budget_gb: float, memory_expansion_factor: float, stream_worker_gb: float,
              max_in_flight: int, progress_every: int, reset_checkpoints: bool = False) -> dict[str, Any]:
    root = Path(checkpoint_root) if checkpoint_root else Path(p2_root) / ".risk_audit_checkpoints" / month.replace("-", "")
    if reset_checkpoints:
        shutil.rmtree(root, ignore_errors=True)
    global _CONFIG
    _CONFIG = {"workers": workers, "tasks_per_child": tasks_per_child, "checkpoint_root": root,
               "read_mode": read_mode, "max_full_bytes": int(max_full_read_gb * 1024 ** 3),
               "memory_budget_gb": memory_budget_gb, "memory_expansion_factor": memory_expansion_factor,
               "stream_worker_gb": stream_worker_gb, "max_in_flight": max_in_flight,
               "progress_every": progress_every}
    risk.scan_p2 = _scan_p2_entry
    risk.scan_p0 = _scan_p0_entry
    return dict(_CONFIG)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel resumable monthly Alpha falsification report")
    parser.add_argument("--p2-root", required=True)
    parser.add_argument("--month", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--labels-root")
    parser.add_argument("--p1-root")
    parser.add_argument("--primary-level", default="B50")
    parser.add_argument("--replication-level", default="B35")
    parser.add_argument("--counterfactual-results")
    parser.add_argument("--batch-size", type=int, default=250_000)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--json-top-n", type=int, default=200)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--enrich-existing", action="store_true")
    parser.add_argument("--workers", type=int, default=0, help="0 uses logical CPU count; pass 24 for a 24-core host")
    parser.add_argument("--tasks-per-child", type=int)
    parser.add_argument("--checkpoint-root")
    parser.add_argument("--reset-checkpoints", action="store_true")
    parser.add_argument("--read-mode", choices=["auto", "full", "stream"], default="auto")
    parser.add_argument("--max-full-read-gb", type=float, default=1.0)
    parser.add_argument("--memory-budget-gb", type=float, default=0.0)
    parser.add_argument("--memory-expansion-factor", type=float, default=6.0)
    parser.add_argument("--stream-worker-gb", type=float, default=1.0)
    parser.add_argument("--max-in-flight", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()
    configure(p2_root=args.p2_root, month=args.month, workers=args.workers,
              tasks_per_child=args.tasks_per_child, checkpoint_root=args.checkpoint_root,
              read_mode=args.read_mode, max_full_read_gb=args.max_full_read_gb,
              memory_budget_gb=args.memory_budget_gb,
              memory_expansion_factor=args.memory_expansion_factor,
              stream_worker_gb=args.stream_worker_gb, max_in_flight=args.max_in_flight,
              progress_every=args.progress_every, reset_checkpoints=args.reset_checkpoints)
    kwargs = dict(report_dir=args.output_dir, labels_root=args.labels_root, p1_root=args.p1_root,
                  primary=args.primary_level, replica=args.replication_level,
                  counterfactual=args.counterfactual_results, progress=args.progress_every)
    if args.enrich_existing:
        risk.enrich(args.p2_root, args.month, **kwargs)
    else:
        risk.generate(args.p2_root, args.month, batch_size=args.batch_size, top_n=args.top_n,
                      json_top_n=args.json_top_n, allow_partial=args.allow_partial, **kwargs)


if __name__ == "__main__":
    main()
