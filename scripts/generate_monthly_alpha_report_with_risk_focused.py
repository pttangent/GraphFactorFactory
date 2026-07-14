#!/usr/bin/env python3
"""Focused B50-only falsification audit.

This is the default economical audit for the frozen January hypotheses. It does
not recalculate Daily Alpha and does not scan the full discovery search space.
Only the six pre-declared intraday mechanism families and the four P0
return-correlation candidates are normalized/residualized.

Optional screening environment variables:
  GFF_RISK_AUDIT_MAX_DAYS=6          evenly spaced dates; 0 means all dates
  GFF_RISK_AUDIT_SNAPSHOT_STRIDE=3   every third decision time; 1 means all
"""
from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import generate_monthly_alpha_report_with_risk_parallel_v2 as runner

base = runner.base
risk = runner.risk

PRIMARY_LEVEL = "B50"
CORE_UNDERREACTION = {
    ("volume_expansion", "15m"),
    ("signed_flow", "30m"),
    ("trade_intensity", "15m"),
    ("trade_intensity", "30m"),
    ("absorption", "30m"),
}
CORE_RETURN_CORR = {("return_corr_raw_1m", "15m")}
CORE_TARGETS = {"target_15m", "target_30m"}


def _candidate_rows(path: Path) -> pd.DataFrame:
    frame = risk.candidates(path)
    if frame.empty:
        return frame
    if "level" in frame:
        frame = frame[frame["level"].astype(str).eq(PRIMARY_LEVEL)]
    return frame


def _name_column(frame: pd.DataFrame) -> pd.Series:
    if "layer_name" in frame:
        return frame["layer_name"].astype(str)
    return pd.Series("", index=frame.index)


def focused_scopes(report: Path):
    frame = _candidate_rows(report / "p2_intraday_scorecard.csv")
    if frame.empty:
        return set(), set()
    names = _name_column(frame)
    targets = frame["target"].astype(str).isin(CORE_TARGETS)

    under_mask = frame["score"].astype(str).eq("daily_underreaction_score") & targets
    under_mask &= pd.Series(
        [(name, str(scale)) in CORE_UNDERREACTION for name, scale in zip(names, frame["scale"])],
        index=frame.index,
    )
    corr_mask = frame["score"].astype(str).eq("daily_consensus_score") & targets
    corr_mask &= pd.Series(
        [(name, str(scale)) in CORE_RETURN_CORR for name, scale in zip(names, frame["scale"])],
        index=frame.index,
    )
    columns = ["layer_id", "scale", "target"]
    under = set(frame.loc[under_mask, columns].astype(str).itertuples(index=False, name=None))
    corr = set(frame.loc[corr_mask, columns].astype(str).itertuples(index=False, name=None))
    return under, corr


def focused_p0_scope(report: Path):
    frame = _candidate_rows(report / "p0_eval" / "p0_eval_combo_scorecard.csv")
    if frame.empty:
        return set()
    names = _name_column(frame)
    mask = names.eq("return_corr_raw_1m")
    mask &= frame["scale"].astype(str).eq("15m")
    mask &= frame["feature"].astype(str).isin({"p0_edge_spillover_signal", "p0_edge_spillover_sum"})
    mask &= frame["target"].astype(str).isin(CORE_TARGETS)
    return set(frame.loc[mask, ["layer_id", "scale", "feature", "target"]].astype(str).itertuples(index=False, name=None))


def _selected_dates(paths: list[Path]) -> set[str] | None:
    maximum = int(os.environ.get("GFF_RISK_AUDIT_MAX_DAYS", "0") or 0)
    dates = sorted({risk.part(path, "date") or "" for path in paths})
    if maximum <= 0 or maximum >= len(dates):
        return None
    positions = np.linspace(0, len(dates) - 1, maximum).round().astype(int)
    return {dates[index] for index in positions}


def _snapshot_stride() -> int:
    return max(1, int(os.environ.get("GFF_RISK_AUDIT_SNAPSHOT_STRIDE", "1") or 1))


def _process_p2_focused(task: dict):
    source, checkpoint = Path(task["path"]), Path(task["checkpoint"])
    fingerprint = base._fingerprint(source)
    cached = base._valid_checkpoint(checkpoint, fingerprint, task["scope_hash"])
    if cached is not None:
        return {"status": "reused", "checkpoint": str(checkpoint), "records": cached["records"],
                "rows": int(cached.get("rows", 0)), "groups": int(cached.get("groups", 0))}

    started = base.time.time()
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
    rows = groups = accepted_times = 0
    previous_time = None
    stride = int(task["snapshot_stride"])

    if required.issubset(columns):
        use_full = base._full_read(source, columns, task["read_mode"], int(task["max_full_bytes"]))
        for timestamp, level, group in base._p2_groups(source, columns, use_full):
            if level != PRIMARY_LEVEL:
                continue
            if previous_time is None or timestamp != previous_time:
                accepted_times += 1
                previous_time = timestamp
            if (accepted_times - 1) % stride:
                continue
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

    records = base._records(states, ["audit", "date", "layer_id", "scale", "level", "target", "signal"])
    payload = {"contract": base.PARALLEL_CONTRACT, "status": "complete", "source_fingerprint": fingerprint,
               "scope_hash": task["scope_hash"], "records": records, "rows": rows, "groups": groups,
               "elapsed_sec": round(base.time.time() - started, 3)}
    base._atomic_json(checkpoint, payload)
    return {"status": "computed", "checkpoint": str(checkpoint), "records": records, "rows": rows, "groups": groups}


def focused_parallel_scan_p2(root: Path, month: str, under_scope, corr_scope, config: dict):
    all_paths = sorted((root / "intraday_relation_features").glob(
        f"date={month}-*/layer_id=*/scale=*/intraday_relation_features.parquet"),
        key=lambda path: path.stat().st_size, reverse=True)
    selected_dates = _selected_dates(all_paths)
    stride = _snapshot_stride()
    tasks = []
    checkpoint_root = Path(config["checkpoint_root"])
    for path in all_paths:
        date = risk.part(path, "date") or ""
        if selected_dates is not None and date not in selected_dates:
            continue
        layer, scale = risk.part(path, "layer_id") or "", risk.part(path, "scale") or ""
        under_targets = sorted({target for item_layer, item_scale, target in under_scope if item_layer == layer and item_scale == scale})
        corr_targets = sorted({target for item_layer, item_scale, target in corr_scope if item_layer == layer and item_scale == scale})
        if not under_targets and not corr_targets:
            continue
        scope_hash = base._scope_hash({"profile": "focused_b50", "under": under_targets, "corr": corr_targets,
                                      "min_sample": risk.MIN_SAMPLE, "snapshot_stride": stride,
                                      "selected_dates": sorted(selected_dates) if selected_dates else "all"})
        tasks.append({"path": str(path), "checkpoint": str(base._checkpoint_path(checkpoint_root, "p2_focused", path)),
                      "scope_hash": scope_hash, "date": date, "layer": layer, "scale": scale,
                      "under_targets": under_targets, "corr_targets": corr_targets,
                      "snapshot_stride": stride, "read_mode": config["read_mode"],
                      "max_full_bytes": config["max_full_bytes"]})

    results, stats = base._run_tasks(tasks, _process_p2_focused, config, "p2-focused")
    states = base._states_from_results(results, ["audit", "date", "layer_id", "scale", "level", "target", "signal"])
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
    stats.update({"audit_profile": "focused_b50", "selected_dates": sorted(selected_dates) if selected_dates else "all",
                  "snapshot_stride": stride, "daily_alpha_scanned": False})
    return r1, r2, stats


def _scan_p2_entry(root: Path, month: str, under_scope, corr_scope, progress: int = 25):
    return focused_parallel_scan_p2(Path(root), month, under_scope, corr_scope, base._CONFIG)


risk.scopes = focused_scopes
risk.p0_scope = focused_p0_scope
base.parallel_scan_p2 = focused_parallel_scan_p2
base._scan_p2_entry = _scan_p2_entry


def main() -> None:
    print(
        "Focused falsification profile: B50 only; six frozen intraday mechanisms; "
        "Daily Alpha excluded. Set GFF_RISK_AUDIT_MAX_DAYS and "
        "GFF_RISK_AUDIT_SNAPSHOT_STRIDE only for a labelled screening run.",
        flush=True,
    )
    base.main()


if __name__ == "__main__":
    main()
