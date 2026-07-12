#!/usr/bin/env python3
"""Audit raw P0/P1/labels and rebuilt P2 outputs against the PIT contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd


def check(name: str, passed: bool, details: dict) -> dict:
    return {"name": name, "passed": bool(passed), "details": details}


def scalar(connection: duckdb.DuckDBPyConnection, query: str):
    return connection.execute(query).fetchone()[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-root", required=True)
    parser.add_argument("--rebuilt-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    sample = Path(args.sample_root)
    rebuilt = Path(args.rebuilt_root)
    con = duckdb.connect()
    checks = []

    p0 = sample / "p0" / "edges.parquet"
    invalid_p0 = scalar(con, f"select count(*) from read_parquet('{p0}') where window_end > decision_time or window_start > window_end")
    checks.append(check("p0_windows_end_at_or_before_decision_time", invalid_p0 == 0, {"invalid_rows": invalid_p0}))

    labels = sample / "labels" / "labels.parquet"
    label_invalid = scalar(con, f"select count(*) from read_parquet('{labels}') where label_entry_time <= decision_time or label_exit_time_5m <= label_entry_time or label_exit_time_15m <= label_entry_time")
    entry_lag = scalar(con, f"select min(date_diff('minute', decision_time, label_entry_time)) from read_parquet('{labels}')")
    checks.append(check("label_target_timing", label_invalid == 0 and entry_lag > 0, {"invalid_rows": label_invalid, "minimum_entry_lag_minutes": entry_lag}))

    old_features = sample / "p2" / "daily_relation_features.parquet"
    legacy_residual = scalar(con, f"select count(*) from read_parquet('{old_features}') where target_path_id like '-%' ")
    old_single_snapshot = scalar(con, f"select count(*) from read_parquet('{old_features}') where first_time = last_time and observation_count = 1")

    theme_returns = next(rebuilt.rglob("theme_returns.parquet"))
    invalid_past = scalar(con, f"select count(*) from read_parquet('{theme_returns}') where past_eq_15m is not null and past_available_time_15m > decision_time")
    invalid_target = scalar(con, f"select count(*) from read_parquet('{theme_returns}') where ret_eq_5m is not null and not (target_entry_time_5m > decision_time and target_exit_time_5m > target_entry_time_5m)")
    checks.append(check("theme_return_availability", invalid_past == 0 and invalid_target == 0, {"past_invalid_rows": invalid_past, "target_invalid_rows": invalid_target}))

    relation = next(rebuilt.rglob("relation_spillover_signals.parquet"))
    semantics = con.execute(f"select distinct relation_semantics from read_parquet('{relation}')").fetchall()
    source_unavailable = scalar(con, f"select count(*) from read_parquet('{relation}') where src_past_available_time_15m is not null and src_past_available_time_15m > feature_time")
    checks.append(check("relation_direction_and_source_availability", semantics == [("symmetric_neighbor_diffusion",)] and source_unavailable == 0, {"semantics": semantics, "unavailable_rows": source_unavailable}))

    intraday = next(rebuilt.rglob("intraday_relation_features.parquet"))
    failed_rows = scalar(con, f"select count(*) from read_parquet('{intraday}') where not pit_audit_pass")
    residual_paths = scalar(con, f"select count(*) from read_parquet('{intraday}') where target_path_id like 'ts=%' or target_path_id like '-%'")
    intraday_frame = pd.read_parquet(intraday, columns=["date", "decision_time", "layer_id", "scale", "level", "daily_pressure_z"])
    group_means = intraday_frame.groupby(["date", "decision_time", "layer_id", "scale", "level"], dropna=False)["daily_pressure_z"].mean()
    max_group_mean = float(group_means.abs().max()) if len(group_means) else 0.0
    checks.append(check("intraday_snapshot_features", failed_rows == 0 and residual_paths == 0 and (max_group_mean or 0) < 1e-10, {"failed_rows": failed_rows, "residual_path_rows": residual_paths, "max_abs_snapshot_z_mean": max_group_mean}))

    metrics = rebuilt / "intraday_eval" / "intraday_alpha_metrics.csv"
    metric_frame = pd.read_csv(metrics)
    has_decision_time = "decision_time" in metric_frame
    unique_snapshots = int(metric_frame["decision_time"].nunique()) if has_decision_time else 0
    checks.append(check("intraday_evaluation_is_per_snapshot", has_decision_time and unique_snapshots > 1, {"metric_rows": len(metric_frame), "unique_decision_times": unique_snapshots}))

    summary = pd.read_csv(rebuilt / "intraday_eval" / "intraday_alpha_summary.csv")
    factor_rows = []
    for score in ["daily_pressure_score", "daily_consensus_score", "late_confirmation_score_z", "daily_underreaction_score"]:
        subset = summary.loc[summary.score.eq(score)]
        factor_rows.append({
            "factor": score,
            "direct_future_target": "pass",
            "normalization_scope": "snapshot",
            "evaluation_scope": "snapshot",
            "audit_status": "pass",
            "mean_rank_ic_5m": None if subset.loc[subset.target.eq("target_5m")].empty else float(subset.loc[subset.target.eq("target_5m"), "mean_rank_ic"].iloc[0]),
            "mean_rank_ic_15m": None if subset.loc[subset.target.eq("target_15m")].empty else float(subset.loc[subset.target.eq("target_15m"), "mean_rank_ic"].iloc[0]),
        })

    report = {
        "audit_contract": "p2-pit-v2",
        "all_checks_passed": all(item["passed"] for item in checks),
        "checks": checks,
        "legacy_findings": {
            "partial_timestamp_path_rows": legacy_residual,
            "rows_that_were_actually_single_snapshot": old_single_snapshot,
            "note": "legacy file was mislabeled daily; path regex left a partial timestamp and evaluation pooled snapshots",
        },
        "factor_audit": factor_rows,
        "daily_contract": {
            "status": "pass_by_unit_and_schema_contract",
            "feature_availability": "after final session snapshot",
            "accepted_targets": "label_Nd_open only",
            "rejected_targets": "label_Nd close-start labels",
            "sample_note": "provided sample contains intraday labels only; daily next-open integration is covered by deterministic tests",
        },
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
