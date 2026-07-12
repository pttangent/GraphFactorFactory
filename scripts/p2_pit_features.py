#!/usr/bin/env python3
"""PIT-safe intraday and end-of-day relation factor transforms/evaluation."""
from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from p2_pit_core import *


def _ensure_signal_components(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["signal"] = pd.to_numeric(frame["signal"], errors="coerce").fillna(0.0)
    frame["relation_edge_count"] = pd.to_numeric(frame.get("relation_edge_count", 1), errors="coerce").fillna(0).clip(lower=0)
    if "signal_sum" not in frame:
        frame["signal_sum"] = frame["signal"] * frame["relation_edge_count"].replace(0, 1)
    if "absolute_signal_sum" not in frame:
        frame["absolute_signal_sum"] = frame["signal"].abs() * frame["relation_edge_count"].replace(0, 1)
    if "positive_source_count" not in frame:
        frame["positive_source_count"] = np.where(frame["signal"] > 0, frame["relation_edge_count"], 0)
    if "negative_source_count" not in frame:
        frame["negative_source_count"] = np.where(frame["signal"] < 0, frame["relation_edge_count"], 0)
    if "positive_signal_sum" not in frame:
        frame["positive_signal_sum"] = frame["signal_sum"].clip(lower=0)
    if "negative_signal_sum" not in frame:
        frame["negative_signal_sum"] = frame["signal_sum"].clip(upper=0)
    return frame


def build_intraday_feature_frame(frame: pd.DataFrame, underreaction_past_horizon: str) -> pd.DataFrame:
    frame = _ensure_signal_components(frame)
    frame["decision_time"] = pd.to_datetime(frame["decision_time"], utc=True, errors="coerce")
    if "date" not in frame:
        frame["date"] = frame["decision_time"].dt.strftime("%Y-%m-%d")
    frame["target_path_id"] = canonical_theme_path(frame["dst_theme_id"])
    frame["feature_time"] = frame["decision_time"]

    group_columns = ["date", "decision_time", "layer_id", "scale", "level", "dst_theme_id", "target_path_id"]
    numeric_sum = [
        "signal_sum", "absolute_signal_sum", "positive_signal_sum", "negative_signal_sum",
        "positive_source_count", "negative_source_count", "relation_edge_count",
    ]
    aggregations: dict[str, str] = {column: "sum" for column in numeric_sum if column in frame}
    aggregations.update({"signal": "mean", "relation_strength_mean": "mean", "feature_time": "max"})
    for column in frame.columns:
        if column in group_columns or column in {"target_path_id", "dst_theme_id"}:
            continue
        if column.startswith("target_") or column.startswith("dst_past_eq_") or column.startswith("dst_past_available_time_"):
            if "time_" in column or "date_" in column:
                aggregations[column] = "max"
            elif pd.api.types.is_numeric_dtype(frame[column]):
                aggregations[column] = "mean"
    result = frame.groupby(group_columns, sort=False, dropna=False).agg(aggregations).reset_index()

    result["observation_count"] = result["relation_edge_count"]
    result["daily_pressure"] = result["signal_sum"]
    result["absolute_pressure"] = result["absolute_signal_sum"]
    result["positive_pressure"] = result["positive_signal_sum"]
    result["negative_pressure"] = result["negative_signal_sum"]
    denominator = result["relation_edge_count"].replace(0, np.nan)
    result["positive_observation_rate"] = result["positive_source_count"] / denominator
    result["negative_observation_rate"] = result["negative_source_count"] / denominator
    result["persistence_proxy"] = (result["positive_source_count"] - result["negative_source_count"]) / denominator
    result["pressure_intensity"] = result["signal_sum"] / denominator

    snapshot_columns = ["date", "decision_time", "layer_id", "scale", "level"]
    result["daily_pressure_z"] = zscore_by_group(result, snapshot_columns, "pressure_intensity")
    result["absolute_pressure_z"] = zscore_by_group(result, snapshot_columns, "absolute_pressure")
    result["relation_edge_count_sum"] = result["relation_edge_count"]
    result["relation_edge_count_sum_z"] = zscore_by_group(result, snapshot_columns, "relation_edge_count_sum")
    result["daily_pressure_score"] = result["daily_pressure_z"] * result["persistence_proxy"].fillna(0.0)
    result["intraday_pressure_score"] = result["daily_pressure_score"]
    result["daily_consensus_score"] = result["daily_pressure_z"] * np.log1p(result["relation_edge_count_sum"].clip(lower=0))
    result["intraday_consensus_score"] = result["daily_consensus_score"]

    preferred_past = f"dst_past_eq_{underreaction_past_horizon}"
    if preferred_past not in result:
        result["expected_pressure_z"] = np.nan
        result["target_pre_response_z"] = np.nan
        result["underreaction_gap_z"] = np.nan
        result["daily_underreaction_score"] = np.nan
        result["late_confirmation_score"] = np.nan
        result["late_confirmation_score_z"] = np.nan
        result["daily_underreaction_status"] = f"missing_{preferred_past}"
    else:
        result["expected_pressure_z"] = zscore_by_group(result, snapshot_columns, "daily_pressure_score")
        result["target_pre_response_z"] = zscore_by_group(result, snapshot_columns, preferred_past)
        result["underreaction_gap_z"] = result["expected_pressure_z"] - result["target_pre_response_z"]
        result["daily_underreaction_score"] = result["underreaction_gap_z"]
        result["intraday_underreaction_score"] = result["underreaction_gap_z"]
        result["late_confirmation_score"] = (
            result["expected_pressure_z"]
            * result["target_pre_response_z"]
            * np.log1p(result["relation_edge_count_sum"].clip(lower=0))
        )
        result["late_confirmation_score_z"] = zscore_by_group(result, snapshot_columns, "late_confirmation_score")
        result["daily_underreaction_status"] = "pit_snapshot_known_past_response"
        result["late_confirmation_status"] = "pit_past_response_alignment"

    audit = pd.Series(True, index=result.index)
    past_available = f"dst_past_available_time_{underreaction_past_horizon}"
    if past_available in result:
        result[past_available] = pd.to_datetime(result[past_available], utc=True, errors="coerce")
        audit &= result[past_available].isna() | (result[past_available] <= result["feature_time"])
    for horizon in DEFAULT_INTRADAY_HORIZONS:
        target = f"target_{horizon}"
        entry = f"target_entry_time_{horizon}"
        exit_column = f"target_exit_time_{horizon}"
        if target not in result:
            continue
        if entry not in result or exit_column not in result:
            audit &= False
            continue
        result[entry] = pd.to_datetime(result[entry], utc=True, errors="coerce")
        result[exit_column] = pd.to_datetime(result[exit_column], utc=True, errors="coerce")
        target_present = result[target].notna()
        audit &= ~target_present | ((result[entry] > result["feature_time"]) & (result[exit_column] > result[entry]))
    result["pit_audit_pass"] = audit
    result["feature_contract"] = "intraday_snapshot"
    return result


def build_temporal_episode_map(theme_ids: pd.Series, temporal_edges: pd.DataFrame) -> pd.DataFrame:
    """Map snapshot-local theme IDs to full-session temporal episodes."""
    nodes = {str(value) for value in theme_ids.dropna().astype(str)}
    parent = {node: node for node in nodes}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a == b:
            return
        parent[max(a, b)] = min(a, b)

    if temporal_edges is not None and not temporal_edges.empty:
        required = {"src_theme_id", "dst_theme_id"}
        if missing := required - set(temporal_edges):
            raise ValueError(f"temporal theme edges missing {sorted(missing)}")
        for left, right in temporal_edges[["src_theme_id", "dst_theme_id"]].dropna().astype(str).itertuples(index=False, name=None):
            nodes.update((left, right))
            parent.setdefault(left, left)
            parent.setdefault(right, right)
            union(left, right)

    components: dict[str, list[str]] = {}
    for node in nodes:
        components.setdefault(find(node), []).append(node)
    episode_for: dict[str, str] = {}
    for members in components.values():
        ordered = sorted(members, key=lambda value: (parse_theme_ts_series(pd.Series([value])).iloc[0], value))
        episode_id = "episode|" + ordered[0]
        for member in members:
            episode_for[member] = episode_id
    return pd.DataFrame({"dst_theme_id": list(episode_for), "theme_episode_id": list(episode_for.values())})


def build_daily_feature_frame(
    frame: pd.DataFrame,
    underreaction_past_horizon: str,
    late_minutes: int,
    temporal_edges: pd.DataFrame | None = None,
) -> pd.DataFrame:
    unsafe_input_targets = [c for c in frame if re.fullmatch(r"target_\d+d", c)]
    if unsafe_input_targets:
        raise ValueError(f"daily EOD features reject close-start labels: {unsafe_input_targets}")
    frame = _ensure_signal_components(frame)
    frame["decision_time"] = pd.to_datetime(frame["decision_time"], utc=True, errors="coerce")
    if "date" not in frame:
        frame["date"] = frame["decision_time"].dt.strftime("%Y-%m-%d")

    if "theme_episode_id" not in frame:
        if temporal_edges is None:
            raise ValueError("daily EOD features require temporal_theme_edges or a precomputed theme_episode_id")
        mapping = build_temporal_episode_map(frame["dst_theme_id"], temporal_edges)
        frame = frame.merge(mapping, on="dst_theme_id", how="left", validate="many_to_one")
    if frame["theme_episode_id"].isna().any():
        raise ValueError("daily episode mapping left unmapped theme IDs")

    session_keys = ["date", "layer_id", "scale"]
    frame["session_close"] = frame.groupby(session_keys, sort=False)["decision_time"].transform("max")
    late = frame["decision_time"] >= (frame["session_close"] - pd.Timedelta(minutes=late_minutes))
    frame["late_signal"] = frame["signal_sum"].where(late, 0.0)
    frame["late_abs_signal"] = frame["absolute_signal_sum"].where(late, 0.0)

    group_columns = ["date", "layer_id", "scale", "level", "theme_episode_id"]
    grouped = frame.groupby(group_columns, sort=False, dropna=False)
    result = grouped.agg(
        first_time=("decision_time", "min"),
        last_time=("decision_time", "max"),
        session_close=("session_close", "max"),
        signal_sum=("signal_sum", "sum"),
        absolute_signal_sum=("absolute_signal_sum", "sum"),
        positive_signal_sum=("positive_signal_sum", "sum"),
        negative_signal_sum=("negative_signal_sum", "sum"),
        positive_source_count=("positive_source_count", "sum"),
        negative_source_count=("negative_source_count", "sum"),
        relation_edge_count=("relation_edge_count", "sum"),
        relation_strength_mean=("relation_strength_mean", "mean"),
        late_signal=("late_signal", "sum"),
        late_abs_signal=("late_abs_signal", "sum"),
    ).reset_index()

    result = result.loc[result["last_time"].eq(result["session_close"])].copy()
    if result.empty:
        return result
    final_rows = frame.loc[frame["decision_time"].eq(frame["session_close"])].copy()
    final_rows = final_rows.sort_values("decision_time").drop_duplicates(group_columns, keep="last")
    final_columns = group_columns.copy()
    for column in frame:
        if column.startswith("target_") or column.startswith("dst_past_eq_") or column.startswith("dst_past_available_time_"):
            final_columns.append(column)
    final_columns = list(dict.fromkeys(final_columns))
    result = result.merge(final_rows[final_columns], on=group_columns, how="left", validate="one_to_one")

    result["target_path_id"] = result["theme_episode_id"]
    result["feature_time"] = result["session_close"]
    result["observation_count"] = result["relation_edge_count"]
    result["daily_pressure"] = result["signal_sum"]
    result["absolute_pressure"] = result["absolute_signal_sum"]
    result["positive_pressure"] = result["positive_signal_sum"]
    result["negative_pressure"] = result["negative_signal_sum"]
    denominator = result["relation_edge_count"].replace(0, np.nan)
    result["positive_observation_rate"] = result["positive_source_count"] / denominator
    result["negative_observation_rate"] = result["negative_source_count"] / denominator
    result["persistence_proxy"] = (result["positive_source_count"] - result["negative_source_count"]) / denominator
    result["pressure_intensity"] = result["daily_pressure"] / denominator
    result["late_signal_sum"] = result["late_signal"]
    result["late_abs_signal_sum"] = result["late_abs_signal"]
    result["late_absolute_share"] = result["late_abs_signal_sum"] / result["absolute_pressure"].replace(0, np.nan)
    result["late_confirmation_score"] = result["late_signal_sum"] * result["late_absolute_share"].fillna(0.0)

    daily_columns = ["date", "layer_id", "scale", "level"]
    for column in ["daily_pressure", "absolute_pressure", "relation_edge_count", "late_confirmation_score"]:
        target_name = "relation_edge_count_sum_z" if column == "relation_edge_count" else column + "_z"
        result[target_name] = zscore_by_group(result, daily_columns, column)
    result["relation_edge_count_sum"] = result["relation_edge_count"]
    result["daily_pressure_score"] = result["daily_pressure_z"] * result["persistence_proxy"].fillna(0.0)
    result["daily_consensus_score"] = result["daily_pressure_z"] * np.log1p(result["relation_edge_count_sum"].clip(lower=0))

    preferred_past = f"dst_past_eq_{underreaction_past_horizon}"
    if preferred_past in result:
        result["expected_pressure_z"] = zscore_by_group(result, daily_columns, "daily_pressure_score")
        result["target_pre_response_z"] = zscore_by_group(result, daily_columns, preferred_past)
        result["underreaction_gap_z"] = result["expected_pressure_z"] - result["target_pre_response_z"]
        result["daily_underreaction_score"] = result["underreaction_gap_z"] * (1.0 + result["late_absolute_share"].fillna(0.0))
        result["daily_underreaction_status"] = "pit_eod_final_snapshot_response"
    else:
        result["expected_pressure_z"] = np.nan
        result["target_pre_response_z"] = np.nan
        result["underreaction_gap_z"] = np.nan
        result["daily_underreaction_score"] = np.nan
        result["daily_underreaction_status"] = f"missing_{preferred_past}"

    audit = result["feature_time"].eq(result["session_close"])
    feature_date = pd.to_datetime(result["date"], errors="coerce").dt.date
    for horizon in DEFAULT_DAILY_HORIZONS:
        target = f"target_{horizon}"
        if target not in result:
            continue
        entry = f"target_entry_date_{horizon}"
        exit_column = f"target_exit_date_{horizon}"
        if entry not in result or exit_column not in result:
            audit &= False
            continue
        entry_date = pd.to_datetime(result[entry], errors="coerce").dt.date
        exit_date = pd.to_datetime(result[exit_column], errors="coerce").dt.date
        present = result[target].notna()
        audit &= ~present | ((entry_date > feature_date) & (exit_date >= entry_date))
    result["feature_contract"] = "end_of_day_episode_next_open_execution"
    result["pit_audit_pass"] = audit
    return result


def _build_feature_one(
    part: Part,
    output_root: str | Path,
    mode: str,
    underreaction_past_horizon: str,
    late_minutes: int,
    skip_existing: bool,
    max_row_groups: int | None,
    temporal_root: str | Path | None = None,
) -> dict:
    started = time.time()
    output_dir = Path(output_root) / f"date={part.date}" / f"layer_id={part.layer_id}" / f"scale={part.scale}"
    filename = "intraday_relation_features.parquet" if mode == "intraday" else "daily_relation_features.parquet"
    output_path = output_dir / filename
    if skip_existing and is_complete(output_dir / "manifest.json"):
        return {"stage": f"{mode}_relation_features", "status": "skipped", "date": part.date, "layer_id": part.layer_id, "scale": part.scale}
    frame = read_partition(part.base, None, max_row_groups)
    if frame.empty:
        output_rows = 0
    else:
        if mode == "intraday":
            output = build_intraday_feature_frame(frame, underreaction_past_horizon)
        else:
            if temporal_root is None:
                raise ValueError("daily mode requires --p1-root with temporal_theme_edges.parquet")
            temporal_path = Path(temporal_root) / f"date={part.date}" / f"layer_id={part.layer_id}" / f"scale={part.scale}" / "temporal_theme_edges.parquet"
            if not temporal_path.exists():
                raise FileNotFoundError(f"missing daily temporal identity file: {temporal_path}")
            temporal = read_partition(temporal_path, None, max_row_groups)
            if "level" in temporal:
                temporal = temporal[temporal["level"].astype(str).isin(frame["level"].astype(str).unique())]
            output = build_daily_feature_frame(frame, underreaction_past_horizon, late_minutes, temporal)
        if not bool(output["pit_audit_pass"].all()):
            failed = int((~output["pit_audit_pass"]).sum())
            raise AssertionError(f"{failed} {mode} feature rows failed PIT audit")
        write_parquet_atomic(output, output_path)
        output_rows = len(output)
    metadata = {
        "stage": f"{mode}_relation_features",
        "status": "complete" if output_rows else "empty",
        "date": part.date,
        "layer_id": part.layer_id,
        "scale": part.scale,
        "output_rows": output_rows,
        "mode": mode,
        "late_minutes": late_minutes if mode == "daily" else None,
        "underreaction_past_horizon": underreaction_past_horizon,
        "normalization_scope": "snapshot_cross_section" if mode == "intraday" else "eod_cross_section",
        "output": str(output_path),
        "elapsed_sec": round(time.time() - started, 3),
    }
    write_manifest(output_dir, metadata)
    return metadata


def _metric_row(subset: pd.DataFrame, score: str, target: str, keys: tuple, key_names: list[str]) -> dict | None:
    values = subset[[score, target]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) < 30:
        return None
    q80, q20 = values[score].quantile(0.8), values[score].quantile(0.2)
    top = values.loc[values[score] >= q80, target].mean()
    bottom = values.loc[values[score] <= q20, target].mean()
    row = dict(zip(key_names, keys))
    rank_ic = (
        np.nan
        if values[score].nunique(dropna=True) < 2 or values[target].nunique(dropna=True) < 2
        else values[score].rank().corr(values[target].rank())
    )
    row.update({"score": score, "target": target, "sample_count": len(values), "rank_ic": rank_ic, "top_minus_bottom": top - bottom})
    return row


def evaluate_intraday_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if "pit_audit_pass" in frame and not bool(frame["pit_audit_pass"].fillna(False).all()):
        raise AssertionError("refusing to evaluate intraday features with failed PIT rows")
    scores = [c for c in SCORES if c in frame]
    targets = [c for c in frame if re.fullmatch(r"target_\d+m", c)]
    key_names = ["date", "decision_time", "layer_id", "scale", "level"]
    rows: list[dict] = []
    for keys, subset in frame.groupby(key_names, dropna=False, sort=False):
        for score in scores:
            for target in targets:
                row = _metric_row(subset, score, target, keys, key_names)
                if row:
                    rows.append(row)
    return pd.DataFrame(rows)


def evaluate_daily_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if "pit_audit_pass" in frame and not bool(frame["pit_audit_pass"].fillna(False).all()):
        raise AssertionError("refusing to evaluate daily features with failed PIT rows")
    scores = [c for c in SCORES if c in frame]
    targets = [c for c in frame if re.fullmatch(r"target_\d+d_open", c)]
    key_names = ["date", "layer_id", "scale", "level"]
    rows: list[dict] = []
    for keys, subset in frame.groupby(key_names, dropna=False, sort=False):
        for score in scores:
            for target in targets:
                row = _metric_row(subset, score, target, keys, key_names)
                if row:
                    rows.append(row)
    return pd.DataFrame(rows)


def _summary(metrics: pd.DataFrame, mode: str) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    group_columns = ["score", "target", "layer_id", "scale", "level"]
    return (
        metrics.groupby(group_columns, sort=False)
        .agg(
            days=("date", "nunique"),
            snapshots=("decision_time", "nunique") if mode == "intraday" else ("date", "nunique"),
            sample_count=("sample_count", "sum"),
            mean_rank_ic=("rank_ic", "mean"),
            mean_spread=("top_minus_bottom", "mean"),
            positive_period_rate=("top_minus_bottom", lambda values: float((values > 0).mean())),
        )
        .reset_index()
    )


def evaluate_feature_root(root: str | Path, output_dir: str | Path, mode: str) -> dict:
    started = time.time()
    filename = "intraday_relation_features.parquet" if mode == "intraday" else "daily_relation_features.parquet"
    frames = [pd.read_parquet(path) for path in Path(root).rglob(filename)]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not frames:
        metadata = {"stage": f"{mode}_feature_eval", "status": "empty", "input_count": 0, "output_rows": 0}
        write_manifest(output_dir, metadata)
        return metadata
    frame = pd.concat(frames, ignore_index=True)
    metrics = evaluate_intraday_frame(frame) if mode == "intraday" else evaluate_daily_frame(frame)
    summary = _summary(metrics, mode)
    metrics_path = output_dir / f"{mode}_alpha_metrics.csv"
    summary_path = output_dir / f"{mode}_alpha_summary.csv"
    metrics.to_csv(metrics_path, index=False)
    summary.to_csv(summary_path, index=False)
    metadata = {
        "stage": f"{mode}_feature_eval",
        "status": "complete" if len(metrics) else "empty",
        "input_count": len(frames),
        "metric_rows": len(metrics),
        "summary_rows": len(summary),
        "output_rows": len(metrics),
        "evaluation_scope": "per_decision_time_cross_section" if mode == "intraday" else "per_date_eod_cross_section",
        "metrics": str(metrics_path),
        "summary": str(summary_path),
        "elapsed_sec": round(time.time() - started, 3),
    }
    write_manifest(output_dir, metadata)
    return metadata
