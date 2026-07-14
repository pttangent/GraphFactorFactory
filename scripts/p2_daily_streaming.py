#!/usr/bin/env python3
"""Single-pass daily relation feature aggregation with bounded memory."""
from __future__ import annotations

import re
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

from p2_pit_core import DEFAULT_DAILY_HORIZONS, iter_time_groups, zscore_by_group
from p2_pit_features import _ensure_signal_components, build_temporal_episode_map

GROUP_COLUMNS = ["date", "layer_id", "scale", "level", "theme_episode_id"]
SUM_COLUMNS = [
    "signal_sum",
    "absolute_signal_sum",
    "positive_signal_sum",
    "negative_signal_sum",
    "positive_source_count",
    "negative_source_count",
    "relation_edge_count",
]
_INTERNAL_STATE_COLUMNS = {"relation_strength_sum", "relation_strength_count"}


def _episode_lookup(temporal_edges: pd.DataFrame | None) -> dict[str, str]:
    if temporal_edges is None:
        return {}
    mapping = build_temporal_episode_map(pd.Series(dtype="string"), temporal_edges)
    if mapping.empty:
        return {}
    return dict(zip(mapping["dst_theme_id"].astype(str), mapping["theme_episode_id"].astype(str)))


def _normalise_snapshot(frame: pd.DataFrame, episode_for: dict[str, str]) -> pd.DataFrame:
    unsafe = [column for column in frame if re.fullmatch(r"target_\d+d", column)]
    if unsafe:
        raise ValueError(f"daily EOD features reject close-start labels: {unsafe}")
    frame = _ensure_signal_components(frame)
    frame["decision_time"] = pd.to_datetime(frame["decision_time"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["decision_time", "dst_theme_id", "layer_id", "scale", "level"]).copy()
    if frame.empty:
        return frame
    if "date" not in frame:
        frame["date"] = frame["decision_time"].dt.strftime("%Y-%m-%d")
    frame["date"] = frame["date"].astype(str)
    frame["layer_id"] = frame["layer_id"].astype(str)
    frame["scale"] = frame["scale"].astype(str)
    frame["level"] = frame["level"].astype(str)
    theme_ids = frame["dst_theme_id"].astype(str).tolist()
    frame["theme_episode_id"] = [episode_for.get(x, "episode|" + x) for x in theme_ids]
    return frame


def _snapshot_aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    aggregations: dict[str, tuple[str, str]] = {
        "first_time": ("decision_time", "min"),
        "last_time": ("decision_time", "max"),
        "relation_strength_sum": ("relation_strength_mean", "sum"),
        "relation_strength_count": ("relation_strength_mean", "count"),
    }
    for column in SUM_COLUMNS:
        aggregations[column] = (column, "sum")
    return frame.groupby(GROUP_COLUMNS, sort=False, dropna=False).agg(**aggregations).reset_index()


def _late_aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(GROUP_COLUMNS, sort=False, dropna=False)
        .agg(late_signal=("signal_sum", "sum"), late_abs_signal=("absolute_signal_sum", "sum"))
        .reset_index()
    )


def _update_state(state: dict, aggregate: pd.DataFrame) -> None:
    for row in aggregate.itertuples(index=False):
        key = tuple(getattr(row, column) for column in GROUP_COLUMNS)
        current = state.get(key)
        if current is None:
            current = {
                "first_time": row.first_time,
                "last_time": row.last_time,
                "relation_strength_sum": 0.0,
                "relation_strength_count": 0,
                **{column: 0.0 for column in SUM_COLUMNS},
            }
            state[key] = current
        current["first_time"] = min(current["first_time"], row.first_time)
        current["last_time"] = max(current["last_time"], row.last_time)
        current["relation_strength_sum"] += float(row.relation_strength_sum) if pd.notna(row.relation_strength_sum) else 0.0
        current["relation_strength_count"] += int(row.relation_strength_count)
        for column in SUM_COLUMNS:
            value = getattr(row, column)
            current[column] += float(value) if pd.notna(value) else 0.0


def _state_frame(state: dict, session_close: pd.Timestamp) -> pd.DataFrame:
    rows: list[dict] = []
    for key, values in state.items():
        row = dict(zip(GROUP_COLUMNS, key))
        row.update({name: value for name, value in values.items() if name not in _INTERNAL_STATE_COLUMNS})
        count = int(values["relation_strength_count"])
        row["relation_strength_mean"] = values["relation_strength_sum"] / count if count else np.nan
        row["session_close"] = session_close
        rows.append(row)
    return pd.DataFrame(rows)


def _finalize_daily(
    result: pd.DataFrame,
    final_snapshot: pd.DataFrame,
    late: pd.DataFrame,
    underreaction_past_horizon: str,
) -> pd.DataFrame:
    if result.empty:
        return result
    result = result.loc[result["last_time"].eq(result["session_close"])].copy()
    if result.empty:
        return result

    result = result.merge(late, on=GROUP_COLUMNS, how="left", validate="one_to_one")
    result[["late_signal", "late_abs_signal"]] = result[["late_signal", "late_abs_signal"]].fillna(0.0)

    final_rows = final_snapshot.sort_values("decision_time").drop_duplicates(GROUP_COLUMNS, keep="last")
    final_columns = GROUP_COLUMNS.copy()
    for column in final_snapshot:
        if column.startswith("target_") or column.startswith("dst_past_eq_") or column.startswith("dst_past_available_time_"):
            final_columns.append(column)
    final_columns = list(dict.fromkeys(final_columns))
    result = result.merge(final_rows[final_columns], on=GROUP_COLUMNS, how="left", validate="one_to_one")

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


def build_daily_feature_frame_streaming(
    source_path: str | Path,
    underreaction_past_horizon: str,
    late_minutes: int,
    temporal_edges: pd.DataFrame | None,
    max_row_groups: int | None = None,
) -> pd.DataFrame:
    episode_for = _episode_lookup(temporal_edges)
    state: dict = {}
    recent: deque[tuple[pd.Timestamp, pd.DataFrame]] = deque()
    final_snapshot = pd.DataFrame()
    session_close: pd.Timestamp | None = None
    partition_identity: tuple[str, str, str] | None = None
    late_delta = pd.Timedelta(minutes=late_minutes)

    for decision_time, raw in iter_time_groups(source_path, None, max_row_groups):
        snapshot = _normalise_snapshot(raw, episode_for)
        if snapshot.empty:
            continue
        if snapshot["date"].nunique(dropna=False) != 1 or snapshot["layer_id"].nunique(dropna=False) != 1 or snapshot["scale"].nunique(dropna=False) != 1:
            raise ValueError(f"daily streaming snapshot mixes date/layer/scale: {source_path}")
        current_identity = (
            str(snapshot["date"].iloc[0]),
            str(snapshot["layer_id"].iloc[0]),
            str(snapshot["scale"].iloc[0]),
        )
        if partition_identity is None:
            partition_identity = current_identity
        elif current_identity != partition_identity:
            raise ValueError(
                f"daily streaming file crosses physical date/layer/scale partitions: "
                f"expected={partition_identity}, found={current_identity}, path={source_path}"
            )

        _update_state(state, _snapshot_aggregate(snapshot))
        recent.append((decision_time, _late_aggregate(snapshot)))
        while recent and recent[0][0] < decision_time - late_delta:
            recent.popleft()
        final_snapshot = snapshot
        session_close = decision_time

    if session_close is None or not state or final_snapshot.empty:
        return pd.DataFrame()
    late_frames = [frame for _, frame in recent]
    if late_frames:
        late = (
            pd.concat(late_frames, ignore_index=True)
            .groupby(GROUP_COLUMNS, sort=False, dropna=False)[["late_signal", "late_abs_signal"]]
            .sum()
            .reset_index()
        )
    else:
        late = pd.DataFrame(columns=GROUP_COLUMNS + ["late_signal", "late_abs_signal"])
    result = _state_frame(state, session_close)
    return _finalize_daily(result, final_snapshot, late, underreaction_past_horizon)
