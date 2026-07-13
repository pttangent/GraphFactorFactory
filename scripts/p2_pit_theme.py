#!/usr/bin/env python3
"""PIT-safe P2 theme return and symmetric relation transforms."""
from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from p2_parallel_runtime import bounded_thread_map
from p2_pit_core import *


def _lookup_time(indexed: pd.DataFrame, decision_time: pd.Timestamp) -> pd.DataFrame:
    try:
        return indexed.loc[[decision_time]].reset_index(drop=True)
    except KeyError:
        return pd.DataFrame(columns=indexed.columns)


def _aggregate_theme_returns(member_frame: pd.DataFrame, label_frame: pd.DataFrame, horizons: list[str]) -> pd.DataFrame | None:
    if member_frame.empty or label_frame.empty:
        return None
    merged = member_frame.merge(
        label_frame,
        left_on=["decision_time", "member_id"],
        right_on=["decision_time", "symbol_id"],
        how="inner",
        copy=False,
    )
    if merged.empty:
        return None
    group_columns = ["decision_time", "layer_id", "scale", "level", "theme_id"]
    merged = merged.sort_values(group_columns + ["core_score"], ascending=[True, True, True, True, True, False])
    grouped = merged.groupby(group_columns, sort=False)

    numeric_labels = [
        column
        for column in merged.columns
        if (column.startswith("label_") or column.startswith("past_label_"))
        and pd.api.types.is_numeric_dtype(merged[column])
    ]
    rename_equal = {
        column: (
            column.replace("past_label_", "past_eq_")
            if column.startswith("past_label_")
            else column.replace("label_", "ret_eq_")
        )
        for column in numeric_labels
    }
    if numeric_labels:
        result = grouped[numeric_labels].mean().rename(columns=rename_equal)
    else:
        result = pd.DataFrame(index=grouped.size().index)

    for horizon in horizons:
        target = f"label_{horizon}"
        if target not in merged:
            continue
        if is_intraday_horizon(horizon):
            entry = f"label_entry_time_{horizon}" if f"label_entry_time_{horizon}" in merged else "label_entry_time"
            exit_column = f"label_exit_time_{horizon}"
            if entry in merged:
                result[f"target_entry_time_{horizon}"] = grouped[entry].max()
            if exit_column in merged:
                result[f"target_exit_time_{horizon}"] = grouped[exit_column].max()
            past = f"past_label_{horizon}"
            if past in merged and f"past_exit_time_{horizon}" in merged:
                result[f"past_available_time_{horizon}"] = grouped[f"past_exit_time_{horizon}"].max()
        else:
            for prefix in ("entry", "exit"):
                source = f"label_{prefix}_date_{horizon}"
                if source in merged:
                    result[f"target_{prefix}_date_{horizon}"] = grouped[source].max()

    if numeric_labels:
        weights = grouped["core_score"].sum().replace(0, np.nan)
        weighted = merged[numeric_labels].multiply(merged["core_score"], axis=0)
        weighted.index = pd.MultiIndex.from_frame(merged[group_columns], names=group_columns)
        weighted_sums = weighted.groupby(level=list(range(len(group_columns))), sort=False).sum(min_count=1)
        core = weighted_sums.div(weights, axis=0)
        core.columns = [
            column.replace("past_label_", "past_core_")
            if column.startswith("past_label_")
            else column.replace("label_", "ret_core_")
            for column in core.columns
        ]
        result = result.join(core)
        del weighted, weighted_sums, core

        top5_rows = grouped.head(5)
        top5 = top5_rows.groupby(group_columns, sort=False)[numeric_labels].mean()
        top5.columns = [
            column.replace("past_label_", "past_top5_")
            if column.startswith("past_label_")
            else column.replace("label_", "ret_top5_")
            for column in top5.columns
        ]
        result = result.join(top5)
    return result.reset_index()


def build_theme_returns_one(
    part: Part,
    labels_root: str | Path,
    output_root: str | Path,
    horizons: list[str],
    levels: set[str] | None,
    skip_existing: bool,
    max_row_groups: int | None,
    inner_workers: int,
) -> dict:
    started = time.time()
    output_dir = Path(output_root) / f"date={part.date}" / f"layer_id={part.layer_id}" / f"scale={part.scale}"
    output_path = output_dir / "theme_returns.parquet"
    if skip_existing and is_complete(output_dir / "manifest.json"):
        return {"stage": "theme_returns", "status": "skipped", "date": part.date, "layer_id": part.layer_id, "scale": part.scale}

    labels = load_labels(label_path(labels_root, part.date), horizons)
    labels_indexed = labels.set_index("decision_time", drop=False).sort_index()
    members = read_partition(
        part.base,
        ["decision_time", "layer_id", "scale", "level", "theme_id", "member_id", "core_score", "rank_in_theme"],
        max_row_groups,
    )
    if members.empty:
        write_manifest(output_dir, {"stage": "theme_returns", "status": "empty", "output_rows": 0})
        return {"stage": "theme_returns", "status": "empty", "date": part.date, "layer_id": part.layer_id, "scale": part.scale}
    if "decision_time" not in members or members["decision_time"].isna().all():
        members["decision_time"] = parse_theme_ts_series(members["theme_id"])
    members["decision_time"] = pd.to_datetime(members["decision_time"], utc=True, errors="coerce")
    members["member_id"] = pd.to_numeric(members["member_id"], errors="coerce").astype("Int64")
    members["core_score"] = pd.to_numeric(members.get("core_score", 0), errors="coerce").fillna(0.0).astype("float32")
    members = members.dropna(subset=["decision_time", "member_id", "theme_id"]).copy()
    members["member_id"] = members["member_id"].astype("int64")
    if "level" not in members:
        members["level"] = "UNKNOWN"
    if levels:
        members = members[members["level"].astype(str).isin(levels)]
    if "layer_id" not in members:
        members["layer_id"] = part.layer_id
    if "scale" not in members:
        members["scale"] = part.scale

    def one(item: tuple[pd.Timestamp, pd.DataFrame]) -> pd.DataFrame | None:
        decision_time, membership = item
        return _aggregate_theme_returns(membership, _lookup_time(labels_indexed, decision_time), horizons)

    groups = members.groupby("decision_time", sort=False, dropna=False)
    if inner_workers > 1:
        frames: Iterable[pd.DataFrame | None] = bounded_thread_map(
            groups,
            inner_workers,
            one,
            max_in_flight=inner_workers * 2,
        )
    else:
        frames = (one(item) for item in groups)
    rows, batches = stream_frames(output_path, frames)

    metadata = {
        "stage": "theme_returns",
        "status": "complete" if rows else "empty",
        "date": part.date,
        "layer_id": part.layer_id,
        "scale": part.scale,
        "output_rows": rows,
        "write_batches": batches,
        "output": str(output_path),
        "past_return_availability": "actual_label_exit_time",
        "input_grouping": "lazy_bounded_decision_time",
        "inner_workers": inner_workers,
        "elapsed_sec": round(time.time() - started, 3),
    }
    write_manifest(output_dir, metadata)
    return metadata


def expand_symmetric_relations(edges: pd.DataFrame) -> pd.DataFrame:
    """P1 relation edges are undirected; emit both neighbor-diffusion directions."""
    if edges.empty:
        return edges
    reverse = edges.copy()
    reverse[["src_theme_id", "dst_theme_id"]] = reverse[["dst_theme_id", "src_theme_id"]].to_numpy()
    expanded = pd.concat([edges, reverse], ignore_index=True)
    keys = [column for column in ["decision_time", "layer_id", "scale", "level", "src_theme_id", "dst_theme_id"] if column in expanded]
    return expanded.drop_duplicates(keys, keep="first")


def _relation_signal_frame(
    edge_frame: pd.DataFrame,
    source_returns: pd.DataFrame,
    target_returns: pd.DataFrame,
    past_horizon: str,
) -> pd.DataFrame | None:
    if edge_frame.empty or source_returns.empty or target_returns.empty:
        return None
    merged = edge_frame.merge(
        source_returns,
        on=["decision_time", "layer_id", "scale", "level", "src_theme_id"],
        how="inner",
        copy=False,
    )
    if merged.empty:
        return None
    merged["source_signal"] = (
        pd.to_numeric(merged["relation_strength"], errors="coerce").fillna(0.0)
        * pd.to_numeric(merged["src_past_return"], errors="coerce").fillna(0.0)
    )
    merged["positive_source_signal"] = merged["source_signal"].clip(lower=0)
    merged["negative_source_signal"] = merged["source_signal"].clip(upper=0)
    merged["absolute_source_signal"] = merged["source_signal"].abs()
    merged["positive_source"] = (merged["source_signal"] > 0).astype("int64")
    merged["negative_source"] = (merged["source_signal"] < 0).astype("int64")
    group_columns = ["decision_time", "layer_id", "scale", "level", "dst_theme_id"]
    aggregate = (
        merged.groupby(group_columns, sort=False)
        .agg(
            signal=("source_signal", "mean"),
            signal_sum=("source_signal", "sum"),
            absolute_signal_sum=("absolute_source_signal", "sum"),
            positive_signal_sum=("positive_source_signal", "sum"),
            negative_signal_sum=("negative_source_signal", "sum"),
            positive_source_count=("positive_source", "sum"),
            negative_source_count=("negative_source", "sum"),
            relation_strength_mean=("relation_strength", "mean"),
            relation_edge_count=("src_theme_id", "size"),
        )
        .reset_index()
    )
    output = aggregate.merge(
        target_returns,
        on=["decision_time", "layer_id", "scale", "level", "dst_theme_id"],
        how="inner",
        copy=False,
    )
    if output.empty:
        return None
    output["feature_time"] = output["decision_time"]
    output["relation_semantics"] = "symmetric_neighbor_diffusion"
    available_column = f"src_past_available_time_{past_horizon}"
    if available_column in merged:
        latest = merged.groupby(group_columns, sort=False)[available_column].max().reset_index()
        output = output.merge(latest, on=group_columns, how="left", copy=False)
        valid = output[available_column].notna()
        if valid.any() and not (output.loc[valid, available_column] <= output.loc[valid, "feature_time"]).all():
            raise AssertionError("source past return is not available at feature_time")
    return output


def relation_spillover_one(
    part: Part,
    returns_root: str | Path,
    output_root: str | Path,
    horizons: list[str],
    past_horizon: str,
    levels: set[str] | None,
    tiers: set[str] | None,
    skip_existing: bool,
    max_row_groups: int | None,
    inner_workers: int,
) -> dict:
    started = time.time()
    returns_path = Path(returns_root) / f"date={part.date}" / f"layer_id={part.layer_id}" / f"scale={part.scale}" / "theme_returns.parquet"
    output_dir = Path(output_root) / f"date={part.date}" / f"layer_id={part.layer_id}" / f"scale={part.scale}"
    output_path = output_dir / "relation_spillover_signals.parquet"
    if skip_existing and is_complete(output_dir / "manifest.json"):
        return {"stage": "relation_spillover", "status": "skipped", "date": part.date, "layer_id": part.layer_id, "scale": part.scale}
    if not returns_path.exists():
        return {"stage": "relation_spillover", "status": "missing_theme_returns", "date": part.date, "layer_id": part.layer_id, "scale": part.scale}

    returns = read_partition(returns_path)
    returns["decision_time"] = pd.to_datetime(returns["decision_time"], utc=True, errors="coerce")
    returns["layer_id"] = returns["layer_id"].astype(str)
    returns["scale"] = returns["scale"].astype(str)
    past_column = f"past_eq_{past_horizon}"
    if past_column not in returns:
        raise ValueError(f"missing {past_column} in {returns_path}")

    source_columns = ["decision_time", "layer_id", "scale", "level", "theme_id", past_column]
    past_available = f"past_available_time_{past_horizon}"
    if past_available in returns:
        source_columns.append(past_available)
    source = returns[source_columns].rename(
        columns={
            "theme_id": "src_theme_id",
            past_column: "src_past_return",
            past_available: f"src_past_available_time_{past_horizon}",
        }
    )

    target_columns = ["decision_time", "layer_id", "scale", "level", "theme_id"]
    rename: dict[str, str] = {"theme_id": "dst_theme_id"}
    for horizon in horizons:
        for prefix in ("ret_eq_", "past_eq_", "target_entry_time_", "target_exit_time_", "past_available_time_", "target_entry_date_", "target_exit_date_"):
            column = f"{prefix}{horizon}"
            if column not in returns:
                continue
            target_columns.append(column)
            if prefix == "ret_eq_":
                rename[column] = f"target_{horizon}"
            elif prefix == "past_eq_":
                rename[column] = f"dst_past_eq_{horizon}"
            elif prefix == "past_available_time_":
                rename[column] = f"dst_past_available_time_{horizon}"
            else:
                rename[column] = column
    target = returns[target_columns].rename(columns=rename)
    target_indexed = target.set_index("decision_time", drop=False).sort_index()
    del returns, target

    edges = read_partition(
        part.base,
        ["decision_time", "layer_id", "scale", "level", "src_theme_id", "dst_theme_id", "relation_strength", "relation_tier", "hard_keep", "edge_count"],
        max_row_groups,
    )
    if "decision_time" not in edges or edges["decision_time"].isna().all():
        edges["decision_time"] = parse_theme_ts_series(edges["src_theme_id"])
    edges["decision_time"] = pd.to_datetime(edges["decision_time"], utc=True, errors="coerce")
    edges = edges.dropna(subset=["decision_time", "src_theme_id", "dst_theme_id"]).copy()
    if levels and "level" in edges:
        edges = edges[edges["level"].astype(str).isin(levels)]
    if tiers and "relation_tier" in edges:
        edges = edges[edges["relation_tier"].astype(str).isin(tiers)]
    if "layer_id" not in edges:
        edges["layer_id"] = part.layer_id
    if "scale" not in edges:
        edges["scale"] = part.scale
    edges["layer_id"] = edges["layer_id"].astype(str)
    edges["scale"] = edges["scale"].astype(str)
    edges_indexed = edges.set_index("decision_time", drop=False).sort_index()
    del edges

    def one(item: tuple[pd.Timestamp, pd.DataFrame]) -> pd.DataFrame | None:
        decision_time, source_at_time = item
        edge_at_time = _lookup_time(edges_indexed, decision_time)
        if edge_at_time.empty:
            return None
        edge_at_time = expand_symmetric_relations(edge_at_time)
        return _relation_signal_frame(
            edge_at_time,
            source_at_time,
            _lookup_time(target_indexed, decision_time),
            past_horizon,
        )

    groups = source.groupby("decision_time", sort=False, dropna=False)
    if inner_workers > 1:
        base_frames: Iterable[pd.DataFrame | None] = bounded_thread_map(
            groups,
            inner_workers,
            one,
            max_in_flight=inner_workers * 2,
        )
    else:
        base_frames = (one(item) for item in groups)

    def frames() -> Iterable[pd.DataFrame | None]:
        for frame in base_frames:
            if frame is not None:
                frame.insert(1, "date", part.date)
            yield frame

    rows, batches = stream_frames(output_path, frames())
    metadata = {
        "stage": "relation_spillover",
        "status": "complete" if rows else "empty",
        "date": part.date,
        "layer_id": part.layer_id,
        "scale": part.scale,
        "past_horizon": past_horizon,
        "relation_semantics": "symmetric_neighbor_diffusion",
        "symmetric_expansion_scope": "single_snapshot",
        "input_grouping": "lazy_bounded_decision_time",
        "inner_workers": inner_workers,
        "output_rows": rows,
        "write_batches": batches,
        "output": str(output_path),
        "elapsed_sec": round(time.time() - started, 3),
    }
    write_manifest(output_dir, metadata)
    return metadata
