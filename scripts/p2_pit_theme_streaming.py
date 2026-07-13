#!/usr/bin/env python3
"""Input-streaming, resumable P2 theme-return and relation-spillover stages."""
from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from p2_checkpoint import file_fingerprint, read_json, stage_checkpoint_valid
from p2_parallel_runtime import bounded_thread_map_ordered
from p2_pit_core import Part, iter_time_groups, label_path, load_labels, merge_time_group_streams, write_manifest
from p2_pit_theme import _aggregate_theme_returns, _relation_signal_frame, expand_symmetric_relations
from p2_streaming_io import stream_frames

THEME_RETURNS_CONTRACT = "theme-returns-stream-v3"
RELATION_SPILLOVER_CONTRACT = "relation-spillover-stream-v3"
_LABEL_CACHE_KEY: tuple | None = None
_LABEL_CACHE_INDEXED: pd.DataFrame | None = None


def _lookup_indexed(indexed: pd.DataFrame, decision_time: pd.Timestamp) -> pd.DataFrame:
    try:
        return indexed.loc[[decision_time]]
    except KeyError:
        return pd.DataFrame(columns=indexed.columns)


def _cached_labels_indexed(path: Path, fingerprint: dict, horizons: list[str]) -> pd.DataFrame:
    """Keep at most one indexed daily label table in each long-lived worker."""
    global _LABEL_CACHE_KEY, _LABEL_CACHE_INDEXED
    key = (
        fingerprint["path"],
        fingerprint["size_bytes"],
        fingerprint["mtime_ns"],
        tuple(horizons),
    )
    if _LABEL_CACHE_KEY != key or _LABEL_CACHE_INDEXED is None:
        labels = load_labels(path, horizons)
        _LABEL_CACHE_INDEXED = labels.set_index("decision_time", drop=False).sort_index()
        _LABEL_CACHE_KEY = key
    return _LABEL_CACHE_INDEXED


def _normalise_membership(group: pd.DataFrame, part: Part, levels: set[str] | None) -> pd.DataFrame:
    group = group.copy()
    group["decision_time"] = pd.to_datetime(group["decision_time"], utc=True, errors="coerce")
    group["member_id"] = pd.to_numeric(group["member_id"], errors="coerce").astype("Int64")
    group["core_score"] = pd.to_numeric(group.get("core_score", 0), errors="coerce").fillna(0.0).astype("float32")
    group = group.dropna(subset=["decision_time", "member_id", "theme_id"])
    if group.empty:
        return group
    group["member_id"] = group["member_id"].astype("int64")
    if "level" not in group:
        group["level"] = "UNKNOWN"
    if levels:
        group = group[group["level"].astype(str).isin(levels)]
    if "layer_id" not in group:
        group["layer_id"] = part.layer_id
    if "scale" not in group:
        group["scale"] = part.scale
    group["layer_id"] = group["layer_id"].astype(str)
    group["scale"] = group["scale"].astype(str)
    if not group.empty:
        if not group["layer_id"].eq(str(part.layer_id)).all():
            raise ValueError(f"mixed layer_id inside membership partition {part.base}")
        if not group["scale"].eq(str(part.scale)).all():
            raise ValueError(f"mixed scale inside membership partition {part.base}")
    return group


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
    labels_file = label_path(labels_root, part.date)
    labels_fingerprint = file_fingerprint(labels_file)
    inputs = {"memberships": file_fingerprint(part.base), "labels": labels_fingerprint}
    config = {
        "horizons": list(horizons),
        "levels": sorted(levels) if levels else None,
        "max_row_groups": max_row_groups,
    }
    if skip_existing and stage_checkpoint_valid(
        output_dir / "manifest.json",
        stage="theme_returns",
        contract_version=THEME_RETURNS_CONTRACT,
        inputs=inputs,
        config=config,
        output_path=output_path,
    ):
        return {"stage": "theme_returns", "status": "skipped", "date": part.date, "layer_id": part.layer_id, "scale": part.scale}

    labels_indexed = _cached_labels_indexed(labels_file, labels_fingerprint, horizons)
    columns = ["decision_time", "layer_id", "scale", "level", "theme_id", "member_id", "core_score", "rank_in_theme"]

    def groups() -> Iterable[tuple[pd.Timestamp, pd.DataFrame]]:
        for decision_time, membership in iter_time_groups(part.base, columns, max_row_groups):
            membership = _normalise_membership(membership, part, levels)
            if not membership.empty:
                yield decision_time, membership

    def one(item: tuple[pd.Timestamp, pd.DataFrame]) -> pd.DataFrame | None:
        decision_time, membership = item
        return _aggregate_theme_returns(membership, _lookup_indexed(labels_indexed, decision_time), horizons)

    if inner_workers > 1:
        frames: Iterable[pd.DataFrame | None] = bounded_thread_map_ordered(
            groups(), inner_workers, one, max_in_flight=inner_workers * 2
        )
    else:
        frames = (one(item) for item in groups())
    rows, batches = stream_frames(output_path, frames)
    metadata = {
        "stage": "theme_returns",
        "stage_contract_version": THEME_RETURNS_CONTRACT,
        "status": "complete" if rows else "empty",
        "date": part.date,
        "layer_id": part.layer_id,
        "scale": part.scale,
        "output_rows": rows,
        "write_batches": batches,
        "output": str(output_path),
        "inputs": inputs,
        "config": config,
        "labels_source": str(labels_file),
        "past_return_availability": "actual_label_exit_time",
        "input_mode": "parquet_batch_time_group_stream",
        "output_order": "decision_time_ascending",
        "maximum_time_groups_in_flight": max(1, inner_workers * 2),
        "elapsed_sec": round(time.time() - started, 3),
    }
    write_manifest(output_dir, metadata)
    return metadata


def _returns_columns(path: Path, horizons: list[str], past_horizon: str) -> list[str]:
    parquet = pq.ParquetFile(path)
    try:
        names = set(parquet.schema.names)
    finally:
        parquet.close()
    columns = ["decision_time", "layer_id", "scale", "level", "theme_id", f"past_eq_{past_horizon}"]
    optional_prefixes = (
        "ret_eq_", "past_eq_", "target_entry_time_", "target_exit_time_",
        "past_available_time_", "target_entry_date_", "target_exit_date_",
    )
    for horizon in horizons:
        for prefix in optional_prefixes:
            column = f"{prefix}{horizon}"
            if column in names:
                columns.append(column)
    return list(dict.fromkeys(column for column in columns if column in names))


def _normalise_returns(group: pd.DataFrame, part: Part) -> pd.DataFrame:
    group = group.copy()
    group["decision_time"] = pd.to_datetime(group["decision_time"], utc=True, errors="coerce")
    group["layer_id"] = group["layer_id"].astype(str)
    group["scale"] = group["scale"].astype(str)
    if not group.empty:
        if not group["layer_id"].eq(str(part.layer_id)).all():
            raise ValueError("mixed layer_id inside theme-return partition")
        if not group["scale"].eq(str(part.scale)).all():
            raise ValueError("mixed scale inside theme-return partition")
    return group


def _source_target(returns: pd.DataFrame, horizons: list[str], past_horizon: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    past_column = f"past_eq_{past_horizon}"
    if past_column not in returns:
        raise ValueError(f"missing {past_column} in streamed theme returns")
    source_columns = ["decision_time", "layer_id", "scale", "level", "theme_id", past_column]
    past_available = f"past_available_time_{past_horizon}"
    if past_available in returns:
        source_columns.append(past_available)
    source = returns[source_columns].rename(columns={
        "theme_id": "src_theme_id",
        past_column: "src_past_return",
        past_available: f"src_past_available_time_{past_horizon}",
    })

    target_columns = ["decision_time", "layer_id", "scale", "level", "theme_id"]
    rename: dict[str, str] = {"theme_id": "dst_theme_id"}
    for horizon in horizons:
        for prefix in (
            "ret_eq_", "past_eq_", "target_entry_time_", "target_exit_time_",
            "past_available_time_", "target_entry_date_", "target_exit_date_",
        ):
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
    return source, returns[target_columns].rename(columns=rename)


def _normalise_edges(edges: pd.DataFrame, part: Part, levels: set[str] | None, tiers: set[str] | None) -> pd.DataFrame:
    edges = edges.copy()
    edges["decision_time"] = pd.to_datetime(edges["decision_time"], utc=True, errors="coerce")
    edges = edges.dropna(subset=["decision_time", "src_theme_id", "dst_theme_id"])
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
    if not edges.empty:
        if not edges["layer_id"].eq(str(part.layer_id)).all():
            raise ValueError(f"mixed layer_id inside relation partition {part.base}")
        if not edges["scale"].eq(str(part.scale)).all():
            raise ValueError(f"mixed scale inside relation partition {part.base}")
    return edges


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
    returns_dir = Path(returns_root) / f"date={part.date}" / f"layer_id={part.layer_id}" / f"scale={part.scale}"
    returns_path = returns_dir / "theme_returns.parquet"
    returns_manifest = returns_dir / "manifest.json"
    output_dir = Path(output_root) / f"date={part.date}" / f"layer_id={part.layer_id}" / f"scale={part.scale}"
    output_path = output_dir / "relation_spillover_signals.parquet"
    config = {
        "horizons": list(horizons),
        "past_horizon": past_horizon,
        "levels": sorted(levels) if levels else None,
        "tiers": sorted(tiers) if tiers else None,
        "max_row_groups": max_row_groups,
    }

    if not returns_path.exists():
        upstream = read_json(returns_manifest)
        if upstream and upstream.get("status") == "empty" and upstream.get("stage_contract_version") == THEME_RETURNS_CONTRACT:
            inputs = {
                "relation_edges": file_fingerprint(part.base),
                "theme_returns_manifest": file_fingerprint(returns_manifest),
            }
            if skip_existing and stage_checkpoint_valid(
                output_dir / "manifest.json",
                stage="relation_spillover",
                contract_version=RELATION_SPILLOVER_CONTRACT,
                inputs=inputs,
                config=config,
                output_path=output_path,
            ):
                return {"stage": "relation_spillover", "status": "skipped", "date": part.date, "layer_id": part.layer_id, "scale": part.scale}
            output_path.unlink(missing_ok=True)
            metadata = {
                "stage": "relation_spillover",
                "stage_contract_version": RELATION_SPILLOVER_CONTRACT,
                "status": "empty",
                "date": part.date,
                "layer_id": part.layer_id,
                "scale": part.scale,
                "output_rows": 0,
                "write_batches": 0,
                "inputs": inputs,
                "config": config,
                "upstream_status": "empty_theme_returns",
                "elapsed_sec": round(time.time() - started, 3),
            }
            write_manifest(output_dir, metadata)
            return metadata
        if upstream and upstream.get("status") == "complete":
            raise FileNotFoundError(f"theme return manifest is complete but output is missing: {returns_path}")
        return {"stage": "relation_spillover", "status": "missing_theme_returns", "date": part.date, "layer_id": part.layer_id, "scale": part.scale}

    inputs = {"relation_edges": file_fingerprint(part.base), "theme_returns": file_fingerprint(returns_path)}
    if skip_existing and stage_checkpoint_valid(
        output_dir / "manifest.json",
        stage="relation_spillover",
        contract_version=RELATION_SPILLOVER_CONTRACT,
        inputs=inputs,
        config=config,
        output_path=output_path,
    ):
        return {"stage": "relation_spillover", "status": "skipped", "date": part.date, "layer_id": part.layer_id, "scale": part.scale}

    return_groups = iter_time_groups(returns_path, _returns_columns(returns_path, horizons, past_horizon), max_row_groups)
    edge_columns = [
        "decision_time", "layer_id", "scale", "level", "src_theme_id", "dst_theme_id",
        "relation_strength", "relation_tier", "hard_keep", "edge_count",
    ]
    edge_groups = iter_time_groups(part.base, edge_columns, max_row_groups)

    def inputs_stream() -> Iterable[tuple[pd.Timestamp, pd.DataFrame, pd.DataFrame]]:
        for decision_time, returns_at_time, edges_at_time in merge_time_group_streams(return_groups, edge_groups):
            returns_at_time = _normalise_returns(returns_at_time, part)
            edges_at_time = _normalise_edges(edges_at_time, part, levels, tiers)
            if not returns_at_time.empty and not edges_at_time.empty:
                yield decision_time, returns_at_time, edges_at_time

    def one(item: tuple[pd.Timestamp, pd.DataFrame, pd.DataFrame]) -> pd.DataFrame | None:
        _, returns_at_time, edges_at_time = item
        source, target = _source_target(returns_at_time, horizons, past_horizon)
        expanded = expand_symmetric_relations(edges_at_time)
        output = _relation_signal_frame(expanded, source, target, past_horizon)
        if output is not None:
            output.insert(1, "date", part.date)
        return output

    if inner_workers > 1:
        frames: Iterable[pd.DataFrame | None] = bounded_thread_map_ordered(
            inputs_stream(), inner_workers, one, max_in_flight=inner_workers * 2
        )
    else:
        frames = (one(item) for item in inputs_stream())
    rows, batches = stream_frames(output_path, frames)
    metadata = {
        "stage": "relation_spillover",
        "stage_contract_version": RELATION_SPILLOVER_CONTRACT,
        "status": "complete" if rows else "empty",
        "date": part.date,
        "layer_id": part.layer_id,
        "scale": part.scale,
        "past_horizon": past_horizon,
        "relation_semantics": "symmetric_neighbor_diffusion",
        "symmetric_expansion_scope": "single_snapshot",
        "input_mode": "dual_sorted_time_stream",
        "output_order": "decision_time_ascending",
        "maximum_time_groups_in_flight": max(1, inner_workers * 2),
        "output_rows": rows,
        "write_batches": batches,
        "output": str(output_path),
        "inputs": inputs,
        "config": config,
        "elapsed_sec": round(time.time() - started, 3),
    }
    write_manifest(output_dir, metadata)
    return metadata
