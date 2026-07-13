#!/usr/bin/env python3
"""Point-in-time-safe P2 theme/relation alpha pipeline core utilities."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("ARROW_NUM_THREADS", "1")
os.environ.setdefault("POLARS_MAX_THREADS", "1")

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PIT_CONTRACT_VERSION = "p2-pit-v2"
DEFAULT_INTRADAY_HORIZONS = ["5m", "15m", "30m", "60m", "120m"]
DEFAULT_DAILY_HORIZONS = ["1d_open", "2d_open", "3d_open", "4d_open", "5d_open", "10d_open", "20d_open", "30d_open"]
DEFAULT_HORIZONS = DEFAULT_INTRADAY_HORIZONS + DEFAULT_DAILY_HORIZONS
SCORES = [
    "daily_pressure_score",
    "daily_underreaction_score",
    "daily_consensus_score",
    "late_confirmation_score_z",
]


@dataclass(frozen=True)
class Part:
    date: str
    layer_id: str
    scale: str
    base: Path


def csvset(value: str | None) -> set[str] | None:
    return None if not value else {x.strip() for x in str(value).split(",") if x.strip()}


def csvlist(value: str | None) -> list[str] | None:
    return None if not value else [x.strip() for x in str(value).split(",") if x.strip()]


def is_intraday_horizon(horizon: str) -> bool:
    return bool(re.fullmatch(r"\d+m", str(horizon)))


def is_daily_open_horizon(horizon: str) -> bool:
    return bool(re.fullmatch(r"\d+d_open", str(horizon)))


def horizon_minutes(horizon: str) -> int:
    if not is_intraday_horizon(horizon):
        raise ValueError(f"not an intraday horizon: {horizon}")
    return int(horizon[:-1])


def ext(path: str | Path, key: str) -> str | None:
    for part in Path(path).parts:
        if part.startswith(key + "="):
            return part.split("=", 1)[1]
    return None


def canonical_theme_path(values: pd.Series) -> pd.Series:
    """Remove only the leading snapshot token; never leave a partial date."""
    return values.astype(str).str.replace(r"^ts=[^|]+\|", "", regex=True)


def parse_theme_ts_series(values: pd.Series) -> pd.Series:
    token = values.astype(str).str.extract(r"^ts=([^|]+)", expand=False)
    compact = token.str.extract(r"(\d{4}-\d{2}-\d{2})T(\d{2})(\d{2})(\d{2})", expand=True)
    parsed = pd.to_datetime(
        compact[0] + " " + compact[1] + ":" + compact[2] + ":" + compact[3],
        utc=True,
        errors="coerce",
        format="%Y-%m-%d %H:%M:%S",
    )
    missing = parsed.isna() & token.notna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(token[missing], utc=True, errors="coerce", format="mixed")
    return parsed


def label_path(root: str | Path, date: str) -> Path:
    root = Path(root)
    candidates = [
        root if root.is_file() else None,
        root / f"date={date}" / "labels.parquet",
        root / "canonical" / f"date={date}" / "labels.parquet",
        root / date / "labels.parquet",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    raise FileNotFoundError(f"labels.parquet not found for date={date} under {root}")


def discover(
    root: str | Path,
    filename: str,
    dates: set[str] | None = None,
    layers: set[str] | None = None,
    scales: set[str] | None = None,
) -> list[Part]:
    parts: list[Part] = []
    for path in Path(root).rglob(filename):
        date, layer, scale = ext(path, "date"), ext(path, "layer_id"), ext(path, "scale")
        if not date or not layer or not scale:
            continue
        if dates and date not in dates:
            continue
        if layers and layer not in layers:
            continue
        if scales and scale not in scales:
            continue
        parts.append(Part(date, layer, scale, path))
    parts.sort(key=lambda part: part.base.stat().st_size, reverse=True)
    return parts


def _arrow_to_pandas(table: pa.Table) -> pd.DataFrame:
    return table.to_pandas(split_blocks=True, self_destruct=True)


def selected_parquet_columns(parquet: pq.ParquetFile, columns: Iterable[str] | None) -> list[str] | None:
    if columns is None:
        return None
    available = set(parquet.schema.names)
    return [column for column in columns if column in available]


def iter_partition_batches(
    path: str | Path,
    columns: Iterable[str] | None = None,
    max_row_groups: int | None = None,
    batch_size: int = 250_000,
) -> Iterator[pd.DataFrame]:
    """Read a parquet partition in bounded batches and always close its handle."""
    parquet = pq.ParquetFile(path)
    try:
        selected = selected_parquet_columns(parquet, columns)
        if selected == []:
            return
        row_group_count = parquet.metadata.num_row_groups
        if max_row_groups is not None:
            row_group_count = min(row_group_count, max_row_groups)
        for row_group in range(row_group_count):
            for batch in parquet.iter_batches(
                row_groups=[row_group],
                columns=selected,
                batch_size=max(1, int(batch_size)),
                use_threads=False,
            ):
                table = pa.Table.from_batches([batch])
                frame = _arrow_to_pandas(table)
                if not frame.empty:
                    yield frame
    finally:
        parquet.close()


def iter_time_groups(
    path: str | Path,
    columns: Iterable[str] | None = None,
    max_row_groups: int | None = None,
    *,
    time_column: str = "decision_time",
    batch_size: int = 250_000,
    utc: bool = True,
) -> Iterator[tuple[pd.Timestamp, pd.DataFrame]]:
    """Yield complete timestamp groups while retaining at most one carry group.

    Input must be globally non-decreasing by ``time_column``. The check is
    deliberate: temporal P1/P2 logic must fail rather than silently reorder a
    full shard in memory.
    """
    carry: pd.DataFrame | None = None
    last_seen: pd.Timestamp | None = None
    for batch in iter_partition_batches(path, columns, max_row_groups, batch_size):
        if time_column not in batch:
            raise ValueError(f"{time_column} missing from {path}")
        batch = batch.copy()
        batch[time_column] = pd.to_datetime(batch[time_column], utc=utc, errors="coerce")
        batch = batch.dropna(subset=[time_column])
        if batch.empty:
            continue
        batch = batch.sort_values(time_column, kind="mergesort")
        batch_min = batch[time_column].iloc[0]
        if last_seen is not None and batch_min < last_seen:
            raise ValueError(f"{path} is not globally sorted by {time_column}; refusing full-file fallback")
        if carry is not None and not carry.empty:
            batch = pd.concat([carry, batch], ignore_index=True, copy=False)
        last_time = batch[time_column].iloc[-1]
        complete = batch.loc[batch[time_column] < last_time]
        carry = batch.loc[batch[time_column].eq(last_time)].copy()
        for timestamp, group in complete.groupby(time_column, sort=False, dropna=False):
            yield timestamp, group
        last_seen = last_time
    if carry is not None and not carry.empty:
        for timestamp, group in carry.groupby(time_column, sort=False, dropna=False):
            yield timestamp, group


def merge_time_group_streams(
    left: Iterable[tuple[pd.Timestamp, pd.DataFrame]],
    right: Iterable[tuple[pd.Timestamp, pd.DataFrame]],
) -> Iterator[tuple[pd.Timestamp, pd.DataFrame, pd.DataFrame]]:
    """Inner-join two sorted timestamp-group streams with constant memory."""
    left_iter, right_iter = iter(left), iter(right)
    try:
        left_item = next(left_iter)
        right_item = next(right_iter)
    except StopIteration:
        return
    while True:
        left_time, left_frame = left_item
        right_time, right_frame = right_item
        if left_time == right_time:
            yield left_time, left_frame, right_frame
            try:
                left_item = next(left_iter)
                right_item = next(right_iter)
            except StopIteration:
                return
        elif left_time < right_time:
            try:
                left_item = next(left_iter)
            except StopIteration:
                return
        else:
            try:
                right_item = next(right_iter)
            except StopIteration:
                return


def read_partition(path: str | Path, columns: Iterable[str] | None = None, max_row_groups: int | None = None) -> pd.DataFrame:
    parquet = pq.ParquetFile(path)
    try:
        selected = selected_parquet_columns(parquet, columns)
        if selected == []:
            return pd.DataFrame()
        if max_row_groups is None:
            table = parquet.read(columns=selected, use_threads=False)
        else:
            row_groups = list(range(min(max_row_groups, parquet.metadata.num_row_groups)))
            if not row_groups:
                return pd.DataFrame()
            table = parquet.read_row_groups(row_groups, columns=selected, use_threads=False)
    finally:
        parquet.close()
    return _arrow_to_pandas(table)


def write_parquet_atomic(frame: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    table = pa.Table.from_pandas(frame, preserve_index=False)
    pq.write_table(table, temporary, compression="zstd", use_dictionary=True)
    del table
    temporary.replace(path)


def write_manifest(directory: str | Path, payload: dict) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {**payload, "pit_contract_version": PIT_CONTRACT_VERSION}
    temporary = directory / "manifest.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    temporary.replace(directory / "manifest.json")


def is_complete(manifest_path: str | Path) -> bool:
    try:
        payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        return (
            payload.get("status") == "complete"
            and int(payload.get("output_rows", 0)) > 0
            and payload.get("pit_contract_version") == PIT_CONTRACT_VERSION
        )
    except Exception:
        return False


def stream_frames(path: str | Path, frames: Iterable[pd.DataFrame | None]) -> tuple[int, int]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    if temporary.exists():
        temporary.unlink()
    writer: pq.ParquetWriter | None = None
    schema: pa.Schema | None = None
    rows = batches = 0
    try:
        for frame in frames:
            if frame is None or frame.empty:
                continue
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(temporary, schema, compression="zstd", use_dictionary=True)
            elif table.schema != schema:
                table = table.cast(schema)
            writer.write_table(table)
            rows += len(frame)
            batches += 1
            del table
    finally:
        if writer is not None:
            writer.close()
    if rows:
        os.replace(temporary, path)
    elif temporary.exists():
        temporary.unlink()
    return rows, batches


def zscore_by_group(frame: pd.DataFrame, group_columns: list[str], column: str) -> pd.Series:
    def transform(series: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(series, errors="coerce")
        std = numeric.std(ddof=0)
        if not np.isfinite(std) or std == 0:
            return pd.Series(0.0, index=series.index)
        return (numeric - numeric.mean()) / std

    return frame.groupby(group_columns, sort=False, dropna=False)[column].transform(transform)


def load_labels(path: str | Path, horizons: list[str]) -> pd.DataFrame:
    """Load targets and construct only fully-realized past intraday returns."""
    parquet = pq.ParquetFile(path)
    try:
        names = set(parquet.schema.names)
        valid = [horizon for horizon in horizons if f"label_{horizon}" in names]
        if not valid:
            raise ValueError(f"no requested label columns in {path}; requested={horizons}")

        columns = ["decision_time", "symbol_id"]
        if "label_entry_time" in names:
            columns.append("label_entry_time")
        for horizon in valid:
            columns.append(f"label_{horizon}")
            for candidate in (
                f"label_entry_time_{horizon}",
                f"label_exit_time_{horizon}",
                f"label_entry_date_{horizon}",
                f"label_exit_date_{horizon}",
            ):
                if candidate in names:
                    columns.append(candidate)
        columns = list(dict.fromkeys(columns))
        table = parquet.read(columns=columns, use_threads=False)
    finally:
        parquet.close()

    frame = _arrow_to_pandas(table)
    frame["decision_time"] = pd.to_datetime(frame["decision_time"], utc=True, errors="coerce")
    frame["symbol_id"] = pd.to_numeric(frame["symbol_id"], errors="coerce").astype("Int64")
    frame = frame.dropna(subset=["decision_time", "symbol_id"]).copy()
    frame["symbol_id"] = frame["symbol_id"].astype("int64")
    if "label_entry_time" in frame:
        frame["label_entry_time"] = pd.to_datetime(frame["label_entry_time"], utc=True, errors="coerce")

    for horizon in valid:
        if not is_intraday_horizon(horizon):
            continue
        target = f"label_{horizon}"
        exit_column = f"label_exit_time_{horizon}"
        if exit_column not in frame:
            raise ValueError(
                f"{exit_column} is required for PIT-safe past_{horizon}; "
                "refusing to infer availability from the nominal horizon"
            )
        frame[exit_column] = pd.to_datetime(frame[exit_column], utc=True, errors="coerce")
        entry_column = f"label_entry_time_{horizon}" if f"label_entry_time_{horizon}" in frame else "label_entry_time"
        if entry_column in frame:
            frame[entry_column] = pd.to_datetime(frame[entry_column], utc=True, errors="coerce")

        past_columns = ["symbol_id", "decision_time", target, exit_column]
        if entry_column in frame:
            past_columns.append(entry_column)
        past = frame[past_columns].copy()
        rename = {
            "decision_time": f"past_source_decision_time_{horizon}",
            target: f"past_label_{horizon}",
            exit_column: "decision_time",
        }
        if entry_column in past:
            rename[entry_column] = f"past_entry_time_{horizon}"
        past = past.rename(columns=rename)
        past[f"past_exit_time_{horizon}"] = past["decision_time"]
        frame = frame.merge(past, on=["decision_time", "symbol_id"], how="left", validate="many_to_one", copy=False)
        available = frame[f"past_exit_time_{horizon}"].notna()
        if available.any() and not (frame.loc[available, f"past_exit_time_{horizon}"] <= frame.loc[available, "decision_time"]).all():
            raise AssertionError(f"past_label_{horizon} contains a value unavailable at decision_time")

    return frame
