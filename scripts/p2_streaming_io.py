#!/usr/bin/env python3
"""Fail-clean, row-group-buffered streaming I/O for P2 stages."""
from __future__ import annotations

import os
import shutil
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def _target_rows(value: int | None) -> int:
    if value is not None:
        return max(1, int(value))
    try:
        return max(1, int(os.environ.get("GFF_PARQUET_TARGET_ROWS", "100000")))
    except (TypeError, ValueError):
        return 100_000


def _minimum_free_gb() -> float:
    try:
        return max(0.0, float(os.environ.get("GFF_MIN_FREE_GB", "0")))
    except (TypeError, ValueError):
        return 0.0


def _check_disk_fuse(path: Path, minimum_free_gb: float) -> None:
    if minimum_free_gb <= 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(path.parent).free / (1024**3)
    if free_gb < minimum_free_gb:
        raise OSError(
            f"streamed parquet disk fuse: {free_gb:.2f}GB free < "
            f"{minimum_free_gb:.2f}GB required for {path}"
        )


def stream_frames(
    path: str | Path,
    frames: Iterable[pd.DataFrame | None],
    *,
    target_rows: int | None = None,
) -> tuple[int, int]:
    """Write ordered frames atomically with bounded row-group compaction.

    Snapshot frames are accumulated only until ``target_rows`` and then flushed
    as one or more Parquet row groups. Existing final output is left untouched
    on failure; stale final output is removed only after a successful empty run.
    """
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(output) + ".tmp")
    temporary.unlink(missing_ok=True)
    target = _target_rows(target_rows)
    minimum_free_gb = _minimum_free_gb()
    _check_disk_fuse(output, minimum_free_gb)

    writer: pq.ParquetWriter | None = None
    schema: pa.Schema | None = None
    buffered: list[pa.Table] = []
    buffered_rows = 0
    rows = writes = 0

    def ensure_writer(table_schema: pa.Schema) -> pq.ParquetWriter:
        nonlocal writer, schema
        if writer is None:
            schema = table_schema
            writer = pq.ParquetWriter(
                temporary,
                schema,
                compression="zstd",
                use_dictionary=True,
            )
        return writer

    def flush() -> None:
        nonlocal buffered, buffered_rows, rows, writes
        if not buffered:
            return
        _check_disk_fuse(output, minimum_free_gb)
        table = buffered[0] if len(buffered) == 1 else pa.concat_tables(buffered)
        ensure_writer(table.schema).write_table(table, row_group_size=target)
        rows += table.num_rows
        writes += max(1, (table.num_rows + target - 1) // target)
        buffered = []
        buffered_rows = 0
        del table

    try:
        for frame in frames:
            if frame is None or frame.empty:
                continue
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if schema is not None and table.schema != schema:
                table = table.cast(schema)
            elif schema is None and buffered and table.schema != buffered[0].schema:
                table = table.cast(buffered[0].schema)

            if table.num_rows >= target:
                flush()
                _check_disk_fuse(output, minimum_free_gb)
                ensure_writer(table.schema).write_table(table, row_group_size=target)
                rows += table.num_rows
                writes += max(1, (table.num_rows + target - 1) // target)
                del table
                continue

            buffered.append(table)
            buffered_rows += table.num_rows
            if buffered_rows >= target:
                flush()
        flush()
        if writer is not None:
            writer.close()
            writer = None
        if rows:
            os.replace(temporary, output)
        else:
            temporary.unlink(missing_ok=True)
            output.unlink(missing_ok=True)
        return rows, writes
    except BaseException:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)
        raise
