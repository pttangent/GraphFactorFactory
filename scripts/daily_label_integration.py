#!/usr/bin/env python3
"""Inject next-open-executable daily labels with bounded-memory Parquet I/O."""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("ARROW_NUM_THREADS", "1")

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from p2_parallel_runtime import bounded_thread_map


def _stream_merge_label_file(
    label_path: Path,
    daily_for_date: pd.DataFrame,
    injected: list[str],
    batch_size: int,
) -> tuple[int, int, int]:
    """Rewrite one label file without materialising the whole file in Pandas."""
    if daily_for_date.empty:
        return 0, 0, 0

    parquet = pq.ParquetFile(label_path)
    existing = set(parquet.schema.names)
    replace_columns = [column for column in injected if column in existing]
    temporary = Path(str(label_path) + ".tmp")
    temporary.unlink(missing_ok=True)
    writer: pq.ParquetWriter | None = None
    output_schema: pa.Schema | None = None
    rows = batches = 0
    daily = daily_for_date.copy()
    daily["symbol_id"] = pd.to_numeric(daily["symbol_id"], errors="raise").astype("int64")

    try:
        for batch in parquet.iter_batches(batch_size=max(1, int(batch_size)), use_threads=False):
            intraday = batch.to_pandas(split_blocks=True, self_destruct=True)
            if intraday.empty:
                continue
            intraday["symbol_id"] = pd.to_numeric(intraday["symbol_id"], errors="raise").astype("int64")
            if replace_columns:
                intraday = intraday.drop(columns=replace_columns)
            merged = intraday.merge(
                daily,
                on="symbol_id",
                how="left",
                sort=False,
                validate="many_to_one",
                copy=False,
            )
            if len(merged) != len(intraday):
                raise AssertionError(f"daily label merge changed row count for {label_path}")
            table = pa.Table.from_pandas(merged, preserve_index=False)
            if writer is None:
                output_schema = table.schema
                writer = pq.ParquetWriter(
                    temporary,
                    output_schema,
                    compression="zstd",
                    use_dictionary=True,
                )
            elif table.schema != output_schema:
                table = table.cast(output_schema)
            writer.write_table(table)
            rows += len(merged)
            batches += 1
            del intraday, merged, table
    except BaseException:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)
        raise
    else:
        if writer is not None:
            writer.close()
        if rows:
            os.replace(temporary, label_path)
        else:
            temporary.unlink(missing_ok=True)
    finally:
        gc.collect()

    return (1 if rows else 0), rows, batches


def _process_label_file(task: tuple[Path, pd.DataFrame, list[str], int]) -> tuple[int, int, int]:
    return _stream_merge_label_file(*task)


def prepare_daily_labels(daily: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    required_daily = {"date", "stable_symbol_id", "next_trade_date"}
    required_mapping = {"symbol", "symbol_id"}
    if missing := required_daily - set(daily):
        raise ValueError(f"daily labels missing {sorted(missing)}")
    if missing := required_mapping - set(mapping):
        raise ValueError(f"symbol mapping missing {sorted(missing)}")

    merged = daily.merge(
        mapping[["symbol", "symbol_id"]].drop_duplicates("symbol"),
        left_on="stable_symbol_id",
        right_on="symbol",
        how="inner",
        validate="many_to_one",
    )
    output = merged[["date", "symbol_id"]].copy()
    output["date"] = output["date"].astype("string")
    output["symbol_id"] = pd.to_numeric(output["symbol_id"], errors="raise").astype("int64")

    horizon_columns: list[str] = []
    for column in merged:
        match = re.fullmatch(r"next_open_to_t(\d+)_close", column)
        if not match:
            continue
        horizon = int(match.group(1))
        suffix = f"{horizon}d_open"
        target = f"label_{suffix}"
        output[target] = pd.to_numeric(merged[column], errors="coerce").astype("float32")
        output[f"label_entry_date_{suffix}"] = merged["next_trade_date"].astype("string")
        exit_source = "next_trade_date" if horizon == 1 else f"t{horizon}_trade_date"
        if exit_source not in merged:
            raise ValueError(f"daily labels missing {exit_source} for {target}")
        output[f"label_exit_date_{suffix}"] = merged[exit_source].astype("string")
        horizon_columns.append(target)

    if not horizon_columns:
        raise ValueError("no next_open_to_tNd_close columns found; close-start labels are intentionally rejected")
    output["daily_label_execution_policy"] = pd.Series("next_session_open", index=output.index, dtype="string")
    duplicates = output.duplicated(["date", "symbol_id"], keep=False)
    if duplicates.any():
        raise ValueError(f"daily labels contain {int(duplicates.sum())} duplicate date/symbol rows")
    return output


def inject_daily_labels(
    labels_root: Path,
    daily_labels_path: Path,
    mapping_path: Path,
    month: str | None = None,
    workers: int = 8,
    batch_size: int = 250_000,
) -> dict:
    """Inject labels using bounded threads and batch-streamed file rewrites."""
    daily = pd.read_parquet(daily_labels_path)
    mapping = pd.read_parquet(mapping_path)
    prepared = prepare_daily_labels(daily, mapping)
    pattern = f"date={month}-*" if month else "date=*"
    files = sorted(path / "labels.parquet" for path in labels_root.glob(pattern) if (path / "labels.parquet").exists())
    injected = [column for column in prepared if column.startswith("label_") or column == "daily_label_execution_policy"]

    prepared_by_date = {
        str(date): subset.drop(columns="date").copy()
        for date, subset in prepared.groupby("date", sort=False)
    }
    tasks: list[tuple[Path, pd.DataFrame, list[str], int]] = []
    for label_path in files:
        date = label_path.parent.name.split("=", 1)[1]
        daily_for_date = prepared_by_date.get(date)
        if daily_for_date is not None and not daily_for_date.empty:
            tasks.append((label_path, daily_for_date, injected, batch_size))

    if not tasks:
        return {
            "updated_files": 0,
            "rows": 0,
            "batches": 0,
            "execution_policy": "next_session_open",
            "injected_columns": injected,
            "workers": 0,
            "parallel_backend": "bounded_threads",
            "input_mode": "parquet_batch_stream",
        }

    worker_count = max(1, min(int(workers), len(tasks), os.cpu_count() or 1))
    updated = rows = batches = 0
    for count, row_count, batch_count in bounded_thread_map(
        tasks,
        worker_count,
        _process_label_file,
        max_in_flight=worker_count,
    ):
        updated += count
        rows += row_count
        batches += batch_count

    return {
        "updated_files": updated,
        "rows": rows,
        "batches": batches,
        "batch_size": batch_size,
        "execution_policy": "next_session_open",
        "injected_columns": injected,
        "workers": worker_count,
        "parallel_backend": "bounded_threads",
        "input_mode": "parquet_batch_stream",
        "atomic_rewrite": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-root", required=True)
    parser.add_argument("--daily-labels", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--month")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=250_000)
    args = parser.parse_args()
    result = inject_daily_labels(
        Path(args.labels_root),
        Path(args.daily_labels),
        Path(args.mapping),
        args.month,
        args.workers,
        args.batch_size,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
