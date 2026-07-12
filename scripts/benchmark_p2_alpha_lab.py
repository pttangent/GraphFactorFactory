#!/usr/bin/env python3
"""Benchmark P2 alpha-lab implementations on real parquet inputs.

This is not a correctness smoke test.  It is intended to compare runtime shape
for competing implementations on the same real partition:

1. old_bad_same_times: reproduces the historical mistake where full-day
   membership is merged against each label decision_time chunk.  To keep the
   benchmark bounded, it uses the same number of label chunks as the sampled
   membership row groups.  The real full-day bug is worse because it can repeat
   this across all intraday label chunks.
2. aligned_iw<N>: current intended architecture: membership[decision_time=t]
   joins only labels[decision_time=t], then streams each result batch to parquet.

The expected production conclusion from the included real-data benchmark is:
use inner_workers=1 and saturate CPU by outer partition/date workers.  Inner
threads can be revisited only if a machine has spare CPU and low memory pressure.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_HORIZONS = ["5m", "15m", "30m", "60m", "120m"]


def mins(horizon: str) -> int:
    if not horizon.endswith("m"):
        raise ValueError(f"only minute horizons are supported: {horizon}")
    return int(horizon[:-1])


def rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2


def read_partition(path: Path, columns: list[str], max_row_groups: int | None) -> pd.DataFrame:
    pf = pq.ParquetFile(path)
    names = set(pf.schema.names)
    cols = [c for c in columns if c in names]
    if max_row_groups is None:
        return pd.read_parquet(path, columns=cols)
    tabs = [pf.read_row_group(i, columns=cols) for i in range(min(max_row_groups, pf.metadata.num_row_groups))]
    return pa.concat_tables(tabs).to_pandas() if tabs else pd.DataFrame()


def load_labels(path: Path, horizons: list[str]) -> tuple[pd.DataFrame, list[str]]:
    names = set(pq.ParquetFile(path).schema.names)
    label_cols = [f"label_{h}" for h in horizons if f"label_{h}" in names]
    if not label_cols:
        raise ValueError(f"no requested label_* columns found in {path}")
    df = pd.read_parquet(path, columns=["decision_time", "symbol_id"] + label_cols)
    df["decision_time"] = pd.to_datetime(df["decision_time"], utc=True)
    df["symbol_id"] = pd.to_numeric(df["symbol_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["decision_time", "symbol_id"]).copy()
    df["symbol_id"] = df["symbol_id"].astype("int64")
    for h in horizons:
        col = f"label_{h}"
        if col in df:
            past = df[["decision_time", "symbol_id", col]].copy()
            past["decision_time"] = past["decision_time"] + pd.Timedelta(minutes=mins(h))
            past = past.rename(columns={col: f"past_label_{h}"})
            df = df.merge(past, on=["decision_time", "symbol_id"], how="left")
    alpha_cols = [c for c in df.columns if c.startswith("label_") or c.startswith("past_label_")]
    return df, alpha_cols


def prep_memberships(path: Path, max_row_groups: int | None) -> pd.DataFrame:
    mem = read_partition(
        path,
        ["decision_time", "layer_id", "scale", "level", "theme_id", "member_id", "core_score", "rank_in_theme"],
        max_row_groups,
    )
    mem["decision_time"] = pd.to_datetime(mem["decision_time"], utc=True)
    mem["member_id"] = pd.to_numeric(mem["member_id"], errors="coerce").astype("Int64")
    mem["core_score"] = pd.to_numeric(mem.get("core_score", 0), errors="coerce").fillna(0.0)
    mem = mem.dropna(subset=["decision_time", "member_id", "theme_id"]).copy()
    mem["member_id"] = mem["member_id"].astype("int64")
    return mem


def calc_returns(df: pd.DataFrame, alpha_cols: list[str]) -> tuple[pd.DataFrame | None, int]:
    if df.empty:
        return None, 0
    group_cols = ["decision_time", "layer_id", "scale", "level", "theme_id"]
    df = df.sort_values(group_cols + ["core_score"], ascending=[True, True, True, True, True, False])
    grouped = df.groupby(group_cols, sort=False)

    result = grouped[alpha_cols].mean()
    result.columns = [
        c.replace("past_label_", "past_eq_") if c.startswith("past_label_") else c.replace("label_", "ret_eq_")
        for c in result.columns
    ]

    weights = grouped["core_score"].sum().replace(0, np.nan)
    for c in alpha_cols:
        weighted = df[c] * df["core_score"]
        name = c.replace("past_label_", "past_core_") if c.startswith("past_label_") else c.replace("label_", "ret_core_")
        result[name] = weighted.groupby([df[col] for col in group_cols], sort=False).sum() / weights

    top5 = grouped.head(5).groupby(group_cols, sort=False)[alpha_cols].mean()
    top5.columns = [
        c.replace("past_label_", "past_top5_") if c.startswith("past_label_") else c.replace("label_", "ret_top5_")
        for c in top5.columns
    ]
    return result.join(top5).reset_index(), len(df)


def stream_write(path: Path, frames) -> tuple[int, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    if tmp.exists():
        tmp.unlink()
    writer = None
    schema = None
    rows = 0
    batches = 0
    try:
        for frame in frames:
            if frame is None or frame.empty:
                continue
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(tmp, schema, compression="zstd")
            elif table.schema != schema:
                table = table.cast(schema)
            writer.write_table(table)
            rows += len(frame)
            batches += 1
    finally:
        if writer is not None:
            writer.close()
    if rows:
        os.replace(tmp, path)
    elif tmp.exists():
        tmp.unlink()
    return rows, batches


def run_aligned(mem: pd.DataFrame, labels: pd.DataFrame, alpha_cols: list[str], out_dir: Path, max_rg: int, inner_workers: int) -> dict:
    labels_by_time = {k: v for k, v in labels.groupby("decision_time", sort=False, dropna=False)}
    groups = list(mem.groupby("decision_time", sort=False, dropna=False))
    peak = rss_mb()
    joined_rows = 0

    def one(item):
        nonlocal peak
        dt, mem_chunk = item
        labels_chunk = labels_by_time.get(dt)
        if labels_chunk is None:
            return None, 0
        joined = mem_chunk.merge(
            labels_chunk,
            left_on=["decision_time", "member_id"],
            right_on=["decision_time", "symbol_id"],
            how="inner",
        )
        peak = max(peak, rss_mb())
        out, jr = calc_returns(joined, alpha_cols)
        peak = max(peak, rss_mb())
        return out, jr

    start = time.perf_counter()
    if inner_workers <= 1:
        frames = []
        for group in groups:
            frame, jr = one(group)
            joined_rows += jr
            frames.append(frame)
        output_rows, batches = stream_write(out_dir / f"aligned_rg{max_rg}_iw{inner_workers}.parquet", frames)
    else:
        def frame_iter():
            nonlocal joined_rows
            with ThreadPoolExecutor(max_workers=inner_workers) as executor:
                futures = [executor.submit(one, group) for group in groups]
                for future in as_completed(futures):
                    frame, jr = future.result()
                    joined_rows += jr
                    yield frame
        output_rows, batches = stream_write(out_dir / f"aligned_rg{max_rg}_iw{inner_workers}.parquet", frame_iter())

    elapsed = time.perf_counter() - start
    return {
        "variant": f"aligned_iw{inner_workers}",
        "max_row_groups": max_rg,
        "membership_rows": int(len(mem)),
        "time_groups": int(len(groups)),
        "joined_rows": int(joined_rows),
        "output_rows": int(output_rows),
        "write_batches": int(batches),
        "elapsed_sec": round(elapsed, 6),
        "joined_rows_per_sec": round(joined_rows / elapsed, 3) if elapsed else None,
        "rss_peak_mb_approx": round(peak, 3),
    }


def run_old_bad_same_times(mem: pd.DataFrame, labels: pd.DataFrame, alpha_cols: list[str], out_dir: Path, max_rg: int) -> dict:
    # Bounded reproduction of the old shape.  It repeats full sampled membership
    # once for each sampled decision_time.  The real full-day bug repeats against
    # all label decision_time chunks and is therefore worse.
    labels_by_time = {k: v for k, v in labels.groupby("decision_time", sort=False, dropna=False)}
    times = list(mem["decision_time"].drop_duplicates())
    mem_no_time = mem.drop(columns=["decision_time"])
    peak = rss_mb()
    joined_rows = 0
    frames = []
    start = time.perf_counter()
    for dt in times:
        labels_chunk = labels_by_time.get(dt)
        if labels_chunk is None:
            continue
        joined = mem_no_time.merge(labels_chunk, left_on=["member_id"], right_on=["symbol_id"], how="inner")
        peak = max(peak, rss_mb())
        out, jr = calc_returns(joined, alpha_cols)
        joined_rows += jr
        frames.append(out)
        peak = max(peak, rss_mb())
    output_rows, batches = stream_write(out_dir / f"old_bad_same_times_rg{max_rg}.parquet", frames)
    elapsed = time.perf_counter() - start
    return {
        "variant": "old_bad_same_times",
        "max_row_groups": max_rg,
        "membership_rows": int(len(mem)),
        "time_groups": int(len(times)),
        "joined_rows": int(joined_rows),
        "output_rows": int(output_rows),
        "write_batches": int(batches),
        "elapsed_sec": round(elapsed, 6),
        "joined_rows_per_sec": round(joined_rows / elapsed, 3) if elapsed else None,
        "rss_peak_mb_approx": round(peak, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-data speed benchmark for P2 theme_returns architecture")
    parser.add_argument("--membership", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--row-groups", default="5,20")
    parser.add_argument("--inner-workers", default="1,2,4,8")
    parser.add_argument("--include-old", action="store_true", help="Run bounded old-bug reproduction for comparison")
    parser.add_argument("--horizons", default=",".join(DEFAULT_HORIZONS))
    args = parser.parse_args()

    row_groups = [int(x) for x in args.row_groups.split(",") if x.strip()]
    inner_workers = [int(x) for x in args.inner_workers.split(",") if x.strip()]
    horizons = [x.strip() for x in args.horizons.split(",") if x.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    labels, alpha_cols = load_labels(args.labels, horizons)
    results = []
    for max_rg in row_groups:
        mem = prep_memberships(args.membership, max_rg)
        if args.include_old:
            gc.collect()
            results.append(run_old_bad_same_times(mem, labels, alpha_cols, args.out_dir, max_rg))
            print(json.dumps(results[-1], ensure_ascii=False), flush=True)
        for iw in inner_workers:
            gc.collect()
            results.append(run_aligned(mem, labels, alpha_cols, args.out_dir, max_rg, iw))
            print(json.dumps(results[-1], ensure_ascii=False), flush=True)

    (args.out_dir / "benchmark_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(results).to_csv(args.out_dir / "benchmark_results.csv", index=False)


if __name__ == "__main__":
    main()
