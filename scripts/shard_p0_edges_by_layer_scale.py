#!/usr/bin/env python3
"""Physically shard P0 canonical edge files by date/layer/scale.

This is a preprocessing step for large-scale P1. It prevents each P1 worker
from loading an entire trading day with every layer/scale in memory. The output
layout is worker-friendly:

    <out-root>/date=YYYY-MM-DD/layer_id=<L>/scale=<S>/edges.parquet

The script reads input parquet files in batches and appends to per-shard parquet
writers, so it does not need to materialize a full date in pandas memory.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


EDGE_COLUMNS = ["decision_time", "layer_id", "src_id", "dst_id", "weight"]
OPTIONAL_SCALE_COLUMNS = ["lookback_minutes", "scale"]


@dataclass
class ShardStats:
    date: str
    layer_id: str
    scale: str
    rows: int = 0
    batches: int = 0
    path: str = ""


def parse_dates(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {x.strip() for x in value.split(",") if x.strip()}


def infer_date(path: Path, explicit_date: str | None = None) -> str:
    if explicit_date:
        return explicit_date
    for part in [path.name, *[p.name for p in path.parents]]:
        m = re.search(r"date=(\d{4}-\d{2}-\d{2})", part)
        if m:
            return m.group(1)
    return "unknown"


def find_edge_files(p0_root: Path, dates: set[str] | None = None) -> list[Path]:
    if p0_root.is_file():
        return [p0_root]
    files = sorted(p0_root.rglob("edges.parquet"))
    if dates is not None:
        files = [f for f in files if infer_date(f) in dates]
    return files


def safe_token(x: object) -> str:
    s = str(x)
    return "".join(ch if ch.isalnum() or ch in "=-_." else "_" for ch in s)


def normalize_batch(batch: pa.RecordBatch, date: str) -> pd.DataFrame:
    names = set(batch.schema.names)
    missing = [c for c in EDGE_COLUMNS if c not in names]
    if missing:
        raise KeyError(f"missing required columns {missing}; available={batch.schema.names}")

    cols = EDGE_COLUMNS + [c for c in OPTIONAL_SCALE_COLUMNS if c in names]
    df = batch.select(cols).to_pandas(types_mapper=None)
    df = df[df["src_id"] != df["dst_id"]].copy()
    df["date"] = date
    if "lookback_minutes" in df.columns:
        lookback = pd.to_numeric(df["lookback_minutes"], errors="coerce")
        df["scale"] = lookback.map(lambda x: f"{int(x)}m" if pd.notna(x) else "default")
        df = df.drop(columns=["lookback_minutes"])
    elif "scale" not in df.columns:
        df["scale"] = "default"
    df["layer_id"] = df["layer_id"].astype("int64")
    df["src_id"] = df["src_id"].astype("int64")
    df["dst_id"] = df["dst_id"].astype("int64")
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0).astype("float64")
    df["scale"] = df["scale"].astype(str)
    return df[["date", "decision_time", "layer_id", "scale", "src_id", "dst_id", "weight"]]


class ShardWriterPool:
    def __init__(self, out_root: Path, compression: str = "zstd") -> None:
        self.out_root = out_root
        self.compression = compression
        self.writers: dict[tuple[str, str, str], pq.ParquetWriter] = {}
        self.stats: dict[tuple[str, str, str], ShardStats] = {}

    def write_group(self, date: str, layer_id: object, scale: object, df: pd.DataFrame) -> None:
        layer_token = safe_token(layer_id)
        scale_token = safe_token(scale)
        key = (date, layer_token, scale_token)
        shard_dir = self.out_root / f"date={date}" / f"layer_id={layer_token}" / f"scale={scale_token}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        path = shard_dir / "edges.parquet"
        out = df[["decision_time", "layer_id", "scale", "src_id", "dst_id", "weight"]].copy()
        table = pa.Table.from_pandas(out, preserve_index=False)
        writer = self.writers.get(key)
        if writer is None:
            writer = pq.ParquetWriter(path, table.schema, compression=self.compression)
            self.writers[key] = writer
            self.stats[key] = ShardStats(date=date, layer_id=layer_token, scale=scale_token, path=str(path))
        else:
            table = table.cast(writer.schema)
        writer.write_table(table)
        self.stats[key].rows += len(out)
        self.stats[key].batches += 1

    def close(self) -> list[ShardStats]:
        for writer in self.writers.values():
            writer.close()
        self.writers.clear()
        return sorted(self.stats.values(), key=lambda s: (s.date, s.layer_id, s.scale))


def shard_file(
    file_path: Path,
    out_root: Path,
    explicit_date: str | None,
    batch_size: int,
    max_batches: int | None,
    compression: str,
) -> list[ShardStats]:
    date = infer_date(file_path, explicit_date)
    pf = pq.ParquetFile(file_path)
    columns = [c for c in EDGE_COLUMNS + OPTIONAL_SCALE_COLUMNS if c in pf.schema.names]
    pool = ShardWriterPool(out_root, compression=compression)
    try:
        for batch_idx, batch in enumerate(pf.iter_batches(batch_size=batch_size, columns=columns), start=1):
            if max_batches is not None and batch_idx > max_batches:
                break
            df = normalize_batch(batch, date)
            if df.empty:
                continue
            for (layer_id, scale), grp in df.groupby(["layer_id", "scale"], sort=False):
                pool.write_group(date, layer_id, scale, grp)
            print(json.dumps({"file": str(file_path), "date": date, "batch": batch_idx, "rows": len(df)}, default=str), flush=True)
    finally:
        stats = pool.close()
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Shard P0 edges by date/layer/scale for multi-worker P1.")
    ap.add_argument("--p0-root", required=True, help="P0 canonical root or a single edges.parquet file")
    ap.add_argument("--out-root", required=True, help="Output root for physical edge shards")
    ap.add_argument("--dates", default=None, help="Comma-separated YYYY-MM-DD filters when p0-root is a directory")
    ap.add_argument("--date", default=None, help="Explicit date when p0-root is a standalone file")
    ap.add_argument("--batch-size", type=int, default=1_000_000)
    ap.add_argument("--max-batches", type=int, default=None, help="Smoke-test cap")
    ap.add_argument("--compression", default="zstd")
    args = ap.parse_args()

    p0_root = Path(args.p0_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    files = find_edge_files(p0_root, parse_dates(args.dates))
    if not files:
        raise FileNotFoundError(f"no edges.parquet found under {p0_root}")

    all_stats: list[ShardStats] = []
    for f in files:
        all_stats.extend(shard_file(f, out_root, args.date, args.batch_size, args.max_batches, args.compression))

    stats_df = pd.DataFrame([asdict(s) for s in all_stats])
    stats_df.to_csv(out_root / "shard_manifest.csv", index=False)
    (out_root / "shard_manifest.json").write_text(json.dumps([asdict(s) for s in all_stats], indent=2), encoding="utf-8")
    print(json.dumps({"edge_files": len(files), "shards": len(all_stats), "manifest": str(out_root / "shard_manifest.csv")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
