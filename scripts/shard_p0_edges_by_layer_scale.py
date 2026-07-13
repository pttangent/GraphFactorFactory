#!/usr/bin/env python3
"""Atomically shard P0 canonical edges by date/layer/real scale."""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

EDGE_COLUMNS = ["decision_time", "layer_id", "src_id", "dst_id", "weight"]
OPTIONAL_SCALE_COLUMNS = ["lookback_minutes", "scale"]
SHARD_CONTRACT_VERSION = "p0-edge-shards-v2"


@dataclass
class ShardStats:
    date: str
    layer_id: str
    scale: str
    rows: int = 0
    batches: int = 0
    path: str = ""


def parse_dates(value: str | None) -> set[str] | None:
    return None if not value else {item.strip() for item in value.split(",") if item.strip()}


def infer_date(path: Path, explicit_date: str | None = None) -> str:
    if explicit_date:
        return explicit_date
    for part in [path.name, *[parent.name for parent in path.parents]]:
        match = re.search(r"date=(\d{4}-\d{2}-\d{2})", part)
        if match:
            return match.group(1)
    return "unknown"


def find_edge_files(p0_root: Path, dates: set[str] | None = None) -> list[Path]:
    files = [p0_root] if p0_root.is_file() else sorted(p0_root.rglob("edges.parquet"))
    return [path for path in files if dates is None or infer_date(path) in dates]


def safe_token(value: object) -> str:
    return "".join(character if character.isalnum() or character in "=-_." else "_" for character in str(value))


def normalize_batch(batch: pa.RecordBatch, date: str, allow_default_scale: bool) -> pd.DataFrame:
    names = set(batch.schema.names)
    missing = [column for column in EDGE_COLUMNS if column not in names]
    if missing:
        raise KeyError(f"missing required columns {missing}; available={batch.schema.names}")
    columns = EDGE_COLUMNS + [column for column in OPTIONAL_SCALE_COLUMNS if column in names]
    frame = batch.select(columns).to_pandas(split_blocks=True, self_destruct=True)
    frame = frame[frame["src_id"] != frame["dst_id"]].copy()
    frame["date"] = date
    if "lookback_minutes" in frame:
        lookback = pd.to_numeric(frame["lookback_minutes"], errors="coerce")
        derived = lookback.map(lambda value: f"{int(value)}m" if pd.notna(value) else None)
        if "scale" in frame:
            supplied = frame["scale"].astype("string")
            mismatch = derived.notna() & supplied.notna() & supplied.ne(derived)
            if mismatch.any():
                raise ValueError(f"scale/lookback mismatch in {date}: {int(mismatch.sum())} rows")
        frame["scale"] = derived
        frame = frame.drop(columns=["lookback_minutes"])
    elif "scale" not in frame:
        if not allow_default_scale:
            raise ValueError("P0 edges have neither scale nor lookback_minutes; refusing silent scale=default")
        frame["scale"] = "default"
    frame["scale"] = frame["scale"].astype("string")
    missing_scale = frame["scale"].isna() | frame["scale"].str.strip().eq("")
    if missing_scale.any():
        if not allow_default_scale:
            raise ValueError(f"missing scale in {date}: {int(missing_scale.sum())} rows")
        frame.loc[missing_scale, "scale"] = "default"
    frame["decision_time"] = pd.to_datetime(frame["decision_time"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["decision_time"])
    frame["layer_id"] = pd.to_numeric(frame["layer_id"], errors="raise").astype("int64")
    frame["src_id"] = pd.to_numeric(frame["src_id"], errors="raise").astype("int64")
    frame["dst_id"] = pd.to_numeric(frame["dst_id"], errors="raise").astype("int64")
    frame["weight"] = pd.to_numeric(frame["weight"], errors="coerce").fillna(0.0).astype("float32")
    return frame[["date", "decision_time", "layer_id", "scale", "src_id", "dst_id", "weight"]]


class ShardWriterPool:
    def __init__(self, out_root: Path, compression: str = "zstd") -> None:
        self.out_root = out_root
        self.compression = compression
        self.writers: dict[tuple[str, str, str], pq.ParquetWriter] = {}
        self.temp_paths: dict[tuple[str, str, str], Path] = {}
        self.final_paths: dict[tuple[str, str, str], Path] = {}
        self.stats: dict[tuple[str, str, str], ShardStats] = {}
        self.last_time: dict[tuple[str, str, str], pd.Timestamp] = {}

    def write_group(self, date: str, layer_id: object, scale: object, frame: pd.DataFrame) -> None:
        layer_token, scale_token = safe_token(layer_id), safe_token(scale)
        key = (date, layer_token, scale_token)
        frame = frame.sort_values("decision_time", kind="mergesort")
        current_min = frame["decision_time"].iloc[0]
        if key in self.last_time and current_min < self.last_time[key]:
            raise ValueError(f"input is not chronological for shard {key}")
        self.last_time[key] = frame["decision_time"].iloc[-1]
        shard_dir = self.out_root / f"date={date}" / f"layer_id={layer_token}" / f"scale={scale_token}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        final_path = shard_dir / "edges.parquet"
        temp_path = Path(str(final_path) + ".tmp")
        output = frame[["decision_time", "layer_id", "scale", "src_id", "dst_id", "weight"]]
        table = pa.Table.from_pandas(output, preserve_index=False)
        writer = self.writers.get(key)
        if writer is None:
            temp_path.unlink(missing_ok=True)
            writer = pq.ParquetWriter(temp_path, table.schema, compression=self.compression, use_dictionary=True)
            self.writers[key] = writer
            self.temp_paths[key] = temp_path
            self.final_paths[key] = final_path
            self.stats[key] = ShardStats(date=date, layer_id=layer_token, scale=scale_token, path=str(final_path))
        elif table.schema != writer.schema:
            table = table.cast(writer.schema)
        writer.write_table(table)
        self.stats[key].rows += len(output)
        self.stats[key].batches += 1

    def close(self, commit: bool) -> list[ShardStats]:
        for writer in self.writers.values():
            writer.close()
        self.writers.clear()
        for key, temp_path in self.temp_paths.items():
            if commit and self.stats[key].rows:
                os.replace(temp_path, self.final_paths[key])
            else:
                temp_path.unlink(missing_ok=True)
        return sorted(self.stats.values(), key=lambda stat: (stat.date, stat.layer_id, stat.scale))


def shard_date_files(
    date: str,
    files: list[Path],
    out_root: Path,
    batch_size: int,
    max_batches: int | None,
    compression: str,
    allow_default_scale: bool,
) -> list[ShardStats]:
    pool = ShardWriterPool(out_root, compression)
    success = False
    batches_seen = 0
    try:
        for file_path in files:
            parquet = pq.ParquetFile(file_path)
            columns = [column for column in EDGE_COLUMNS + OPTIONAL_SCALE_COLUMNS if column in parquet.schema.names]
            for batch in parquet.iter_batches(batch_size=batch_size, columns=columns, use_threads=False):
                batches_seen += 1
                if max_batches is not None and batches_seen > max_batches:
                    break
                frame = normalize_batch(batch, date, allow_default_scale)
                for (layer_id, scale), group in frame.groupby(["layer_id", "scale"], sort=False):
                    pool.write_group(date, layer_id, scale, group)
            if max_batches is not None and batches_seen >= max_batches:
                break
        success = True
    finally:
        stats = pool.close(commit=success)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Shard P0 edges by date/layer/real scale for P1 workers.")
    parser.add_argument("--p0-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--dates")
    parser.add_argument("--date")
    parser.add_argument("--batch-size", type=int, default=250_000)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--allow-default-scale", action="store_true", help="Only for legacy data; disables strict scale identity")
    args = parser.parse_args()

    p0_root, out_root = Path(args.p0_root), Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    files = find_edge_files(p0_root, parse_dates(args.dates))
    if not files:
        raise FileNotFoundError(f"no edges.parquet found under {p0_root}")
    by_date: dict[str, list[Path]] = defaultdict(list)
    for file_path in files:
        by_date[infer_date(file_path, args.date)].append(file_path)
    all_stats: list[ShardStats] = []
    for date, date_files in sorted(by_date.items()):
        all_stats.extend(shard_date_files(date, date_files, out_root, args.batch_size, args.max_batches, args.compression, args.allow_default_scale))
    manifest = {
        "shard_contract_version": SHARD_CONTRACT_VERSION,
        "status": "complete" if all_stats else "empty",
        "input_files": len(files),
        "shards": len(all_stats),
        "allow_default_scale": args.allow_default_scale,
        "stats": [asdict(stat) for stat in all_stats],
    }
    pd.DataFrame(manifest["stats"]).to_csv(out_root / "shard_manifest.csv", index=False)
    temporary = out_root / "shard_manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(temporary, out_root / "shard_manifest.json")
    print(json.dumps({"edge_files": len(files), "shards": len(all_stats), "strict_scale": not args.allow_default_scale}, indent=2), flush=True)


if __name__ == "__main__":
    main()
