#!/usr/bin/env python3
"""Direct, resumable P0 alpha extraction from canonical daily edge files.

This module intentionally avoids the old ``p0_alpha_shards`` materialization.
Each worker owns one trade date, scans the canonical edge file once, and writes
only the final node, spillover, and graph-state features.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from p2_alpha_pit_features import (
    DEFAULT_INTRADAY_HORIZONS,
    PIT_CONTRACT_VERSION,
    csvlist,
    csvset,
    ext,
    iter_time_groups,
    label_path,
    load_labels,
    write_manifest,
)
from p2_parallel_runtime import collect_process_map

DIRECT_CONTRACT_VERSION = "p0-canonical-direct-v1"
CANONICAL_COLUMNS = [
    "decision_time",
    "window_start",
    "window_end",
    "layer_id",
    "scale",
    "lookback_minutes",
    "src_id",
    "dst_id",
    "weight",
]
STAGE_LAYOUT = {
    "node": ("p0_node_features", "p0_node_features.parquet", "p0_node_features"),
    "spillover": (
        "p0_edge_spillover",
        "p0_edge_spillover_features.parquet",
        "p0_edge_spillover",
    ),
    "graph": ("p0_graph_state", "p0_graph_state_features.parquet", "p0_graph_state"),
}


@dataclass(frozen=True)
class DatePart:
    date: str
    base: Path


def _source_fingerprint(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def discover_canonical_dates(root: str | Path, dates: set[str] | None = None) -> list[DatePart]:
    by_date: dict[str, list[Path]] = {}
    for path in Path(root).rglob("edges.parquet"):
        date = ext(path, "date")
        if not date or (dates and date not in dates):
            continue
        if ext(path, "layer_id") or ext(path, "scale"):
            continue
        by_date.setdefault(date, []).append(path)

    parts: list[DatePart] = []
    for date, paths in sorted(by_date.items()):
        if len(paths) != 1:
            sample = ", ".join(str(path) for path in paths[:5])
            raise ValueError(
                f"expected exactly one canonical edges.parquet for date={date}; "
                f"found {len(paths)}: {sample}"
            )
        parts.append(DatePart(date, paths[0]))
    parts.sort(key=lambda part: part.base.stat().st_size, reverse=True)
    return parts


def _normalise_snapshot(
    frame: pd.DataFrame,
    date: str,
    layers: set[str] | None,
    scales: set[str] | None,
) -> pd.DataFrame:
    required = {"decision_time", "layer_id", "src_id", "dst_id", "weight"}
    if missing := required - set(frame):
        raise ValueError(f"canonical P0 edges for {date} missing {sorted(missing)}")

    frame = frame.copy()
    frame["decision_time"] = pd.to_datetime(frame["decision_time"], utc=True, errors="coerce")
    frame["src_id"] = pd.to_numeric(frame["src_id"], errors="coerce").astype("Int64")
    frame["dst_id"] = pd.to_numeric(frame["dst_id"], errors="coerce").astype("Int64")
    frame = frame.dropna(subset=["decision_time", "src_id", "dst_id"]).copy()
    frame = frame[frame["src_id"] != frame["dst_id"]].copy()
    frame["src_id"] = frame["src_id"].astype("int64")
    frame["dst_id"] = frame["dst_id"].astype("int64")
    frame["layer_id"] = pd.to_numeric(frame["layer_id"], errors="raise").astype("int64").astype(str)

    supplied = frame["scale"].astype("string") if "scale" in frame else pd.Series(pd.NA, index=frame.index, dtype="string")
    if "lookback_minutes" in frame:
        lookback = pd.to_numeric(frame["lookback_minutes"], errors="coerce")
        derived = lookback.map(lambda value: f"{int(value)}m" if pd.notna(value) else pd.NA).astype("string")
        mismatch = supplied.notna() & derived.notna() & supplied.ne(derived)
        if mismatch.any():
            raise ValueError(f"scale/lookback mismatch in {date}: {int(mismatch.sum())} rows")
        frame["scale"] = derived.where(derived.notna(), supplied)
    elif "scale" in frame:
        frame["scale"] = supplied
    else:
        raise ValueError("canonical P0 edges require scale or lookback_minutes")

    missing_scale = frame["scale"].isna() | frame["scale"].astype(str).str.strip().eq("")
    if missing_scale.any():
        raise ValueError(f"missing real scale in {date}: {int(missing_scale.sum())} rows")
    frame["scale"] = frame["scale"].astype(str)
    frame["weight"] = pd.to_numeric(frame["weight"], errors="coerce").fillna(0.0).astype("float32")
    frame["abs_weight"] = frame["weight"].abs()

    if "window_end" in frame:
        frame["window_end"] = pd.to_datetime(frame["window_end"], utc=True, errors="coerce")
        valid = frame["window_end"].notna()
        if valid.any() and not (frame.loc[valid, "window_end"] <= frame.loc[valid, "decision_time"]).all():
            raise AssertionError("P0 edge window_end exceeds decision_time")

    if layers:
        frame = frame[frame["layer_id"].isin(layers)]
    if scales:
        frame = frame[frame["scale"].isin(scales)]
    return frame


def _labels_at(indexed: pd.DataFrame, decision_time: pd.Timestamp) -> pd.DataFrame:
    try:
        return indexed.loc[[decision_time]].reset_index(drop=True)
    except KeyError:
        return pd.DataFrame(columns=indexed.columns)


def _node_features(edges: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame | None:
    if target.empty:
        return None
    keys = ["decision_time", "layer_id", "scale"]
    source = (
        edges.groupby(keys + ["src_id"], sort=False)
        .agg(
            src_edge_count=("dst_id", "size"),
            src_weight_sum=("abs_weight", "sum"),
            src_weight_mean=("abs_weight", "mean"),
            src_weight_max=("abs_weight", "max"),
        )
        .reset_index()
        .rename(columns={"src_id": "symbol_id"})
    )
    destination = (
        edges.groupby(keys + ["dst_id"], sort=False)
        .agg(
            dst_edge_count=("src_id", "size"),
            dst_weight_sum=("abs_weight", "sum"),
            dst_weight_mean=("abs_weight", "mean"),
            dst_weight_max=("abs_weight", "max"),
        )
        .reset_index()
        .rename(columns={"dst_id": "symbol_id"})
    )
    features = source.merge(destination, on=keys + ["symbol_id"], how="outer", copy=False)
    numeric = [column for column in features if column.endswith(("_count", "_sum", "_mean", "_max"))]
    features[numeric] = features[numeric].fillna(0)
    features["p0_total_edge_count"] = features["src_edge_count"] + features["dst_edge_count"]
    features["p0_total_weight_sum"] = features["src_weight_sum"] + features["dst_weight_sum"]
    output = features.merge(target, on=["decision_time", "symbol_id"], how="inner", copy=False)
    if output.empty:
        return None
    output["pit_audit_pass"] = True
    return output


def _spillover_features(
    edges: pd.DataFrame,
    subset: pd.DataFrame,
    past_horizon: str,
    target_columns: list[str],
    metadata_columns: list[str],
) -> pd.DataFrame | None:
    if subset.empty:
        return None
    past = f"past_label_{past_horizon}"
    past_exit = f"past_exit_time_{past_horizon}"
    source = subset[["decision_time", "symbol_id", past, past_exit]].rename(
        columns={
            "symbol_id": "src_id",
            past: "src_past_return",
            past_exit: "src_past_available_time",
        }
    )
    rename = {
        "symbol_id": "dst_id",
        **{column: "target_" + column.replace("label_", "") for column in target_columns},
    }
    for column in metadata_columns:
        match = re.fullmatch(r"label_(entry|exit)_time_(\d+m)", column)
        if match:
            rename[column] = f"target_{match.group(1)}_time_{match.group(2)}"
    selected = list(dict.fromkeys(["decision_time", "symbol_id"] + target_columns + metadata_columns))
    target = subset[selected].rename(columns=rename)

    merged = edges.merge(source, on=["decision_time", "src_id"], how="inner", copy=False)
    if merged.empty:
        return None
    valid = merged["src_past_available_time"].notna()
    if valid.any() and not (
        merged.loc[valid, "src_past_available_time"] <= merged.loc[valid, "decision_time"]
    ).all():
        raise AssertionError("P0 source past return unavailable at decision_time")
    merged["edge_signal"] = merged["weight"] * pd.to_numeric(
        merged["src_past_return"], errors="coerce"
    ).fillna(0.0)
    aggregate = (
        merged.groupby(["decision_time", "layer_id", "scale", "dst_id"], sort=False)
        .agg(
            p0_edge_spillover_signal=("edge_signal", "mean"),
            p0_edge_spillover_sum=("edge_signal", "sum"),
            p0_edge_count=("src_id", "size"),
            p0_edge_abs_weight=("abs_weight", "sum"),
            p0_edge_mean_abs_weight=("abs_weight", "mean"),
        )
        .reset_index()
    )
    output = aggregate.merge(target, on=["decision_time", "dst_id"], how="inner", copy=False)
    if output.empty:
        return None
    output["pit_audit_pass"] = True
    return output


def _graph_state(edges: pd.DataFrame) -> pd.DataFrame:
    nodes = np.unique(np.concatenate([edges["src_id"].to_numpy(), edges["dst_id"].to_numpy()]))
    return pd.DataFrame(
        [
            {
                "decision_time": edges["decision_time"].iloc[0],
                "layer_id": edges["layer_id"].iloc[0],
                "scale": edges["scale"].iloc[0],
                "edge_count": len(edges),
                "active_node_count": len(nodes),
                "avg_abs_weight": float(edges["abs_weight"].mean()),
                "sum_abs_weight": float(edges["abs_weight"].sum()),
                "max_abs_weight": float(edges["abs_weight"].max()),
                "density_proxy": len(edges) / max(len(nodes), 1),
                "pit_audit_pass": True,
            }
        ]
    )


class PartitionWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.writer: pq.ParquetWriter | None = None
        self.schema: pa.Schema | None = None
        self.rows = 0
        self.batches = 0

    def write(self, frame: pd.DataFrame | None) -> None:
        if frame is None or frame.empty:
            return
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if self.writer is None:
            self.schema = table.schema
            self.writer = pq.ParquetWriter(
                self.path,
                self.schema,
                compression="zstd",
                use_dictionary=False,
            )
        elif table.schema != self.schema:
            table = table.cast(self.schema)
        self.writer.write_table(table)
        self.rows += len(frame)
        self.batches += 1
        del table

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None


class WriterRegistry:
    def __init__(self, staging_root: Path, features: set[str]) -> None:
        self.staging_root = staging_root
        self.features = features
        self.writers: dict[tuple[str, str, str], PartitionWriter] = {}

    def write(self, feature: str, layer: str, scale: str, frame: pd.DataFrame | None) -> None:
        if feature not in self.features or frame is None or frame.empty:
            return
        stage_dir, filename, _ = STAGE_LAYOUT[feature]
        key = (feature, layer, scale)
        writer = self.writers.get(key)
        if writer is None:
            path = self.staging_root / stage_dir / f"layer_id={layer}" / f"scale={scale}" / filename
            writer = PartitionWriter(path)
            self.writers[key] = writer
        writer.write(frame)

    def close(self) -> None:
        for writer in self.writers.values():
            writer.close()

    def write_manifests(self, date: str, source: Path) -> list[str]:
        output_files: list[str] = []
        for (feature, layer, scale), writer in sorted(self.writers.items()):
            stage_dir, filename, stage_name = STAGE_LAYOUT[feature]
            directory = self.staging_root / stage_dir / f"layer_id={layer}" / f"scale={scale}"
            write_manifest(
                directory,
                {
                    "stage": stage_name,
                    "status": "complete" if writer.rows else "empty",
                    "date": date,
                    "layer_id": layer,
                    "scale": scale,
                    "output_rows": writer.rows,
                    "write_batches": writer.batches,
                    "input_mode": "canonical_date_single_pass",
                    "source": str(source),
                },
            )
            output_files.append(str(Path(stage_dir) / f"date={date}" / f"layer_id={layer}" / f"scale={scale}" / filename))
        return output_files

    def row_summary(self) -> dict[str, int]:
        totals = {feature: 0 for feature in self.features}
        for (feature, _, _), writer in self.writers.items():
            totals[feature] += writer.rows
        return totals


def _status_path(out_root: Path, date: str) -> Path:
    return out_root / "p0_direct_status" / f"date={date}" / "manifest.json"


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _checkpoint_valid(
    out_root: Path,
    part: DatePart,
    fingerprint: dict,
    label_fingerprint: dict | None,
    features: set[str],
    horizons: list[str],
    past_horizon: str,
    layers: set[str] | None,
    scales: set[str] | None,
) -> bool:
    payload = _read_json(_status_path(out_root, part.date))
    if not payload:
        return False
    if payload.get("direct_contract_version") != DIRECT_CONTRACT_VERSION:
        return False
    if payload.get("pit_contract_version") != PIT_CONTRACT_VERSION:
        return False
    if payload.get("status") not in {"complete", "empty"}:
        return False
    expected = {
        "source_fingerprint": fingerprint,
        "label_fingerprint": label_fingerprint,
        "features": sorted(features),
        "horizons": list(horizons),
        "past_horizon": past_horizon,
        "layers": sorted(layers) if layers else None,
        "scales": sorted(scales) if scales else None,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            return False
    return all((out_root / relative).exists() for relative in payload.get("output_files", []))


def _write_status(out_root: Path, date: str, payload: dict) -> None:
    directory = _status_path(out_root, date).parent
    directory.mkdir(parents=True, exist_ok=True)
    body = {
        **payload,
        "direct_contract_version": DIRECT_CONTRACT_VERSION,
        "pit_contract_version": PIT_CONTRACT_VERSION,
    }
    temporary = directory / "manifest.json.tmp"
    temporary.write_text(json.dumps(body, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(temporary, directory / "manifest.json")


def _check_free_space(out_root: Path, minimum_free_gb: float) -> float:
    out_root.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(out_root).free / (1024**3)
    if free_gb < minimum_free_gb:
        raise OSError(
            f"P0 direct pipeline disk fuse: {free_gb:.2f}GB free < {minimum_free_gb:.2f}GB required"
        )
    return free_gb


def _commit_date(out_root: Path, staging_root: Path, date: str, features: set[str]) -> None:
    for feature in sorted(features):
        stage_dir, _, _ = STAGE_LAYOUT[feature]
        staged = staging_root / stage_dir
        final = out_root / stage_dir / f"date={date}"
        if final.exists():
            shutil.rmtree(final)
        if staged.exists():
            final.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged, final)
    shutil.rmtree(staging_root, ignore_errors=True)


def process_date(
    part: DatePart,
    labels_root: str,
    out_root: str,
    horizons: list[str],
    past_horizon: str,
    layers: set[str] | None,
    scales: set[str] | None,
    features: set[str],
    skip_existing: bool,
    max_row_groups: int | None,
    batch_size: int,
    min_free_gb: float,
    disk_check_every: int,
) -> dict:
    started = time.time()
    root = Path(out_root)
    fingerprint = _source_fingerprint(part.base)
    labels_file = label_path(labels_root, part.date) if features & {"node", "spillover"} else None
    label_fingerprint = _source_fingerprint(labels_file) if labels_file is not None else None
    if skip_existing and _checkpoint_valid(
        root,
        part,
        fingerprint,
        label_fingerprint,
        features,
        horizons,
        past_horizon,
        layers,
        scales,
    ):
        return {"date": part.date, "status": "skipped", "elapsed_sec": 0.0}

    free_at_start = _check_free_space(root, min_free_gb)
    staging_root = root / ".p0_direct_staging" / f"date={part.date}"
    shutil.rmtree(staging_root, ignore_errors=True)
    staging_root.mkdir(parents=True, exist_ok=True)
    registry = WriterRegistry(staging_root, features)

    labels_indexed: pd.DataFrame | None = None
    target_columns: list[str] = []
    metadata_columns: list[str] = []
    snapshots = groups_seen = 0
    try:
        if labels_file is not None:
            labels = load_labels(labels_file, horizons)
            if "spillover" in features:
                past = f"past_label_{past_horizon}"
                past_exit = f"past_exit_time_{past_horizon}"
                if past not in labels or past_exit not in labels:
                    raise ValueError(f"missing {past} or {past_exit} for date={part.date}")
                target_columns = [column for column in labels if re.fullmatch(r"label_\d+m", column)]
                metadata_columns = [
                    column
                    for column in labels
                    if column == "label_entry_time"
                    or re.fullmatch(r"label_(?:entry|exit)_time_\d+m", column)
                ]
            labels_indexed = labels.set_index("decision_time", drop=False).sort_index()

        for decision_time, raw in iter_time_groups(
            part.base,
            CANONICAL_COLUMNS,
            max_row_groups,
            time_column="decision_time",
            batch_size=batch_size,
        ):
            snapshots += 1
            if disk_check_every > 0 and snapshots % disk_check_every == 0:
                _check_free_space(root, min_free_gb)
            edges = _normalise_snapshot(raw, part.date, layers, scales)
            if edges.empty:
                continue
            target = (
                _labels_at(labels_indexed, decision_time)
                if labels_indexed is not None
                else pd.DataFrame()
            )
            for (layer, scale), group in edges.groupby(["layer_id", "scale"], sort=False):
                groups_seen += 1
                layer, scale = str(layer), str(scale)
                if "node" in features:
                    registry.write("node", layer, scale, _node_features(group, target))
                if "spillover" in features:
                    registry.write(
                        "spillover",
                        layer,
                        scale,
                        _spillover_features(
                            group,
                            target,
                            past_horizon,
                            target_columns,
                            metadata_columns,
                        ),
                    )
                if "graph" in features:
                    registry.write("graph", layer, scale, _graph_state(group))
            del raw, edges, target
            if snapshots % 25 == 0:
                gc.collect()

        registry.close()
        output_files = registry.write_manifests(part.date, part.base)
        row_summary = registry.row_summary()
        _commit_date(root, staging_root, part.date, features)
        status = "complete" if output_files else "empty"
        payload = {
            "status": status,
            "date": part.date,
            "source_fingerprint": fingerprint,
            "label_fingerprint": label_fingerprint,
            "features": sorted(features),
            "horizons": list(horizons),
            "past_horizon": past_horizon,
            "layers": sorted(layers) if layers else None,
            "scales": sorted(scales) if scales else None,
            "output_files": output_files,
            "output_rows": row_summary,
            "snapshots": snapshots,
            "layer_scale_groups": groups_seen,
            "input_mode": "one_date_one_scan_no_physical_alpha_shards",
            "batch_size": batch_size,
            "free_gb_at_start": round(free_at_start, 3),
            "elapsed_sec": round(time.time() - started, 3),
        }
        _write_status(root, part.date, payload)
        return payload
    except BaseException as exc:
        registry.close()
        shutil.rmtree(staging_root, ignore_errors=True)
        _write_status(
            root,
            part.date,
            {
                "status": "failed",
                "date": part.date,
                "source_fingerprint": fingerprint,
                "label_fingerprint": label_fingerprint,
                "features": sorted(features),
                "horizons": list(horizons),
                "past_horizon": past_horizon,
                "layers": sorted(layers) if layers else None,
                "scales": sorted(scales) if scales else None,
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_sec": round(time.time() - started, 3),
            },
        )
        raise


def run_direct(
    p0_root: str | Path,
    labels_root: str | Path,
    out_root: str | Path,
    dates: set[str] | None,
    layers: set[str] | None,
    scales: set[str] | None,
    horizons: list[str],
    past_horizon: str,
    features: set[str],
    workers: int,
    skip_existing: bool,
    max_row_groups: int | None,
    batch_size: int,
    min_free_gb: float,
    disk_check_every: int,
) -> list[dict]:
    unknown = features - set(STAGE_LAYOUT)
    if unknown:
        raise ValueError(f"unknown P0 direct features: {sorted(unknown)}")
    parts = discover_canonical_dates(p0_root, dates)
    if not parts:
        raise FileNotFoundError(f"no canonical date-level edges.parquet found under {p0_root}")
    worker_count = max(1, min(int(workers), len(parts)))
    results = collect_process_map(
        parts,
        worker_count,
        process_date,
        str(labels_root),
        str(out_root),
        horizons,
        past_horizon,
        layers,
        scales,
        features,
        skip_existing,
        max_row_groups,
        batch_size,
        min_free_gb,
        disk_check_every,
        max_in_flight=worker_count,
        max_tasks_per_child=1,
    )
    summary_path = Path(out_root) / "p0_direct_run_summary.json"
    temporary = Path(str(summary_path) + ".tmp")
    temporary.write_text(
        json.dumps(
            {
                "direct_contract_version": DIRECT_CONTRACT_VERSION,
                "pit_contract_version": PIT_CONTRACT_VERSION,
                "workers": worker_count,
                "results": results,
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )
    os.replace(temporary, summary_path)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Direct P0 alpha extraction without physical alpha shards")
    parser.add_argument("--p0-root", required=True)
    parser.add_argument("--labels-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--dates")
    parser.add_argument("--layers")
    parser.add_argument("--scales")
    parser.add_argument("--horizons", default=",".join(DEFAULT_INTRADAY_HORIZONS))
    parser.add_argument("--past-horizon", default="15m")
    parser.add_argument("--features", default="node,spillover,graph")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-row-groups", type=int)
    parser.add_argument("--batch-size", type=int, default=500_000)
    parser.add_argument("--min-free-gb", type=float, default=50.0)
    parser.add_argument("--disk-check-every", type=int, default=25)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    horizons = [
        horizon
        for horizon in (csvlist(args.horizons) or DEFAULT_INTRADAY_HORIZONS)
        if horizon.endswith("m")
    ]
    results = run_direct(
        args.p0_root,
        args.labels_root,
        args.out_root,
        csvset(args.dates),
        csvset(args.layers),
        csvset(args.scales),
        horizons,
        args.past_horizon,
        csvset(args.features) or set(STAGE_LAYOUT),
        args.workers,
        args.skip_existing,
        args.max_row_groups,
        args.batch_size,
        args.min_free_gb,
        args.disk_check_every,
    )
    print(json.dumps({"result": results}, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
