from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from graphfactorfactory.application.causality import audit_source_events
from graphfactorfactory.application.pipeline import _process_chunk
from graphfactorfactory.application.pit import decision_grid, filter_regular_session
from graphfactorfactory.domain.config import BuildConfig
from graphfactorfactory.domain.layers import LAYERS, LAYER_BY_ID
from graphfactorfactory.infrastructure.schemas import EDGE_SCHEMA, NODE_SCHEMA, SNAPSHOT_SCHEMA
from graphfactorfactory.infrastructure.store import CanonicalGraphStore
from graphfactorfactory.infrastructure.writer import DayWriter
from graphfactorfactory.ports.node_source import NodeFactorSource


RETURN_CORR_LAYER_IDS = (1, 14, 15)
RETURN_CORR_LAYERS = tuple(LAYER_BY_ID[layer_id] for layer_id in RETURN_CORR_LAYER_IDS)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_baseline_day(source_root: Path, output_root: Path, trade_date: str) -> tuple[Path, Path]:
    source_day = source_root / "canonical" / f"date={trade_date}"
    output_day = output_root / "canonical" / f"date={trade_date}"
    if not source_day.exists():
        raise FileNotFoundError(source_day)
    required = [source_day / f"{name}.parquet" for name in ("edges", "node_features", "snapshots", "labels")]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Baseline day is incomplete: {missing}")
    if output_day.exists():
        shutil.rmtree(output_day)
    output_day.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_day, output_day)
    return source_day, output_day


def _copy_dimensions_and_manifest(source_root: Path, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    source_dimensions = source_root / "dimensions"
    if not source_dimensions.exists():
        raise FileNotFoundError(source_dimensions)
    shutil.copytree(source_dimensions, output_root / "dimensions", dirs_exist_ok=True)
    source_manifest = source_root / "manifest.json"
    if source_manifest.exists() and not (output_root / "manifest.json").exists():
        shutil.copy2(source_manifest, output_root / "manifest.json")


def _write_layer_dimension(output_root: Path) -> None:
    layers = pd.DataFrame([
        {"layer_id": 0, "name": "multiplex", "family": "multiplex", "directed": False, "lag_bars": 0, "columns": ""},
        *[
            {
                "layer_id": layer.layer_id,
                "name": layer.name,
                "family": layer.family,
                "directed": layer.directed,
                "lag_bars": layer.lag_bars,
                "columns": ",".join(layer.columns),
            }
            for layer in LAYERS
        ],
    ])
    path = output_root / "dimensions" / "layers.parquet"
    temp = path.with_suffix(".parquet.tmp")
    layers.to_parquet(temp, index=False, compression="zstd")
    os.replace(temp, path)


def _table_without_layers(table: pa.Table, layer_ids: Iterable[int]) -> pa.Table:
    ids = pa.array(list(layer_ids), type=pa.int16())
    keep = pc.invert(pc.is_in(table["layer_id"], value_set=ids))
    return table.filter(keep)


def _validate_unique(frame: pd.DataFrame, keys: list[str], name: str) -> None:
    if frame.duplicated(keys).any():
        examples = frame.loc[frame.duplicated(keys, keep=False), keys].head(10).to_dict("records")
        raise ValueError(f"Duplicate {name} keys after ReturnCorr patch: {examples}")


def _atomic_merge_parquet(
    baseline_path: Path,
    patch_path: Path,
    schema: pa.Schema,
    layer_ids: tuple[int, ...],
    unique_keys: list[str],
) -> dict[str, int]:
    baseline = pq.read_table(baseline_path)
    if "date" in baseline.column_names:
        baseline = baseline.drop(["date"])
    patch = pq.read_table(patch_path)
    if "date" in patch.column_names:
        patch = patch.drop(["date"])
    preserved = _table_without_layers(baseline, layer_ids)
    patch_ids = set(pc.unique(patch["layer_id"]).to_pylist()) if patch.num_rows else set()
    missing = set(layer_ids) - patch_ids
    if missing:
        raise ValueError(f"Patch file {patch_path.name} is missing layers {sorted(missing)}")
    merged = pa.concat_tables([preserved, patch], promote_options="default").cast(schema, safe=False)
    frame = merged.to_pandas()
    _validate_unique(frame, unique_keys, baseline_path.stem)
    temp_path = baseline_path.with_suffix(".parquet.tmp")
    pq.write_table(
        merged,
        temp_path,
        compression="zstd",
        compression_level=6,
        use_dictionary=True,
        write_statistics=True,
        row_group_size=250_000 if baseline_path.stem != "snapshots" else 10_000,
    )
    pq.ParquetFile(temp_path).metadata
    os.replace(temp_path, baseline_path)
    return {
        "baseline_rows": baseline.num_rows,
        "preserved_rows": preserved.num_rows,
        "patch_rows": patch.num_rows,
        "merged_rows": merged.num_rows,
    }


def _benchmark_coverage(events: pd.DataFrame, trade_date: str, config: BuildConfig) -> pd.DataFrame:
    rows = []
    snapshot_count = len(decision_grid(events, config))
    for symbol in config.return_corr_benchmarks:
        subset = events.loc[events["symbol"].astype(str).eq(symbol), "ret_5m"] if "ret_5m" in events else pd.Series(dtype=float)
        available = int(pd.to_numeric(subset, errors="coerce").notna().sum())
        rows.append({
            "date": trade_date,
            "snapshot_count": snapshot_count,
            "benchmark_symbol": symbol,
            "available_points": available,
            "missing_ratio": 1.0 - available / max(len(events[events["symbol"].astype(str).eq(symbol)]), 1),
            "fallback_used": available < config.return_corr_min_benchmark_points,
        })
    return pd.DataFrame(rows)


def _update_coverage_report(output_root: Path, coverage: pd.DataFrame, trade_date: str) -> Path:
    qa_root = output_root / "qa"
    qa_root.mkdir(parents=True, exist_ok=True)
    path = qa_root / "return_corr_benchmark_coverage.csv"
    if path.exists():
        existing = pd.read_csv(path)
        existing = existing[existing["date"].astype(str) != str(trade_date)]
        coverage = pd.concat([existing, coverage], ignore_index=True)
    coverage.sort_values(["date", "benchmark_symbol"]).to_csv(path, index=False)
    return path


def _update_patch_manifest(
    output_root: Path,
    trade_date: str,
    config: BuildConfig,
    source: NodeFactorSource,
    merge_stats: dict[str, dict[str, int]],
    labels_sha256: str,
) -> Path:
    path = output_root / "manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {"dates": {}}
    history = manifest.setdefault("return_corr_patch_history", [])
    history = [item for item in history if str(item.get("date")) != str(trade_date)]
    history.append({
        "date": trade_date,
        "layer_ids": list(RETURN_CORR_LAYER_IDS),
        "layer_names": [layer.name for layer in RETURN_CORR_LAYERS],
        "config_hash": config.config_hash,
        "source_fingerprint": asdict(source.fingerprint()),
        "merge_stats": merge_stats,
        "labels_sha256": labels_sha256,
        "multiplex_layer_0": "stale_disabled_until_full_rebuild",
    })
    manifest["return_corr_patch_history"] = sorted(history, key=lambda item: str(item["date"]))
    manifest["config"] = config.to_dict()
    manifest["config_hash"] = config.config_hash
    manifest["multiplex_status"] = {
        "layer_id": 0,
        "status": "stale_disabled_until_full_rebuild",
        "reason": "ReturnCorr layers 1/14/15 were patched without rebuilding the all-layer multiplex.",
    }
    day_root = output_root / "canonical" / f"date={trade_date}"
    manifest.setdefault("dates", {})[trade_date] = {
        name: pq.ParquetFile(day_root / f"{name}.parquet").metadata.num_rows
        for name in ("edges", "node_features", "snapshots", "labels")
    }
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(manifest, indent=2, default=str))
    os.replace(temp, path)
    return path


class ReturnCorrPatchPipeline:
    """Rebuild only ReturnCorr layers into a new graph store and merge them atomically.

    Layer 0 is intentionally not rebuilt. The output manifest marks multiplex stale so
    downstream code must not treat it as production-complete until a full Phase 0 run.
    """

    def __init__(
        self,
        source: NodeFactorSource,
        source_store: str | Path,
        output_store: str | Path,
        config: BuildConfig,
        *,
        max_workers: int = 1,
    ):
        self.source = source
        self.source_root = Path(source_store).expanduser().resolve()
        self.output_root = Path(output_store).expanduser().resolve()
        if self.source_root == self.output_root:
            raise ValueError("ReturnCorr patch output must differ from the source graph store")
        self.config = config
        self.max_workers = max(1, int(max_workers))

    def build_date(self, trade_date: str) -> dict:
        _copy_dimensions_and_manifest(self.source_root, self.output_root)
        source_day, output_day = _copy_baseline_day(self.source_root, self.output_root, trade_date)
        _write_layer_dimension(self.output_root)
        symbols = pd.read_parquet(self.output_root / "dimensions" / "symbols.parquet").sort_values("symbol_id")

        events = filter_regular_session(self.source.load_date(trade_date), self.config)
        if events.empty:
            raise ValueError(f"No regular-session rows for {trade_date}")
        audit_source_events(events)
        keep_symbols = set(symbols["symbol"].astype(str)) | set(self.config.return_corr_benchmarks)
        events = events[events["symbol"].astype(str).isin(keep_symbols)].copy()
        events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True)
        events["available_time"] = pd.to_datetime(events["available_time"], utc=True)

        coverage = _benchmark_coverage(events, trade_date, self.config)
        if bool(coverage["fallback_used"].all()):
            import logging
            logging.warning(f"All configured ReturnCorr benchmarks are unavailable on {trade_date}. Falling back to cross-sectional median residualization.")

        graph_decisions = decision_grid(events, self.config)
        graph_decisions = graph_decisions[:: max(1, self.config.graph_step_minutes // 5)]
        chunks = np.array_split(graph_decisions, min(self.max_workers, max(len(graph_decisions), 1)))
        tasks = []
        for chunk in chunks:
            if len(chunk) == 0:
                continue
            min_t = pd.Timestamp(chunk[0])
            min_t = min_t.tz_localize("UTC") if min_t.tzinfo is None else min_t.tz_convert("UTC")
            max_t = pd.Timestamp(chunk[-1])
            max_t = max_t.tz_localize("UTC") if max_t.tzinfo is None else max_t.tz_convert("UTC")
            chunk_start = min_t - pd.Timedelta(minutes=self.config.graph_window_minutes)
            chunk_data = events[(events["timestamp"] > chunk_start) & (events["available_time"] <= max_t)].copy()
            tasks.append((list(chunk), chunk_data, self.config, symbols, RETURN_CORR_LAYERS, False))

        temp_root = Path(tempfile.mkdtemp(prefix=f"gff_return_corr_{trade_date}_", dir=str(self.output_root)))
        try:
            with DayWriter(temp_root, trade_date, self.config.parquet_compression, self.config.parquet_compression_level) as writer:
                if self.max_workers == 1:
                    results = [_process_chunk(task) for task in tasks]
                else:
                    from concurrent.futures import ProcessPoolExecutor
                    with ProcessPoolExecutor(max_workers=self.max_workers) as pool:
                        results = list(pool.map(_process_chunk, tasks))
                for chunk_results in results:
                    for products, _ in chunk_results:
                        writer.write_edges(products.edges)
                        writer.write_node_features(products.node_features)
                        writer.write_snapshots(products.snapshots)

            patch_day = temp_root / "canonical" / f"date={trade_date}"
            labels_before = _file_sha256(output_day / "labels.parquet")
            merge_stats = {
                "edges": _atomic_merge_parquet(output_day / "edges.parquet", patch_day / "edges.parquet", EDGE_SCHEMA, RETURN_CORR_LAYER_IDS, ["decision_time", "layer_id", "src_id", "dst_id"]),
                "node_features": _atomic_merge_parquet(output_day / "node_features.parquet", patch_day / "node_features.parquet", NODE_SCHEMA, RETURN_CORR_LAYER_IDS, ["decision_time", "layer_id", "symbol_id"]),
                "snapshots": _atomic_merge_parquet(output_day / "snapshots.parquet", patch_day / "snapshots.parquet", SNAPSHOT_SCHEMA, RETURN_CORR_LAYER_IDS, ["decision_time", "layer_id"]),
            }
            labels_after = _file_sha256(output_day / "labels.parquet")
            if labels_before != labels_after:
                raise ValueError("labels.parquet changed during ReturnCorr patch")

            coverage_path = _update_coverage_report(self.output_root, coverage, trade_date)
            manifest_path = _update_patch_manifest(self.output_root, trade_date, self.config, self.source, merge_stats, labels_after)
            catalog = CanonicalGraphStore(self.output_root, self.config).finalize_catalog()
            return {
                "date": trade_date,
                "source_day": str(source_day),
                "output_day": str(output_day),
                "layers": list(RETURN_CORR_LAYER_IDS),
                "merge_stats": merge_stats,
                "coverage_report": str(coverage_path),
                "manifest": str(manifest_path),
                "catalog": str(catalog),
                "labels_sha256": labels_after,
                "multiplex_status": "stale_disabled_until_full_rebuild",
            }
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)
