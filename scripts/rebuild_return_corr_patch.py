from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import tempfile

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from graphfactorfactory.application.causality import audit_source_events
from graphfactorfactory.application.pipeline import _process_chunk
from graphfactorfactory.application.pit import decision_grid, filter_regular_session
from graphfactorfactory.domain.config import BuildConfig
from graphfactorfactory.domain.layers import LAYERS
from graphfactorfactory.infrastructure.nodefactorfactory.monthpack_source import (
    BoundMonthNodeFactorSource,
    MonthPackNodeFactorSource,
)
from graphfactorfactory.infrastructure.store import CanonicalGraphStore

RETURN_LAYER_IDS = (1, 14, 15)
RETURN_LAYERS = tuple(layer for layer in LAYERS if layer.layer_id in RETURN_LAYER_IDS)


def _sql_path(path: Path) -> str:
    return str(path).replace("'", "''")


def _layer_dimensions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "layer_id": 0,
                "name": "multiplex",
                "family": "multiplex",
                "directed": False,
                "lag_bars": 0,
                "columns": "",
                "transform": "multiplex",
            },
            *[
                {
                    "layer_id": layer.layer_id,
                    "name": layer.name,
                    "family": layer.family,
                    "directed": layer.directed,
                    "lag_bars": layer.lag_bars,
                    "columns": ",".join(layer.columns),
                    "transform": layer.transform,
                }
                for layer in LAYERS
            ],
        ]
    )


def _validate_patch(day_root: Path) -> None:
    for name in ("edges", "node_features", "snapshots"):
        path = day_root / f"{name}.parquet"
        if not path.exists() or pq.ParquetFile(path).metadata.num_rows <= 0:
            raise RuntimeError(f"missing or empty patch table: {path}")
    snapshots = pd.read_parquet(day_root / "snapshots.parquet", columns=["layer_id"])
    found = set(snapshots["layer_id"].astype(int).unique())
    if found != set(RETURN_LAYER_IDS):
        raise RuntimeError(f"patch layer mismatch: expected {RETURN_LAYER_IDS}, found {sorted(found)}")


def _atomic_merge_table(target: Path, patch: Path, layer_ids: tuple[int, ...], compression: str) -> None:
    if not target.exists():
        raise FileNotFoundError(target)
    temporary = target.with_suffix(target.suffix + ".patching")
    temporary.unlink(missing_ok=True)
    ids = ",".join(map(str, layer_ids))
    target_sql = _sql_path(target)
    patch_sql = _sql_path(patch)
    temporary_sql = _sql_path(temporary)
    sql = (
        f"COPY (SELECT * FROM read_parquet('{target_sql}') WHERE layer_id NOT IN ({ids}) "
        f"UNION ALL BY NAME SELECT * FROM read_parquet('{patch_sql}')) "
        f"TO '{temporary_sql}' (FORMAT PARQUET, COMPRESSION {compression.upper()})"
    )
    connection = duckdb.connect()
    try:
        connection.execute(sql)
    finally:
        connection.close()
    pq.ParquetFile(temporary)
    os.replace(temporary, target)


def _assert_non_target_identical(before: Path, after: Path, layer_ids: tuple[int, ...]) -> None:
    ids = ",".join(map(str, layer_ids))
    before_sql = _sql_path(before)
    after_sql = _sql_path(after)
    sql = f"""
        SELECT COUNT(*)
        FROM (
            (SELECT * FROM read_parquet('{before_sql}') WHERE layer_id NOT IN ({ids})
             EXCEPT ALL
             SELECT * FROM read_parquet('{after_sql}') WHERE layer_id NOT IN ({ids}))
            UNION ALL
            (SELECT * FROM read_parquet('{after_sql}') WHERE layer_id NOT IN ({ids})
             EXCEPT ALL
             SELECT * FROM read_parquet('{before_sql}') WHERE layer_id NOT IN ({ids}))
        )
    """
    connection = duckdb.connect()
    try:
        differences = int(connection.execute(sql).fetchone()[0])
    finally:
        connection.close()
    if differences:
        raise RuntimeError(f"non-ReturnCorr rows changed between {before} and {after}: {differences}")


def _record_patch(graph_root: Path, trade_date: str, config: BuildConfig) -> None:
    manifest_path = graph_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"dates": {}}
    history = manifest.setdefault("patch_history", [])
    history.append(
        {
            "trade_date": trade_date,
            "layer_ids": list(RETURN_LAYER_IDS),
            "reason": "exact raw/market-residual/cross-sectional ReturnCorr rebuild",
            "config_hash": config.config_hash,
            "multiplex_status": "stale_requires_full_rebuild",
        }
    )
    manifest["multiplex_status"] = "stale_requires_full_rebuild"
    manifest["return_corr_patch_version"] = "exact_correlation_v1"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))


def build_patch_date(*, source, graph_root: Path, trade_date: str, config: BuildConfig, workers: int) -> None:
    symbols_path = graph_root / "dimensions" / "symbols.parquet"
    if not symbols_path.exists():
        raise FileNotFoundError(f"stable symbol mapping not found: {symbols_path}")
    symbols = pd.read_parquet(symbols_path).sort_values("symbol_id")
    if symbols["symbol_id"].duplicated().any() or symbols["symbol"].duplicated().any():
        raise RuntimeError("symbols.parquet contains duplicate symbol_id or symbol")

    events = filter_regular_session(source.load_date(trade_date), config)
    if events.empty:
        raise ValueError(f"No regular-session rows for {trade_date}")
    audit_source_events(events)
    known = set(symbols["symbol"].astype(str))
    events = events[events["symbol"].astype(str).isin(known)].copy()
    events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True)
    events["available_time"] = pd.to_datetime(events["available_time"], utc=True)

    decisions = decision_grid(events, config)
    decisions = decisions[:: max(1, config.graph_step_minutes // 5)]
    chunks = [chunk for chunk in np.array_split(decisions, max(1, workers)) if len(chunk)]
    tasks = []
    for chunk in chunks:
        first = pd.Timestamp(chunk[0])
        first = first.tz_localize("UTC") if first.tzinfo is None else first.tz_convert("UTC")
        last = pd.Timestamp(chunk[-1])
        last = last.tz_localize("UTC") if last.tzinfo is None else last.tz_convert("UTC")
        chunk_data = events[
            (events["timestamp"] > first - pd.Timedelta(minutes=config.graph_window_minutes))
            & (events["available_time"] <= last)
        ].copy()
        tasks.append((list(chunk), chunk_data, config, symbols, RETURN_LAYERS, False))

    with tempfile.TemporaryDirectory(prefix=f"gff-returncorr-{trade_date}-") as temp:
        patch_root = Path(temp) / "graph_store"
        patch_store = CanonicalGraphStore(patch_root, replace(config, store_labels=False))
        patch_store.initialize_dimensions(symbols, _layer_dimensions())
        with patch_store.open_day(trade_date) as writer:
            with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
                futures = [pool.submit(_process_chunk, task) for task in tasks]
                for future in as_completed(futures):
                    for products, _ in future.result():
                        writer.write_edges(products.edges)
                        writer.write_node_features(products.node_features)
                        writer.write_snapshots(products.snapshots)
        patch_day = patch_root / "canonical" / f"date={trade_date}"
        _validate_patch(patch_day)

        target_day = graph_root / "canonical" / f"date={trade_date}"
        backup = graph_root / "patch_backups" / f"date={trade_date}"
        backup.mkdir(parents=True, exist_ok=True)
        names = ("edges", "node_features", "snapshots")
        for name in names:
            shutil.copy2(target_day / f"{name}.parquet", backup / f"{name}.parquet")
        try:
            for name in names:
                target = target_day / f"{name}.parquet"
                before = backup / f"{name}.parquet"
                _atomic_merge_table(target, patch_day / target.name, RETURN_LAYER_IDS, config.parquet_compression)
                _assert_non_target_identical(before, target, RETURN_LAYER_IDS)
        except Exception:
            for name in names:
                shutil.copy2(backup / f"{name}.parquet", target_day / f"{name}.parquet")
            raise

    _record_patch(graph_root, trade_date, config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Atomically rebuild only ReturnCorr layers 1/14/15")
    parser.add_argument("--source-monthpack-root", required=True)
    parser.add_argument("--graph-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dates", nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    graph_root = Path(args.graph_root).expanduser().resolve()
    config = BuildConfig.from_yaml(args.config)
    source_root = Path(args.source_monthpack_root).expanduser().resolve()
    monthpack = MonthPackNodeFactorSource(source_root)

    for trade_date in args.dates:
        month = trade_date[:7]
        source = BoundMonthNodeFactorSource(monthpack, month)
        build_patch_date(source=source, graph_root=graph_root, trade_date=trade_date, config=config, workers=args.workers)
        print(f"patched {trade_date}")

    store = CanonicalGraphStore(graph_root, config)
    store.initialize_dimensions(pd.read_parquet(graph_root / "dimensions" / "symbols.parquet"), _layer_dimensions())
    store.finalize_catalog()


if __name__ == "__main__":
    main()
