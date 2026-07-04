from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pyarrow.parquet as pq

from graphfactorfactory.domain.config import BuildConfig
from graphfactorfactory.domain.records import SourceFingerprint
from graphfactorfactory.infrastructure.writer import DayWriter


class CanonicalGraphStore:
    def __init__(self, root: str | Path, config: BuildConfig):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.config = config

    def initialize_dimensions(self, symbols: pd.DataFrame, layers: pd.DataFrame) -> None:
        dimensions = self.root / "dimensions"
        dimensions.mkdir(exist_ok=True)
        symbols.to_parquet(dimensions / "symbols.parquet", index=False, compression="zstd")
        layers.to_parquet(dimensions / "layers.parquet", index=False, compression="zstd")

    def open_day(self, trade_date: str) -> DayWriter:
        day_root = self.root / "canonical" / f"date={trade_date}"
        if day_root.exists():
            for path in day_root.glob("*"):
                path.unlink()
        return DayWriter(self.root, trade_date, self.config.parquet_compression, self.config.parquet_compression_level)

    def finalize_catalog(self) -> Path:
        catalog_path = self.root / "graphfactorfactory.duckdb"
        connection = duckdb.connect(str(catalog_path))
        root = str(self.root).replace("'", "''")
        try:
            connection.execute(f"CREATE OR REPLACE VIEW symbols AS SELECT * FROM read_parquet('{root}/dimensions/symbols.parquet')")
            connection.execute(f"CREATE OR REPLACE VIEW layers AS SELECT * FROM read_parquet('{root}/dimensions/layers.parquet')")
            for name in ("edges", "node_features", "snapshots", "labels"):
                glob_path = f"{root}/canonical/date=*/{'labels.parquet' if name == 'labels' else name + '.parquet'}"
                try:
                    connection.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{glob_path}', hive_partitioning=true)")
                except duckdb.IOException:
                    pass
            connection.execute("CREATE TABLE IF NOT EXISTS metadata(key VARCHAR PRIMARY KEY, value VARCHAR)")
            connection.execute("INSERT OR REPLACE INTO metadata VALUES ('schema_version', 'graphfactorfactory-canonical-v1')")
        finally:
            connection.close()
        return catalog_path

    def write_manifest(
        self,
        trade_date: str,
        source_fingerprint: SourceFingerprint,
        config: BuildConfig,
        universe_count: int,
        node_feature_columns: list[str],
        split_source_metadata=None,
    ) -> Path:
        manifest_path = self.root / "manifest.json"
        existing: dict[str, Any] = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"dates": {}}
        existing.update({
            "schema_version": "graphfactorfactory-canonical-v1",
            "storage_policy": {
                "nodefactorfactory_source": "external_not_duplicated",
                "canonical_graph_store": "lossless_retained_edges_nodes_snapshots",
                "qlib": "on_demand_view_not_materialized_by_default",
            },
            "source": asdict(source_fingerprint),
            "config": config.to_dict(),
            "config_hash": config.config_hash,
            "universe_count": universe_count,
            "node_feature_columns": node_feature_columns,
            "label_adjustment": {
                "policy": "split_adjusted_target_only" if split_source_metadata else "raw",
                "split_source": asdict(split_source_metadata) if split_source_metadata else None,
            },
        })
        existing.setdefault("dates", {})[trade_date] = self.count_date_rows(trade_date)
        manifest_path.write_text(json.dumps(existing, indent=2, default=str))
        return manifest_path

    def count_date_rows(self, trade_date: str) -> dict[str, int]:
        day_root = self.root / "canonical" / f"date={trade_date}"
        result = {}
        for name in ("edges", "node_features", "snapshots", "labels"):
            path = day_root / f"{name}.parquet"
            result[name] = pq.ParquetFile(path).metadata.num_rows if path.exists() else 0
        return result
