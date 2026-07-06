from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from graphfactorfactory.infrastructure.schemas import EDGE_SCHEMA, NODE_SCHEMA, SNAPSHOT_SCHEMA


def _arrow_table(frame: pd.DataFrame, schema: pa.Schema) -> pa.Table:
    undeclared = sorted(set(frame.columns).difference(schema.names))
    if undeclared:
        raise ValueError(f"Parquet schema would discard undeclared columns: {undeclared}")
    if frame.empty:
        return pa.Table.from_pylist([], schema=schema)
    return pa.Table.from_pandas(frame, schema=schema, preserve_index=False, safe=False)


class DayWriter:
    def __init__(self, root: Path, trade_date: str, compression: str, compression_level: int):
        self.day_root = root / "canonical" / f"date={trade_date}"
        self.day_root.mkdir(parents=True, exist_ok=True)
        options = dict(compression=compression, compression_level=compression_level, use_dictionary=True, write_statistics=True)
        self.edge_writer = pq.ParquetWriter(self.day_root / "edges.parquet", EDGE_SCHEMA, **options)
        self.node_writer = pq.ParquetWriter(self.day_root / "node_features.parquet", NODE_SCHEMA, **options)
        self.snapshot_writer = pq.ParquetWriter(self.day_root / "snapshots.parquet", SNAPSHOT_SCHEMA, **options)
        self.label_path = self.day_root / "labels.parquet"
        self.counts = {"edges": 0, "node_features": 0, "snapshots": 0, "labels": 0}

    def write_edges(self, frame: pd.DataFrame) -> None:
        if not frame.empty:
            self.edge_writer.write_table(_arrow_table(frame, EDGE_SCHEMA), row_group_size=250_000)
            self.counts["edges"] += len(frame)

    def write_node_features(self, frame: pd.DataFrame) -> None:
        if not frame.empty:
            self.node_writer.write_table(_arrow_table(frame, NODE_SCHEMA), row_group_size=250_000)
            self.counts["node_features"] += len(frame)

    def write_snapshots(self, frame: pd.DataFrame) -> None:
        if not frame.empty:
            self.snapshot_writer.write_table(_arrow_table(frame, SNAPSHOT_SCHEMA), row_group_size=10_000)
            self.counts["snapshots"] += len(frame)

    def write_labels(self, frame: pd.DataFrame) -> None:
        output = frame.copy()
        for column in output:
            if column.startswith("label_") and pd.api.types.is_numeric_dtype(output[column]):
                output[column] = output[column].astype("float32")
        output.to_parquet(self.label_path, index=False, compression="zstd", compression_level=6)
        self.counts["labels"] = len(output)

    def close(self) -> None:
        self.edge_writer.close(); self.node_writer.close(); self.snapshot_writer.close()
        (self.day_root / "row_counts.json").write_text(json.dumps(self.counts, indent=2))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
