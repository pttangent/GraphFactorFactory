from __future__ import annotations

import glob
import hashlib
import json
from pathlib import Path
from typing import Sequence

import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from graphfactorfactory.application.causality import IDENTITY_COLUMNS
from graphfactorfactory.domain.records import SourceFingerprint
from graphfactorfactory.ports.node_source import NodeFactorSource


class ParquetNodeFactorSource(NodeFactorSource):
    def __init__(self, path_or_glob: str | Path):
        raw = str(path_or_glob)
        candidate = Path(raw).expanduser()
        if candidate.is_dir():
            self.paths = tuple(sorted(str(path) for path in candidate.rglob("*.parquet")))
        elif any(token in raw for token in "*?["):
            self.paths = tuple(sorted(glob.glob(raw, recursive=True)))
        elif candidate.exists():
            self.paths = (str(candidate.resolve()),)
        else:
            raise FileNotFoundError(path_or_glob)
        if not self.paths:
            raise FileNotFoundError(f"No parquet files found for {path_or_glob}")
        self.dataset = ds.dataset(self.paths, format="parquet")

    def fingerprint(self) -> SourceFingerprint:
        schema_sha = hashlib.sha256(str(self.dataset.schema).encode()).hexdigest()
        metadata_records = []
        total_rows = 0
        total_bytes = 0
        factor_versions: set[str] = set()
        trade_versions: set[str] = set()
        for raw_path in self.paths:
            path = Path(raw_path)
            metadata = pq.ParquetFile(path).metadata
            total_rows += metadata.num_rows
            total_bytes += path.stat().st_size
            metadata_records.append((str(path), path.stat().st_size, metadata.num_rows, path.stat().st_mtime_ns))
            schema_names = set(pq.read_schema(path).names)
            for column, target in (("factor_semantics_version", factor_versions), ("trade_semantics_version", trade_versions)):
                if column in schema_names:
                    values = pd.read_parquet(path, columns=[column])[column].dropna().astype(str).unique()
                    target.update(values.tolist())
        file_hash = hashlib.sha256(json.dumps(metadata_records, sort_keys=True).encode()).hexdigest()
        return SourceFingerprint(paths=self.paths, total_bytes=total_bytes, total_rows=total_rows, schema_sha256=schema_sha, file_metadata_sha256=file_hash, factor_semantics_versions=tuple(sorted(factor_versions)), trade_semantics_versions=tuple(sorted(trade_versions)))

    def available_dates(self) -> list[str]:
        table = self.dataset.to_table(columns=["trade_date"])
        return sorted(pd.Series(table.column("trade_date").to_pylist()).dropna().astype(str).unique().tolist())

    def load_date(self, trade_date: str, columns: Sequence[str] | None = None) -> pd.DataFrame:
        table = self.dataset.to_table(columns=list(columns) if columns else None, filter=ds.field("trade_date") == trade_date)
        frame = table.to_pandas()
        required = {"symbol", "timestamp", "available_time"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"NodeFactorFactory data missing columns: {sorted(missing)}")
        frame["symbol"] = frame["symbol"].astype(str)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame["available_time"] = pd.to_datetime(frame["available_time"], utc=True)
        return frame

    def numeric_feature_columns(self) -> list[str]:
        result = []
        for field in self.dataset.schema:
            if field.name in IDENTITY_COLUMNS or field.name.startswith("label_"):
                continue
            if str(field.type).startswith(("int", "uint", "float", "double", "decimal")):
                result.append(field.name)
        return result
