from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from graphfactorfactory.domain.records import SourceFingerprint
from graphfactorfactory.ports.node_source import NodeFactorSource
from .parquet_source import ParquetNodeFactorSource

class MonthPackNodeFactorSource:
    def __init__(self, root_dir: str | Path):
        self.root = Path(root_dir).expanduser().resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"MonthPack root not found: {self.root}")

    def _get_month_dir(self, month: str) -> Path:
        return self.root / f"month={month}"

    def available_months(self) -> list[str]:
        months = []
        for p in self.root.glob("month=*"):
            if p.is_dir():
                months.append(p.name.split("=")[1])
        return sorted(months)

    def available_dates(self, month: str) -> list[str]:
        month_dir = self._get_month_dir(month)
        if not month_dir.exists():
            return []
        nf_dir = month_dir / "node_factors_5m"
        dates = []
        if nf_dir.exists():
            for p in nf_dir.glob("date=*"):
                if p.is_dir():
                    dates.append(p.name.split("=")[1])
        return sorted(dates)

    def load_node_factors(self, month: str, trade_date: str, columns: Sequence[str] | None = None) -> pd.DataFrame:
        date_dir = self._get_month_dir(month) / "node_factors_5m" / f"date={trade_date}"
        if not date_dir.exists():
            raise FileNotFoundError(f"Node factors not found for {trade_date} in {month}")
        source = ParquetNodeFactorSource(date_dir)
        return source.load_date(trade_date, columns)

    def load_trade_flow(self, month: str, trade_date: str) -> pd.DataFrame:
        # For future use or if needed by GraphFactorFactory
        date_dir = self._get_month_dir(month) / "trade_flow_1m" / f"date={trade_date}"
        if not date_dir.exists():
            raise FileNotFoundError(f"Trade flow not found for {trade_date} in {month}")
        dataset = ds.dataset(date_dir, format="parquet")
        return dataset.to_table().to_pandas()

    def load_bars(self, month: str, trade_date: str) -> pd.DataFrame:
        # For future use
        date_dir = self._get_month_dir(month) / "raw_1m" / f"date={trade_date}"
        if not date_dir.exists():
            raise FileNotFoundError(f"Bars not found for {trade_date} in {month}")
        dataset = ds.dataset(date_dir, format="parquet")
        return dataset.to_table().to_pandas()

    def load_metadata(self) -> pd.DataFrame:
        # If there's a global metadata file, load it. NFF doesn't have a standard global one in monthpack yet, 
        # usually in D:\DEV\USStock_Proj\metadata
        pass

    def fingerprint(self, month: str) -> SourceFingerprint:
        # Generate fingerprint based on node_factors_5m
        month_dir = self._get_month_dir(month)
        nf_dir = month_dir / "node_factors_5m"
        if not nf_dir.exists():
            raise FileNotFoundError(f"No node factors found for month {month}")
        source = ParquetNodeFactorSource(nf_dir)
        return source.fingerprint()


class BoundMonthNodeFactorSource(NodeFactorSource):
    """Adapter to bind a specific month to the standard NodeFactorSource interface."""
    def __init__(self, pack_source: MonthPackNodeFactorSource, month: str):
        self.pack_source = pack_source
        self.month = month

    def fingerprint(self) -> SourceFingerprint:
        return self.pack_source.fingerprint(self.month)

    def available_dates(self) -> list[str]:
        return self.pack_source.available_dates(self.month)

    def load_date(self, trade_date: str, columns: Sequence[str] | None = None) -> pd.DataFrame:
        return self.pack_source.load_node_factors(self.month, trade_date, columns)

    def numeric_feature_columns(self) -> list[str]:
        # Peek at the first available date to get the schema
        dates = self.available_dates()
        if not dates:
            return []
        nf_dir = self.pack_source._get_month_dir(self.month) / "node_factors_5m"
        source = ParquetNodeFactorSource(nf_dir)
        return source.numeric_feature_columns()
