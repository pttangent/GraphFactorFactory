from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class SplitSourceMetadata:
    path: str
    sha256: str
    raw_row_count: int
    normalized_event_count: int
    collapsed_duplicate_ratio_count: int
    min_date: str | None
    max_date: str | None


class SplitAdjustmentSource:
    REQUIRED_COLUMNS = {"symbol", "date", "from", "to"}

    def __init__(self, csv_path: str | Path):
        self.path = Path(csv_path).expanduser().resolve()
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        frame = pd.read_csv(self.path, encoding="utf-8-sig")
        missing = self.REQUIRED_COLUMNS.difference(frame.columns)
        if missing:
            raise ValueError(f"Split CSV missing columns: {sorted(missing)}")
        frame = frame.rename(columns={"date": "effective_date"}).copy()
        frame["symbol"] = frame["symbol"].astype(str).str.upper().str.strip()
        frame["effective_date"] = pd.to_datetime(frame["effective_date"], errors="coerce").dt.date
        frame["from"] = pd.to_numeric(frame["from"], errors="coerce")
        frame["to"] = pd.to_numeric(frame["to"], errors="coerce")
        frame["share_multiplier"] = frame["to"] / frame["from"]
        frame = frame.replace([float("inf"), float("-inf")], pd.NA)
        frame = frame.dropna(subset=["symbol", "effective_date", "share_multiplier"])
        frame = frame[frame["share_multiplier"] > 0]
        self.raw_row_count = int(len(frame))
        frame["ratio_key"] = frame["share_multiplier"].round(12)
        deduplicated = frame.drop_duplicates(["symbol", "effective_date", "ratio_key"])
        self.collapsed_duplicate_ratio_count = int(len(frame) - len(deduplicated))
        normalized = deduplicated.groupby(["symbol", "effective_date"], as_index=False).agg(
            share_multiplier=("share_multiplier", "prod"),
            component_count=("share_multiplier", "size"),
        )
        self.frame = normalized.sort_values(["symbol", "effective_date"]).reset_index(drop=True)

    @property
    def metadata(self) -> SplitSourceMetadata:
        digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        dates = self.frame["effective_date"]
        return SplitSourceMetadata(
            path=str(self.path),
            sha256=digest,
            raw_row_count=self.raw_row_count,
            normalized_event_count=int(len(self.frame)),
            collapsed_duplicate_ratio_count=self.collapsed_duplicate_ratio_count,
            min_date=str(dates.min()) if len(dates) else None,
            max_date=str(dates.max()) if len(dates) else None,
        )

    def cumulative_multiplier(self, symbols: pd.Series, entry_times: pd.Series, exit_times: pd.Series) -> pd.Series:
        result = pd.Series(1.0, index=symbols.index, dtype="float64")
        entry_dates = pd.to_datetime(entry_times, utc=True).dt.date
        exit_dates = pd.to_datetime(exit_times, utc=True).dt.date
        lookup = {symbol: group for symbol, group in self.frame.groupby("symbol", sort=False)}
        normalized_symbols = symbols.astype(str).str.upper()
        for symbol, indexes in normalized_symbols.groupby(normalized_symbols).groups.items():
            events = lookup.get(symbol)
            if events is None:
                continue
            event_dates = events["effective_date"].to_numpy()
            multipliers = events["share_multiplier"].to_numpy(dtype="float64")
            for index in indexes:
                mask = (event_dates > entry_dates.loc[index]) & (event_dates <= exit_dates.loc[index])
                if mask.any():
                    result.loc[index] = float(multipliers[mask].prod())
        return result
