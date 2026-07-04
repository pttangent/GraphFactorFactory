from __future__ import annotations

import pandas as pd

from graphfactorfactory.application.causality import audit_pit_panel
from graphfactorfactory.domain.config import BuildConfig


def filter_regular_session(frame: pd.DataFrame, config: BuildConfig) -> pd.DataFrame:
    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
    result["available_time"] = pd.to_datetime(result["available_time"], utc=True)
    local = result["timestamp"].dt.tz_convert(config.market_timezone)
    local_time = local.dt.strftime("%H:%M")
    local_dates = local.dt.date
    result = result[(local_time >= config.market_open) & (local_time < config.market_close)].copy()
    result["session_date"] = pd.Series(local_dates, index=frame.index).loc[result.index].astype(str)
    return result


def decision_grid(events: pd.DataFrame, config: BuildConfig) -> pd.DatetimeIndex:
    if events.empty:
        return pd.DatetimeIndex([], tz="UTC")
    available = pd.to_datetime(events["available_time"], utc=True)
    start = available.min().ceil(config.frequency)
    end = available.max().floor(config.frequency)
    return pd.date_range(start, end, freq=config.frequency)


def build_point_in_time_panel(events: pd.DataFrame, config: BuildConfig, stale_tolerance: str = "15min") -> pd.DataFrame:
    frame = events.copy()
    frame["symbol"] = frame["symbol"].astype(str)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True).astype("datetime64[ns, UTC]")
    frame["available_time"] = pd.to_datetime(frame["available_time"], utc=True).astype("datetime64[ns, UTC]")
    decisions = decision_grid(frame, config)
    symbols = pd.Index(sorted(frame["symbol"].unique()), name="symbol")
    grid = pd.MultiIndex.from_product([decisions, symbols], names=["decision_time", "symbol"]).to_frame(index=False)
    source = frame.rename(columns={"timestamp": "source_timestamp", "available_time": "source_available_time"})
    source = source.sort_values(["source_available_time", "symbol", "source_timestamp"])
    grid = grid.sort_values(["decision_time", "symbol"])
    panel = pd.merge_asof(grid, source, left_on="decision_time", right_on="source_available_time", by="symbol", direction="backward", tolerance=pd.Timedelta(stale_tolerance))
    panel = panel[panel["close"].notna()].copy()
    audit_pit_panel(panel)
    return panel.sort_values(["decision_time", "symbol"]).reset_index(drop=True)
