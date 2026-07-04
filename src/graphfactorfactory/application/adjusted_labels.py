from __future__ import annotations

import pandas as pd


def build_split_adjusted_labels(panel, horizons_minutes, split_source, bar_minutes=5):
    base = panel[["decision_time", "symbol", "close"]].copy()
    base["decision_time"] = pd.to_datetime(base["decision_time"], utc=True)
    prices = base.rename(columns={"decision_time": "price_time", "close": "price"})
    result = base[["decision_time", "symbol"]].copy()
    entry_time = result["decision_time"] + pd.Timedelta(minutes=bar_minutes)
    entry_price = result.assign(price_time=entry_time).merge(prices, on=["symbol", "price_time"], how="left", validate="many_to_one")["price"]
    result["label_entry_time"] = entry_time
    result["label_entry_price"] = entry_price.astype("float32")
    result["label_adjustment_policy"] = "split_adjusted_target_only"
    for horizon in horizons_minutes:
        exit_time = entry_time + pd.Timedelta(minutes=horizon)
        exit_price = result[["decision_time", "symbol"]].assign(price_time=exit_time).merge(prices, on=["symbol", "price_time"], how="left", validate="many_to_one")["price"]
        multiplier = split_source.cumulative_multiplier(result["symbol"], entry_time, exit_time)
        result[f"label_exit_time_{horizon}m"] = exit_time
        result[f"label_exit_price_raw_{horizon}m"] = exit_price.astype("float32")
        result[f"label_split_multiplier_{horizon}m"] = multiplier.astype("float32")
        result[f"label_raw_{horizon}m"] = (exit_price / entry_price - 1.0).astype("float32")
        result[f"label_split_adj_{horizon}m"] = (exit_price * multiplier / entry_price - 1.0).astype("float32")
        result[f"label_{horizon}m"] = result[f"label_split_adj_{horizon}m"]
    if result.duplicated(["decision_time", "symbol"]).any():
        raise AssertionError("Label primary key is not unique")
    return result
