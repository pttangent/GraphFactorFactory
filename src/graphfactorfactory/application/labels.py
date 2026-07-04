from __future__ import annotations

import pandas as pd


def build_forward_labels(panel: pd.DataFrame, horizons_minutes: tuple[int, ...], bar_minutes: int = 5) -> pd.DataFrame:
    required = {"decision_time", "symbol", "close"}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"Label input missing columns: {sorted(missing)}")
    base = panel[["decision_time", "symbol", "close"]].copy()
    base["decision_time"] = pd.to_datetime(base["decision_time"], utc=True)
    price = base.rename(columns={"decision_time": "price_time", "close": "price"})
    output = base[["decision_time", "symbol"]].copy()
    entry_time = output["decision_time"] + pd.Timedelta(minutes=bar_minutes)
    entry = output.assign(price_time=entry_time).merge(price, on=["symbol", "price_time"], how="left", validate="many_to_one")["price"]
    output["label_entry_time"] = entry_time
    output["label_entry_price"] = entry.astype("float32")
    for horizon in horizons_minutes:
        exit_time = entry_time + pd.Timedelta(minutes=horizon)
        exit_price = output[["decision_time", "symbol"]].assign(price_time=exit_time).merge(price, on=["symbol", "price_time"], how="left", validate="many_to_one")["price"]
        output[f"label_exit_time_{horizon}m"] = exit_time
        output[f"label_exit_price_{horizon}m"] = exit_price.astype("float32")
        output[f"label_{horizon}m"] = (exit_price.to_numpy() / entry.to_numpy() - 1.0).astype("float32")
    if output.duplicated(["decision_time", "symbol"]).any():
        raise AssertionError("Label primary key is not unique")
    return output
