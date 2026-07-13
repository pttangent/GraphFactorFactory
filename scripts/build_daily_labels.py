#!/usr/bin/env python3
"""Build cross-day labels with an explicit execution-time contract.

For an end-of-day feature that is only known after the final intraday snapshot,
``close_t -> future close`` is not executable. The live-safe labels produced
here therefore include ``next_open_to_tNd_close``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import pyzipper
except ImportError:  # optional encrypted input support
    pyzipper = None

LOG = logging.getLogger("build_daily_labels")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def generate_zip_password(filename: str) -> bytes:
    salt = "vvtr123!@#qwe"
    return hashlib.sha256(f"{filename}{salt}".encode("utf-8")).hexdigest().encode("utf-8")


def read_price_input(path: Path) -> pd.DataFrame:
    if path.is_file():
        return pd.read_parquet(path)
    zips = sorted(path.rglob("*.zip"))
    if zips:
        if pyzipper is None:
            raise RuntimeError("pyzipper is required to read encrypted ZIP inputs")
        frames = []
        for zip_path in zips:
            with pyzipper.AESZipFile(zip_path) as archive:
                names = [name for name in archive.namelist() if not name.endswith("/")]
                if not names:
                    continue
                with archive.open(names[0], pwd=generate_zip_password(zip_path.name)) as source:
                    frames.append(pd.read_csv(source))
        if not frames:
            raise FileNotFoundError(f"no readable rows under {path}")
        return pd.concat(frames, ignore_index=True)
    files = sorted(path.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet or encrypted ZIP files under {path}")
    return pd.concat((pd.read_parquet(file) for file in files), ignore_index=True)


def require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"missing required columns: {missing}; available={list(frame.columns)}")


def build_daily_labels(
    prices: pd.DataFrame,
    *,
    date_col: str = "date",
    stable_id_col: str = "stable_symbol_id",
    symbol_col: str | None = "symbol",
    open_col: str = "open",
    close_col: str = "close",
    max_horizon: int | None = None,
) -> pd.DataFrame:
    required = [date_col, stable_id_col, open_col, close_col] + ([symbol_col] if symbol_col else [])
    require_columns(prices, required)
    frame = prices.copy()
    frame[date_col] = pd.to_datetime(frame[date_col], utc=True, errors="coerce").dt.date
    frame[open_col] = pd.to_numeric(frame[open_col], errors="coerce")
    frame[close_col] = pd.to_numeric(frame[close_col], errors="coerce")
    frame = frame.dropna(subset=[date_col, stable_id_col, open_col, close_col])
    frame = frame[(frame[open_col] > 0) & (frame[close_col] > 0)].copy()
    duplicates = frame.duplicated([date_col, stable_id_col], keep=False)
    if duplicates.any():
        print(f"Dropping {duplicates.sum()} duplicate daily OHLC rows for stable symbol/date")
        frame = frame.drop_duplicates([date_col, stable_id_col], keep="last")
    frame = frame.sort_values([stable_id_col, date_col]).copy()
    grouped = frame.groupby(stable_id_col, sort=False)

    counts = grouped.size()
    inferred_max = max(int(counts.max()) - 1, 1)
    max_horizon = inferred_max if max_horizon is None else min(max_horizon, inferred_max)
    frame["close_t"] = frame[close_col]
    frame["open_t"] = frame[open_col]
    frame["next_trade_date"] = grouped[date_col].shift(-1)
    frame["next_open"] = grouped[open_col].shift(-1)
    frame["next_close"] = grouped[close_col].shift(-1)
    frame["close_to_next_open"] = frame["next_open"] / frame["close_t"] - 1.0
    frame["next_open_to_close"] = frame["next_close"] / frame["next_open"] - 1.0
    frame["close_to_next_close"] = frame["next_close"] / frame["close_t"] - 1.0

    output: dict[str, Any] = {
        "date": pd.to_datetime(frame[date_col]).dt.strftime("%Y-%m-%d"),
        "stable_symbol_id": frame[stable_id_col].astype(str),
        "open_t": frame["open_t"],
        "close_t": frame["close_t"],
        "next_trade_date": pd.to_datetime(frame["next_trade_date"]).dt.strftime("%Y-%m-%d"),
        "next_open": frame["next_open"],
        "next_close": frame["next_close"],
        "close_to_next_open": frame["close_to_next_open"],
        "next_open_to_close": frame["next_open_to_close"],
        "close_to_next_close": frame["close_to_next_close"],
        "next_open_to_t1_close": frame["next_open_to_close"],
        "t1_trade_date": pd.to_datetime(frame["next_trade_date"]).dt.strftime("%Y-%m-%d"),
    }
    if symbol_col:
        output["symbol"] = frame[symbol_col].astype(str)

    for horizon in range(2, max_horizon + 1):
        trade_date = grouped[date_col].shift(-horizon)
        future_open = grouped[open_col].shift(-horizon)
        future_close = grouped[close_col].shift(-horizon)
        output[f"t{horizon}_trade_date"] = pd.to_datetime(trade_date).dt.strftime("%Y-%m-%d")
        output[f"t{horizon}_close_return"] = future_close / frame["close_t"] - 1.0
        output[f"t{horizon}_open_return"] = future_open / frame["close_t"] - 1.0
        output[f"next_open_to_t{horizon}_close"] = future_close / frame["next_open"] - 1.0

    labels = pd.DataFrame(output)
    labels["daily_feature_available_after"] = labels["date"]
    labels["daily_execution_policy"] = "next_session_open"
    labels = labels.dropna(subset=["next_open_to_t1_close"]).reset_index(drop=True)
    return labels


def validate_labels(labels: pd.DataFrame, max_abs_return: float, allow_extreme_returns: bool) -> dict[str, Any]:
    return_columns = [
        column
        for column in labels
        if column in {"close_to_next_open", "next_open_to_close", "close_to_next_close"}
        or column.endswith("_return")
        or re.fullmatch(r"next_open_to_t\d+_close", column)
    ]
    report: dict[str, Any] = {
        "rows": len(labels),
        "dates": labels["date"].nunique(),
        "stable_symbols": labels["stable_symbol_id"].nunique(),
        "daily_execution_policy": "next_session_open",
        "live_safe_target_pattern": "next_open_to_tNd_close",
        "return_columns": {},
    }
    errors = []
    for column in return_columns:
        values = pd.to_numeric(labels[column], errors="coerce")
        extreme_count = int((values.abs() > max_abs_return).sum())
        report["return_columns"][column] = {
            "count": int(values.notna().sum()),
            "mean": float(values.mean()) if values.notna().any() else None,
            "max_abs": float(values.abs().max()) if values.notna().any() else None,
            "extreme_count": extreme_count,
        }
        if extreme_count:
            errors.append(f"{column}: {extreme_count}")
    if errors and not allow_extreme_returns:
        raise ValueError("daily label sanity check failed: " + "; ".join(errors))
    return report


def make_mock_prices() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.date_range("2026-01-01", "2026-06-30", freq="B")
    rows = []
    for index in range(100):
        price = 50 + index / 10
        for date in dates:
            open_price = max(0.5, price * (1 + rng.normal(0, 0.01)))
            close_price = max(0.5, open_price * (1 + rng.normal(0, 0.015)))
            rows.append({"date": date, "stable_symbol_id": f"MOCK_{index:04d}", "symbol": f"MOCK_{index:04d}", "open": open_price, "close": close_price})
            price = close_price
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PIT-safe next-open daily forward-return labels")
    parser.add_argument("--raw-daily-prices")
    parser.add_argument("--out-path", required=True)
    parser.add_argument("--date-col", default="date")
    parser.add_argument("--stable-id-col", default="stable_symbol_id")
    parser.add_argument("--symbol-col", default="symbol")
    parser.add_argument("--open-col", default="open")
    parser.add_argument("--close-col", default="close")
    parser.add_argument("--max-horizon", type=int)
    parser.add_argument("--max-abs-return", type=float, default=5.0)
    parser.add_argument("--allow-extreme-returns", action="store_true")
    parser.add_argument("--allow-mock", action="store_true")
    parser.add_argument("--report-path")
    args = parser.parse_args()

    if args.raw_daily_prices:
        prices = read_price_input(Path(args.raw_daily_prices))
        source = args.raw_daily_prices
        is_mock = False
    elif args.allow_mock:
        prices = make_mock_prices()
        source = "mock"
        is_mock = True
    else:
        raise SystemExit("missing --raw-daily-prices; use --allow-mock only for smoke tests")

    labels = build_daily_labels(
        prices,
        date_col=args.date_col,
        stable_id_col=args.stable_id_col,
        symbol_col=args.symbol_col.strip() or None,
        open_col=args.open_col,
        close_col=args.close_col,
        max_horizon=args.max_horizon,
    )
    report = validate_labels(labels, args.max_abs_return, args.allow_extreme_returns)
    report.update({"source": source, "is_mock": is_mock})
    output = Path(args.out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(output, index=False)
    report_path = Path(args.report_path) if args.report_path else output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    LOG.info("wrote %d rows to %s", len(labels), output)


if __name__ == "__main__":
    main()
