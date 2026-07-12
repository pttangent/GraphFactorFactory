#!/usr/bin/env python3
"""Build stable cross-day daily forward-return labels.

This script intentionally refuses to create mock data unless --allow-mock is
explicitly set.  Daily labels are only valid when they are built on a stable
cross-date identifier such as stable_symbol_id, ticker, FIGI, or another
canonical ID.  Do not join date-local symbol_id across days.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOG = logging.getLogger("build_daily_labels")


def read_parquet_input(path: Path) -> pd.DataFrame:
    if path.is_dir():
        files = sorted(path.rglob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files under {path}")
        LOG.info("Loading %d parquet files from %s", len(files), path)
        return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    LOG.info("Loading parquet file %s", path)
    return pd.read_parquet(path)


def ensure_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}; available={list(df.columns)}")


def build_daily_labels(
    prices_df: pd.DataFrame,
    date_col: str,
    stable_id_col: str,
    symbol_col: str | None,
    open_col: str,
    close_col: str,
) -> pd.DataFrame:
    required = [date_col, stable_id_col, open_col, close_col]
    if symbol_col:
        required.append(symbol_col)
    ensure_columns(prices_df, required)

    df = prices_df.copy()
    df[date_col] = pd.to_datetime(df[date_col]).dt.date
    df[open_col] = pd.to_numeric(df[open_col], errors="coerce")
    df[close_col] = pd.to_numeric(df[close_col], errors="coerce")
    df = df.dropna(subset=[date_col, stable_id_col, open_col, close_col]).copy()
    df = df[(df[open_col] > 0) & (df[close_col] > 0)].copy()

    dup = int(df.duplicated([date_col, stable_id_col]).sum())
    if dup > 0:
        raise ValueError(
            f"Found {dup} duplicate rows for ({date_col}, {stable_id_col}); "
            "daily label builder requires one OHLC row per stable symbol per date."
        )

    df = df.sort_values([stable_id_col, date_col]).copy()
    group = df.groupby(stable_id_col, sort=False)

    df["next_trade_date"] = group[date_col].shift(-1)
    df["t3_trade_date"] = group[date_col].shift(-3)
    df["t5_trade_date"] = group[date_col].shift(-5)
    df["next_open"] = group[open_col].shift(-1)
    df["next_close"] = group[close_col].shift(-1)
    df["t3_close"] = group[close_col].shift(-3)
    df["t5_close"] = group[close_col].shift(-5)

    df["close_t"] = df[close_col]
    df["open_t"] = df[open_col]
    df["close_to_next_open"] = df["next_open"] / df["close_t"] - 1.0
    df["next_open_to_close"] = df["next_close"] / df["next_open"] - 1.0
    df["close_to_next_close"] = df["next_close"] / df["close_t"] - 1.0
    df["t3_close_return"] = df["t3_close"] / df["close_t"] - 1.0
    df["t5_close_return"] = df["t5_close"] / df["close_t"] - 1.0

    out = pd.DataFrame({
        "date": pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d"),
        "stable_symbol_id": df[stable_id_col].astype(str),
        "open_t": df["open_t"],
        "close_t": df["close_t"],
        "next_trade_date": pd.to_datetime(df["next_trade_date"]).dt.strftime("%Y-%m-%d"),
        "next_open": df["next_open"],
        "next_close": df["next_close"],
        "close_to_next_open": df["close_to_next_open"],
        "next_open_to_close": df["next_open_to_close"],
        "close_to_next_close": df["close_to_next_close"],
        "t3_trade_date": pd.to_datetime(df["t3_trade_date"]).dt.strftime("%Y-%m-%d"),
        "t3_close_return": df["t3_close_return"],
        "t5_trade_date": pd.to_datetime(df["t5_trade_date"]).dt.strftime("%Y-%m-%d"),
        "t5_close_return": df["t5_close_return"],
    })
    if symbol_col:
        out.insert(2, "symbol", df[symbol_col].astype(str).values)

    # Keep rows with at least a T+1 close label.  T+3/T+5 can be NaN near the end.
    out = out.dropna(subset=["close_to_next_close"]).reset_index(drop=True)
    return out


def make_mock_prices() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.date_range("2026-01-01", "2026-06-30", freq="B")
    symbols = [f"MOCK_{i:04d}" for i in range(1, 101)]
    records: list[dict[str, Any]] = []
    for i, symbol in enumerate(symbols, 1):
        price = 50.0 + i * 0.1
        for d in dates:
            overnight = rng.normal(0, 0.01)
            intraday = rng.normal(0, 0.015)
            open_px = max(0.5, price * (1 + overnight))
            close_px = max(0.5, open_px * (1 + intraday))
            records.append({
                "date": d.strftime("%Y-%m-%d"),
                "stable_symbol_id": f"MOCK_{i:04d}",
                "symbol": symbol,
                "open": open_px,
                "close": close_px,
            })
            price = close_px
    return pd.DataFrame(records)


def validate_labels(labels: pd.DataFrame, max_abs_return: float, allow_extreme_returns: bool) -> dict[str, Any]:
    ret_cols = [
        "close_to_next_open",
        "next_open_to_close",
        "close_to_next_close",
        "t3_close_return",
        "t5_close_return",
    ]
    report: dict[str, Any] = {
        "rows": int(len(labels)),
        "dates": int(labels["date"].nunique()) if "date" in labels else 0,
        "stable_symbols": int(labels["stable_symbol_id"].nunique()) if "stable_symbol_id" in labels else 0,
        "max_abs_return_threshold": float(max_abs_return),
        "return_columns": {},
    }
    extreme_messages: list[str] = []
    for col in ret_cols:
        s = pd.to_numeric(labels[col], errors="coerce")
        max_abs = float(s.abs().max(skipna=True)) if s.notna().any() else float("nan")
        cnt = int((s.abs() > max_abs_return).sum())
        report["return_columns"][col] = {
            "count": int(s.notna().sum()),
            "mean": float(s.mean(skipna=True)) if s.notna().any() else None,
            "max_abs": max_abs,
            "extreme_count": cnt,
        }
        if cnt > 0:
            extreme_messages.append(f"{col}: {cnt} rows exceed abs(return)>{max_abs_return}")
    if extreme_messages and not allow_extreme_returns:
        raise ValueError(
            "Daily label sanity check failed; possible split/ID/price issue. "
            + "; ".join(extreme_messages)
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cross-day daily labels from stable-symbol OHLC prices.")
    parser.add_argument("--raw-daily-prices", default=None, help="Parquet file or directory with daily OHLC rows.")
    parser.add_argument("--out-path", required=True, help="Output daily_labels.parquet path.")
    parser.add_argument("--date-col", default="date")
    parser.add_argument("--stable-id-col", default="stable_symbol_id")
    parser.add_argument("--symbol-col", default="symbol", help="Set to empty string if no symbol column exists.")
    parser.add_argument("--open-col", default="open")
    parser.add_argument("--close-col", default="close")
    parser.add_argument("--max-abs-return", type=float, default=5.0, help="Fail if any label return exceeds this abs threshold unless allowed.")
    parser.add_argument("--allow-extreme-returns", action="store_true")
    parser.add_argument("--allow-mock", action="store_true", help="Explicitly generate mock labels for pipeline smoke tests only.")
    parser.add_argument("--report-path", default=None, help="Optional JSON label report path.")
    args = parser.parse_args()

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report_path) if args.report_path else out_path.with_suffix(".report.json")

    if args.raw_daily_prices:
        prices = read_parquet_input(Path(args.raw_daily_prices))
        source = str(args.raw_daily_prices)
        is_mock = False
    elif args.allow_mock:
        LOG.warning("Generating explicit MOCK daily labels. Do not use this for alpha research.")
        prices = make_mock_prices()
        source = "mock"
        is_mock = True
    else:
        raise SystemExit(
            "Missing --raw-daily-prices. Refusing to generate mock daily_labels.parquet. "
            "Use --allow-mock only for smoke tests."
        )

    symbol_col = args.symbol_col.strip() or None
    labels = build_daily_labels(
        prices,
        date_col=args.date_col,
        stable_id_col=args.stable_id_col,
        symbol_col=symbol_col,
        open_col=args.open_col,
        close_col=args.close_col,
    )
    report = validate_labels(labels, args.max_abs_return, args.allow_extreme_returns)
    report.update({
        "source": source,
        "is_mock": is_mock,
        "date_col": args.date_col,
        "stable_id_col": args.stable_id_col,
        "symbol_col": symbol_col,
        "open_col": args.open_col,
        "close_col": args.close_col,
        "output": str(out_path),
    })

    if is_mock and "mock" not in out_path.name.lower():
        LOG.warning("Mock labels are being written to a non-mock-looking filename: %s", out_path)

    labels.to_parquet(out_path, index=False)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    LOG.info("Wrote %d daily label rows to %s", len(labels), out_path)
    LOG.info("Wrote label report to %s", report_path)


if __name__ == "__main__":
    main()
