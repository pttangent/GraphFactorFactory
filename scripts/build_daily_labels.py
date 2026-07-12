#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOG = logging.getLogger(__name__)

def build_daily_labels(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a dataframe of daily OHLC prices with ['date', 'stable_symbol_id', 'symbol', 'open', 'close'],
    computes the required daily label metrics.
    """
    # Sort chronologically
    df = prices_df.sort_values(by=["stable_symbol_id", "date"]).copy()
    
    # Next day features
    df["next_open"] = df.groupby("stable_symbol_id")["open"].shift(-1)
    df["next_close"] = df.groupby("stable_symbol_id")["close"].shift(-1)
    
    # T+3 and T+5 close
    df["t3_close"] = df.groupby("stable_symbol_id")["close"].shift(-3)
    df["t5_close"] = df.groupby("stable_symbol_id")["close"].shift(-5)
    
    # Return features
    df["close_t"] = df["close"]
    df["close_to_next_open"] = (df["next_open"] / df["close_t"]) - 1.0
    df["next_open_to_close"] = (df["next_close"] / df["next_open"]) - 1.0
    df["close_to_next_close"] = (df["next_close"] / df["close_t"]) - 1.0
    
    df["t3_close_return"] = (df["t3_close"] / df["close_t"]) - 1.0
    df["t5_close_return"] = (df["t5_close"] / df["close_t"]) - 1.0
    
    # Select target columns
    cols = [
        "date",
        "stable_symbol_id",
        "symbol",
        "close_t",
        "next_open",
        "next_close",
        "close_to_next_open",
        "next_open_to_close",
        "close_to_next_close",
        "t3_close_return",
        "t5_close_return"
    ]
    return df[cols].dropna(subset=["close_to_next_close"])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-daily-prices", help="Path to daily OHLC parquet file", required=False)
    parser.add_argument("--out-path", required=True, help="Path to output daily_labels.parquet")
    args = parser.parse_args()
    
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not args.raw_daily_prices:
        LOG.warning("No --raw-daily-prices provided. Generating mock daily_labels.parquet for pipeline validation.")
        # Mock implementation since raw daily EOD data location is TBD
        dates = pd.date_range("2026-01-01", "2026-06-30", freq="B")
        symbols = [f"SYM_{i}" for i in range(1, 101)]
        records = []
        for d in dates:
            for i, s in enumerate(symbols):
                records.append({
                    "date": d,
                    "stable_symbol_id": i + 1000,
                    "symbol": s,
                    "open": 100.0 + np.random.normal(0, 1),
                    "close": 100.0 + np.random.normal(0, 1)
                })
        df = pd.DataFrame(records)
    else:
        LOG.info(f"Loading prices from {args.raw_daily_prices}")
        df = pd.read_parquet(args.raw_daily_prices)
        
    LOG.info("Computing daily labels (next_open, t3/t5 returns)...")
    labels_df = build_daily_labels(df)
    
    LOG.info(f"Writing {len(labels_df)} daily label rows to {out_path}")
    labels_df.to_parquet(out_path, index=False)
    LOG.info("Done.")

if __name__ == "__main__":
    main()
