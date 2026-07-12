#!/usr/bin/env python3
"""Inject next-open-executable daily labels into intraday label shards."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def prepare_daily_labels(daily: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    required_daily = {"date", "stable_symbol_id", "next_trade_date"}
    required_mapping = {"symbol", "symbol_id"}
    if missing := required_daily - set(daily):
        raise ValueError(f"daily labels missing {sorted(missing)}")
    if missing := required_mapping - set(mapping):
        raise ValueError(f"symbol mapping missing {sorted(missing)}")

    merged = daily.merge(mapping[["symbol", "symbol_id"]].drop_duplicates("symbol"), left_on="stable_symbol_id", right_on="symbol", how="inner", validate="many_to_one")
    output = merged[["date", "symbol_id"]].copy()
    output["date"] = output["date"].astype(str)

    horizon_columns: list[str] = []
    for column in merged:
        match = re.fullmatch(r"next_open_to_t(\d+)_close", column)
        if not match:
            continue
        horizon = int(match.group(1))
        suffix = f"{horizon}d_open"
        target = f"label_{suffix}"
        output[target] = pd.to_numeric(merged[column], errors="coerce").astype("float32")
        output[f"label_entry_date_{suffix}"] = merged["next_trade_date"].astype(str)
        exit_source = "next_trade_date" if horizon == 1 else f"t{horizon}_trade_date"
        if exit_source not in merged:
            raise ValueError(f"daily labels missing {exit_source} for {target}")
        output[f"label_exit_date_{suffix}"] = merged[exit_source].astype(str)
        horizon_columns.append(target)

    if not horizon_columns:
        raise ValueError("no next_open_to_tNd_close columns found; close-start labels are intentionally rejected")
    output["daily_label_execution_policy"] = "next_session_open"
    return output


def inject_daily_labels(labels_root: Path, daily_labels_path: Path, mapping_path: Path, month: str | None = None) -> dict:
    daily = pd.read_parquet(daily_labels_path)
    mapping = pd.read_parquet(mapping_path)
    prepared = prepare_daily_labels(daily, mapping)
    pattern = f"date={month}-*" if month else "date=*"
    files = sorted(path / "labels.parquet" for path in labels_root.glob(pattern) if (path / "labels.parquet").exists())
    updated = rows = 0
    injected = [c for c in prepared if c.startswith("label_") or c == "daily_label_execution_policy"]
    for label_path in files:
        date = label_path.parent.name.split("=", 1)[1]
        daily_for_date = prepared.loc[prepared.date.eq(date)].drop(columns="date")
        if daily_for_date.empty:
            continue
        intraday = pd.read_parquet(label_path)
        replace_columns = [column for column in injected if column in intraday]
        if replace_columns:
            intraday = intraday.drop(columns=replace_columns)
        merged = intraday.merge(daily_for_date, on="symbol_id", how="left", validate="many_to_one")
        temporary = label_path.with_suffix(".parquet.tmp")
        merged.to_parquet(temporary, index=False)
        temporary.replace(label_path)
        updated += 1
        rows += len(merged)
    return {"updated_files": updated, "rows": rows, "execution_policy": "next_session_open", "injected_columns": injected}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-root", required=True)
    parser.add_argument("--daily-labels", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--month")
    args = parser.parse_args()
    result = inject_daily_labels(Path(args.labels_root), Path(args.daily_labels), Path(args.mapping), args.month)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
