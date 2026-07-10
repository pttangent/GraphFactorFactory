import os
import pandas as pd
import numpy as np
from pathlib import Path

dates = [
    "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09",
    "2026-01-12", "2026-01-13", "2026-01-14", "2026-01-15", "2026-01-16",
    "2026-01-20", "2026-01-21", "2026-01-22", "2026-01-23", "2026-01-26",
    "2026-01-27", "2026-01-28", "2026-01-29", "2026-01-30", "2026-02-02"
]

FLATTENED_ROOT = Path("artifacts/p2_alpha_lab/flattened")
LABELS_ROOT = Path(r"D:\DEV\US-Stock\GraphFactorFactory\data\graph_store_6m\canonical")
OUT_DIR = Path("artifacts/p2_alpha_lab")

horizons = ["5m", "15m", "30m", "60m", "120m"]

all_theme_returns = []

for d in dates:
    print(f"Building theme returns for {d}...")
    mem_path = FLATTENED_ROOT / f"date={d}" / "theme_memberships.parquet"
    lbl_path = LABELS_ROOT / f"date={d}" / "labels.parquet"
    
    if not mem_path.exists() or not lbl_path.exists():
        print(f"  Missing data for {d}, skipping.")
        continue
        
    df_mem = pd.read_parquet(mem_path)
    df_lbl = pd.read_parquet(lbl_path)
    
    # Merge memberships with labels
    # member_id maps to symbol_id
    df_mem['member_id'] = df_mem['member_id'].astype(int)
    df_lbl['symbol_id'] = df_lbl['symbol_id'].astype(int)
    
    # Ensure datetime matching
    df_mem['decision_time'] = pd.to_datetime(df_mem['decision_time'], utc=True)
    df_lbl['decision_time'] = pd.to_datetime(df_lbl['decision_time'], utc=True)
    
    # We also need to match on decision_time. P1 memberships should have decision_time.
    df = pd.merge(df_mem, df_lbl, left_on=["decision_time", "member_id"], right_on=["decision_time", "symbol_id"], how="inner")
    
    if df.empty:
        print(f"  No overlapping labels for {d}.")
        continue
        
    # We want to group by: decision_time, layer_id, scale, level, theme_id
    group_cols = ["decision_time", "layer_id", "scale", "level", "theme_id"]
    
    # Pre-sort by core_score descending to easily get top 5 / top 10
    df = df.sort_values("core_score", ascending=False)
    
    def agg_theme(g):
        res = {}
        for h in horizons:
            col = f"label_{h}"
            if col not in g.columns:
                continue
            
            vals = g[col]
            weights = g["core_score"]
            # Equal weight
            res[f"ret_eq_{h}"] = vals.mean()
            # Core weighted
            w_sum = weights.sum()
            res[f"ret_core_{h}"] = (vals * weights).sum() / w_sum if w_sum > 0 else np.nan
            # Top 5 core equal weight
            res[f"ret_top5_{h}"] = vals.head(5).mean()
            # Top 10 core equal weight
            res[f"ret_top10_{h}"] = vals.head(10).mean()
        return pd.Series(res)

    theme_rets = df.groupby(group_cols).apply(agg_theme).reset_index()
    all_theme_returns.append(theme_rets)
    print(f"  Generated {len(theme_rets)} theme return records.")

if all_theme_returns:
    final_df = pd.concat(all_theme_returns, ignore_index=True)
    final_df.to_parquet(OUT_DIR / "theme_returns.parquet", index=False)
    print(f"Successfully saved theme_returns.parquet with {len(final_df)} rows.")
else:
    print("No data processed.")
