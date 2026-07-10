import os
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

dates = [
    "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09",
    "2026-01-12", "2026-01-13", "2026-01-14", "2026-01-15", "2026-01-16",
    "2026-01-20", "2026-01-21", "2026-01-22", "2026-01-23", "2026-01-26",
    "2026-01-27", "2026-01-28", "2026-01-29", "2026-01-30", "2026-02-02"
]

FLATTENED_ROOT = Path("artifacts/p2_alpha_lab/flattened")
LABELS_ROOT = Path(r"D:\DEV\US-Stock\GraphFactorFactory\data\graph_store_6m\canonical")
OUT_DIR = Path("artifacts/p2_alpha_lab/theme_returns_by_date")
OUT_DIR.mkdir(parents=True, exist_ok=True)
horizons = ["5m", "15m", "30m", "60m", "120m"]

def process_date(d):
    print(f"[{d}] Starting...")
    out_file = OUT_DIR / f"date={d}.parquet"
    if out_file.exists():
        print(f"[{d}] Already exists, skipping.")
        return True
        
    mem_path = FLATTENED_ROOT / f"date={d}" / "theme_memberships.parquet"
    lbl_path = LABELS_ROOT / f"date={d}" / "labels.parquet"
    if not mem_path.exists() or not lbl_path.exists():
        return False
        
    df_mem = pd.read_parquet(mem_path)
    df_lbl = pd.read_parquet(lbl_path)
    
    df_mem['member_id'] = df_mem['member_id'].astype(int)
    df_lbl['symbol_id'] = df_lbl['symbol_id'].astype(int)
    df_mem['decision_time'] = pd.to_datetime(df_mem['decision_time'], utc=True)
    df_lbl['decision_time'] = pd.to_datetime(df_lbl['decision_time'], utc=True)
    
    df = pd.merge(df_mem, df_lbl, left_on=["decision_time", "member_id"], right_on=["decision_time", "symbol_id"], how="inner")
    
    if df.empty: return False
        
    group_cols = ["decision_time", "layer_id", "scale", "level", "theme_id"]
    df = df.sort_values("core_score", ascending=False)
    
    avail_horizons = [h for h in horizons if f"label_{h}" in df.columns]
    for h in avail_horizons:
        df[f'weighted_{h}'] = df[f'label_{h}'] * df['core_score']
        
    grouped = df.groupby(group_cols)
    res = grouped[[f"label_{h}" for h in avail_horizons]].mean()
    res.columns = [f"ret_eq_{h}" for h in avail_horizons]
    
    w_sum = grouped['core_score'].sum()
    for h in avail_horizons:
        res[f"ret_core_{h}"] = grouped[f'weighted_{h}'].sum() / w_sum
        
    top10 = grouped.head(10).groupby(group_cols)[[f"label_{h}" for h in avail_horizons]].mean()
    top10.columns = [f"ret_top10_{h}" for h in avail_horizons]
    
    top5 = grouped.head(5).groupby(group_cols)[[f"label_{h}" for h in avail_horizons]].mean()
    top5.columns = [f"ret_top5_{h}" for h in avail_horizons]
    
    theme_rets = res.join(top5).join(top10).reset_index()
    theme_rets.to_parquet(out_file, index=False)
    print(f"[{d}] Done! ({len(theme_rets)} rows)")
    return True

if __name__ == '__main__':
    print("Building theme returns with multiprocessing...")
    with ProcessPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(process_date, d) for d in dates]
        for fut in as_completed(futures):
            fut.result()
            
    print("All dates processed.")
