import pandas as pd
import numpy as np
from pathlib import Path
import scipy.stats as stats

FLATTENED_ROOT = Path("artifacts/p2_alpha_lab/flattened")
LABELS_ROOT = Path(r"D:\DEV\US-Stock\GraphFactorFactory\data\graph_store_6m\canonical")
OUT_DIR = Path("artifacts/p2_alpha_lab")

all_dates = [d.name.split('=')[1] for d in FLATTENED_ROOT.iterdir() if d.is_dir() and 'date=' in d.name]
all_results = []

for d in all_dates:
    print(f"Processing core-peripheral for {d}...")
    mem_path = FLATTENED_ROOT / f"date={d}" / "theme_memberships.parquet"
    lbl_path = LABELS_ROOT / f"date={d}" / "labels.parquet"
    
    if not mem_path.exists() or not lbl_path.exists():
        continue
        
    df_mem = pd.read_parquet(mem_path)
    df_lbl = pd.read_parquet(lbl_path)
    
    df_mem['decision_time'] = pd.to_datetime(df_mem['decision_time'], utc=True)
    df_lbl['decision_time'] = pd.to_datetime(df_lbl['decision_time'], utc=True)
    df_mem['member_id'] = df_mem['member_id'].astype(int)
    df_lbl['symbol_id'] = df_lbl['symbol_id'].astype(int)
    
    df_lbl_past = df_lbl[['decision_time', 'symbol_id', 'label_15m']].copy()
    df_lbl_past['decision_time'] = df_lbl_past['decision_time'] + pd.Timedelta(minutes=15)
    df_lbl_past = df_lbl_past.rename(columns={'label_15m': 'past_ret_15m'})
    
    df_lbl_combined = pd.merge(df_lbl, df_lbl_past, on=['decision_time', 'symbol_id'], how='inner')
    df = pd.merge(df_mem, df_lbl_combined, left_on=['decision_time', 'member_id'], right_on=['decision_time', 'symbol_id'], how='inner')
    
    if df.empty:
        continue
        
    # Vectorized computation
    group_cols = ['decision_time', 'layer_id', 'scale', 'theme_id']
    
    # Sort by group and core_score descending
    df = df.sort_values(group_cols + ['core_score'], ascending=[True, True, True, True, False])
    
    # Count members per group
    counts = df.groupby(group_cols).size()
    # Filter groups with >= 5 members
    valid_groups = counts[counts >= 5].index
    df = df[df.set_index(group_cols).index.isin(valid_groups)]
    
    if df.empty:
        continue
        
    # Assign rank within group
    df['rank'] = df.groupby(group_cols).cumcount()
    df['group_size'] = df.groupby(group_cols)['rank'].transform('size')
    
    # Identify core and peripheral
    df['is_core'] = df['rank'] < np.maximum(1, (df['group_size'] * 0.2).astype(int))
    df['is_peri'] = df['rank'] >= (df['group_size'] - np.maximum(1, (df['group_size'] * 0.5).astype(int)))
    
    # Calculate core past return per group
    core_past_ret = df[df['is_core']].groupby(group_cols)['past_ret_15m'].mean().rename('core_past_ret')
    # Calculate max core score per group
    max_core = df.groupby(group_cols)['core_score'].max().rename('max_core_score')
    
    # Join back to peri members
    peri_df = df[df['is_peri']].copy()
    peri_df = peri_df.join(core_past_ret, on=group_cols)
    peri_df = peri_df.join(max_core, on=group_cols)
    
    # Signal
    peri_df['peri_signal'] = peri_df['core_past_ret'] * (peri_df['max_core_score'] - peri_df['core_score'])
    
    res = peri_df[['decision_time', 'symbol_id', 'peri_signal', 'label_5m', 'label_15m', 'label_30m', 'label_60m']]
    all_results.append(res)

if all_results:
    final = pd.concat(all_results, ignore_index=True)
    
    horizons = ["5m", "15m", "30m", "60m"]
    ic_results = []
    
    for h in horizons:
        col = f"label_{h}"
        if col in final.columns:
            valid = final.dropna(subset=['peri_signal', col])
            if not valid.empty:
                ic, p = stats.spearmanr(valid['peri_signal'], valid[col])
                q_top = valid['peri_signal'].quantile(0.8)
                q_bot = valid['peri_signal'].quantile(0.2)
                top_ret = valid[valid['peri_signal'] >= q_top][col].mean()
                bot_ret = valid[valid['peri_signal'] <= q_bot][col].mean()
                spread = top_ret - bot_ret
                
                ic_results.append({
                    "horizon": h,
                    "sample_count": len(valid),
                    "rank_ic": ic,
                    "p_value": p,
                    "top_quintile_ret": top_ret,
                    "bottom_quintile_ret": bot_ret,
                    "long_short_spread": spread
                })
                
    res_df = pd.DataFrame(ic_results)
    res_df.to_csv(OUT_DIR / "core_peripheral_ic.csv", index=False)
    print("Saved core_peripheral_ic.csv")
    print(res_df)
else:
    print("No valid core-peripheral data found.")
