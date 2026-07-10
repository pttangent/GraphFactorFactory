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
    
    # Create past labels by shifting decision_time forward
    df_lbl_past = df_lbl[['decision_time', 'symbol_id', 'label_15m']].copy()
    # If decision_time was 09:30, its forward 15m return is the past 15m return at 09:45
    df_lbl_past['decision_time'] = df_lbl_past['decision_time'] + pd.Timedelta(minutes=15)
    df_lbl_past = df_lbl_past.rename(columns={'label_15m': 'past_ret_15m'})
    
    # Merge current forward labels and past labels
    df_lbl_combined = pd.merge(df_lbl, df_lbl_past, on=['decision_time', 'symbol_id'], how='inner')
    
    # Merge with memberships
    df = pd.merge(df_mem, df_lbl_combined, left_on=['decision_time', 'member_id'], right_on=['decision_time', 'symbol_id'], how='inner')
    
    if df.empty:
        continue
        
    # For each theme (decision_time, layer_id, scale, theme_id), identify core and peripheral
    def calc_diffusion(g):
        g = g.sort_values('core_score', ascending=False)
        n = len(g)
        if n < 5:
            return pd.DataFrame()
            
        n_core = max(1, int(n * 0.2))
        n_peri = max(1, int(n * 0.5))
        
        core_members = g.head(n_core)
        peri_members = g.tail(n_peri).copy()
        
        # Calculate core past return
        core_past_ret = core_members['past_ret_15m'].mean()
        
        # Signal for peripheral
        # peripheral_score = core_past_ret * (max_core_score - member_core_score)
        max_core = g['core_score'].max()
        peri_members['peri_signal'] = core_past_ret * (max_core - peri_members['core_score'])
        
        return peri_members[['decision_time', 'symbol_id', 'peri_signal', 'label_5m', 'label_15m', 'label_30m', 'label_60m']]
        
    res = df.groupby(['decision_time', 'layer_id', 'scale', 'theme_id']).apply(calc_diffusion).reset_index(drop=True)
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
