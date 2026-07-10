import pandas as pd
import numpy as np
from pathlib import Path
import scipy.stats as stats

FLATTENED_ROOT = Path("artifacts/p2_alpha_lab/flattened")
RETURNS_PATH = Path("artifacts/p2_alpha_lab/theme_returns.parquet")
OUT_DIR = Path("artifacts/p2_alpha_lab")

print("Loading theme returns...")
df_ret = pd.read_parquet(RETURNS_PATH)
df_ret['decision_time'] = pd.to_datetime(df_ret['decision_time'], utc=True)

# Build a lookup for forward returns of any theme at any time
# We'll use ret_core_15m as the past return proxy.
ret_lookup = df_ret.set_index(['decision_time', 'theme_id'])['ret_core_15m'].to_dict()

all_dates = [d.name.split('=')[1] for d in FLATTENED_ROOT.iterdir() if d.is_dir() and 'date=' in d.name]
all_results = []

for d in all_dates:
    print(f"Processing spillover for {d}...")
    rel_path = FLATTENED_ROOT / f"date={d}" / "theme_relation_edges.parquet"
    temp_path = FLATTENED_ROOT / f"date={d}" / "temporal_theme_edges.parquet"
    
    if not rel_path.exists() or not temp_path.exists():
        continue
        
    df_rel = pd.read_parquet(rel_path)
    df_temp = pd.read_parquet(temp_path)
    
    # We only care about fuzzy relations for spillover (not hard_keep tree edges)
    # relation_tier tells us if it's fuzzy
    df_rel = df_rel[df_rel['hard_keep'] == False]
    if df_rel.empty:
        continue
        
    df_rel['decision_time'] = pd.to_datetime(df_rel['decision_time'], utc=True)
    df_temp['dst_time'] = pd.to_datetime(df_temp['dst_time'], utc=True)
    
    # We need A's past return. A is src_theme_id in df_rel.
    # To get A's past return, we find A_prev in df_temp where dst_theme_id == A
    # Wait, there could be multiple A_prev for A. Let's take the one with max continuation_strength.
    idx = df_temp.groupby(['dst_time', 'dst_theme_id'])['continuation_strength'].idxmax()
    best_temp = df_temp.loc[idx]
    
    # Map best_temp (dst_theme_id) -> src_theme_id and src_time
    temp_map = best_temp.set_index(['dst_time', 'dst_theme_id'])[['src_theme_id', 'src_time']]
    
    def get_past_return(row):
        t_curr = row['decision_time']
        A = row['src_theme_id']
        key = (t_curr, A)
        if key in temp_map.index:
            A_prev = temp_map.loc[key, 'src_theme_id']
            t_prev = pd.to_datetime(temp_map.loc[key, 'src_time'], utc=True)
            # The past return of A is the forward 15m return of A_prev at t_prev
            return ret_lookup.get((t_prev, A_prev), np.nan)
        return np.nan
        
    df_rel['src_past_ret_15m'] = df_rel.apply(get_past_return, axis=1)
    
    # Drop rows without past return
    df_rel = df_rel.dropna(subset=['src_past_ret_15m'])
    
    # Calculate spillover score for B (dst_theme_id)
    # score = relation_strength * src_past_ret_15m
    df_rel['spillover_signal'] = df_rel['relation_strength'] * df_rel['src_past_ret_15m']
    
    # Aggregate signal by dst_theme_id (B)
    signal_df = df_rel.groupby(['decision_time', 'dst_theme_id'])['spillover_signal'].sum().reset_index()
    signal_df = signal_df.rename(columns={'dst_theme_id': 'theme_id'})
    
    # Merge B's forward return from df_ret
    day_ret = df_ret[df_ret['decision_time'].dt.strftime('%Y-%m-%d') == d]
    merged = pd.merge(signal_df, day_ret, on=['decision_time', 'theme_id'], how='inner')
    all_results.append(merged)

if all_results:
    final = pd.concat(all_results, ignore_index=True)
    
    # Calculate IC per horizon
    horizons = ["5m", "15m", "30m", "60m"]
    ic_results = []
    
    for h in horizons:
        col = f"ret_core_{h}"
        if col in final.columns:
            valid = final.dropna(subset=['spillover_signal', col])
            if not valid.empty:
                ic, p = stats.spearmanr(valid['spillover_signal'], valid[col])
                # Calculate spread (Top 20% vs Bottom 20%)
                q_top = valid['spillover_signal'].quantile(0.8)
                q_bot = valid['spillover_signal'].quantile(0.2)
                top_ret = valid[valid['spillover_signal'] >= q_top][col].mean()
                bot_ret = valid[valid['spillover_signal'] <= q_bot][col].mean()
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
    res_df.to_csv(OUT_DIR / "relation_spillover_ic.csv", index=False)
    print("Saved relation_spillover_ic.csv")
    print(res_df)
else:
    print("No valid spillover data found.")
