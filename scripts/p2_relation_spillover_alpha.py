import pandas as pd
import numpy as np
from pathlib import Path
import scipy.stats as stats
from concurrent.futures import ProcessPoolExecutor, as_completed

FLATTENED_ROOT = Path("artifacts/p2_alpha_lab/flattened")
RETURNS_PATH = Path("artifacts/p2_alpha_lab/theme_returns.parquet")
OUT_DIR = Path("artifacts/p2_alpha_lab")

print("Loading theme returns...")
df_ret = pd.read_parquet(RETURNS_PATH)
df_ret['decision_time'] = pd.to_datetime(df_ret['decision_time'], utc=True)
ret_lookup = df_ret.set_index(['decision_time', 'theme_id'])['ret_core_15m'].to_dict()

all_dates = [d.name.split('=')[1] for d in FLATTENED_ROOT.iterdir() if d.is_dir() and 'date=' in d.name]

def process_date(d):
    rel_path = FLATTENED_ROOT / f"date={d}" / "theme_relation_edges.parquet"
    temp_path = FLATTENED_ROOT / f"date={d}" / "temporal_theme_edges.parquet"
    if not rel_path.exists() or not temp_path.exists():
        return None
        
    df_rel = pd.read_parquet(rel_path)
    df_temp = pd.read_parquet(temp_path)
    df_rel = df_rel[df_rel['hard_keep'] == False]
    if df_rel.empty:
        return None
        
    df_rel['decision_time'] = pd.to_datetime(df_rel['decision_time'], utc=True)
    df_temp['dst_time'] = pd.to_datetime(df_temp['dst_time'], utc=True)
    
    idx = df_temp.groupby(['dst_time', 'dst_theme_id'])['continuation_strength'].idxmax()
    best_temp = df_temp.loc[idx]
    temp_map = best_temp.set_index(['dst_time', 'dst_theme_id'])[['src_theme_id', 'src_time']]
    
    def get_past_return(row):
        t_curr = row['decision_time']
        A = row['src_theme_id']
        key = (t_curr, A)
        if key in temp_map.index:
            A_prev = temp_map.loc[key, 'src_theme_id']
            t_prev = pd.to_datetime(temp_map.loc[key, 'src_time'], utc=True)
            return ret_lookup.get((t_prev, A_prev), np.nan)
        return np.nan
        
    df_rel['src_past_ret_15m'] = df_rel.apply(get_past_return, axis=1)
    df_rel = df_rel.dropna(subset=['src_past_ret_15m'])
    df_rel['spillover_signal'] = df_rel['relation_strength'] * df_rel['src_past_ret_15m']
    
    signal_df = df_rel.groupby(['decision_time', 'dst_theme_id'])['spillover_signal'].sum().reset_index()
    signal_df = signal_df.rename(columns={'dst_theme_id': 'theme_id'})
    
    day_ret = df_ret[df_ret['decision_time'].dt.strftime('%Y-%m-%d') == d]
    merged = pd.merge(signal_df, day_ret, on=['decision_time', 'theme_id'], how='inner')
    return merged

if __name__ == '__main__':
    all_results = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_date, d) for d in all_dates]
        for fut in as_completed(futures):
            r = fut.result()
            if r is not None:
                all_results.append(r)
                
    if all_results:
        final = pd.concat(all_results, ignore_index=True)
        horizons = ["5m", "15m", "30m", "60m"]
        ic_results = []
        for h in horizons:
            col = f"ret_core_{h}"
            if col in final.columns:
                valid = final.dropna(subset=['spillover_signal', col])
                if not valid.empty:
                    ic, p = stats.spearmanr(valid['spillover_signal'], valid[col])
                    q_top = valid['spillover_signal'].quantile(0.8)
                    q_bot = valid['spillover_signal'].quantile(0.2)
                    top_ret = valid[valid['spillover_signal'] >= q_top][col].mean()
                    bot_ret = valid[valid['spillover_signal'] <= q_bot][col].mean()
                    spread = top_ret - bot_ret
                    ic_results.append({
                        "horizon": h, "sample_count": len(valid), "rank_ic": ic,
                        "p_value": p, "top_quintile_ret": top_ret,
                        "bottom_quintile_ret": bot_ret, "long_short_spread": spread
                    })
        res_df = pd.DataFrame(ic_results)
        res_df.to_csv(OUT_DIR / "relation_spillover_ic.csv", index=False)
