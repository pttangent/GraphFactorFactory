import pandas as pd
import numpy as np
from pathlib import Path
import scipy.stats as stats

FLATTENED_ROOT = Path("artifacts/p2_alpha_lab/flattened")
THEME_RETS_ROOT = Path("artifacts/p2_alpha_lab/theme_returns_by_date")
OUT_DIR = Path("artifacts/p2_alpha_lab")

all_dates = [d.name.split('=')[1] for d in FLATTENED_ROOT.iterdir() if d.is_dir() and 'date=' in d.name]
all_dates.sort()

seen_themes = set()

def process_date(d):
    global seen_themes
    print(f"[Birth] [{d}] Starting...", flush=True)
    nodes_path = FLATTENED_ROOT / f"date={d}" / "theme_nodes.parquet"
    rets_path = THEME_RETS_ROOT / f"date={d}.parquet"
    
    if not nodes_path.exists() or not rets_path.exists():
        return None
        
    df_nodes = pd.read_parquet(nodes_path)
    df_rets = pd.read_parquet(rets_path)
    
    df_nodes['decision_time'] = pd.to_datetime(df_nodes['decision_time'], utc=True)
    df_rets['decision_time'] = pd.to_datetime(df_rets['decision_time'], utc=True)
    
    df_nodes['theme_id_full'] = df_nodes['layer_id'].astype(str) + "_" + df_nodes['scale'].astype(str) + "_" + df_nodes['theme_id'].astype(str)
    
    chunk_results = []
    
    for dt, nodes_group in df_nodes.groupby('decision_time'):
        current_themes = set(nodes_group['theme_id_full'])
        new_themes = current_themes - seen_themes
        seen_themes.update(current_themes)
        
        if not new_themes:
            continue
            
        nodes_group = nodes_group[nodes_group['theme_id_full'].isin(new_themes)].copy()
        nodes_group['is_new'] = 1.0
        
        fut_rets = df_rets[df_rets['decision_time'] == dt]
        if fut_rets.empty: continue
            
        m = pd.merge(nodes_group, fut_rets, on=['decision_time', 'layer_id', 'scale', 'theme_id'], how='inner')
        if not m.empty:
            chunk_results.append(m[['decision_time', 'layer_id', 'scale', 'theme_id', 'is_new', 'ret_eq_5m', 'ret_eq_15m', 'ret_eq_30m', 'ret_eq_60m']])
            
    if chunk_results:
        final_res = pd.concat(chunk_results, ignore_index=True)
        print(f"[Birth] [{d}] Done! ({len(final_res)} rows)", flush=True)
        return final_res
    return None

if __name__ == '__main__':
    all_results = []
    print("Running theme birth alpha sequentially...", flush=True)
    for d in all_dates:
        r = process_date(d)
        if r is not None:
            all_results.append(r)
            
    if all_results:
        final = pd.concat(all_results, ignore_index=True)
        
        rets_dfs = []
        for d in all_dates:
            p = THEME_RETS_ROOT / f"date={d}.parquet"
            if p.exists():
                rets_dfs.append(pd.read_parquet(p))
        if rets_dfs:
            all_rets = pd.concat(rets_dfs, ignore_index=True)
            
            horizons = ["5m", "15m", "30m", "60m"]
            stats_results = []
            for h in horizons:
                col = f"ret_eq_{h}"
                if col in final.columns and col in all_rets.columns:
                    new_mean = final[col].mean()
                    all_mean = all_rets[col].mean()
                    spread = new_mean - all_mean
                    stats_results.append({
                        "horizon": h, "new_theme_count": len(final),
                        "new_theme_avg_ret": new_mean, "market_avg_ret": all_mean,
                        "excess_return": spread
                    })
            res_df = pd.DataFrame(stats_results)
            res_df.to_csv(OUT_DIR / "theme_birth_stats.csv", index=False)
            print("theme_birth_stats.csv saved!", flush=True)
