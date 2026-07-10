import pandas as pd
import numpy as np
from pathlib import Path
import scipy.stats as stats
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

FLATTENED_ROOT = Path("artifacts/p2_alpha_lab/flattened")
THEME_RETS_ROOT = Path("artifacts/p2_alpha_lab/theme_returns_by_date")
OUT_DIR = Path("artifacts/p2_alpha_lab")

all_dates = [d.name.split('=')[1] for d in FLATTENED_ROOT.iterdir() if d.is_dir() and 'date=' in d.name]

def process_date(d, load_lock):
    try:
        edges_path = FLATTENED_ROOT / f"date={d}" / "theme_relation_edges.parquet"
        rets_path = THEME_RETS_ROOT / f"date={d}.parquet"
        
        if not edges_path.exists() or not rets_path.exists():
            return None
            
        with load_lock:
            df_edges = pd.read_parquet(edges_path)
            df_rets = pd.read_parquet(rets_path)
            
            df_edges['decision_time'] = pd.to_datetime(df_edges['decision_time'], utc=True)
            df_rets['decision_time'] = pd.to_datetime(df_rets['decision_time'], utc=True)
        
        df_rets_past = df_rets[['decision_time', 'theme_id', 'layer_id', 'scale', 'ret_eq_15m']].copy()
        df_rets_past['decision_time'] = df_rets_past['decision_time'] + pd.Timedelta(minutes=15)
        df_rets_past = df_rets_past.rename(columns={'ret_eq_15m': 'theme_A_past_ret_15m'})
        
        chunk_results = []
        
        for dt, edges_group in df_edges.groupby('decision_time'):
            past_rets = df_rets_past[df_rets_past['decision_time'] == dt]
            fut_rets = df_rets[df_rets['decision_time'] == dt]
            
            if past_rets.empty or fut_rets.empty:
                continue
                
            m1 = pd.merge(edges_group, past_rets, left_on=['decision_time', 'layer_id', 'scale', 'theme_A'], right_on=['decision_time', 'layer_id', 'scale', 'theme_id'], how='inner')
            if m1.empty: continue
                
            m1['spillover_signal'] = m1['fuzzy_relation_score'] * m1['theme_A_past_ret_15m']
            
            agg = m1.groupby(['decision_time', 'layer_id', 'scale', 'theme_B'])['spillover_signal'].mean().reset_index()
            
            m2 = pd.merge(agg, fut_rets, left_on=['decision_time', 'layer_id', 'scale', 'theme_B'], right_on=['decision_time', 'layer_id', 'scale', 'theme_id'], how='inner')
            if not m2.empty:
                chunk_results.append(m2)
                
        if chunk_results:
            final_res = pd.concat(chunk_results, ignore_index=True)
            print(f"[Spillover] [{d}] Done! ({len(final_res)} rows)", flush=True)
            return final_res
        return None
    except Exception as e:
        print(f"[{d}] ERROR: {str(e)}", flush=True)
        return None

if __name__ == '__main__':
    all_results = []
    print("Running spillover alpha concurrently with Lock...", flush=True)
    m = multiprocessing.Manager()
    load_lock = m.Lock()
    
    with ProcessPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(process_date, d, load_lock) for d in all_dates]
        for future in as_completed(futures):
            r = future.result()
            if r is not None:
                all_results.append(r)
                
    if all_results:
        final = pd.concat(all_results, ignore_index=True)
        horizons = ["5m", "15m", "30m", "60m"]
        ic_results = []
        for h in horizons:
            col = f"ret_eq_{h}"
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
        print("relation_spillover_ic.csv saved!", flush=True)
