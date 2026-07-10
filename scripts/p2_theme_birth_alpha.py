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

all_dates = [d.name.split('=')[1] for d in FLATTENED_ROOT.iterdir() if d.is_dir() and 'date=' in d.name]

def process_date(d):
    temp_path = FLATTENED_ROOT / f"date={d}" / "temporal_theme_edges.parquet"
    if not temp_path.exists():
        return None
        
    df_temp = pd.read_parquet(temp_path)
    df_temp['dst_time'] = pd.to_datetime(df_temp['dst_time'], utc=True)
    has_predecessor = set(zip(df_temp['dst_time'], df_temp['dst_theme_id']))
    
    day_ret = df_ret[df_ret['decision_time'].dt.strftime('%Y-%m-%d') == d].copy()
    day_ret['is_new_theme'] = day_ret.apply(lambda r: 0 if (r['decision_time'], r['theme_id']) in has_predecessor else 1, axis=1)
    return day_ret

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
                valid = final.dropna(subset=['is_new_theme', col])
                if not valid.empty:
                    ic, p = stats.spearmanr(valid['is_new_theme'], valid[col])
                    new_ret = valid[valid['is_new_theme'] == 1][col].mean()
                    old_ret = valid[valid['is_new_theme'] == 0][col].mean()
                    spread = new_ret - old_ret
                    ic_results.append({
                        "horizon": h, "sample_count": len(valid), "rank_ic": ic,
                        "p_value": p, "new_theme_ret": new_ret,
                        "old_theme_ret": old_ret, "long_short_spread": spread
                    })
        res_df = pd.DataFrame(ic_results)
        res_df.to_csv(OUT_DIR / "theme_birth_ic.csv", index=False)
