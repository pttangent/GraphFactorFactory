import os
import glob
import pandas as pd
from pathlib import Path

dates = [
    "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09",
    "2026-01-12", "2026-01-13", "2026-01-14", "2026-01-15", "2026-01-16",
    "2026-01-20", "2026-01-21", "2026-01-22", "2026-01-23", "2026-01-26",
    "2026-01-27", "2026-01-28", "2026-01-29", "2026-01-30", "2026-02-02"
]

P1_ROOT = Path("artifacts/p1_b50_b35_sharded")
OUT_DIR = Path("artifacts/p2_alpha_lab/flattened")

targets = [
    "theme_nodes.parquet",
    "theme_memberships.parquet",
    "theme_relation_edges.parquet",
    "temporal_theme_edges.parquet"
]

for d in dates:
    print(f"Flattening date {d}...")
    p1_date_dir = P1_ROOT / f"date={d}"
    if not p1_date_dir.exists():
        continue
        
    date_out_dir = OUT_DIR / f"date={d}"
    date_out_dir.mkdir(parents=True, exist_ok=True)
    if all((date_out_dir / t).exists() for t in targets):
        print(f'  Skipping {d}, already flattened.')
        continue
    
    shard_dirs = glob.glob(str(p1_date_dir / "layer_id=*" / "scale=*"))
    
    for target in targets:
        dfs = []
        for sd in shard_dirs:
            fp = os.path.join(sd, target)
            if os.path.exists(fp):
                try:
                    df = pd.read_parquet(fp)
                    if not df.empty:
                        # Some parquet files (like temporal) already have layer_id, scale columns
                        dfs.append(df)
                except Exception as e:
                    print(f"Error reading {fp}: {e}")
                    
        if dfs:
            merged = pd.concat(dfs, ignore_index=True)
            merged.to_parquet(date_out_dir / target, index=False)
            print(f"  Saved {target}: {len(merged)} rows")
        else:
            print(f"  Warning: No data for {target} in date {d}")

print("Flattening completed.")
