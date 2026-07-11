import os
import tarfile
from pathlib import Path
import subprocess

nff_root = Path(r'P:\US-Stock\NodeFactorFactory\canonical')
p0_canon_root = Path(r'D:\DEV\US-Stock\GraphFactorFactory\data\graph_store_6m\canonical')
p0_edge_root = Path(r'D:\DEV\US-Stock\GraphFactorFactory\data\p0_edge_shards')
p1_root = Path(r'C:\GFF_Cache\p1_b50_b35_sharded')
p2_tr_root = Path(r'C:\GFF_Cache\p2_alpha_lab\theme_returns')
p2_rs_root = Path(r'C:\GFF_Cache\p2_alpha_lab\relation_spillover')
p2_df_root = Path(r'C:\GFF_Cache\p2_alpha_lab\daily_relation_features')

out_root = Path(r'C:\GFF_Cache\smokerun_january_full')
out_root.mkdir(parents=True, exist_ok=True)

dates = ['2026-01-02', '2026-01-05', '2026-01-06', '2026-01-07', '2026-01-08',
         '2026-01-09', '2026-01-12', '2026-01-13', '2026-01-14', '2026-01-15',
         '2026-01-16', '2026-01-20', '2026-01-21', '2026-01-22', '2026-01-23',
         '2026-01-26', '2026-01-27', '2026-01-28', '2026-01-29', '2026-01-30']

def pack_day(date_str):
    print(f"Packing full data for {date_str}...")
    tar_path = out_root / f"{date_str}_full.tar.gz"
    
    with tarfile.open(tar_path, "w:gz") as tar:
        def try_add(path, arc_name):
            if path.exists():
                tar.add(path, arcname=arc_name)

        try_add(nff_root / f"date={date_str}", f"nff_canonical/date={date_str}")
        try_add(p0_canon_root / f"date={date_str}", f"p0_canonical/date={date_str}")
        try_add(p0_edge_root / f"date={date_str}", f"p0_edges/date={date_str}")
        try_add(p1_root / f"date={date_str}", f"p1_output/date={date_str}")
        try_add(p2_tr_root / f"date={date_str}", f"p2_theme_returns/date={date_str}")
        try_add(p2_rs_root / f"date={date_str}", f"p2_relation_spillover/date={date_str}")
        try_add(p2_df_root / f"date={date_str}", f"p2_daily_relation_features/date={date_str}")
            
    print(f"Splitting {tar_path.name} into 48MB chunks...")
    subprocess.run(["python", "scripts/split_tar.py", str(tar_path)])
    tar_path.unlink()

for d in dates:
    pack_day(d)

print("All January full smokerun days packed successfully!")
