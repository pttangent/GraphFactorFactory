import os
import tarfile
from pathlib import Path
import subprocess
import shutil

p0_canon_root = Path(r'D:\DEV\US-Stock\GraphFactorFactory\data\graph_store_6m\canonical')
p0_edge_root = Path(r'D:\DEV\US-Stock\GraphFactorFactory\data\p0_edge_shards')
p1_root = Path(r'C:\GFF_Cache\p1_b50_b35_sharded')

# Output to C drive to avoid filling up D drive!
out_root = Path(r'C:\GFF_Cache\smokerun_january')
out_root.mkdir(parents=True, exist_ok=True)

dates = ['2026-01-02', '2026-01-05', '2026-01-06', '2026-01-07', '2026-01-08',
         '2026-01-09', '2026-01-12', '2026-01-13', '2026-01-14', '2026-01-15',
         '2026-01-16', '2026-01-20', '2026-01-21', '2026-01-22', '2026-01-23',
         '2026-01-26', '2026-01-27', '2026-01-28', '2026-01-29', '2026-01-30']

def pack_day(date_str):
    print(f"Packing data for {date_str}...")
    tar_path = out_root / f"{date_str}_smokerun.tar.gz"
    
    with tarfile.open(tar_path, "w:gz") as tar:
        # P0 Canonical
        d_canon = p0_canon_root / f"date={date_str}"
        if d_canon.exists():
            tar.add(d_canon, arcname=f"p0_canonical/date={date_str}")
        
        # P0 Edges
        d_edges = p0_edge_root / f"date={date_str}"
        if d_edges.exists():
            tar.add(d_edges, arcname=f"p0_edges/date={date_str}")
            
        # P1 output
        d_p1 = p1_root / f"date={date_str}"
        if d_p1.exists():
            tar.add(d_p1, arcname=f"p1_output/date={date_str}")
            
    print(f"Splitting {tar_path.name} into 48MB chunks...")
    subprocess.run(["python", "scripts/split_tar.py", str(tar_path)])
    
    tar_path.unlink() # Delete the large single tar file after splitting

for d in dates:
    pack_day(d)

print("All January smokerun days packed successfully!")
