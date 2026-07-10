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
P0_LABELS_ROOT = Path(r"D:\DEV\US-Stock\GraphFactorFactory\data\graph_store_6m\canonical")
OUT_DIR = Path("artifacts/p2_alpha_lab")
OUT_DIR.mkdir(parents=True, exist_ok=True)

inventory = []

for d in dates:
    row = {"date": d}
    
    # Check P0 Labels
    labels_path = P0_LABELS_ROOT / f"date={d}" / "labels.parquet"
    if labels_path.exists():
        row["has_labels"] = True
        try:
            df_lbl = pd.read_parquet(labels_path)
            horizons = [c.replace("label_", "") for c in df_lbl.columns if c.startswith("label_") and "raw" not in c and "exit" not in c and "adj" not in c and "split" not in c and "time" not in c]
            row["available_forward_horizons"] = ",".join(horizons)
        except Exception as e:
            row["available_forward_horizons"] = "ERROR"
    else:
        row["has_labels"] = False
        row["available_forward_horizons"] = ""
        
    # Check P1 completeness
    p1_date_dir = P1_ROOT / f"date={d}"
    if p1_date_dir.exists():
        row["has_p1"] = True
        
        # Count rows in shards
        node_rows = 0
        member_rows = 0
        relation_rows = 0
        temporal_rows = 0
        
        shard_dirs = glob.glob(str(p1_date_dir / "layer_id=*" / "scale=*"))
        row["shards_count"] = len(shard_dirs)
        
        for sd in shard_dirs:
            try:
                if os.path.exists(os.path.join(sd, "theme_nodes.parquet")):
                    node_rows += len(pd.read_parquet(os.path.join(sd, "theme_nodes.parquet"), columns=[]))
                if os.path.exists(os.path.join(sd, "theme_memberships.parquet")):
                    member_rows += len(pd.read_parquet(os.path.join(sd, "theme_memberships.parquet"), columns=[]))
                if os.path.exists(os.path.join(sd, "theme_relation_edges.parquet")):
                    relation_rows += len(pd.read_parquet(os.path.join(sd, "theme_relation_edges.parquet"), columns=[]))
                if os.path.exists(os.path.join(sd, "temporal_theme_edges.parquet")):
                    temporal_rows += len(pd.read_parquet(os.path.join(sd, "temporal_theme_edges.parquet"), columns=[]))
            except Exception:
                pass
                
        row["theme_nodes_rows"] = node_rows
        row["membership_rows"] = member_rows
        row["relation_rows"] = relation_rows
        row["temporal_rows"] = temporal_rows
        row["p1_complete"] = (node_rows > 0 and member_rows > 0)
    else:
        row["has_p1"] = False
        row["shards_count"] = 0
        row["p1_complete"] = False
        
    inventory.append(row)
    print(f"Processed {d} - P1 Complete: {row.get('p1_complete')}")

df_inv = pd.DataFrame(inventory)
df_inv.to_csv(OUT_DIR / "inventory.csv", index=False)
print(f"Inventory saved to {OUT_DIR / 'inventory.csv'}")
print(df_inv)
