#!/usr/bin/env python3
import argparse
import json
import logging
import concurrent.futures
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOG = logging.getLogger(__name__)

def check_partition(part_dir: Path) -> dict:
    errors = []
    
    # 2. Check manifest
    manifest_path = part_dir / "manifest.json"
    if not manifest_path.exists():
        errors.append("Missing manifest.json")
    
    # 3. Check theme_memberships
    mem_path = part_dir / "theme_memberships.parquet"
    if not mem_path.exists() or mem_path.stat().st_size < 100:
        errors.append("Missing or empty theme_memberships.parquet")
        
    # 4. Check theme_relation_edges
    rel_path = part_dir / "theme_relation_edges.parquet"
    if not rel_path.exists() or rel_path.stat().st_size < 100:
        errors.append("Missing or empty theme_relation_edges.parquet")
        
    # 5, 6, 7. Check leaf size and summary
    sum_path = part_dir / "p1_b50_b35_summary.parquet"
    if not sum_path.exists():
        errors.append("Missing p1_b50_b35_summary.parquet")
    else:
        try:
            df = pd.read_parquet(sum_path)
            if "level" in df.columns and "leaf_count" in df.columns:
                b50 = df[df["level"] == "B50"]
                b35 = df[df["level"] == "B35"]
                
                if not b50.empty and b50["leaf_count"].max() > 50:
                    errors.append(f"B50 leaf size exceeded 50 (max {b50['leaf_count'].max()})")
                if not b35.empty and b35["leaf_count"].max() > 35:
                    errors.append(f"B35 leaf size exceeded 35 (max {b35['leaf_count'].max()})")
                
                if df["leaf_count"].sum() == 0:
                    errors.append("Summary has 0 leaf_count overall")
        except Exception as e:
            errors.append(f"Failed to read summary parquet: {str(e)}")

    return {"partition": str(part_dir.relative_to(part_dir.parents[2])), "errors": errors}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p1-root", required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    p1_root = Path(args.p1_root)
    main_manifest = p1_root / "run_p1_parallel_manifest.json"
    
    # 1. Check main manifest
    if not main_manifest.exists():
        LOG.error(f"Main manifest not found at {main_manifest}")
        raise SystemExit(1)
        
    with open(main_manifest, "r", encoding="utf-8") as f:
        meta = json.load(f)
        if meta.get("tasks_failed", 0) > 0 or len(meta.get("failed", [])) > 0:
            LOG.error("run_p1_parallel_manifest.json indicates FAILED tasks!")
            LOG.error(f"Failed count: {meta.get('tasks_failed')}")
            raise SystemExit(1)
            
    LOG.info("Main manifest QC passed (0 failed tasks).")
    
    # Find all partitions
    LOG.info("Scanning for partition directories...")
    # Assuming structure: out_root/date=*/layer_id=*/scale=*/
    partitions = [d for d in p1_root.rglob("scale=*") if d.is_dir()]
    LOG.info(f"Found {len(partitions)} partitions to QC.")
    
    failed_partitions = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(check_partition, p): p for p in partitions}
        for i, fut in enumerate(concurrent.futures.as_completed(futures)):
            res = fut.result()
            if res["errors"]:
                failed_partitions.append(res)
            
            if (i + 1) % 500 == 0:
                LOG.info(f"QC Progress: {i + 1} / {len(partitions)} checked.")

    if failed_partitions:
        LOG.error(f"QC FAILED. {len(failed_partitions)} partitions have errors.")
        for f in failed_partitions[:10]:
            LOG.error(f"  [{f['partition']}] -> {', '.join(f['errors'])}")
        if len(failed_partitions) > 10:
            LOG.error(f"  ... and {len(failed_partitions) - 10} more.")
        raise SystemExit(1)
        
    LOG.info(f"All {len(partitions)} partitions passed QC successfully!")

if __name__ == "__main__":
    main()
