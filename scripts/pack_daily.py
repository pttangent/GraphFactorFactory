import argparse
import zipfile
import shutil
from pathlib import Path

def pack_day(date_str, p1_root, p2_root, out_dir):
    p1_day = p1_root / f"date={date_str}"
    p2_day = p2_root / f"date={date_str}"
    
    if not p1_day.exists() or not p2_day.exists():
        return False
        
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"{date_str}.zip"
    
    # Don't repack if already exists
    if zip_path.exists():
        return False
        
    print(f"Packing {date_str} to {zip_path}...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Pack Phase 1
        for file_path in p1_day.rglob("*"):
            if file_path.is_file():
                arcname = f"phase1/{file_path.relative_to(p1_day)}"
                zf.write(file_path, arcname)
                
        # Pack Phase 2
        for file_path in p2_day.rglob("*"):
            if file_path.is_file():
                arcname = f"phase2/{file_path.relative_to(p2_day)}"
                zf.write(file_path, arcname)
                
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p1-root", default="outputs/theme_discovery_phase1")
    parser.add_argument("--p2-root", default="outputs/theme_temporal_phase2")
    parser.add_argument("--out-dir", default="outputs/packed_daily")
    args = parser.parse_args()
    
    p1_root = Path(args.p1_root)
    p2_root = Path(args.p2_root)
    out_dir = Path(args.out_dir)
    
    if not p1_root.exists() or not p2_root.exists():
        print("Roots not found.")
        return
        
    dates = sorted([d.name.split("=")[1] for d in p1_root.glob("date=*") if (d / "_SUCCESS").exists()])
    for d in dates:
        # Check if p2 is also successful (by checking if _SUCCESS or similar marker exists in state?)
        # Or just check if the directory exists since P2 is run after P1
        pack_day(d, p1_root, p2_root, out_dir)

if __name__ == "__main__":
    main()
