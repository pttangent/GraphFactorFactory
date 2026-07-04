import os
import glob
import pyarrow.parquet as pq
import concurrent.futures

def check_file(path):
    try:
        # Just try to read the metadata and one column chunk to verify integrity
        pq.read_table(path, columns=["symbol_id"])
        return path, True
    except Exception as e:
        return path, False

def main():
    print("Scanning for corrupted parquet files in 2026-06...")
    files = glob.glob(r"C:\nodefactor_work\month_packs\month=2026-06\**\*.parquet", recursive=True)
    corrupted = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        for path, ok in executor.map(check_file, files):
            if not ok:
                corrupted.append(path)
                
    if corrupted:
        print(f"Found {len(corrupted)} corrupted files. Deleting them...")
        for p in corrupted:
            os.remove(p)
        print("Deleted corrupted files. Running robocopy to restore them...")
        import subprocess
        subprocess.run(["robocopy", r"P:\US-Stock\NodeFactorFactory\warehouse\month_packs\month=2026-06", r"C:\nodefactor_work\month_packs\month=2026-06", "/E", "/MT:32", "/NFL", "/NDL", "/NJH", "/NJS", "/NP"])
        print("Repair complete.")
    else:
        print("No corrupted files found.")

if __name__ == "__main__":
    main()
