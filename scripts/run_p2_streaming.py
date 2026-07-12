import os
import sys
import time
import shutil
import subprocess
import threading
from pathlib import Path
from queue import Queue

NAS_P0_ROOT = Path(r"P:\US-Stock\GFF_Full_Workspace\graph_store_6m\canonical")
NAS_P1_ROOT = Path(r"P:\US-Stock\GFF_Full_Workspace\p1_b50_b35_sharded")

LOCAL_WORKSPACE = Path(r"D:\GFF_Streaming_Workspace")
LOCAL_P0 = LOCAL_WORKSPACE / "p0"
LOCAL_P1 = LOCAL_WORKSPACE / "p1"
LOCAL_P2_OUT = LOCAL_WORKSPACE / "p2_out"

NAS_P2_OUT = Path(r"P:\US-Stock\GFF_Full_Workspace\p2_alpha_full_run")

MONTHS = ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06"]

def robocopy_dir(src: Path, dst: Path):
    if not src.exists(): return
    dst.parent.mkdir(parents=True, exist_ok=True)
    # robocopy exit codes: 0-7 are success/normal, >=8 is error
    cmd = ["robocopy", str(src), str(dst), "/E", "/MT:16", "/R:3", "/W:1", "/NFL", "/NDL", "/NJH", "/NJS"]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode >= 8:
        print(f"[ERROR] robocopy failed for {src}: {r.stderr.decode('utf-8', errors='ignore')}")
        raise SystemExit(1)

def pull_month(month: str):
    print(f"[{time.strftime('%H:%M:%S')}] [PRODUCER] PRE-FETCHING data for {month} to local D: drive...")
    LOCAL_P0.mkdir(parents=True, exist_ok=True)
    LOCAL_P1.mkdir(parents=True, exist_ok=True)
    
    # Copy P0 and Labels
    if NAS_P0_ROOT.exists():
        for p0_dir in NAS_P0_ROOT.glob(f"date={month}-*"):
            robocopy_dir(p0_dir, LOCAL_P0 / p0_dir.name)
            
    # Copy P1
    if NAS_P1_ROOT.exists():
        for p1_dir in NAS_P1_ROOT.glob(f"date={month}-*"):
            robocopy_dir(p1_dir, LOCAL_P1 / p1_dir.name)
            
    print(f"[{time.strftime('%H:%M:%S')}] [PRODUCER] Finished pre-fetching {month}")

def run_p2_month(month: str):
    print(f"[{time.strftime('%H:%M:%S')}] [CONSUMER] Starting 24-core P2 pipeline for {month}...")
    
    dates = [d.name.split('=')[1] for d in LOCAL_P0.glob(f"date={month}-*")]
    if not dates:
        print(f"[{time.strftime('%H:%M:%S')}] [CONSUMER] No dates found for {month}, skipping.")
        return
        
    dates_str = ",".join(sorted(dates))
    
    cmd = [
        sys.executable, "scripts/run_p2_24core_scheduler.py",
        "--p0-root", str(LOCAL_P0),
        "--labels-root", str(LOCAL_P0),
        "--p1-root", str(LOCAL_P1),
        "--p2-root", str(LOCAL_P2_OUT),  # Output to LOCAL SSD to avoid 24-core network I/O contention!
        "--dates", dates_str,
        "--layers", "3,6,8,9,11",
        "--scales", "15m,30m",
        "--profile", "max",
        "--cores", "24",
        "--target-cpu", "1.0",
        "--inner-workers", "1",
        "--skip-existing"
    ]
    
    env = os.environ.copy()
    r = subprocess.run(cmd, env=env, stdout=sys.stdout, stderr=sys.stderr)
    if r.returncode != 0:
        print(f"[ERROR] Pipeline failed for {month}!")
        raise SystemExit(1)
        
    print(f"[{time.strftime('%H:%M:%S')}] [CONSUMER] P2 Pipeline finished for {month}.")
        
    print(f"[{time.strftime('%H:%M:%S')}] [CONSUMER] Cleaning up local inputs for {month}...")
    for d in LOCAL_P0.glob(f"date={month}-*"):
        shutil.rmtree(d, ignore_errors=True)
    for d in LOCAL_P1.glob(f"date={month}-*"):
        shutil.rmtree(d, ignore_errors=True)
    
    print(f"[{time.strftime('%H:%M:%S')}] [CONSUMER] {month} fully processed and local workspace cleaned!")

def main():
    q = Queue(maxsize=1)
    
    def producer():
        for month in MONTHS:
            pull_month(month)
            # Will block here if consumer is busy with previous month, exactly as requested
            q.put(month)
        q.put(None)
        
    def consumer():
        while True:
            month = q.get()
            if month is None:
                break
            run_p2_month(month)
            q.task_done()
            
    prod_thread = threading.Thread(target=producer)
    cons_thread = threading.Thread(target=consumer)
    
    print(f"[{time.strftime('%H:%M:%S')}] === STARTING 6-MONTH P2 STREAMING PIPELINE ===")
    prod_thread.start()
    cons_thread.start()
    
    prod_thread.join()
    cons_thread.join()
    print(f"[{time.strftime('%H:%M:%S')}] === ALL MONTHS COMPLETED SUCCESSFULLY ===")
    print(f"[{time.strftime('%H:%M:%S')}] === STARTING GLOBAL EVALUATION ON FULL DATASET ===")
    
    # Run global evaluate-daily on the accumulated local output
    cmd = [sys.executable, "scripts/p2_alpha_daily_features.py", "evaluate-daily", 
           "--features-root", str(LOCAL_P2_OUT / "daily_relation_features"),
           "--out-dir", str(LOCAL_P2_OUT / "daily_relation_eval_global")]
    subprocess.run(cmd, env=os.environ.copy())
    
    print(f"[{time.strftime('%H:%M:%S')}] === PIPELINE FULLY TERMINATED ===")

if __name__ == '__main__':
    main()
