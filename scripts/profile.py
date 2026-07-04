import psutil
import time
import subprocess
import csv
import sys
import threading
import argparse
from pathlib import Path

def run_and_profile():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-month", type=str, required=True)
    parser.add_argument("--max-workers", type=int, default=26, help="Max workers for ProcessPoolExecutor")
    args = parser.parse_args()

    print(f"Starting pipeline profiling with {args.max_workers} workers...")
    output_csv = Path("data/graph_store/profiling.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Relative_Seconds', 'CPU_Percent', 'RAM_MB', 'Disk_Read_MB', 'Disk_Write_MB'])
        
        # Start pipeline
        import os
        env = os.environ.copy()
        env["PYTHONPATH"] = "src"
        cmd = f'{sys.executable} scripts/run_phase0.py --target-month {args.target_month} --max-workers {args.max_workers}'
        process = subprocess.Popen(cmd, shell=True, env=env, cwd=str(Path("d:/DEV/US-Stock/GraphFactorFactory")))
        
        start_time = time.time()
        disk_start = psutil.disk_io_counters()
        
        try:
            while process.poll() is None:
                # Sleep a bit to sample exactly every 1 second
                time.sleep(1.0)
                
                cpu = psutil.cpu_percent(interval=None)
                ram = psutil.virtual_memory().used / (1024 * 1024)
                disk_now = psutil.disk_io_counters()
                
                read_mb = (disk_now.read_bytes - disk_start.read_bytes) / (1024 * 1024) if disk_now and disk_start else 0
                write_mb = (disk_now.write_bytes - disk_start.write_bytes) / (1024 * 1024) if disk_now and disk_start else 0
                
                writer.writerow([round(time.time() - start_time, 1), cpu, ram, round(read_mb, 2), round(write_mb, 2)])
                f.flush()
                disk_start = disk_now
        except KeyboardInterrupt:
            process.terminate()
            
    print(f"Profiling finished. Saved to {output_csv}")

if __name__ == "__main__":
    run_and_profile()
