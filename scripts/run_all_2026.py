import json
import subprocess
import sys
import time
from pathlib import Path

months = ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06"]

print("=== BEGINNING 2026 FULL PIPELINE ===", flush=True)

# 1. Run Phase 0 (which calls Phase 1 internally for each month)
# Note: Since the Phase 1 script runs month-by-month if we call it, we can just call it directly!
print("=== [STEP 1] PHASE 1: THEME DISCOVERY ===", flush=True)
for m in months:
    y, mo = m.split("-")
    if mo in ["01", "03", "05", "07", "08", "10", "12"]: last_day = "31"
    elif mo in ["04", "06", "09", "11"]: last_day = "30"
    elif mo == "02": last_day = "28"
    
    date_start = f"{m}-01"
    date_end = f"{m}-{last_day}"
    
    print(f"Running Phase 1 for {m}...", flush=True)
    t0 = time.time()
    subprocess.run([
        sys.executable, "scripts/run_theme_discovery_phase1.py",
        "--date-start", date_start,
        "--date-end", date_end,
        "--graph-root", "data/graph_store",
        "--out-root", "outputs/theme_discovery_phase1",
        "--max-snapshot-workers", "26"
    ], check=True)
    print(f"Phase 1 for {m} completed in {time.time() - t0:.1f}s", flush=True)

# 2. Run Phase 2 (Intraday Lifecycle Rebuild)
print("\n=== [STEP 2] PHASE 2: INTRADAY LIFECYCLE REBUILD ===", flush=True)
t0 = time.time()
subprocess.run([
    sys.executable, "scripts/run_phase2_from_earliest.py",
    "--phase1-root", "outputs/theme_discovery_phase1",
    "--phase2-root", "outputs/theme_temporal_phase2"
], check=True)
print(f"Phase 2 Intraday Rebuild completed in {time.time() - t0:.1f}s", flush=True)

# 3. Run Phase 2 (Cross-night Carryover A/B)
print("\n=== [STEP 3] PHASE 2: CROSS-NIGHT CARRYOVER A/B ===", flush=True)
t0 = time.time()
subprocess.run([
    sys.executable, "scripts/run_monthly_carryover_ab.py",
    "--config", "configs/monthly_carryover_ab_2026.json",
    "--phase1-root", "outputs/theme_discovery_phase1",
    "--max-workers", "4",
    "--resume"
], check=True)
print(f"Phase 2 Cross-night completed in {time.time() - t0:.1f}s", flush=True)

print("\n=== ALL 2026 PIPELINES COMPLETED SUCCESSFULLY ===", flush=True)
