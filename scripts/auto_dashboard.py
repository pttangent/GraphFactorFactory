import time
import subprocess
import os
import math
from pathlib import Path
from datetime import datetime

repo_dir = Path(r"d:\DEV\US-Stock\GraphFactorFactory")
dashboard_path = repo_dir / "dashboard.md"
canonical_dir = repo_dir / "data" / "graph_store" / "canonical"
themes_dir = repo_dir / "data" / "graph_store" / "themes"

TOTAL_DAYS = {
    '2026-01': 20,
    '2026-02': 19,
    '2026-03': 22,
    '2026-04': 21,
    '2026-05': 20,
    '2026-06': 22
}

def build_progress_bar(completed, total, elapsed_str, length=10):
    completed = min(completed, total)
    ratio = completed / total if total > 0 else 0
    filled = math.floor(ratio * length)
    bar = "█" * filled + "░" * (length - filled)
    percent = int(ratio * 100)
    return f"[{bar}] {percent}% ({completed}/{total}: {elapsed_str})"

def get_progress_stats(dir_path, month, target_file):
    if not dir_path.exists():
        return 0, "N/A", "0m"
    
    dirs = [d for d in dir_path.iterdir() if d.is_dir() and d.name.startswith(f"date={month}")]
    
    mtimes = []
    for d in dirs:
        file_path = d / target_file
        if file_path.exists():
            mtimes.append(file_path.stat().st_mtime)
            
    count = len(mtimes)
    if count == 0:
        return 0, "N/A", "0m"
        
    mtimes.sort()
    
    deltas = []
    for i in range(1, len(mtimes)):
        delta = mtimes[i] - mtimes[i-1]
        # Ignore pauses longer than 60 minutes
        if delta > 0 and delta < 3600:
            deltas.append(delta)
            
    if deltas:
        avg_speed = sum(deltas) / len(deltas)
        speed_str = f"{avg_speed/60:.1f} min/d"
        total_active_seconds = sum(deltas) + avg_speed
        elapsed_str = f"{total_active_seconds/60:.1f}m"
    else:
        speed_str = "N/A"
        elapsed_str = "N/A"
        
    return count, speed_str, elapsed_str

def update_and_push():
    lines = [
        "# GraphFactorFactory Two-Phase Progress Dashboard",
        f"**Generated UTC:** `{datetime.utcnow().isoformat()}`",
        "",
        "| Month | Phase 0 Progress (Graph) | P0 Speed | Phase 1 Progress (Theme) | P1 Speed |",
        "| --- | --- | --- | --- | --- |"
    ]
    
    for month, total in TOTAL_DAYS.items():
        p0_count, p0_speed, p0_elapsed = get_progress_stats(canonical_dir, month, "edges.parquet")
        p1_count, p1_speed, p1_elapsed = get_progress_stats(themes_dir, month, "themes.parquet")
        
        p0_str = build_progress_bar(p0_count, total, p0_elapsed)
        p1_str = build_progress_bar(p1_count, total, p1_elapsed)
        
        lines.append(f"| {month} | {p0_str} | {p0_speed} | {p1_str} | {p1_speed} |")
        
    dashboard_path.write_text("\n".join(lines), encoding="utf-8")
    
    try:
        subprocess.run(["git", "branch", "dashboard_upadate"], cwd=str(repo_dir), capture_output=True)
        subprocess.run(["git", "checkout", "dashboard_upadate"], cwd=str(repo_dir), capture_output=True)
        subprocess.run(["git", "add", "dashboard.md"], cwd=str(repo_dir), capture_output=True)
        subprocess.run(["git", "commit", "-m", f"Auto-update dashboard {datetime.utcnow().isoformat()}"], cwd=str(repo_dir), capture_output=True)
        subprocess.run(["git", "push", "-u", "origin", "dashboard_upadate"], cwd=str(repo_dir), capture_output=True)
        subprocess.run(["git", "checkout", "-"], cwd=str(repo_dir), capture_output=True)
    except Exception as e:
        print(f"Error pushing: {e}")

if __name__ == "__main__":
    update_and_push()
    while True:
        time.sleep(300)
        update_and_push()
