import argparse
import concurrent.futures
import json
import logging
import multiprocessing
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

import pandas as pd
import psutil

from graphfactorfactory.application.pipeline import GraphFactorPipeline
from graphfactorfactory.domain.config import BuildConfig
from graphfactorfactory.infrastructure.nodefactorfactory.monthpack_source import MonthPackNodeFactorSource, BoundMonthNodeFactorSource
from graphfactorfactory.infrastructure.store import CanonicalGraphStore
from graphfactorfactory.themes.pipeline import ThemeDiscoveryPipeline, ThemeDiscoveryConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase0Runner")


def monitor_worker(worker_name: str, pid: int, start_time: float, result_dict: dict):
    """Monitors CPU and RAM for a given PID until it exits or finishes."""
    try:
        proc = psutil.Process(pid)
        cpu_usages = []
        mem_usages = []
        while proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE:
            try:
                cpu_usages.append(proc.cpu_percent(interval=1.0))
                mem_usages.append(proc.memory_info().rss / (1024 * 1024))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
        elapsed = time.time() - start_time
        avg_cpu = sum(cpu_usages) / len(cpu_usages) if cpu_usages else 0.0
        avg_mem = sum(mem_usages) / len(mem_usages) if mem_usages else 0.0
        peak_mem = max(mem_usages) if mem_usages else 0.0
        result_dict[worker_name] = {
            "elapsed_seconds": elapsed,
            "avg_cpu_percent": avg_cpu,
            "avg_ram_mb": avg_mem,
            "peak_ram_mb": peak_mem
        }
        logger.info(f"[{worker_name}] Finished in {elapsed:.2f}s | CPU: {avg_cpu:.1f}% | Peak RAM: {peak_mem:.1f} MB")
    except Exception as e:
        logger.error(f"Error monitoring {worker_name}: {e}")


def process_graph_date(args):
    trade_date, month, temp_monthpack_dir, output_dir, config_path, lock = args
    start_time = time.time()
    pid = os.getpid()
    logger.info(f"[Graph Worker {pid}] Starting date {trade_date}")
    
    # We will self-monitor by launching a thread
    stats = {}
    monitor_thread = threading.Thread(target=monitor_worker, args=(f"Graph-{trade_date}", pid, start_time, stats), daemon=True)
    monitor_thread.start()

    try:
        pack_source = MonthPackNodeFactorSource(temp_monthpack_dir)
        source = BoundMonthNodeFactorSource(pack_source, month)
        config = BuildConfig.from_yaml(config_path) if config_path else BuildConfig()
        
        # We need a custom store that locks during manifest write
        class LockedCanonicalGraphStore(CanonicalGraphStore):
            def write_manifest(self, *args, **kwargs):
                with lock:
                    return super().write_manifest(*args, **kwargs)
            def finalize_catalog(self, *args, **kwargs):
                pass

        store = LockedCanonicalGraphStore(output_dir, config)
        pipeline = GraphFactorPipeline(source, store, config)
        result = pipeline.build_date(trade_date)
        return {"trade_date": trade_date, "success": True, "result": result, "stats": stats}
    except Exception as e:
        import traceback
        logger.error(f"[Graph Worker {pid}] Failed on {trade_date}: {e}\n{traceback.format_exc()}")
        return {"trade_date": trade_date, "success": False, "error": str(e), "stats": stats}


def background_copy_month(source_root: Path, temp_root: Path, month: str, force_skip: bool = False):
    logger.info(f"[Background Copy] Requested copy for MonthPack {month}...")
    if force_skip:
        logger.info(f"[Background Copy] Skipping copy for {month} because force_skip is True (User manually prepared data).")
        return
    src_dir = source_root / f"month={month}"
    dst_dir = temp_root / f"month={month}"
    if not src_dir.exists():
        logger.warning(f"[Background Copy] Source not found: {src_dir}")
        return
    if dst_dir.exists() and (dst_dir / "node_factors_5m").exists():
        logger.info(f"[Background Copy] Destination already exists and seems ready: {dst_dir}. Skipping copy.")
        return
    try:
        import subprocess
        # use robocopy for fast multi-threaded copy on windows
        res = subprocess.run(["robocopy", str(src_dir), str(dst_dir), "/E", "/MT:32", "/NFL", "/NDL", "/NJH", "/NJS", "/NP"], capture_output=True)
        # robocopy exit codes < 8 are considered success
        if res.returncode >= 8:
            logger.error(f"[Background Copy] Robocopy failed for {month} with code {res.returncode}: {res.stdout.decode('utf-8', errors='ignore')}")
        else:
            logger.info(f"[Background Copy] Finished copying {month}.")
    except Exception as e:
        logger.error(f"[Background Copy] Failed for {month}: {e}")


def update_dashboard(output_dir: Path, month: str, graph_results: list, theme_results: list):
    dashboard_path = output_dir.parent / "dashboard.md"
    now = pd.Timestamp.now('UTC').isoformat()
    lines = [
        "# GraphFactorFactory Phase 0 Dashboard",
        f"Generated UTC: `{now}`",
        "",
        f"## Current Month: {month}",
        f"- Target output: `{output_dir}`",
        f"- Graph Dates Completed: {sum(1 for r in graph_results if r['success'])} / {len(graph_results)}",
        f"- Theme Dates Completed: {len(theme_results)}",
        "",
        "## Worker Performance (Graph Stage)",
        "| Date | Success | Elapsed (s) | Avg CPU % | Peak RAM (MB) |",
        "| --- | --- | ---: | ---: | ---: |"
    ]
    for r in graph_results:
        date = r["trade_date"]
        success = r["success"]
        stats = r.get("stats", {}).get(f"Graph-{date}", {})
        elapsed = stats.get("elapsed_seconds", 0)
        cpu = stats.get("avg_cpu_percent", 0)
        ram = stats.get("peak_ram_mb", 0)
        lines.append(f"| {date} | {success} | {elapsed:.1f} | {cpu:.1f}% | {ram:.1f} |")
    
def push_dashboard_loop(output_dir: Path):
    dashboard_path = output_dir.parent / "dashboard.md"
    repo_dir = Path(r"d:\DEV\US-Stock\GraphFactorFactory")
    # Ensure branch exists
    subprocess.run(["git", "branch", "dashboard_upadate"], cwd=str(repo_dir), capture_output=True)
    while True:
        time.sleep(300)
        if dashboard_path.exists():
            try:
                subprocess.run(["git", "checkout", "dashboard_upadate"], cwd=str(repo_dir), capture_output=True)
                # We need to copy the dashboard to the repo root to commit it easily, or just add its path
                # Since the prompt said "在dashboard显示产物的路径", let's put dashboard at repo root.
                target_dashboard = repo_dir / "dashboard.md"
                shutil.copy2(dashboard_path, target_dashboard)
                subprocess.run(["git", "add", "dashboard.md"], cwd=str(repo_dir), capture_output=True)
                subprocess.run(["git", "commit", "-m", "Auto-update dashboard"], cwd=str(repo_dir), capture_output=True)
                subprocess.run(["git", "push", "-u", "origin", "dashboard_upadate"], cwd=str(repo_dir), capture_output=True)
                # Switch back to the previous branch (smoke_run)
                subprocess.run(["git", "checkout", "-"], cwd=str(repo_dir), capture_output=True)
                logger.info("[Dashboard Push] Successfully pushed dashboard update.")
            except Exception as e:
                logger.error(f"[Dashboard Push] Failed to push dashboard: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-monthpack-root", default=r"P:\US-Stock\NodeFactorFactory\warehouse\month_packs")
    parser.add_argument("--temp-monthpack-root", default=r"C:\nodefactor_work\month_packs")
    parser.add_argument("--output-root", default=r"D:\DEV\US-Stock\GraphFactorFactory\data\graph_store")
    parser.add_argument("--target-month", default="2026-06")
    parser.add_argument("--next-month", default="2026-05")
    parser.add_argument("--config", default=r"D:\DEV\US-Stock\GraphFactorFactory\configs\default.yaml")
    parser.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() - 2))
    parser.add_argument("--skip-copy", action="store_true", default=True, help="Skip copying from HDD (assume user manually copied data)")
    args = parser.parse_args()

    source_root = Path(args.source_monthpack_root)
    temp_root = Path(args.temp_monthpack_root)
    output_root = Path(args.output_root)

    # Initialize dashboard immediately so user can see status
    output_root.mkdir(parents=True, exist_ok=True)
    update_dashboard(output_root, args.target_month, [], [])

    # Start the 5-minute auto-push thread
    threading.Thread(target=push_dashboard_loop, args=(output_root,), daemon=True).start()

    # 1. Start background copy for the NEXT month
    if args.next_month:
        bg_thread = threading.Thread(target=background_copy_month, args=(source_root, temp_root, args.next_month, args.skip_copy), daemon=True)
        bg_thread.start()

    # 2. Ensure TARGET month is copied (blocking)
    logger.info(f"Ensuring target month {args.target_month} is copied from {source_root} to {temp_root}...")
    background_copy_month(source_root, temp_root, args.target_month, force_skip=args.skip_copy)

    pack_source = MonthPackNodeFactorSource(temp_root)
    dates = pack_source.available_dates(args.target_month)
    if not dates:
        logger.error(f"No dates found for {args.target_month} in {temp_root}")
        return

    logger.info(f"Found {len(dates)} dates for {args.target_month}. Starting Graph Pipeline with {args.workers} workers...")
    
    manager = multiprocessing.Manager()
    lock = manager.Lock()

    graph_results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_graph_date, (d, args.target_month, str(temp_root), str(output_root), args.config, lock)): d for d in dates}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            graph_results.append(res)
            update_dashboard(output_root, args.target_month, graph_results, [])

    # Check for graph failures
    failed = [r["trade_date"] for r in graph_results if not r["success"]]
    if failed:
        logger.error(f"Graph Pipeline failed for dates: {failed}. Aborting Themes.")
        return

    # 3. Sequential Theme Pipeline
    logger.info("Graph Pipeline completed successfully.")
    logger.info("Finalizing DuckDB catalog for Graph Store...")
    config = BuildConfig.from_yaml(args.config) if args.config else BuildConfig()
    CanonicalGraphStore(output_root, config).finalize_catalog()

    # 4. Run Theme Discovery Phase (Sequential because DuckDB takes huge memory)
    theme_config = ThemeDiscoveryConfig(run_id=f"run_{args.target_month}")
    # Using the same output_root for themes as per standard architecture
    theme_pipeline = ThemeDiscoveryPipeline(output_root, output_root / "themes", theme_config)
    
    theme_start_time = time.time()
    # Execute theme discovery
    theme_results = theme_pipeline.run(date_start=f"{args.target_month}-01", date_end=f"{args.target_month}-31")
    theme_elapsed = time.time() - theme_start_time
    logger.info(f"Theme Pipeline complete in {theme_elapsed:.2f}s.")

    update_dashboard(output_root, args.target_month, graph_results, theme_results)
    
    logger.info(f"Phase 0 completely finished for {args.target_month}.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
