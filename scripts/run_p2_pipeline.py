#!/usr/bin/env python3
import argparse
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOG = logging.getLogger(__name__)

# Constants
P1_ROOT = r"C:\GFF_Cache\p1_b50_b35_sharded"
P0_LABELS_ROOT = r"D:\DEV\US-Stock\GraphFactorFactory\data\graph_store_6m\canonical"
P2_OUT_ROOT = r"C:\GFF_Cache\p2_alpha_lab"

ROUNDS = {
    "1": {
        "dates": "2026-01-07",
        "layers": "9",
        "scales": "15m,30m",
        "levels": "B50,B35"
    },
    "2": {
        "dates": "2026-01-07,2026-01-08,2026-01-09",
        "layers": "6,8,9",
        "scales": "15m,30m",
        "levels": "B50,B35"
    },
    "3": {
        "dates": "2026-01-07,2026-01-08,2026-01-09,2026-01-12,2026-01-13,2026-01-14,2026-01-15,2026-01-16,2026-01-20,2026-01-21",
        "layers": "3,6,8,9,11",
        "scales": "15m,30m",
        "levels": "B50,B35"
    }
}

def run_cmd(cmd: list):
    LOG.info("Executing: " + " ".join(cmd))
    res = subprocess.run(cmd)
    if res.returncode != 0:
        LOG.error(f"Command failed with exit code {res.returncode}")
        raise SystemExit(res.returncode)

def run_qc(workers: int):
    LOG.info("=== Phase: P1 QC ===")
    run_cmd(["python", "scripts/p1_qc.py", "--p1-root", P1_ROOT, "--workers", str(workers)])

def run_p2_lab(r_params: dict, workers: int):
    LOG.info("=== Phase: P2 Intraday Alpha Lab ===")
    
    dates_arg = ["--dates", r_params["dates"]] if r_params["dates"] else []
    
    # 1. Build Theme Returns
    run_cmd([
        "python", "scripts/p2_alpha_daily_features.py", "build-theme-returns",
        "--p1-root", P1_ROOT,
        "--labels-root", P0_LABELS_ROOT,
        "--out-root", f"{P2_OUT_ROOT}\\theme_returns",
        "--layers", r_params["layers"],
        "--scales", r_params["scales"],
        "--levels", r_params["levels"],
        "--workers", str(workers)
    ] + dates_arg)
    
    # 2. Relation Spillover
    run_cmd([
        "python", "scripts/p2_alpha_daily_features.py", "relation-spillover",
        "--p1-root", P1_ROOT,
        "--theme-returns-root", f"{P2_OUT_ROOT}\\theme_returns",
        "--out-root", f"{P2_OUT_ROOT}\\relation_spillover",
        "--layers", r_params["layers"],
        "--scales", r_params["scales"],
        "--levels", r_params["levels"],
        "--past-horizon", "15m",
        "--workers", str(workers)
    ] + dates_arg)
    
    # 3. Daily Relation Features
    run_cmd([
        "python", "scripts/p2_alpha_daily_features.py", "daily-relation-features",
        "--signals-root", f"{P2_OUT_ROOT}\\relation_spillover",
        "--out-root", f"{P2_OUT_ROOT}\\daily_relation_features",
        "--layers", r_params["layers"],
        "--scales", r_params["scales"],
        "--levels", r_params["levels"],
        "--workers", str(workers)
    ] + dates_arg)
    
    # 4. Evaluate Daily (Proxy Eval)
    run_cmd([
        "python", "scripts/p2_alpha_daily_features.py", "evaluate-daily",
        "--features-root", f"{P2_OUT_ROOT}\\daily_relation_features",
        "--out-dir", f"{P2_OUT_ROOT}\\daily_relation_eval"
    ])

def run_true_daily_alpha(workers: int):
    LOG.info("=== Phase: True Daily Alpha Validation ===")
    
    # Build daily labels first
    labels_out = f"{P2_OUT_ROOT}\\daily_labels.parquet"
    run_cmd([
        "python", "scripts/build_daily_labels.py",
        "--out-path", labels_out
    ])
    
    LOG.info("Daily labels generated. Ready for true daily alpha downstream ingestion!")

def main():
    parser = argparse.ArgumentParser(description="P2 Pipeline Orchestrator")
    parser.add_argument("--round", choices=["qc", "1", "2", "3", "4"], required=True, 
                        help="Execution round (qc=QC only, 1/2/3=Lab increments, 4=Daily Labels)")
    parser.add_argument("--workers", type=int, default=16, help="Worker count for parallelism")
    args = parser.parse_args()

    if args.round == "qc":
        run_qc(args.workers)
    elif args.round in ["1", "2", "3"]:
        run_p2_lab(ROUNDS[args.round], args.workers)
    elif args.round == "4":
        run_true_daily_alpha(args.workers)

if __name__ == "__main__":
    main()
