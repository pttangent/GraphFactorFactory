import argparse
import json
import logging
import multiprocessing
import shutil
import time
from pathlib import Path

import pandas as pd
from dataclasses import replace

from graphfactorfactory.application.pipeline import GraphFactorPipeline
from graphfactorfactory.domain.config import BuildConfig
from graphfactorfactory.domain.layers import LAYER_SCALES
from graphfactorfactory.infrastructure.nodefactorfactory.parquet_source import ParquetNodeFactorSource
from graphfactorfactory.infrastructure.store import CanonicalGraphStore
from graphfactorfactory.themes.pipeline import ThemeDiscoveryConfig, ThemeDiscoveryPipeline
from scripts.run_phase01_production import run_day

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FULL_QA")

def _et_mask(series, start="10:00", end="10:15"):
    local = pd.to_datetime(series, utc=True).dt.tz_convert("America/New_York")
    hhmm = local.dt.strftime("%H:%M")
    return (hhmm >= start) & (hhmm <= end)

def execute_pipeline(month_pack_root, graph_root, theme_root, workers, config_path, overwrite=False):
    date = "2026-06-16"
    graph_root = Path(graph_root).resolve()
    theme_root = Path(theme_root).resolve()
    if overwrite:
        shutil.rmtree(graph_root, ignore_errors=True)
        shutil.rmtree(theme_root, ignore_errors=True)
        
    config = BuildConfig.from_yaml(config_path)
    config = replace(config, frequency="1min", market_open="09:30", market_close="10:15", graph_step_minutes=1)
    
    glob_pattern = str(Path(month_pack_root) / "month=*" / "node_factors_1m" / "date=*" / "*.parquet")
    source = ParquetNodeFactorSource(glob_pattern)
    store = CanonicalGraphStore(graph_root, config)
    pipe = GraphFactorPipeline(source, store, config)
    pipe.max_threads = max(1, workers)
    pipe.task_chunk_size = 1
    
    logger.info("Executing Phase 0")
    pipe.build_date(date)
    
    theme_config = ThemeDiscoveryConfig(run_id="run_qa_full", frame_minutes=1)
    theme_pipe = ThemeDiscoveryPipeline(graph_root, theme_root, theme_config)
    run_day(theme_pipe, graph_root / "canonical" / f"date={date}", workers)
    theme_pipe.store.build_read_models()
    return date, config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--month-pack-root", required=True)
    parser.add_argument("--graph-root", required=True)
    parser.add_argument("--theme-root", required=True)
    parser.add_argument("--config", default="configs/phase0_ab_selected_v1.yaml")
    parser.add_argument("--workers", type=int, default=22)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    
    date, config = execute_pipeline(args.month_pack_root, args.graph_root, args.theme_root, args.workers, args.config, args.overwrite)
    logger.info("Pipeline execution complete. Running Analyzers...")
    
    from scripts.qa_suite.core_analyzers import run_analyzers
    run_analyzers(args.graph_root, args.theme_root, date)
    
    # Zip the package
    import zipfile
    day_theme = Path(args.theme_root).resolve() / f"date={date}"
    zip_path = Path.cwd() / "QApack.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in day_theme.glob("*.csv"):
            zf.write(f, f.name)
        if (day_theme / "report.json").exists():
            zf.write(day_theme / "report.json", "report.json")
            
    logger.info(f"Successfully generated {zip_path}")
    
if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
