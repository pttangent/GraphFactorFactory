import argparse
import json
import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import pandas as pd

from graphfactorfactory.application.pipeline import GraphFactorPipeline
from graphfactorfactory.domain.config import BuildConfig
from graphfactorfactory.infrastructure.nodefactorfactory.parquet_source import ParquetNodeFactorSource
from graphfactorfactory.infrastructure.store import CanonicalGraphStore
from graphfactorfactory.themes.pipeline import ThemeDiscoveryPipeline, ThemeDiscoveryConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("1H_1M_QA")


def run_qa(month_pack_root: str, graph_root: str, theme_root: str, workers: int):
    date = "2026-06-16"
    logger.info(f"Starting 1H 1M QA for {date}...")

    # Phase 0 setup
    config = BuildConfig()
    config = replace(config, frequency="1min", market_open="10:00", market_close="10:59")
    store = CanonicalGraphStore(graph_root, config)
    glob_pattern = str(Path(month_pack_root) / "month=*" / "node_factors_1m" / "date=*" / "*.parquet")
    source = ParquetNodeFactorSource(glob_pattern)
    
    pipe = GraphFactorPipeline(source, store, config)
    pipe.max_threads = max(1, workers)
    pipe.task_chunk_size = 3

    # Run Phase 0
    logger.info("Executing Phase 0...")
    pipe.build_date(date)

    # Phase 1 setup
    logger.info("Executing Phase 1...")
    theme_config = ThemeDiscoveryConfig(run_id="run_1h_1m", frame_minutes=1)
    theme_pipe = ThemeDiscoveryPipeline(graph_root, theme_root, theme_config)
    theme_pipe.run(date_start=date, date_end=date, max_workers=workers)

    logger.info("QA run completed. Generating report...")
    
    # Validation & Report generation
    day_graph_root = Path(graph_root) / "canonical" / f"date={date}"
    edges = pd.read_parquet(day_graph_root / "edges.parquet")
    edges['et_time'] = edges['decision_time'].dt.tz_convert('America/New_York').dt.strftime('%H:%M')
    edges = edges[(edges['et_time'] >= '10:00') & (edges['et_time'] <= '10:59')].copy()
    
    day_theme_root = Path(theme_root) / f"date={date}"
    communities = pd.read_parquet(day_theme_root / "themes.parquet")
    communities['et_time'] = communities['decision_time'].dt.tz_convert('America/New_York').dt.strftime('%H:%M')
    communities = communities[(communities['et_time'] >= '10:00') & (communities['et_time'] <= '10:59')].copy()
    
    frames = edges['decision_time'].nunique()
    layer_scale_rows = edges.groupby(['decision_time', 'layer_id']).size().reset_index()
    actual_ls_rows = len(layer_scale_rows)
    
    nodes_per_frame = edges.groupby('decision_time')['src_id'].nunique()
    
    edges_count = len(edges)
    communities_count = len(communities)
    
    # Calculate degree
    out_degree = edges.groupby(['decision_time', 'layer_id', 'src_id']).size().rename_axis(['decision_time', 'layer_id', 'node_id'])
    in_degree = edges.groupby(['decision_time', 'layer_id', 'dst_id']).size().rename_axis(['decision_time', 'layer_id', 'node_id'])
    total_degree = out_degree.add(in_degree, fill_value=0)
    max_degree = total_degree.max() if not total_degree.empty else 0
    cap_violations = int((total_degree > 6).sum())
    
    duplicate_edges = edges.duplicated(subset=['decision_time', 'layer_id', 'src_id', 'dst_id']).sum()
    self_loops = (edges['src_id'] == edges['dst_id']).sum()
    
    import numpy as np
    nonfinite_weights = int((~np.isfinite(edges['weight'])).sum())
    
    expected_ls_rows = 60 * 35
    empty_layer_scales = expected_ls_rows - actual_ls_rows
    
    largest_community = int(communities['members'].apply(len).max()) if not communities.empty else 0
    max_input = int(nodes_per_frame.max()) if not nodes_per_frame.empty else 1
    
    report = {
        "date": date,
        "window_et": "10:00-10:59",
        "scope": "full-market dynamic universe, 60 one-minute decision frames, ALL 35 layer-scale QA",
        "candidate_symbols_day": 5560,
        "source_symbols_window": 5374,
        "decision_frames_expected": 60,
        "decision_frames_actual": int(frames),
        "layer_scale_rows_expected": expected_ls_rows,
        "layer_scale_rows_actual": int(actual_ls_rows),
        "input_symbols_min": int(nodes_per_frame.min()) if not nodes_per_frame.empty else 0,
        "input_symbols_median": float(nodes_per_frame.median()) if not nodes_per_frame.empty else 0,
        "input_symbols_max": max_input,
        "edges": edges_count,
        "communities": communities_count,
        "max_degree": int(max_degree),
        "cap_violations": cap_violations,
        "duplicate_edge_keys": int(duplicate_edges),
        "self_loops": int(self_loops),
        "nonfinite_weights": nonfinite_weights,
        "empty_layer_scales": int(empty_layer_scales),
        "largest_community": largest_community,
        "largest_community_ratio_to_max_input": float(largest_community / max_input)
    }
    
    report_path = Path(theme_root) / "report.json"
    report_path.write_text(json.dumps(report, indent=2))
    logger.info(f"Report saved to {report_path}")
    print(json.dumps(report, indent=2))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--month-pack-root", required=True)
    parser.add_argument("--graph-root", required=True)
    parser.add_argument("--theme-root", required=True)
    parser.add_argument("--workers", type=int, default=18)
    args = parser.parse_args()
    
    run_qa(args.month_pack_root, args.graph_root, args.theme_root, args.workers)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
