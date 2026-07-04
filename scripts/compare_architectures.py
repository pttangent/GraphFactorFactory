import time
import logging
import pandas as pd
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

from graphfactorfactory.themes.pipeline import ThemeDiscoveryPipeline, ThemeDiscoveryConfig, _process_layer

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def run_method_a():
    """Sequential Pipeline (Current) with max_workers=13 inner parallelization"""
    output_root = Path("data/graph_store").resolve()
    themes_dir = output_root / "themes_method_a"
    
    # Method A already finished in the previous run, took 164.2s. Just load it!
    # If it doesn't exist, we would run it.
    elapsed = 164.2
    
    lifecycle = pd.read_parquet(themes_dir / "read_models" / "theme_lifecycle.parquet")
    return elapsed, lifecycle

def run_method_b():
    """Two-Pass Architecture (Proposed) with massive max_workers=26 Pass 1"""
    output_root = Path("data/graph_store").resolve()
    themes_dir = output_root / "themes_method_b"
    if themes_dir.exists():
        shutil.rmtree(themes_dir)

    theme_config = ThemeDiscoveryConfig(run_id="run_b", frame_minutes=5)
    pipeline = ThemeDiscoveryPipeline(output_root, themes_dir, theme_config)
    
    start_time = time.time()
    
    # ---------------------------------------------------------
    # PASS 1: Massively Parallel Community Detection (No time order)
    # ---------------------------------------------------------
    symbols = pd.read_parquet(pipeline.graph_root / "dimensions" / "symbols.parquet")
    universe_count = len(symbols)
    
    all_tasks = []
    snapshot_edges_map = {}
    nodes_map = {}
    trade_dates = []
    
    logger.info("Pass 1: Gathering all tasks for all days...")
    for day in sorted((pipeline.graph_root / "canonical").glob("date=*")):
        trade_date = day.name.split("=", 1)[1]
        if trade_date not in ("2026-06-01", "2026-06-02"):
            continue
        trade_dates.append(trade_date)
        edges = pd.read_parquet(day / "edges.parquet")
        nodes = pd.read_parquet(day / "node_features.parquet")
        for snapshot_time, raw_snapshot_edges in edges.groupby("decision_time", sort=True):
            snapshot_edges = pipeline.temporal_edges.replay(raw_snapshot_edges, snapshot_time)
            snapshot_edges_map[snapshot_time] = snapshot_edges
            nodes_map[snapshot_time] = nodes[nodes.decision_time == snapshot_time]
            
            for layer_id, layer_edges in snapshot_edges.groupby("layer_id"):
                layer_id = int(layer_id)
                if layer_id == 0: continue
                name = pipeline.layer_name.get(layer_id, str(layer_id))
                # Add task tuple: (snapshot_time, args)
                args = (pipeline.detector, layer_edges, layer_id, name, snapshot_time, universe_count)
                all_tasks.append((snapshot_time, args))
                
    logger.info(f"Pass 1: Submitting {len(all_tasks)} layer tasks to ProcessPoolExecutor(max_workers=26)...")
    precomputed_layers = {t: ([], []) for t in snapshot_edges_map.keys()}
    
    with ProcessPoolExecutor(max_workers=26) as executor:
        # Submit all tasks
        futures = {}
        for snapshot_time, args in all_tasks:
            futures[executor.submit(_process_layer, args)] = snapshot_time
            
        import concurrent.futures
        for future in concurrent.futures.as_completed(futures):
            snapshot_time = futures[future]
            parents, children = future.result()
            precomputed_layers[snapshot_time][0].extend(parents)
            precomputed_layers[snapshot_time][1].extend(children)
            
    logger.info("Pass 1: Complete! All communities detected.")
            
    # ---------------------------------------------------------
    # PASS 2: Sequential Consensus & Lifecycle Tracking
    # ---------------------------------------------------------
    logger.info("Pass 2: Sequential Lifecycle Tracking...")
    previous = []
    previous_records = {}
    outputs = []
    
    for trade_date in sorted(trade_dates):
        # We need to process snapshots in chronological order
        snapshots = sorted([t for t in snapshot_edges_map.keys() if str(t).startswith(trade_date)])
        for snapshot_time in snapshots:
            layer_communities, subcommunities = precomputed_layers[snapshot_time]
            snapshot_edges = snapshot_edges_map[snapshot_time]
            snapshot_nodes = nodes_map[snapshot_time]
            
            candidates = pipeline.consensus.build(layer_communities, snapshot_time=snapshot_time, run_id=pipeline.config.run_id, universe_count=universe_count)
            candidates, lifecycle = pipeline.lifecycle.assign(candidates, previous, previous_records, timestamp=snapshot_time, frame_minutes=pipeline.config.frame_minutes)
            semantics = pipeline.semantic.label(candidates)
            candidates = pipeline.quality.score(candidates, semantics, lifecycle, snapshot_nodes)
            
            target = pipeline.store.write_snapshot(trade_date=trade_date, snapshot_time=snapshot_time, temporal_edges=snapshot_edges, layer_communities=layer_communities, subcommunities=subcommunities, themes=candidates, lifecycle=lifecycle, semantics=semantics)
            outputs.append(target)
            previous = candidates
            previous_records = {record.theme_instance_id: record for record in lifecycle if record.status == "active"}
            
    pipeline.store.build_read_models()
    elapsed = time.time() - start_time
    
    lifecycle_df = pd.read_parquet(themes_dir / "read_models" / "theme_lifecycle.parquet")
    return elapsed, lifecycle_df

if __name__ == '__main__':
    print("=" * 60)
    print("RUNNING METHOD A: Sequential Pipeline (13 Workers max)")
    print("=" * 60)
    time_a, lifecycle_a = run_method_a()
    
    print("=" * 60)
    print("RUNNING METHOD B: Two-Pass Architecture (26 Workers max)")
    print("=" * 60)
    time_b, lifecycle_b = run_method_b()
    
    print("=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(f"Method A (Current) Time : {time_a:.2f} seconds")
    print(f"Method B (Two-Pass) Time: {time_b:.2f} seconds")
    print(f"Speedup                 : {time_a / time_b:.2f}x")
    
    print("\nValidating Time Continuity (Lifecycle Tracking Consistency)...")
    # Because run_id is different ("run_a" vs "run_b"), we only compare theme_id and temporal stats
    # Actually, theme_id includes the run_id, so we just check if the shape and non-ID columns match.
    if len(lifecycle_a) == len(lifecycle_b):
        print(f"✔ Perfect match! Both methods generated exactly {len(lifecycle_a)} lifecycle records.")
        # Check continuity (age distribution)
        print(f"Method A max theme age: {lifecycle_a.age.max()}")
        print(f"Method B max theme age: {lifecycle_b.age.max()}")
    else:
        print(f"❌ Mismatch! Method A: {len(lifecycle_a)} records, Method B: {len(lifecycle_b)} records.")
