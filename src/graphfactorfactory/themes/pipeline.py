from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from .community import LeidenCommunityDetector
from .consensus import ConsensusThemeBuilder
from .semantic_quality import MetadataSemanticLabeler, ThemeQualityScorer
from .store import ThemeStore
from .temporal import ThemeLifecycleTracker
from .temporal_edges import TemporalEdgeConfig, TemporalEdgeReplay


import time
import logging
from concurrent.futures import ProcessPoolExecutor

logger = logging.getLogger(__name__)

def _process_theme_chunk(args):
    chunk_times, chunk_edges, temporal_edges_builder, detector, consensus_builder, layer_names, universe_count, run_id = args
    results = []
    
    for snapshot_time in chunk_times:
        raw_snapshot_edges = chunk_edges[chunk_edges["decision_time"] == snapshot_time]
        snapshot_edges = temporal_edges_builder.replay(raw_snapshot_edges, snapshot_time)
        
        all_parents, all_children = [], []
        for layer_id, layer_edges in snapshot_edges.groupby("layer_id"):
            layer_id = int(layer_id)
            if layer_id == 0: continue
            name = layer_names.get(layer_id, str(layer_id))
            parents, children = detector.detect_hierarchy(layer_edges, layer_id=layer_id, layer_name=name, snapshot_time=snapshot_time, universe_count=universe_count)
            all_parents.extend(parents)
            all_children.extend(children)
        
        # Run consensus building in parallel!
        candidates = consensus_builder.build(all_parents + all_children, snapshot_time=snapshot_time, run_id=run_id, universe_count=universe_count)
        results.append((snapshot_time, snapshot_edges, all_parents, all_children, candidates))
        
    return results

@dataclass(frozen=True)
class ThemeDiscoveryConfig:
    run_id: str = "gff_theme_run"
    frame_minutes: int = 15
    leiden_resolution: float = 1.0
    subcommunity_resolution: float = 1.4
    min_members: int = 3
    min_subcommunity_size: int = 30
    min_consensus_score: float = 0.35
    min_distinct_families: int = 2
    min_overlap: float = 0.5
    market_mode_max_member_ratio: float = 0.15
    temporal_enter_threshold: float = 0.75
    temporal_exit_threshold: float = 0.65
    temporal_smoothing_alpha: float = 0.6
    temporal_missing_grace_frames: int = 1
    seed: int = 20260704


class ThemeDiscoveryPipeline:
    def __init__(self, graph_store_root, theme_store_root, config: ThemeDiscoveryConfig, metadata: pd.DataFrame | None = None):
        self.graph_root = Path(graph_store_root).expanduser().resolve()
        self.store = ThemeStore(theme_store_root)
        self.config = config
        layers = pd.read_parquet(self.graph_root / "dimensions" / "layers.parquet")
        self.layer_name = dict(zip(layers.layer_id.astype(int), layers.name.astype(str)))
        self.layer_family = dict(zip(layers.name.astype(str), layers.family.astype(str)))
        self.detector = LeidenCommunityDetector(resolution=config.leiden_resolution, seed=config.seed, min_members=config.min_members, market_mode_max_member_ratio=config.market_mode_max_member_ratio, sub_resolution=config.subcommunity_resolution, min_subcommunity_size=config.min_subcommunity_size)
        self.consensus = ConsensusThemeBuilder(layer_weights={name: 1.0 for name in self.layer_family}, family_map=self.layer_family, min_consensus_score=config.min_consensus_score, min_members=config.min_members, min_distinct_families=config.min_distinct_families, market_mode_max_member_ratio=config.market_mode_max_member_ratio, seed=config.seed)
        self.lifecycle = ThemeLifecycleTracker(config.min_overlap)
        self.semantic = MetadataSemanticLabeler(metadata)
        self.quality = ThemeQualityScorer()
        self.temporal_edges = TemporalEdgeReplay(TemporalEdgeConfig(config.temporal_enter_threshold, config.temporal_exit_threshold, config.temporal_smoothing_alpha, config.temporal_missing_grace_frames))

    def run(self, date_start=None, date_end=None, max_workers=26):
        symbols = pd.read_parquet(self.graph_root / "dimensions" / "symbols.parquet")
        universe_count = len(symbols)
        previous = []
        previous_records = {}
        outputs = []
        
        import concurrent.futures
        
        logger.info(f"Starting Two-Pass Theme Discovery with {max_workers}x parallel ProcessPool (Universe: {universe_count})...")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for day in sorted((self.graph_root / "canonical").glob("date=*")):
                trade_date = day.name.split("=", 1)[1]
                if date_start and trade_date < date_start:
                    continue
                if date_end and trade_date > date_end:
                    continue
                
                logger.info(f"[{trade_date}] Starting Theme Discovery (Two-Pass)...")
                t0 = time.time()
                edges = pd.read_parquet(day / "edges.parquet")
                nodes = pd.read_parquet(day / "node_features.parquet")
                
                import numpy as np
                # PASS 1: Vectorized Temporal Edge Replay & Task Gathering
                snapshot_times = sorted(edges["decision_time"].unique())
                nodes_map = {t: nodes[nodes.decision_time == t] for t in snapshot_times}
                
                chunks = np.array_split(snapshot_times, max_workers)
                chunk_tasks = []
                for chunk in chunks:
                    if len(chunk) == 0: continue
                    chunk_times = list(chunk)
                    chunk_edges = edges[edges["decision_time"].isin(chunk_times)].copy()
                    args = (chunk_times, chunk_edges, self.temporal_edges, self.detector, self.consensus, self.layer_name, universe_count, self.config.run_id)
                    chunk_tasks.append(args)
                
                # PASS 2: Massively Parallel Community Detection (ProcessPool)
                precomputed_layers = {t: (None, [], [], []) for t in snapshot_times}
                
                futures = {executor.submit(_process_theme_chunk, task): task for task in chunk_tasks}
                for future in concurrent.futures.as_completed(futures):
                    try:
                        chunk_results = future.result()
                        for snapshot_time, snapshot_edges, parents, children, candidates in chunk_results:
                            precomputed_layers[snapshot_time] = (snapshot_edges, parents, children, candidates)
                    except Exception as e:
                        logger.error(f"Error processing chunk: {e}")
                    
                # PASS 3: Sequential Lifecycle Tracking & Batch Accumulation
                for snapshot_time in snapshot_times:
                    snapshot_edges, parents, children, candidates = precomputed_layers[snapshot_time]
                    if snapshot_edges is None:
                        continue
                        
                    layer_communities = parents
                    subcommunities = children
                    snapshot_nodes = nodes_map[snapshot_time]
                    candidates, lifecycle = self.lifecycle.assign(candidates, previous, previous_records, timestamp=snapshot_time, frame_minutes=self.config.frame_minutes)
                    semantics = self.semantic.label(candidates)
                    candidates = self.quality.score(candidates, semantics, lifecycle, snapshot_nodes)
                    
                    self.store.accumulate_snapshot(
                        snapshot_time=snapshot_time,
                        temporal_edges=snapshot_edges,
                        layer_communities=layer_communities,
                        subcommunities=subcommunities,
                        themes=candidates,
                        lifecycle=lifecycle,
                        semantics=semantics
                    )
                    
                    previous = candidates
                    previous_records = {record.theme_instance_id: record for record in lifecycle if record.status == "active"}
                
                # Batch Write for the Day
                target = self.store.write_day(trade_date)
                if target:
                    outputs.append(target)
                logger.info(f"[{trade_date}] Finished Theme Discovery in {time.time() - t0:.1f}s")
                
        self.store.build_read_models()
        return outputs
