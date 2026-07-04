from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from .community import LeidenCommunityDetector
from .consensus import ConsensusThemeBuilder
from .semantic_quality import MetadataSemanticLabeler, ThemeQualityScorer
from .store import ThemeStore
from .temporal import ThemeLifecycleTracker
from .temporal_edges import TemporalEdgeConfig, TemporalEdgeReplay

logger = logging.getLogger(__name__)


def _leaf_communities(parents, children):
    split_ids = {c.parent_community_id for c in children if c.parent_community_id is not None}
    return [p for p in parents if p.community_id not in split_ids] + list(children)


def _process_theme_chunk(args):
    chunk_times, chunk_edges, detector, consensus_builder, layer_names, universe_count, run_id = args
    results = []
    for snapshot_time in chunk_times:
        snapshot_edges = chunk_edges[chunk_edges["decision_time"] == snapshot_time]
        all_parents, all_children, consensus_inputs = [], [], []
        for layer_id, layer_edges in snapshot_edges.groupby("layer_id"):
            layer_id = int(layer_id)
            if layer_id == 0:
                continue
            name = layer_names.get(layer_id, str(layer_id))
            parents, children = detector.detect_hierarchy(
                layer_edges,
                layer_id=layer_id,
                layer_name=name,
                snapshot_time=snapshot_time,
                universe_count=universe_count,
            )
            all_parents.extend(parents)
            all_children.extend(children)
            consensus_inputs.extend(_leaf_communities(parents, children))
        candidates = consensus_builder.build(
            consensus_inputs,
            snapshot_time=snapshot_time,
            run_id=run_id,
            universe_count=universe_count,
        )
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
    min_layer_support_ratio: float = 0.10
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
        self.detector = LeidenCommunityDetector(
            resolution=config.leiden_resolution,
            seed=config.seed,
            min_members=config.min_members,
            market_mode_max_member_ratio=config.market_mode_max_member_ratio,
            sub_resolution=config.subcommunity_resolution,
            min_subcommunity_size=config.min_subcommunity_size,
        )
        self.consensus = ConsensusThemeBuilder(
            layer_weights={name: 1.0 for name in self.layer_family},
            family_map=self.layer_family,
            min_consensus_score=config.min_consensus_score,
            min_members=config.min_members,
            min_distinct_families=config.min_distinct_families,
            market_mode_max_member_ratio=config.market_mode_max_member_ratio,
            min_layer_support_ratio=config.min_layer_support_ratio,
            seed=config.seed,
        )
        self.lifecycle = ThemeLifecycleTracker(config.min_overlap)
        self.semantic = MetadataSemanticLabeler(metadata)
        self.quality = ThemeQualityScorer()
        self.temporal_edges = TemporalEdgeReplay(
            TemporalEdgeConfig(
                config.temporal_enter_threshold,
                config.temporal_exit_threshold,
                config.temporal_smoothing_alpha,
                config.temporal_missing_grace_frames,
            )
        )

    def run(self, date_start=None, date_end=None, max_workers=26):
        import concurrent.futures
        import numpy as np

        symbols = pd.read_parquet(self.graph_root / "dimensions" / "symbols.parquet")
        universe_count = len(symbols)
        previous = []
        previous_records = {}
        outputs = []

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for day in sorted((self.graph_root / "canonical").glob("date=*")):
                trade_date = day.name.split("=", 1)[1]
                if date_start and trade_date < date_start:
                    continue
                if date_end and trade_date > date_end:
                    continue
                t0 = time.time()
                edges = pd.read_parquet(day / "edges.parquet")
                nodes = pd.read_parquet(day / "node_features.parquet")
                snapshot_times = sorted(edges["decision_time"].unique())
                nodes_map = {t: nodes[nodes.decision_time == t] for t in snapshot_times}

                replayed_parts = []
                for snapshot_time in snapshot_times:
                    raw_edges = edges[edges["decision_time"] == snapshot_time]
                    replayed = self.temporal_edges.replay(raw_edges, snapshot_time)
                    if not replayed.empty:
                        replayed_parts.append(replayed)
                replayed_edges = pd.concat(replayed_parts, ignore_index=True) if replayed_parts else edges.iloc[0:0].copy()

                chunks = np.array_split(snapshot_times, max_workers)
                tasks = []
                for chunk in chunks:
                    if len(chunk) == 0:
                        continue
                    times = list(chunk)
                    chunk_edges = replayed_edges[replayed_edges["decision_time"].isin(times)].copy()
                    tasks.append((times, chunk_edges, self.detector, self.consensus, self.layer_name, universe_count, self.config.run_id))

                precomputed = {t: (None, [], [], []) for t in snapshot_times}
                futures = [executor.submit(_process_theme_chunk, task) for task in tasks]
                for future in concurrent.futures.as_completed(futures):
                    for result in future.result():
                        precomputed[result[0]] = result[1:]

                for snapshot_time in snapshot_times:
                    snapshot_edges, parents, children, candidates = precomputed[snapshot_time]
                    if snapshot_edges is None:
                        continue
                    candidates, lifecycle = self.lifecycle.assign(
                        candidates,
                        previous,
                        previous_records,
                        timestamp=snapshot_time,
                        frame_minutes=self.config.frame_minutes,
                    )
                    semantics = self.semantic.label(candidates)
                    candidates = self.quality.score(candidates, semantics, lifecycle, nodes_map[snapshot_time])
                    self.store.accumulate_snapshot(
                        snapshot_time=snapshot_time,
                        temporal_edges=snapshot_edges,
                        layer_communities=parents,
                        subcommunities=children,
                        themes=candidates,
                        lifecycle=lifecycle,
                        semantics=semantics,
                    )
                    previous = candidates
                    previous_records = {r.theme_instance_id: r for r in lifecycle if r.status == "active"}

                target = self.store.write_day(trade_date)
                if target:
                    outputs.append(target)
                logger.info(f"[{trade_date}] finished in {time.time() - t0:.1f}s")

        self.store.build_read_models()
        return outputs
