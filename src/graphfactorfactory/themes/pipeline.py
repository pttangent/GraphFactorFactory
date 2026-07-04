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

    def run(self, date_start=None, date_end=None):
        symbols = pd.read_parquet(self.graph_root / "dimensions" / "symbols.parquet")
        universe_count = len(symbols)
        previous = []
        previous_records = {}
        outputs = []
        for day in sorted((self.graph_root / "canonical").glob("date=*")):
            trade_date = day.name.split("=", 1)[1]
            if date_start and trade_date < date_start:
                continue
            if date_end and trade_date > date_end:
                continue
            edges = pd.read_parquet(day / "edges.parquet")
            nodes = pd.read_parquet(day / "node_features.parquet")
            for snapshot_time, raw_snapshot_edges in edges.groupby("decision_time", sort=True):
                snapshot_edges = self.temporal_edges.replay(raw_snapshot_edges, snapshot_time)
                layer_communities = []
                subcommunities = []
                for layer_id, layer_edges in snapshot_edges.groupby("layer_id"):
                    layer_id = int(layer_id)
                    if layer_id == 0:
                        continue
                    name = self.layer_name.get(layer_id, str(layer_id))
                    parents, children = self.detector.detect_hierarchy(layer_edges, layer_id=layer_id, layer_name=name, snapshot_time=snapshot_time, universe_count=universe_count)
                    layer_communities.extend(parents)
                    subcommunities.extend(children)
                candidates = self.consensus.build(layer_communities, snapshot_time=snapshot_time, run_id=self.config.run_id, universe_count=universe_count)
                candidates, lifecycle = self.lifecycle.assign(candidates, previous, previous_records, timestamp=snapshot_time, frame_minutes=self.config.frame_minutes)
                semantics = self.semantic.label(candidates)
                snapshot_nodes = nodes[nodes.decision_time == snapshot_time]
                candidates = self.quality.score(candidates, semantics, lifecycle, snapshot_nodes)
                target = self.store.write_snapshot(trade_date=trade_date, snapshot_time=snapshot_time, temporal_edges=snapshot_edges, layer_communities=layer_communities, subcommunities=subcommunities, themes=candidates, lifecycle=lifecycle, semantics=semantics)
                outputs.append(target)
                previous = candidates
                previous_records = {record.theme_instance_id: record for record in lifecycle if record.status == "active"}
        self.store.build_read_models()
        return outputs
