from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from dataclasses import asdict

import numpy as np
import pandas as pd

from .community import LeidenCommunityDetector
from .consensus import ConsensusThemeBuilder
from .semantic_quality import MetadataSemanticLabeler, ThemeQualityScorer
from .pipeline import ThemeDiscoveryConfig

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
        results.append((snapshot_time, all_parents, all_children, candidates))
    return results

class ThemeStorePhase1:
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._accumulator = []

    def accumulate_snapshot(self, *, snapshot_time, layer_communities, subcommunities, themes, semantics):
        self._accumulator.append({
            "layer_communities": pd.DataFrame([asdict(item) for item in layer_communities]) if layer_communities else pd.DataFrame(),
            "subcommunities": pd.DataFrame([asdict(item) for item in subcommunities]) if subcommunities else pd.DataFrame(),
            "themes": pd.DataFrame([{**asdict(item), "quality_breakdown": json.dumps(item.quality_breakdown, sort_keys=True)} for item in themes]) if themes else pd.DataFrame(),
            "semantics": pd.DataFrame([asdict(item) for item in semantics]) if semantics else pd.DataFrame()
        })

    def write_day(self, trade_date):
        if not self._accumulator:
            return None
        target = self.root / f"date={trade_date}"
        target.mkdir(parents=True, exist_ok=True)
        
        for key in ["layer_communities", "subcommunities", "themes", "semantics"]:
            frames = [item[key] for item in self._accumulator if not item[key].empty]
            if frames:
                pd.concat(frames, ignore_index=True).to_parquet(target / f"{key}.parquet", index=False)
                
        self._accumulator = []
        return target

class ThemeDiscoveryPhase1Pipeline:
    def __init__(self, graph_store_root, theme_store_root, config: ThemeDiscoveryConfig, metadata: pd.DataFrame | None = None):
        self.graph_root = Path(graph_store_root).expanduser().resolve()
        self.store = ThemeStorePhase1(theme_store_root)
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
        self.semantic = MetadataSemanticLabeler(metadata)
        self.quality = ThemeQualityScorer()

    def run(self, date_start=None, date_end=None, max_workers=26):
        symbols = pd.read_parquet(self.graph_root / "dimensions" / "symbols.parquet")
        universe_count = len(symbols)
        outputs = []

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for day in sorted((self.graph_root / "canonical").glob("date=*")):
                trade_date = day.name.split("=", 1)[1]
                if date_start and trade_date < date_start:
                    continue
                if date_end and trade_date > date_end:
                    continue
                
                # Resume logic: skip if _SUCCESS exists
                success_marker = self.store.root / f"date={trade_date}" / "_SUCCESS"
                if success_marker.exists():
                    logger.info(f"[{trade_date}] Skipping Phase 1, _SUCCESS marker exists.")
                    continue

                t0 = time.time()
                edges = pd.read_parquet(day / "edges.parquet")
                nodes = pd.read_parquet(day / "node_features.parquet")
                snapshot_times = sorted(edges["decision_time"].unique())
                nodes_map = {t: nodes[nodes.decision_time == t] for t in snapshot_times}

                chunks = np.array_split(snapshot_times, max_workers)
                tasks = []
                for chunk in chunks:
                    if len(chunk) == 0:
                        continue
                    times = list(chunk)
                    chunk_edges = edges[edges["decision_time"].isin(times)].copy()
                    tasks.append((times, chunk_edges, self.detector, self.consensus, self.layer_name, universe_count, self.config.run_id))

                precomputed = {t: (None, [], []) for t in snapshot_times}
                
                import concurrent.futures
                futures = [executor.submit(_process_theme_chunk, task) for task in tasks]
                for future in concurrent.futures.as_completed(futures):
                    for result in future.result():
                        precomputed[result[0]] = result[1:]

                for snapshot_time in snapshot_times:
                    parents, children, candidates = precomputed[snapshot_time]
                    if not parents and not children and not candidates:
                        continue
                    
                    semantics = self.semantic.label(candidates)
                    candidates = self.quality.score(candidates, semantics, [], nodes_map[snapshot_time])
                    
                    self.store.accumulate_snapshot(
                        snapshot_time=snapshot_time,
                        layer_communities=parents,
                        subcommunities=children,
                        themes=candidates,
                        semantics=semantics,
                    )

                target = self.store.write_day(trade_date)
                if target:
                    (target / "_SUCCESS").write_text("success", encoding="utf-8")
                    outputs.append(target)
                logger.info(f"[{trade_date}] Phase 1 finished in {time.time() - t0:.1f}s")

        return outputs
