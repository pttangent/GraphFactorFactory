from __future__ import annotations

from collections import defaultdict
from itertools import combinations
import hashlib
import pandas as pd

from .community import LeidenCommunityDetector
from .models import LayerCommunity, ThemeCandidate


_DISABLED_PRICE_LAYERS = {"return_corr", "return_corr_cross_sectional_residual"}


class ConsensusThemeBuilder:
    def __init__(self, *, layer_weights=None, family_map=None, min_consensus_score=0.35, min_members=3, min_distinct_families=2, market_mode_max_member_ratio=0.15, seed=20260704):
        self.layer_weights = layer_weights or {}
        self.family_map = family_map or {}
        self.min_consensus_score = min_consensus_score
        self.min_members = min_members
        self.min_distinct_families = min_distinct_families
        self.market_mode_max_member_ratio = market_mode_max_member_ratio
        self.seed = seed

    def build(self, communities: list[LayerCommunity], *, snapshot_time, run_id: str, universe_count: int) -> list[ThemeCandidate]:
        pair_scores = defaultdict(float)
        pair_layers = defaultdict(set)
        for community in communities:
            if community.is_market_mode or community.layer_name in _DISABLED_PRICE_LAYERS:
                continue
            weight = float(self.layer_weights.get(community.layer_name, 1.0))
            if weight <= 0:
                continue
            for left, right in combinations(sorted(community.members), 2):
                pair_scores[(left, right)] += weight
                pair_layers[(left, right)].add(community.layer_name)
        rows = [(a, b, score) for (a, b), score in pair_scores.items() if score >= self.min_consensus_score]
        if not rows:
            return []
        edges = pd.DataFrame(rows, columns=["src_id", "dst_id", "weight"])
        detector = LeidenCommunityDetector(resolution=1.0, seed=self.seed, min_members=self.min_members, market_mode_max_member_ratio=self.market_mode_max_member_ratio)
        consensus = detector.detect(edges, layer_id=-1, layer_name="consensus", snapshot_time=snapshot_time, universe_count=universe_count)
        result = []
        for index, community in enumerate(consensus, start=1):
            members = set(community.members)
            supporting_layers = sorted({layer for pair, layers in pair_layers.items() if set(pair).issubset(members) for layer in layers})
            families = sorted({self.family_map.get(layer, layer) for layer in supporting_layers})
            if len(families) < self.min_distinct_families:
                continue
            scores = [score for pair, score in pair_scores.items() if set(pair).issubset(members)]
            consensus_score = sum(scores) / len(scores) if scores else 0.0
            size_penalty = min(1.0, 8.0 / max(len(members), 1))
            structure_score = min(1.0, consensus_score * (0.6 + 0.4 * size_penalty))
            digest = hashlib.sha1(",".join(map(str, sorted(members))).encode()).hexdigest()[:10]
            instance = f"{pd.Timestamp(snapshot_time).strftime('%Y%m%dT%H%M%S')}_{digest}"
            result.append(ThemeCandidate(instance, f"{run_id}_new_{index:04d}", snapshot_time, tuple(sorted(members)), tuple(supporting_layers), tuple(families), float(consensus_score), float(structure_score), len(members)/max(universe_count,1), community.is_market_mode))
        return result
