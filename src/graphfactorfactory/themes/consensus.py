from __future__ import annotations

from collections import defaultdict
from itertools import combinations
import hashlib
import pandas as pd

from .community import LeidenCommunityDetector
from .models import LayerCommunity, ThemeCandidate


class ConsensusThemeBuilder:
    def __init__(self, *, layer_weights=None, family_map=None, min_consensus_score=0.35, min_members=3, min_distinct_families=2, market_mode_max_member_ratio=0.15, min_layer_support_ratio=0.10, seed=20260704):
        self.layer_weights = layer_weights or {}
        self.family_map = family_map or {}
        self.min_consensus_score = min_consensus_score
        self.min_members = min_members
        self.min_distinct_families = min_distinct_families
        self.market_mode_max_member_ratio = market_mode_max_member_ratio
        self.min_layer_support_ratio = min_layer_support_ratio
        self.seed = seed

    def build(self, communities: list[LayerCommunity], *, snapshot_time, run_id: str, universe_count: int) -> list[ThemeCandidate]:
        """Build a consensus graph from independent mechanism-family support.

        Multiple highly related layers in one family may describe the same
        mechanism. A family therefore contributes at most its strongest layer
        weight to a pair. This prevents activity-heavy families from dominating,
        while allowing a genuine two-family theme to pass a 0.35 threshold when
        the configured universe contains roughly five families.
        """
        pair_layers = defaultdict(set)
        pair_family_weight = defaultdict(dict)

        configured_family_capacity = defaultdict(float)
        for layer, raw_weight in self.layer_weights.items():
            weight = float(raw_weight)
            if weight <= 0:
                continue
            family = self.family_map.get(layer, layer)
            configured_family_capacity[family] = max(configured_family_capacity[family], weight)
        total_family_capacity = sum(configured_family_capacity.values()) or 1.0

        for community in communities:
            if community.is_market_mode:
                continue
            layer = community.layer_name
            family = self.family_map.get(layer, layer)
            weight = float(self.layer_weights.get(layer, 1.0))
            if weight <= 0:
                continue
            for left, right in combinations(sorted(set(community.members)), 2):
                pair = (left, right)
                if layer in pair_layers[pair]:
                    continue
                pair_layers[pair].add(layer)
                previous = pair_family_weight[pair].get(family, 0.0)
                pair_family_weight[pair][family] = max(previous, weight)

        accepted = {}
        for pair, family_weights in pair_family_weight.items():
            if len(family_weights) < self.min_distinct_families:
                continue
            normalized_score = sum(family_weights.values()) / total_family_capacity
            if normalized_score >= self.min_consensus_score:
                accepted[pair] = min(1.0, normalized_score)

        if not accepted:
            return []

        edges = pd.DataFrame(
            [(left, right, score) for (left, right), score in accepted.items()],
            columns=["src_id", "dst_id", "weight"],
        )
        detector = LeidenCommunityDetector(
            resolution=1.0,
            seed=self.seed,
            min_members=self.min_members,
            market_mode_max_member_ratio=self.market_mode_max_member_ratio,
        )
        consensus = detector.detect(
            edges,
            layer_id=-1,
            layer_name="consensus",
            snapshot_time=snapshot_time,
            universe_count=universe_count,
        )

        result = []
        for index, community in enumerate(consensus, start=1):
            members = set(community.members)
            internal_pairs = [pair for pair in accepted if pair[0] in members and pair[1] in members]
            if not internal_pairs:
                continue

            layer_counts = defaultdict(int)
            for pair in internal_pairs:
                for layer in pair_layers[pair]:
                    layer_counts[layer] += 1
            supporting_layers = sorted(
                layer for layer, count in layer_counts.items()
                if count / len(internal_pairs) >= self.min_layer_support_ratio
            )
            families = sorted({self.family_map.get(layer, layer) for layer in supporting_layers})
            if len(families) < self.min_distinct_families:
                continue

            consensus_score = sum(accepted[pair] for pair in internal_pairs) / len(internal_pairs)
            size_penalty = min(1.0, 8.0 / max(len(members), 1))
            structure_score = min(1.0, consensus_score * (0.6 + 0.4 * size_penalty))
            digest = hashlib.sha1(",".join(map(str, sorted(members))).encode()).hexdigest()[:10]
            instance = f"{pd.Timestamp(snapshot_time).strftime('%Y%m%dT%H%M%S')}_{digest}"
            result.append(
                ThemeCandidate(
                    instance,
                    f"{run_id}_new_{index:04d}",
                    snapshot_time,
                    tuple(sorted(members)),
                    tuple(supporting_layers),
                    tuple(families),
                    float(consensus_score),
                    float(structure_score),
                    len(members) / max(universe_count, 1),
                    community.is_market_mode,
                )
            )
        return result
