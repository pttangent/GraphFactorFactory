from __future__ import annotations

from collections import defaultdict
from itertools import combinations
import hashlib
import pandas as pd

from .community import LeidenCommunityDetector
from .models import LayerCommunity, ThemeCandidate


class ConsensusThemeBuilder:
    def __init__(
        self,
        *,
        layer_weights=None,
        family_map=None,
        min_consensus_score=0.35,
        min_members=3,
        min_distinct_families=2,
        market_mode_max_member_ratio=0.15,
        min_layer_support_ratio=0.10,
        seed=20260704,
    ):
        self.layer_weights = layer_weights or {}
        self.family_map = family_map or {}
        self.min_consensus_score = min_consensus_score
        self.min_members = min_members
        self.min_distinct_families = min_distinct_families
        self.market_mode_max_member_ratio = market_mode_max_member_ratio
        self.min_layer_support_ratio = min_layer_support_ratio
        self.seed = seed

    def build(self, communities: list[LayerCommunity], *, snapshot_time, run_id: str, universe_count: int) -> list[ThemeCandidate]:
        """Build themes from genuinely cross-family co-membership edges.

        The previous implementation admitted every single-layer pair because the
        default layer weight (1.0) exceeded the default threshold (0.35). It then
        labelled an entire Leiden community as supported by any layer appearing
        on any internal pair. That made the consensus graph a layer union and
        produced giant, apparently 12/13-layer themes.

        Here an edge must be supported by distinct mechanism families before it
        enters the graph. The score is normalized to [0, 1] by total configured
        layer weight, so ``min_consensus_score`` has stable semantics.
        """
        pair_weight = defaultdict(float)
        pair_layers = defaultdict(set)
        pair_families = defaultdict(set)

        total_weight = sum(float(weight) for weight in self.layer_weights.values() if float(weight) > 0)
        if total_weight <= 0:
            total_weight = 1.0

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
                # A layer may contribute at most once to a pair even if upstream
                # input accidentally contains duplicate communities.
                if layer in pair_layers[pair]:
                    continue
                pair_weight[pair] += weight
                pair_layers[pair].add(layer)
                pair_families[pair].add(family)

        accepted = {}
        for pair, raw_weight in pair_weight.items():
            families = pair_families[pair]
            normalized_score = raw_weight / total_weight
            if len(families) < self.min_distinct_families:
                continue
            if normalized_score < self.min_consensus_score:
                continue
            accepted[pair] = normalized_score

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
                layer
                for layer, count in layer_counts.items()
                if count / len(internal_pairs) >= self.min_layer_support_ratio
            )
            families = sorted({self.family_map.get(layer, layer) for layer in supporting_layers})
            if len(families) < self.min_distinct_families:
                continue

            scores = [accepted[pair] for pair in internal_pairs]
            consensus_score = sum(scores) / len(scores)
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
