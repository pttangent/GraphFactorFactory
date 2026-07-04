from __future__ import annotations

from collections import defaultdict
import hashlib
import pandas as pd

from .community import LeidenCommunityDetector
from .models import ThemeCandidate


class SparseConsensusBuilder:
    def __init__(self, family_map, min_score=0.25, min_families=2, min_members=3, seed=20260704):
        self.family_map = family_map
        self.min_score = min_score
        self.min_families = min_families
        self.min_members = min_members
        self.seed = seed
        self.family_count = max(len(set(family_map.values())), 1)

    def build(self, communities, support_edges, *, snapshot_time, run_id, universe_count):
        memberships = defaultdict(dict)
        layer_names = {}
        for community in communities:
            if community.is_market_mode:
                continue
            layer_id = int(community.layer_id)
            layer_names[layer_id] = community.layer_name
            token = (community.community_id, community.parent_community_id)
            for node in community.members:
                memberships[layer_id][int(node)] = token

        pair_layers = defaultdict(set)
        pair_families = defaultdict(set)
        for raw_layer_id, group in support_edges.groupby("layer_id", sort=False):
            layer_id = int(raw_layer_id)
            membership = memberships.get(layer_id)
            layer_name = layer_names.get(layer_id)
            if not membership or layer_name is None:
                continue
            family = self.family_map.get(layer_name, layer_name)
            src_group = group.src_id.astype(int).map(membership)
            dst_group = group.dst_id.astype(int).map(membership)
            keep = src_group.notna() & dst_group.notna() & src_group.eq(dst_group)
            for left, right in group.loc[keep, ["src_id", "dst_id"]].astype(int).itertuples(index=False, name=None):
                if left == right:
                    continue
                pair = (left, right) if left < right else (right, left)
                pair_layers[pair].add(layer_name)
                pair_families[pair].add(family)

        accepted = {
            pair: min(1.0, len(families) / self.family_count)
            for pair, families in pair_families.items()
            if len(families) >= self.min_families
            and len(families) / self.family_count >= self.min_score
        }
        if not accepted:
            return []

        edges = pd.DataFrame(
            [(left, right, score) for (left, right), score in accepted.items()],
            columns=["src_id", "dst_id", "weight"],
        )
        communities_out = LeidenCommunityDetector(
            resolution=1.0,
            seed=self.seed,
            min_members=self.min_members,
        ).detect(
            edges,
            layer_id=-1,
            layer_name="consensus",
            snapshot_time=snapshot_time,
            universe_count=universe_count,
        )

        result = []
        for index, community in enumerate(communities_out, start=1):
            members = set(community.members)
            internal = [pair for pair in accepted if pair[0] in members and pair[1] in members]
            if not internal:
                continue
            layers = sorted({layer for pair in internal for layer in pair_layers[pair]})
            families = sorted({self.family_map.get(layer, layer) for layer in layers})
            score = sum(accepted[pair] for pair in internal) / len(internal)
            structure = score * (0.6 + 0.4 * min(1.0, 8.0 / len(members)))
            digest = hashlib.sha1(",".join(map(str, sorted(members))).encode()).hexdigest()[:10]
            instance = f"{pd.Timestamp(snapshot_time).strftime('%Y%m%dT%H%M%S')}_{digest}"
            result.append(ThemeCandidate(
                instance,
                f"{run_id}_new_{index:04d}",
                snapshot_time,
                tuple(sorted(members)),
                tuple(layers),
                tuple(families),
                float(score),
                float(structure),
                len(members) / max(universe_count, 1),
                community.is_market_mode,
            ))
        return result
