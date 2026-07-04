from __future__ import annotations

from collections import defaultdict
import pandas as pd
import igraph as ig
import leidenalg

from .models import LayerCommunity


class LeidenCommunityDetector:
    def __init__(self, *, resolution=1.0, seed=20260704, min_members=3, market_mode_max_member_ratio=0.15, sub_resolution=1.4, min_subcommunity_size=30):
        self.resolution = resolution
        self.seed = seed
        self.min_members = min_members
        self.market_mode_max_member_ratio = market_mode_max_member_ratio
        self.sub_resolution = sub_resolution
        self.min_subcommunity_size = min_subcommunity_size

    def detect(self, edges: pd.DataFrame, *, layer_id: int, layer_name: str, snapshot_time, universe_count: int) -> list[LayerCommunity]:
        if edges.empty:
            return []
        nodes = sorted(set(edges.src_id.astype(int)) | set(edges.dst_id.astype(int)))
        index = {node: position for position, node in enumerate(nodes)}
        graph = ig.Graph(n=len(nodes), edges=[(index[int(a)], index[int(b)]) for a, b in zip(edges.src_id, edges.dst_id)], directed=False)
        graph.es["weight"] = edges.weight.astype(float).tolist()
        partition = leidenalg.find_partition(graph, leidenalg.RBConfigurationVertexPartition, weights="weight", resolution_parameter=self.resolution, seed=self.seed, n_iterations=-1)
        grouped = defaultdict(list)
        for node, community in zip(nodes, partition.membership):
            grouped[int(community)].append(int(node))
        result = []
        for community_id, members in sorted(grouped.items()):
            if len(members) < self.min_members:
                continue
            market_mode = len(members) / max(universe_count, 1) > self.market_mode_max_member_ratio
            result.append(LayerCommunity(snapshot_time, layer_id, layer_name, community_id, tuple(sorted(members)), float(partition.modularity), market_mode))
        return result

    def detect_hierarchy(self, edges: pd.DataFrame, *, layer_id: int, layer_name: str, snapshot_time, universe_count: int) -> tuple[list[LayerCommunity], list[LayerCommunity]]:
        parents = self.detect(edges, layer_id=layer_id, layer_name=layer_name, snapshot_time=snapshot_time, universe_count=universe_count)
        children = []
        next_id = 0
        for parent in parents:
            if len(parent.members) < self.min_subcommunity_size:
                continue
            induced = edges[edges.src_id.isin(parent.members) & edges.dst_id.isin(parent.members)]
            detector = LeidenCommunityDetector(resolution=self.sub_resolution, seed=self.seed + parent.community_id + 1, min_members=self.min_members, market_mode_max_member_ratio=1.0, sub_resolution=self.sub_resolution, min_subcommunity_size=self.min_subcommunity_size)
            for child in detector.detect(induced, layer_id=layer_id, layer_name=layer_name, snapshot_time=snapshot_time, universe_count=len(parent.members)):
                children.append(LayerCommunity(snapshot_time, layer_id, layer_name, next_id, child.members, child.modularity, False, parent.community_id))
                next_id += 1
        return parents, children
