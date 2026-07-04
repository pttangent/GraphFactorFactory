from __future__ import annotations

from collections import defaultdict
import pandas as pd
import igraph as ig
import leidenalg
from dataclasses import replace

from .models import LayerCommunity


class LeidenCommunityDetector:
    def __init__(self, *, resolution=1.0, seed=20260704, min_members=3, market_mode_max_member_ratio=0.15, sub_resolution=1.4, min_subcommunity_size=30):
        self.resolution = resolution
        self.seed = seed
        self.min_members = min_members
        self.market_mode_max_member_ratio = market_mode_max_member_ratio
        self.sub_resolution = sub_resolution
        self.min_subcommunity_size = min_subcommunity_size

    def _detect_graph(self, graph: ig.Graph, nodes: list[int], layer_id: int, layer_name: str, snapshot_time, universe_count: int, seed: int, resolution: float, parent_community_id: int | None = None) -> tuple[list[LayerCommunity], dict[int, list[int]]]:
        if graph.vcount() == 0:
            return [], {}
        partition = leidenalg.find_partition(graph, leidenalg.RBConfigurationVertexPartition, weights="weight", resolution_parameter=resolution, seed=seed, n_iterations=2)
        grouped = defaultdict(list)
        grouped_indices = defaultdict(list)
        for index, (node, community) in enumerate(zip(nodes, partition.membership)):
            grouped[int(community)].append(int(node))
            grouped_indices[int(community)].append(index)
        result = []
        for community_id, members in sorted(grouped.items()):
            if len(members) < self.min_members:
                continue
            market_mode = len(members) / max(universe_count, 1) > self.market_mode_max_member_ratio if parent_community_id is None else False
            result.append(LayerCommunity(snapshot_time, layer_id, layer_name, community_id, tuple(sorted(members)), float(partition.modularity), market_mode, parent_community_id))
        return result, grouped_indices

    def detect(self, edges: pd.DataFrame, *, layer_id: int, layer_name: str, snapshot_time, universe_count: int) -> list[LayerCommunity]:
        if edges.empty:
            return []
        src = edges.src_id.astype(int).values
        dst = edges.dst_id.astype(int).values
        weight = edges.weight.astype(float).values
        
        nodes_array = pd.unique(pd.concat([pd.Series(src), pd.Series(dst)]))
        nodes = sorted(nodes_array.tolist())
        index = pd.Series(range(len(nodes)), index=nodes)
        
        src_indices = index.loc[src].values
        dst_indices = index.loc[dst].values
        
        graph = ig.Graph(n=len(nodes), edges=list(zip(src_indices, dst_indices)), directed=False)
        graph.es["weight"] = weight.tolist()

        parents, _ = self._detect_graph(graph, nodes, layer_id, layer_name, snapshot_time, universe_count, self.seed, self.resolution)
        return parents

    def detect_hierarchy(self, edges: pd.DataFrame, *, layer_id: int, layer_name: str, snapshot_time, universe_count: int) -> tuple[list[LayerCommunity], list[LayerCommunity]]:
        if edges.empty:
            return [], []
        # Build graph ONCE using vectorized numpy/pandas instead of slow python zip iteration
        src = edges.src_id.astype(int).values
        dst = edges.dst_id.astype(int).values
        weight = edges.weight.astype(float).values
        
        nodes_array = pd.unique(pd.concat([pd.Series(src), pd.Series(dst)]))
        nodes = sorted(nodes_array.tolist())
        index = pd.Series(range(len(nodes)), index=nodes)
        
        src_indices = index.loc[src].values
        dst_indices = index.loc[dst].values
        
        graph = ig.Graph(n=len(nodes), edges=list(zip(src_indices, dst_indices)), directed=False)
        graph.es["weight"] = weight.tolist()

        # Pass 1: Parents
        parents, grouped_indices = self._detect_graph(graph, nodes, layer_id, layer_name, snapshot_time, universe_count, self.seed, self.resolution)
        
        # Pass 2: Children (Subgraphs)
        children = []
        next_id = 0
        for parent in parents:
            if len(parent.members) < self.min_subcommunity_size:
                continue
            
            # Use extremely fast C++ subgraph extraction!
            subgraph_indices = grouped_indices[parent.community_id]
            subgraph = graph.subgraph(subgraph_indices)
            subgraph_nodes = [nodes[i] for i in subgraph_indices]
            
            child_seed = self.seed + parent.community_id + 1
            child_communities, _ = self._detect_graph(subgraph, subgraph_nodes, layer_id, layer_name, snapshot_time, len(parent.members), child_seed, self.sub_resolution, parent.community_id)
            
            for child in child_communities:
                # Re-assign IDs sequentially
                child = replace(child, community_id=next_id)
                children.append(child)
                next_id += 1
                
        return parents, children
