from __future__ import annotations

from collections import defaultdict
import igraph as ig
import leidenalg
import pandas as pd

from .models import LayerCommunity


class TwoSliceLeidenDetector:
    def __init__(self, resolution=1.0, omega=0.10, seed=20260704, min_members=3, market_mode_max_member_ratio=0.15):
        self.resolution = resolution
        self.omega = omega
        self.seed = seed
        self.min_members = min_members
        self.market_mode_max_member_ratio = market_mode_max_member_ratio
        self.previous_edges = {}

    def detect(self, edges, *, layer_id, layer_name, snapshot_time, universe_count):
        current = edges[["src_id", "dst_id", "weight"]].copy()
        previous = self.previous_edges.get(int(layer_id))
        self.previous_edges[int(layer_id)] = current.copy()
        if previous is None or previous.empty:
            return self._single(current, layer_id, layer_name, snapshot_time, universe_count)

        old_nodes = sorted(set(previous.src_id.astype(int)) | set(previous.dst_id.astype(int)))
        new_nodes = sorted(set(current.src_id.astype(int)) | set(current.dst_id.astype(int)))
        labels = [(0, node) for node in old_nodes] + [(1, node) for node in new_nodes]
        index = {label: idx for idx, label in enumerate(labels)}
        graph_edges, weights = [], []

        for row in previous.itertuples(index=False):
            graph_edges.append((index[(0, int(row.src_id))], index[(0, int(row.dst_id))]))
            weights.append(float(row.weight))
        for row in current.itertuples(index=False):
            graph_edges.append((index[(1, int(row.src_id))], index[(1, int(row.dst_id))]))
            weights.append(float(row.weight))
        for node in sorted(set(old_nodes) & set(new_nodes)):
            graph_edges.append((index[(0, node)], index[(1, node)]))
            weights.append(float(self.omega))

        graph = ig.Graph(n=len(labels), edges=graph_edges, directed=False)
        graph.es["weight"] = weights
        partition = leidenalg.find_partition(
            graph,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=self.resolution,
            seed=self.seed,
            n_iterations=2,
        )

        grouped = defaultdict(list)
        for idx, community_id in enumerate(partition.membership):
            slice_id, node = labels[idx]
            if slice_id == 1:
                grouped[int(community_id)].append(int(node))

        result = []
        next_id = 0
        for members in sorted(grouped.values(), key=lambda values: (-len(values), values)):
            members = tuple(sorted(set(members)))
            if len(members) < self.min_members:
                continue
            market_mode = len(members) / max(universe_count, 1) > self.market_mode_max_member_ratio
            result.append(LayerCommunity(snapshot_time, layer_id, layer_name, next_id, members, float(partition.modularity), market_mode))
            next_id += 1
        return result

    def _single(self, edges, layer_id, layer_name, snapshot_time, universe_count):
        if edges.empty:
            return []
        nodes = sorted(set(edges.src_id.astype(int)) | set(edges.dst_id.astype(int)))
        index = {node: idx for idx, node in enumerate(nodes)}
        graph = ig.Graph(
            n=len(nodes),
            edges=[(index[int(a)], index[int(b)]) for a, b in edges[["src_id", "dst_id"]].itertuples(index=False, name=None)],
            directed=False,
        )
        graph.es["weight"] = edges.weight.astype(float).tolist()
        partition = leidenalg.find_partition(
            graph,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=self.resolution,
            seed=self.seed,
            n_iterations=2,
        )
        grouped = defaultdict(list)
        for node, community_id in zip(nodes, partition.membership):
            grouped[int(community_id)].append(int(node))
        result = []
        for community_id, members in sorted(grouped.items()):
            if len(members) < self.min_members:
                continue
            market_mode = len(members) / max(universe_count, 1) > self.market_mode_max_member_ratio
            result.append(LayerCommunity(snapshot_time, layer_id, layer_name, community_id, tuple(sorted(members)), float(partition.modularity), market_mode))
        return result
