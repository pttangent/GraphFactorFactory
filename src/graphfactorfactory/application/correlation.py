from __future__ import annotations

import numpy as np
from scipy import sparse

from graphfactorfactory.domain.config import BuildConfig


def reciprocal_correlation_graph(values: np.ndarray, config: BuildConfig):
    """Build an exact Pearson-equivalent reciprocal top-k graph.

    ``values`` must contain one standardized trajectory per row.  After L2
    normalization, the dot product is the Pearson correlation of the centered
    trajectories.  Unlike the generic LSH path, every pair is considered
    before reciprocal top-k and degree-cap pruning, matching StockNet's
    ReturnCorr semantics.
    """
    values = np.asarray(values, dtype=np.float32)
    node_count = values.shape[0]
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    normalized = np.divide(values, norms, out=np.zeros_like(values), where=norms > 1e-12)
    scores = normalized @ normalized.T
    np.fill_diagonal(scores, -np.inf)

    directed: dict[tuple[int, int], tuple[float, int]] = {}
    for index in range(node_count):
        row = scores[index]
        eligible = np.flatnonzero(np.isfinite(row) & (row >= config.minimum_similarity))
        if eligible.size == 0:
            continue
        count = min(config.top_k, eligible.size)
        selected_positions = np.argpartition(row[eligible], -count)[-count:]
        selected = eligible[selected_positions]
        selected = selected[np.argsort(row[selected])[::-1]]
        for rank, target in enumerate(selected.tolist(), start=1):
            directed[(index, int(target))] = (float(row[target]), rank)

    reciprocal = []
    for (left, right), (left_weight, left_rank) in directed.items():
        reverse = directed.get((right, left))
        if left < right and reverse is not None:
            right_weight, right_rank = reverse
            reciprocal.append((left, right, (left_weight + right_weight) / 2.0, left_rank, right_rank))

    incident: dict[int, list[tuple[float, int, int]]] = {}
    for left, right, weight, _, _ in reciprocal:
        incident.setdefault(left, []).append((weight, left, right))
        incident.setdefault(right, []).append((weight, left, right))
    keep: set[tuple[int, int]] = set()
    for node_edges in incident.values():
        for _, left, right in sorted(node_edges, reverse=True)[: config.degree_cap]:
            keep.add((left, right))
    kept = [edge for edge in reciprocal if (edge[0], edge[1]) in keep]

    rows: list[int] = []
    columns: list[int] = []
    weights: list[float] = []
    for left, right, weight, _, _ in kept:
        rows.extend((left, right))
        columns.extend((right, left))
        weights.extend((weight, weight))
    adjacency = sparse.csr_matrix(
        (weights, (rows, columns)), shape=(node_count, node_count), dtype=np.float32
    )
    return adjacency, kept, 0
