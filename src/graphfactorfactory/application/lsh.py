from __future__ import annotations

import numpy as np
from scipy import sparse

from graphfactorfactory.domain.config import BuildConfig


def strict_degree_cap(edges: list[tuple[int, int, float, int, int]], degree_cap: int):
    if degree_cap <= 0 or not edges:
        return []
    degree: dict[int, int] = {}
    kept = []
    for edge in sorted(edges, key=lambda item: (-float(item[2]), int(item[0]), int(item[1]))):
        left, right = int(edge[0]), int(edge[1])
        if degree.get(left, 0) >= degree_cap or degree.get(right, 0) >= degree_cap:
            continue
        kept.append(edge)
        degree[left] = degree.get(left, 0) + 1
        degree[right] = degree.get(right, 0) + 1
    return kept


def reciprocal_lsh_graph(values: np.ndarray, config: BuildConfig):
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    values = np.divide(values, norms, out=np.zeros_like(values), where=norms > 1e-12)
    node_count, dimension = values.shape
    bits = 10 if node_count >= 4000 else (8 if node_count >= 1000 else 6)
    projections = np.random.default_rng(20260704 + dimension).standard_normal((dimension, bits), dtype=np.float32)
    signatures = ((values @ projections) > 0).astype(np.uint16)
    codes = np.sum(signatures * (1 << np.arange(bits, dtype=np.uint16)), axis=1)
    buckets: dict[int, list[int]] = {}
    for index, code in enumerate(codes.tolist()):
        buckets.setdefault(code, []).append(index)
    directed: dict[tuple[int, int], tuple[float, int]] = {}
    for index, code in enumerate(codes.tolist()):
        candidates = list(buckets[code])
        radius = 1
        while len(candidates) < config.top_k + 1 and radius <= 2:
            if radius == 1:
                for bit in range(bits):
                    candidates.extend(buckets.get(code ^ (1 << bit), ()))
            else:
                for first in range(bits):
                    for second in range(first + 1, bits):
                        candidates.extend(buckets.get(code ^ (1 << first) ^ (1 << second), ()))
            radius += 1
        candidate_ids = np.asarray(sorted(set(candidates)), dtype=np.int32)
        candidate_ids = candidate_ids[candidate_ids != index]
        if candidate_ids.size == 0:
            continue
        similarities = values[candidate_ids] @ values[index]
        count = min(config.top_k, len(similarities))
        chosen = np.argpartition(similarities, -count)[-count:]
        chosen = chosen[np.argsort(similarities[chosen])[::-1]]
        for rank, position in enumerate(chosen, start=1):
            weight = float(similarities[position])
            if weight >= config.minimum_similarity:
                directed[(index, int(candidate_ids[position]))] = (weight, rank)
    reciprocal = []
    for (left, right), (left_weight, left_rank) in directed.items():
        reverse = directed.get((right, left))
        if left < right and reverse is not None:
            right_weight, right_rank = reverse
            reciprocal.append((left, right, (left_weight + right_weight) / 2.0, left_rank, right_rank))
    kept = strict_degree_cap(reciprocal, config.degree_cap)
    rows, columns, weights = [], [], []
    for left, right, weight, _, _ in kept:
        rows.extend((left, right)); columns.extend((right, left)); weights.extend((weight, weight))
    adjacency = sparse.csr_matrix((weights, (rows, columns)), shape=(node_count, node_count), dtype=np.float32)
    return adjacency, kept, bits
