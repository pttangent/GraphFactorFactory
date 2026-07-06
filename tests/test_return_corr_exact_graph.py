import numpy as np

from graphfactorfactory.application.correlation import reciprocal_correlation_graph
from graphfactorfactory.domain.config import BuildConfig


def test_exact_correlation_graph_keeps_reciprocal_top_neighbors():
    values = np.array(
        [
            [-1.0, 0.0, 1.0, 2.0],
            [-2.0, 0.0, 2.0, 4.0],
            [2.0, 1.0, 0.0, -1.0],
            [0.0, 1.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    config = BuildConfig(top_k=2, degree_cap=2, minimum_similarity=0.5)
    adjacency, edges, bits = reciprocal_correlation_graph(values, config)

    assert bits == 0
    assert adjacency.shape == (4, 4)
    assert any({left, right} == {0, 1} and weight > 0.99 for left, right, weight, _, _ in edges)
    assert adjacency[0, 1] > 0.99
    assert adjacency[1, 0] > 0.99
    assert all(left != right for left, right, *_ in edges)


def test_exact_correlation_graph_is_deterministic():
    rng = np.random.default_rng(7)
    values = rng.normal(size=(20, 12)).astype(np.float32)
    config = BuildConfig(top_k=4, degree_cap=3, minimum_similarity=0.0)

    first = reciprocal_correlation_graph(values, config)[1]
    second = reciprocal_correlation_graph(values, config)[1]
    assert first == second
