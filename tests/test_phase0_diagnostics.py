import pandas as pd

from graphfactorfactory.application.phase0_diagnostics import (
    StrongEdgeThresholds,
    compute_resonance_diagnostics,
    compute_snapshot_diagnostics,
    compute_temporal_diagnostics,
)


def _edges():
    return pd.DataFrame(
        [
            {"decision_time": "2026-06-09T13:30:00Z", "layer_id": 2, "lookback_minutes": 5, "src_id": 0, "dst_id": 1, "weight": 0.9},
            {"decision_time": "2026-06-09T13:30:00Z", "layer_id": 2, "lookback_minutes": 5, "src_id": 1, "dst_id": 2, "weight": 0.8},
            {"decision_time": "2026-06-09T13:31:00Z", "layer_id": 2, "lookback_minutes": 5, "src_id": 0, "dst_id": 1, "weight": 0.95},
            {"decision_time": "2026-06-09T13:31:00Z", "layer_id": 2, "lookback_minutes": 5, "src_id": 2, "dst_id": 3, "weight": 0.7},
            {"decision_time": "2026-06-09T13:31:00Z", "layer_id": 4, "lookback_minutes": 5, "src_id": 0, "dst_id": 1, "weight": 0.85},
        ]
    ).assign(decision_time=lambda x: pd.to_datetime(x.decision_time, utc=True))


def test_snapshot_diagnostics_include_strong_edges_and_coverage():
    edges = _edges()
    thresholds = StrongEdgeThresholds.fit(edges[edges.layer_id == 2])
    result = compute_snapshot_diagnostics(edges, universe_count=5, thresholds=thresholds)
    row = result[(result.layer_id == 2) & (result.decision_time.dt.minute == 30)].iloc[0]
    assert row.edge_count == 2
    assert row.active_nodes == 3
    assert row.node_coverage == 0.6
    assert 0.0 <= row.strong_edge_ratio_q90 <= 1.0
    assert row.weight_p95 >= row.weight_p50


def test_temporal_diagnostics_split_birth_death_and_persistence():
    edges = _edges()
    thresholds = StrongEdgeThresholds.fit(edges[edges.layer_id == 2])
    result = compute_temporal_diagnostics(edges[edges.layer_id == 2], thresholds)
    second = result.sort_values("decision_time").iloc[1]
    assert second.all_birth_rate == 0.5
    assert second.all_death_rate == 0.5
    assert second.all_persistence == 0.5


def test_resonance_counts_cross_layer_edge_support():
    edges = _edges()
    layers = pd.DataFrame(
        [
            {"layer_id": 2, "family": "activity"},
            {"layer_id": 4, "family": "trade_flow"},
        ]
    )
    result = compute_resonance_diagnostics(edges, layers)
    second = result.sort_values("decision_time").iloc[1]
    assert second.resonant_edges_ge2_layers == 1
    assert second.resonant_edges_ge2_families == 1
