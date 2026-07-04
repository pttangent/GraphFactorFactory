import pandas as pd

from graphfactorfactory.themes.consensus import ConsensusThemeBuilder
from graphfactorfactory.themes.models import LayerCommunity
from graphfactorfactory.themes.pipeline import _leaf_communities
from graphfactorfactory.themes.temporal import ThemeLifecycleTracker
from graphfactorfactory.themes.temporal_edges import TemporalEdgeConfig, TemporalEdgeReplay

TS = pd.Timestamp("2025-01-01", tz="UTC")


def test_consensus_requires_distinct_families():
    communities = [
        LayerCommunity(TS, 1, "price", 0, (1, 2, 3), 0.5),
        LayerCommunity(TS, 2, "flow", 0, (1, 2, 3), 0.5),
    ]
    builder = ConsensusThemeBuilder(
        layer_weights={"price": 1.0, "flow": 1.0},
        family_map={"price": "price", "flow": "flow"},
        min_consensus_score=0.5,
        min_distinct_families=2,
    )
    themes = builder.build(communities, snapshot_time=TS, run_id="test", universe_count=10)
    assert len(themes) == 1
    assert themes[0].source_families == ("flow", "price")


def test_single_layer_pair_does_not_enter_consensus_union():
    communities = [LayerCommunity(TS, 1, "price", 0, (1, 2, 3), 0.5)]
    builder = ConsensusThemeBuilder(
        layer_weights={"price": 1.0, "flow": 1.0},
        family_map={"price": "price", "flow": "flow"},
        min_consensus_score=0.35,
        min_distinct_families=2,
    )
    assert builder.build(communities, snapshot_time=TS, run_id="test", universe_count=10) == []


def test_duplicate_layer_community_cannot_fake_cross_layer_support():
    communities = [
        LayerCommunity(TS, 1, "price", 0, (1, 2, 3), 0.5),
        LayerCommunity(TS, 1, "price", 1, (1, 2, 3), 0.5),
    ]
    builder = ConsensusThemeBuilder(
        layer_weights={"price": 1.0, "flow": 1.0},
        family_map={"price": "price", "flow": "flow"},
        min_consensus_score=0.35,
        min_distinct_families=2,
    )
    assert builder.build(communities, snapshot_time=TS, run_id="test", universe_count=10) == []


def test_consensus_uses_children_instead_of_parent_and_children():
    parent = LayerCommunity(TS, 1, "return_corr", 7, tuple(range(100)), 0.5)
    child_a = LayerCommunity(TS, 1, "return_corr", 20, tuple(range(50)), 0.6, False, 7)
    child_b = LayerCommunity(TS, 1, "return_corr", 21, tuple(range(50, 100)), 0.6, False, 7)
    selected = _leaf_communities([parent], [child_a, child_b])
    assert selected == [child_a, child_b]


def test_lifecycle_continuation_preserves_path():
    from graphfactorfactory.themes.models import ThemeCandidate, LifecycleRecord
    previous = ThemeCandidate("a", "path-a", TS, (1, 2, 3), ("price",), ("price",), 0.8, 0.7, 0.3)
    current = ThemeCandidate("b", "new", pd.Timestamp("2025-01-01 00:15", tz="UTC"), (1, 2, 3, 4), ("price",), ("price",), 0.8, 0.7, 0.4)
    prior_record = LifecycleRecord("path-a", "a", previous.snapshot_time, "birth", "active", 1, 15, 1.0, 1.0)
    assigned, records = ThemeLifecycleTracker(0.5).assign([current], [previous], {"a": prior_record}, timestamp=current.snapshot_time, frame_minutes=15)
    assert assigned[0].theme_path_id == "path-a"
    assert records[0].event_type == "continuation"


def test_temporal_edge_hysteresis_keeps_grace_frame():
    replay = TemporalEdgeReplay(TemporalEdgeConfig(0.7, 0.6, 1.0, 1))
    edges = pd.DataFrame([{"decision_time": TS, "window_start": TS, "window_end": TS, "layer_id": 1, "src_id": 1, "dst_id": 2, "weight": 0.8, "src_rank": 1, "dst_rank": 1, "directed": False, "lag_bars": 0, "window_points": 12, "vector_dimension": 12}])
    assert len(replay.replay(edges, TS)) == 1
    grace = replay.replay(edges.iloc[0:0], pd.Timestamp("2025-01-01 00:15", tz="UTC"))
    assert len(grace) == 1
    assert grace.iloc[0].temporal_status == "grace"
