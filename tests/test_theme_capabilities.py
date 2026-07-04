import pandas as pd

from graphfactorfactory.themes.consensus import ConsensusThemeBuilder
from graphfactorfactory.themes.models import LayerCommunity
from graphfactorfactory.themes.temporal import ThemeLifecycleTracker
from graphfactorfactory.themes.temporal_edges import TemporalEdgeConfig, TemporalEdgeReplay


def test_consensus_requires_distinct_families():
    communities = [
        LayerCommunity(pd.Timestamp("2025-01-01", tz="UTC"), 1, "price", 0, (1, 2, 3), 0.5),
        LayerCommunity(pd.Timestamp("2025-01-01", tz="UTC"), 2, "flow", 0, (1, 2, 3), 0.5),
    ]
    builder = ConsensusThemeBuilder(
        layer_weights={"price": 1.0, "flow": 1.0},
        family_map={"price": "price", "flow": "flow"},
        min_consensus_score=0.5,
        min_distinct_families=2,
    )
    themes = builder.build(communities, snapshot_time=pd.Timestamp("2025-01-01", tz="UTC"), run_id="test", universe_count=10)
    assert len(themes) == 1
    assert themes[0].source_families == ("flow", "price")


def test_lifecycle_continuation_preserves_path():
    from graphfactorfactory.themes.models import ThemeCandidate, LifecycleRecord
    previous = ThemeCandidate("a", "path-a", pd.Timestamp("2025-01-01", tz="UTC"), (1, 2, 3), ("price",), ("price",), 0.8, 0.7, 0.3)
    current = ThemeCandidate("b", "new", pd.Timestamp("2025-01-01 00:15", tz="UTC"), (1, 2, 3, 4), ("price",), ("price",), 0.8, 0.7, 0.4)
    prior_record = LifecycleRecord("path-a", "a", previous.snapshot_time, "birth", "active", 1, 15, 1.0, 1.0)
    assigned, records = ThemeLifecycleTracker(0.5).assign([current], [previous], {"a": prior_record}, timestamp=current.snapshot_time, frame_minutes=15)
    assert assigned[0].theme_path_id == "path-a"
    assert records[0].event_type == "continuation"


def test_temporal_edge_hysteresis_keeps_grace_frame():
    replay = TemporalEdgeReplay(TemporalEdgeConfig(0.7, 0.6, 1.0, 1))
    edges = pd.DataFrame([{"decision_time": pd.Timestamp("2025-01-01", tz="UTC"), "window_start": pd.Timestamp("2025-01-01", tz="UTC"), "window_end": pd.Timestamp("2025-01-01", tz="UTC"), "layer_id": 1, "src_id": 1, "dst_id": 2, "weight": 0.8, "src_rank": 1, "dst_rank": 1, "directed": False, "lag_bars": 0, "window_points": 12, "vector_dimension": 12}])
    assert len(replay.replay(edges, edges.iloc[0].decision_time)) == 1
    grace = replay.replay(edges.iloc[0:0], pd.Timestamp("2025-01-01 00:15", tz="UTC"))
    assert len(grace) == 1
    assert grace.iloc[0].temporal_status == "grace"
