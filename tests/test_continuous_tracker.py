from types import SimpleNamespace

from graphfactorfactory.themes.continuous_tracker import ContinuousThemeTracker, ContinuousTrackingConfig


def _theme(members):
    return SimpleNamespace(
        members=tuple(members),
        families=("price", "flow"),
        layers=("return_corr",),
        structure_score=0.7,
    )


def test_repeated_state_only_increments_timestamp_age():
    tracker = ContinuousThemeTracker(ContinuousTrackingConfig(method="stocknet_j035"))
    ids = tracker.step([_theme([1, 2, 3])], timestamp="t0", graph_state_hash="A")
    tracker.step([_theme([1, 2, 3])], timestamp="t1", graph_state_hash="A")
    path = tracker.paths[ids[0]]
    assert path.age_states == 1
    assert path.age_timestamps == 2


def test_adjacent_state_inherits_lifecycle():
    tracker = ContinuousThemeTracker(ContinuousTrackingConfig(method="stocknet_j035"))
    first = tracker.step([_theme([1, 2, 3, 4])], timestamp="t0", graph_state_hash="A")
    second = tracker.step([_theme([1, 2, 3, 5])], timestamp="t1", graph_state_hash="B")
    assert second == first


def test_overnight_transition_preserves_path_when_supported():
    tracker = ContinuousThemeTracker(ContinuousTrackingConfig(method="hybrid", overnight_threshold_addition=0.02))
    first = tracker.step([_theme([1, 2, 3, 4, 5])], timestamp="close", graph_state_hash="A")
    second = tracker.step(
        [_theme([1, 2, 3, 4, 8])],
        timestamp="next_open",
        graph_state_hash="B",
        transition_type="overnight",
    )
    assert second == first
    assert tracker.paths[first[0]].transitions[-1]["transition_type"] == "overnight"
