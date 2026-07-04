import pandas as pd

from graphfactorfactory.themes.temporal_edges_v2 import (
    LayerRelativeTemporalReplay,
    LayerTemporalPolicy,
)
from graphfactorfactory.themes.temporal_community import TwoSliceLeidenDetector


TS0 = pd.Timestamp("2026-01-07 15:10", tz="UTC")
TS1 = pd.Timestamp("2026-01-07 15:11", tz="UTC")


def _edges(ts, rows):
    base = []
    for src, dst, weight in rows:
        base.append({
            "decision_time": ts,
            "window_start": ts,
            "window_end": ts,
            "layer_id": 1,
            "src_id": src,
            "dst_id": dst,
            "weight": weight,
            "src_rank": 1,
            "dst_rank": 1,
            "directed": False,
            "lag_bars": 0,
            "window_points": 12,
            "vector_dimension": 12,
        })
    return pd.DataFrame(base)


def test_temporal_replay_separates_observed_and_prior():
    replay = LayerRelativeTemporalReplay(
        default=LayerTemporalPolicy(
            enter_quantile=0.0,
            exit_quantile=0.0,
            prior_lambda=0.2,
            smoothing_alpha=1.0,
            grace_frames=1,
        )
    )
    first = replay.replay(_edges(TS0, [(1, 2, 0.8)]), TS0)
    assert first.iloc[0].raw_observed_weight == 0.8
    assert first.iloc[0].temporal_prior_weight == 0.0

    grace = replay.replay(_edges(TS1, []), TS1)
    assert grace.iloc[0].temporal_status == "prior_only"
    assert grace.iloc[0].raw_observed_weight == 0.0
    assert grace.iloc[0].window_points == 0


def test_two_slice_detector_preserves_stable_cluster_identity():
    detector = TwoSliceLeidenDetector(
        resolution=1.0,
        omega=0.35,
        min_members=2,
        market_mode_max_member_ratio=0.9,
    )
    first = detector.detect(
        _edges(TS0, [(1, 2, 1.0), (2, 3, 1.0), (4, 5, 1.0), (5, 6, 1.0)]),
        layer_id=1,
        layer_name="test",
        snapshot_time=TS0,
        universe_count=10,
    )
    second = detector.detect(
        _edges(TS1, [(1, 2, 1.0), (2, 3, 0.9), (4, 5, 1.0), (5, 6, 0.9)]),
        layer_id=1,
        layer_name="test",
        snapshot_time=TS1,
        universe_count=10,
    )
    first_sets = {frozenset(item.members) for item in first}
    second_sets = {frozenset(item.members) for item in second}
    assert frozenset({1, 2, 3}) in first_sets
    assert frozenset({1, 2, 3}) in second_sets
    assert frozenset({4, 5, 6}) in second_sets
