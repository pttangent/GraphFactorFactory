import pandas as pd

from graphfactorfactory.themes.graph_state_clock import GraphStateClock


def edges(weight=0.8):
    return pd.DataFrame([
        {"layer_id": 1, "src_id": 1, "dst_id": 2, "weight": weight},
        {"layer_id": 1, "src_id": 2, "dst_id": 3, "weight": 0.7},
    ])


def test_duplicate_snapshot_keeps_same_state_index():
    clock = GraphStateClock()
    changed0, state0, hash0 = clock.observe(edges())
    changed1, state1, hash1 = clock.observe(edges())
    assert changed0 is True
    assert changed1 is False
    assert state0 == state1 == 0
    assert hash0 == hash1


def test_weight_change_advances_state_index():
    clock = GraphStateClock()
    clock.observe(edges())
    changed, state, _ = clock.observe(edges(0.81))
    assert changed is True
    assert state == 1
