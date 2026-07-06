import pandas as pd
import pytest

from graphfactorfactory.application.correlation import reciprocal_correlation_graph
from graphfactorfactory.application.pit import decision_grid
from graphfactorfactory.application.lsh import strict_degree_cap
from graphfactorfactory.domain.config import BuildConfig
from graphfactorfactory.infrastructure.schemas import EDGE_SCHEMA, NODE_SCHEMA, SNAPSHOT_SCHEMA
from graphfactorfactory.infrastructure.writer import _arrow_table
from graphfactorfactory.themes.production_replay import infer_frame_minutes

def test_strict_degree_cap():
    edges=[(0,1,.99,1,1),(0,2,.98,1,1),(1,2,.97,1,1),(2,3,.96,1,1)]
    kept=strict_degree_cap(edges,1)
    degree={}
    for left,right,*_ in kept:
        degree[left]=degree.get(left,0)+1
        degree[right]=degree.get(right,0)+1
    assert max(degree.values(),default=0)<=1

def test_actual_cadence():
    times=['2026-06-16T13:30:00Z','2026-06-16T13:35:00Z','2026-06-16T13:40:00Z']
    assert infer_frame_minutes(times,15)==5


def test_decision_grid_stops_at_regular_session_close():
    events = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-06-16T13:30:00Z", "2026-06-16T19:55:00Z"], utc=True
            ),
            "available_time": pd.to_datetime(
                ["2026-06-16T13:31:00Z", "2026-06-16T20:15:00Z"], utc=True
            ),
        }
    )
    decisions = decision_grid(events, BuildConfig(frequency="5min"))
    assert len(decisions) == 78
    assert decisions[0] == pd.Timestamp("2026-06-16T13:35:00Z")
    assert decisions[-1] == pd.Timestamp("2026-06-16T20:00:00Z")


def test_phase0_schemas_preserve_scale_and_parameter_contract():
    common = {
        "lookback_minutes",
        "scale_role",
        "decision_step_minutes",
        "top_k",
        "degree_cap",
        "minimum_similarity",
    }
    assert common <= set(EDGE_SCHEMA.names)
    assert common <= set(NODE_SCHEMA.names)
    assert common <= set(SNAPSHOT_SCHEMA.names)
    assert {
        "node_coverage",
        "isolated_node_ratio",
        "degree_cap_saturation",
        "transform",
    } <= set(SNAPSHOT_SCHEMA.names)


def test_parquet_writer_rejects_silent_detail_loss():
    frame = pd.DataFrame({"decision_time": [], "undeclared_diagnostic": []})
    with pytest.raises(ValueError, match="undeclared_diagnostic"):
        _arrow_table(frame, EDGE_SCHEMA)


def test_exact_correlation_enforces_both_endpoint_caps():
    values = [[1.0, 0.0], [0.99, 0.01], [0.98, 0.02], [0.97, 0.03]]
    _, kept, _ = reciprocal_correlation_graph(
        values,
        BuildConfig(top_k=3, degree_cap=1, minimum_similarity=-1.0),
    )
    degree = {}
    for left, right, *_ in kept:
        degree[left] = degree.get(left, 0) + 1
        degree[right] = degree.get(right, 0) + 1
    assert max(degree.values(), default=0) <= 1


def test_phase0_decisions_are_partitioned_into_small_dynamic_chunks():
    from graphfactorfactory.application.pipeline import _partition_decisions

    decisions = list(range(11))
    chunks = _partition_decisions(decisions, chunk_size=3)
    assert chunks == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10]]
