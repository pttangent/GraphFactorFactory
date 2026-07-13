from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_b50_b35_theme_forest_streaming import TemporalConfig
from build_b50_b35_theme_forest_streaming_v2 import temporal_edges_fast
from daily_label_integration import _stream_merge_label_file
from p2_parallel_runtime import bounded_thread_map_ordered
from p2_p0_graph_alpha import Part as P0Part
from p2_p0_graph_alpha import graph_state_one
from p2_pit_core import iter_time_groups, merge_time_group_streams
from run_p1_sharded_parallel import P2_MEASURED_GB_PER_PROCESS, resolved_workers
from run_p2_24core_scheduler import build_plan, scope_name


def test_time_group_stream_keeps_split_snapshot_whole(tmp_path: Path):
    path = tmp_path / "groups.parquet"
    frame = pd.DataFrame({
        "decision_time": ["2026-01-02T14:30:00Z"] * 5 + ["2026-01-02T14:31:00Z"] * 4,
        "value": range(9),
    })
    frame.to_parquet(path, index=False, row_group_size=9)
    groups = list(iter_time_groups(path, batch_size=3))
    assert [len(group) for _, group in groups] == [5, 4]


def test_time_group_stream_rejects_cross_batch_time_reversal(tmp_path: Path):
    path = tmp_path / "unsorted.parquet"
    pd.DataFrame({
        "decision_time": ["2026-01-02T14:31:00Z", "2026-01-02T14:30:00Z"],
        "value": [1, 2],
    }).to_parquet(path, index=False, row_group_size=1)
    with pytest.raises(ValueError, match="not globally sorted"):
        list(iter_time_groups(path, batch_size=1))


def test_dual_time_stream_is_inner_join():
    t0 = pd.Timestamp("2026-01-02T14:30:00Z")
    t1 = pd.Timestamp("2026-01-02T14:31:00Z")
    left = [(t0, pd.DataFrame({"x": [0]})), (t1, pd.DataFrame({"x": [1]}))]
    right = [(t1, pd.DataFrame({"y": [1]}))]
    merged = list(merge_time_group_streams(left, right))
    assert len(merged) == 1 and merged[0][0] == t1


def test_ordered_bounded_threads_keep_snapshot_order():
    def slow(value: int) -> int:
        time.sleep((4 - value) * 0.005)
        return value

    output = list(bounded_thread_map_ordered(range(4), 4, slow, max_in_flight=4))
    assert output == [0, 1, 2, 3]


def test_fast_temporal_match_selects_maximum_overlap_successor():
    previous = [("a", {1, 2, 3, 4}), ("b", {8, 9})]
    current = [("x", {1, 2, 3, 5}), ("y", {4, 8, 9})]
    rows = temporal_edges_fast("t0", "t1", "3", "30m", previous, current, "B50", TemporalConfig())
    mapping = {row["src_theme_id"]: row["dst_theme_id"] for row in rows}
    assert mapping == {"a": "x", "b": "y"}


def test_24core_plans_and_month_scope_are_resource_bounded():
    plan = build_plan(24, 1.0, "balanced", 0, 128.0, 24.0)
    assert plan["build-theme-returns"].estimated_slots == 24
    assert plan["relation-spillover"].estimated_slots == 24
    assert all(stage.estimated_peak_ram_gb <= 93.6 for stage in plan.values())
    assert resolved_workers(0, 24, 128.0, 24.0, P2_MEASURED_GB_PER_PROCESS) == 24
    assert scope_name("2026-01-02,2026-01-03") == "202601"
    assert scope_name("2026-01-31,2026-02-02") == "selected"


def test_daily_label_injection_streams_batches_and_preserves_rows(tmp_path: Path):
    label_path = tmp_path / "labels.parquet"
    original = pd.DataFrame({
        "decision_time": pd.to_datetime([
            "2026-01-02T14:30:00Z",
            "2026-01-02T14:30:00Z",
            "2026-01-02T14:31:00Z",
            "2026-01-02T14:31:00Z",
            "2026-01-02T14:32:00Z",
            "2026-01-02T14:32:00Z",
        ], utc=True),
        "symbol_id": [1, 2, 1, 2, 1, 2],
        "label_5m": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    })
    original.to_parquet(label_path, index=False, row_group_size=3)
    daily = pd.DataFrame({
        "symbol_id": [1, 2],
        "label_1d_open": pd.Series([0.01, -0.02], dtype="float32"),
        "label_entry_date_1d_open": pd.Series(["2026-01-05", "2026-01-05"], dtype="string"),
        "label_exit_date_1d_open": pd.Series(["2026-01-05", "2026-01-05"], dtype="string"),
        "daily_label_execution_policy": pd.Series(["next_session_open", "next_session_open"], dtype="string"),
    })
    injected = [column for column in daily if column != "symbol_id"]
    updated, rows, batches = _stream_merge_label_file(label_path, daily, injected, batch_size=2)
    result = pd.read_parquet(label_path)
    assert (updated, rows, batches) == (1, 6, 4)
    assert len(result) == len(original)
    assert result["label_5m"].tolist() == original["label_5m"].tolist()
    assert result.groupby("symbol_id")["label_1d_open"].first().to_dict() == pytest.approx({1: 0.01, 2: -0.02})


def test_p0_graph_state_does_not_split_snapshot_at_row_group(tmp_path: Path):
    shard = tmp_path / "date=2026-01-02" / "layer_id=3" / "scale=30m" / "edges.parquet"
    shard.parent.mkdir(parents=True)
    frame = pd.DataFrame({
        "decision_time": pd.to_datetime(
            ["2026-01-02T14:30:00Z"] * 5 + ["2026-01-02T14:31:00Z"] * 4,
            utc=True,
        ),
        "layer_id": [3] * 9,
        "scale": ["30m"] * 9,
        "src_id": [1, 1, 2, 2, 3, 1, 2, 3, 4],
        "dst_id": [2, 3, 3, 4, 4, 2, 3, 4, 1],
        "weight": [0.1] * 9,
    })
    frame.to_parquet(shard, index=False, row_group_size=2)
    output_root = tmp_path / "out"
    result = graph_state_one(P0Part("2026-01-02", "3", "30m", shard), str(output_root), None)
    output = pd.read_parquet(output_root / "date=2026-01-02" / "layer_id=3" / "scale=30m" / "p0_graph_state_features.parquet")
    assert result["output_rows"] == 2
    assert output["edge_count"].tolist() == [5, 4]
