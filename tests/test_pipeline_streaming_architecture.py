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
from p2_parallel_runtime import bounded_thread_map_ordered
from p2_pit_core import iter_time_groups, merge_time_group_streams
from run_p1_sharded_parallel import resolved_workers
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
    assert resolved_workers(0, 24, 128.0, 24.0, 4.5) == 20
    assert scope_name("2026-01-02,2026-01-03") == "202601"
    assert scope_name("2026-01-31,2026-02-02") == "selected"
