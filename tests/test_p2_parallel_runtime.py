from __future__ import annotations

import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# The scheduler only needs these constants at import time. Stubbing avoids
# importing the full Pandas/PyArrow pipeline in this focused resource test.
stub = types.ModuleType("p2_alpha_pit_features")
stub.DEFAULT_HORIZONS = ["5m", "15m", "30m", "1d_open"]
stub.DEFAULT_INTRADAY_HORIZONS = ["5m", "15m", "30m"]
stub.PIT_CONTRACT_VERSION = "p2-pit-v2"
sys.modules.setdefault("p2_alpha_pit_features", stub)

from p2_parallel_runtime import (
    bounded_thread_map,
    bounded_thread_map_ordered,
    resolve_max_tasks_per_child,
)
from run_p2_24core_scheduler import build_plan


def test_24core_128gb_plan_uses_nested_threads_without_oom_process_count():
    plan = build_plan(24, 1.0, "balanced", 0, 128.0, 24.0)
    theme = plan["build-theme-returns"]
    relation = plan["relation-spillover"]

    assert (theme.workers, theme.inner_workers, theme.estimated_slots) == (6, 4, 24)
    assert (relation.workers, relation.inner_workers, relation.estimated_slots) == (6, 4, 24)
    assert max(stage.workers for stage in plan.values()) <= 24
    assert all(stage.estimated_peak_ram_gb <= 93.6 for stage in plan.values())


def test_safe_profile_leaves_more_ram_than_balanced():
    safe = build_plan(24, 1.0, "safe", 0, 128.0, 24.0)
    balanced = build_plan(24, 1.0, "balanced", 0, 128.0, 24.0)
    assert safe["build-theme-returns"].workers <= balanced["build-theme-returns"].workers
    assert safe["relation-spillover"].workers <= balanced["relation-spillover"].workers


def test_worker_recycling_has_unambiguous_semantics(monkeypatch):
    monkeypatch.delenv("GFF_MAX_TASKS_PER_CHILD", raising=False)
    assert resolve_max_tasks_per_child(None) == 8
    monkeypatch.setenv("GFF_MAX_TASKS_PER_CHILD", "6")
    assert resolve_max_tasks_per_child(None) == 6
    assert resolve_max_tasks_per_child(8) == 8
    assert resolve_max_tasks_per_child(1) == 1
    assert resolve_max_tasks_per_child(0) is None


def test_bounded_thread_map_does_not_materialize_entire_input():
    consumed: list[int] = []

    def source():
        for value in range(100):
            consumed.append(value)
            yield value

    iterator = bounded_thread_map(source(), 2, lambda value: value * 2, max_in_flight=2)
    first = next(iterator)
    assert first in {0, 2}
    assert len(consumed) <= 3
    list(iterator)
    assert len(consumed) == 100


def test_ordered_thread_map_keeps_order_and_bounded_reorder_window():
    consumed: list[int] = []

    def source():
        for value in range(20):
            consumed.append(value)
            yield value

    def slow(value: int) -> int:
        if value == 0:
            time.sleep(0.05)
        return value

    iterator = bounded_thread_map_ordered(source(), 4, slow, max_in_flight=4)
    first = next(iterator)
    assert first == 0
    assert len(consumed) <= 5
    assert [first, *list(iterator)] == list(range(20))
