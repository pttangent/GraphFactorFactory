from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_boundary_tasks_include_cross_month():
    module = load_script("monthly_orchestrator", "scripts/run_monthly_carryover_ab.py")
    dates = ["2025-01-30", "2025-01-31", "2025-02-03"]
    tasks = module.build_tasks({"random_seed": 7}, dates)
    assert [(task.date_from, task.date_to) for task in tasks] == [
        ("2025-01-30", "2025-01-31"),
        ("2025-01-31", "2025-02-03"),
    ]


def test_bridge_candidates_never_cross_layers():
    module = load_script("carryover_worker", "scripts/run_monthly_carryover_task.py")
    previous = [
        {"id": "p1", "layer": 1, "members": {1, 2, 3}, "core": {1, 2, 3}, "size": 3},
    ]
    current = [
        {"id": "wrong", "layer": 2, "members": {1, 2, 3}, "core": {1, 2, 3}, "size": 3},
        {"id": "right", "layer": 1, "members": {1, 2}, "core": {1, 2}, "size": 2},
    ]
    candidates = module.bridge_candidates(previous, current, entry=0.5)
    assert len(candidates) == 1
    assert candidates[0]["current_index"] == 1
    assert candidates[0]["layer"] == 1


def test_earliest_invalid_detects_missing_and_stale(tmp_path: Path):
    module = load_script("phase2_earliest", "scripts/run_phase2_from_earliest.py")
    phase1 = tmp_path / "phase1"
    phase2 = tmp_path / "phase2"
    for trade_date in ("2025-01-02", "2025-01-03"):
        source = phase1 / f"date={trade_date}" / "layer_communities.parquet"
        source.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"snapshot_time": []}).to_parquet(source)
    assert module.earliest_invalid(phase1, phase2) == "2025-01-02"

    for trade_date in ("2025-01-02", "2025-01-03"):
        day = phase2 / f"date={trade_date}"
        state = phase2 / "_state" / f"date={trade_date}"
        day.mkdir(parents=True, exist_ok=True)
        state.mkdir(parents=True, exist_ok=True)
        (day / "_SUCCESS").write_text("success")
        (state / "_SUCCESS").write_text("success")
    assert module.earliest_invalid(phase1, phase2) is None


def test_frozen_2025_config_has_seven_distinct_arms():
    config = json.loads((REPO_ROOT / "configs/monthly_carryover_ab_2025.json").read_text(encoding="utf-8"))
    assert list(config["arms"]) == ["A", "B", "C", "D9", "D11", "D13", "D15"]
    assert config["arms"]["D9"]["max_dormant_states"] == 3
    assert config["arms"]["D13"]["revival_fingerprint"] == 0.22
    assert config["arms"]["D15"]["revival_fingerprint"] == 0.30
    assert config["arms"]["D15"]["breadth_expansion"] == 0.10
