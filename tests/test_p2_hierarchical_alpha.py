from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "p2_hierarchical_alpha.py"
sys.path.insert(0, str(REPO_ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("p2_hierarchical_alpha", SCRIPT)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def _touch_membership(root: Path, date: str, layer: int, scale: str) -> None:
    path = (
        root
        / f"date={date}"
        / f"layer_id={layer}"
        / f"scale={scale}"
        / "theme_memberships.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"theme_id": [], "level": [], "member_id": []}
    ).to_parquet(path, index=False)


def test_worker_plan_uses_all_24_workers_with_88gb() -> None:
    plan = module._plan_workers(216, 24, 88.0)
    assert plan["effective_workers"] == 24
    assert 3.0 < plan["worker_memory_gb"] < 3.2
    assert plan["worker_memory_gb"] * 24 < 88.0


def test_full_panel_discovers_every_common_layer_scale(tmp_path: Path) -> None:
    dates = ["2026-01-06", "2026-01-07", "2026-01-08"]
    panel = [(1, "15m"), (1, "30m"), (2, "5m"), (2, "15m")]
    for date in dates:
        for layer, scale in panel:
            _touch_membership(tmp_path, date, layer, scale)

    rows = module._discover_full_panel(
        tmp_path,
        dates,
        {1: "return_corr", 2: "volume"},
        allow_missing=False,
    )

    assert len(rows) == len(dates) * len(panel)
    assert {(row["layer_id"], row["scale"]) for row in rows} == set(panel)


def test_full_panel_fails_on_silent_missing_partition(tmp_path: Path) -> None:
    dates = ["2026-01-06", "2026-01-07", "2026-01-08"]
    for date in dates:
        _touch_membership(tmp_path, date, 1, "15m")
    _touch_membership(tmp_path, dates[0], 2, "30m")
    _touch_membership(tmp_path, dates[1], 2, "30m")

    with pytest.raises(RuntimeError, match="incomplete three-day P1 panel"):
        module._discover_full_panel(
            tmp_path,
            dates,
            {},
            allow_missing=False,
        )


def test_source_has_no_focused_partition_whitelist() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "FOCUSED" not in text
    assert 'rglob("theme_memberships.parquet")' in text
    assert '"no_sampling": True' in text
