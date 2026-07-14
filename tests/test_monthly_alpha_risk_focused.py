from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import generate_monthly_alpha_report_with_risk_focused as focused


def test_focused_scope_keeps_only_frozen_b50_intraday_hypotheses(tmp_path: Path):
    report = tmp_path / "report"
    (report / "p0_eval").mkdir(parents=True)
    pd.DataFrame(
        [
            {"layer_id": "4", "layer_name": "volume_expansion", "scale": "15m", "level": "B50", "score": "daily_underreaction_score", "target": "target_15m", "candidate_pass": True},
            {"layer_id": "4", "layer_name": "volume_expansion", "scale": "15m", "level": "B35", "score": "daily_underreaction_score", "target": "target_15m", "candidate_pass": True},
            {"layer_id": "9", "layer_name": "unlisted_layer", "scale": "15m", "level": "B50", "score": "daily_underreaction_score", "target": "target_15m", "candidate_pass": True},
            {"layer_id": "1", "layer_name": "return_corr_raw_1m", "scale": "15m", "level": "B50", "score": "daily_consensus_score", "target": "target_30m", "candidate_pass": True},
            {"layer_id": "1", "layer_name": "return_corr_raw_1m", "scale": "15m", "level": "B50", "score": "daily_consensus_score", "target": "target_60m", "candidate_pass": True},
        ]
    ).to_csv(report / "p2_intraday_scorecard.csv", index=False)
    pd.DataFrame(
        [
            {"layer_id": "1", "layer_name": "return_corr_raw_1m", "scale": "15m", "feature": "p0_edge_spillover_signal", "target": "target_15m", "candidate_pass": True},
            {"layer_id": "1", "layer_name": "return_corr_raw_1m", "scale": "15m", "feature": "p0_edge_spillover_sum", "target": "target_30m", "candidate_pass": True},
            {"layer_id": "1", "layer_name": "return_corr_raw_1m", "scale": "30m", "feature": "p0_edge_spillover_signal", "target": "target_15m", "candidate_pass": True},
        ]
    ).to_csv(report / "p0_eval" / "p0_eval_combo_scorecard.csv", index=False)

    under, corr = focused.focused_scopes(report)
    assert under == {("4", "15m", "target_15m")}
    assert corr == {("1", "15m", "target_30m")}
    assert focused.focused_p0_scope(report) == {
        ("1", "15m", "p0_edge_spillover_signal", "target_15m"),
        ("1", "15m", "p0_edge_spillover_sum", "target_30m"),
    }


def test_focused_worker_computes_only_b50_and_supports_snapshot_stride(tmp_path: Path):
    path = tmp_path / "features.parquet"
    rows = []
    for minute in (30, 35, 40, 45):
        for level in ("B50", "B35"):
            for index in range(40):
                own = (index - 20) / 20
                pressure = np.sin(index / 8)
                rows.append(
                    {
                        "decision_time": pd.Timestamp(f"2026-01-02T14:{minute}:00Z"),
                        "level": level,
                        "expected_pressure_z": pressure,
                        "target_pre_response_z": own,
                        "daily_underreaction_score": pressure - own,
                        "daily_consensus_score": pressure,
                        "target_15m": pressure - own,
                        "pit_audit_pass": True,
                    }
                )
    pd.DataFrame(rows).to_parquet(path, index=False)
    checkpoint = tmp_path / "checkpoint.json"
    task = {
        "path": str(path), "checkpoint": str(checkpoint), "scope_hash": "focused-test",
        "date": "2026-01-02", "layer": "4", "scale": "15m",
        "under_targets": ["target_15m"], "corr_targets": [],
        "snapshot_stride": 2, "read_mode": "full", "max_full_bytes": 10**9,
    }
    result = focused._process_p2_focused(task)
    assert result["status"] == "computed"
    assert result["groups"] == 2
    assert result["rows"] == 80
    assert result["records"]
    assert {row["level"] for row in result["records"]} == {"B50"}
    reused = focused._process_p2_focused(task)
    assert reused["status"] == "reused"
