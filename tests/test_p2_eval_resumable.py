from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from p2_eval_streaming import EVAL_CONTRACT_VERSION, evaluate_feature_root, merge_evaluation_states
from p2_exact_eval import evaluate_intraday_frame_exact
from p2_pit_features import evaluate_intraday_frame


def test_exact_intraday_kernel_matches_legacy_with_nan_inf_ties_and_constant():
    rng = np.random.default_rng(31)
    count = 48
    frame = pd.DataFrame(
        {
            "date": ["2026-01-02"] * count,
            "decision_time": [pd.Timestamp("2026-01-02T14:30:00Z")] * count,
            "layer_id": ["3"] * count,
            "scale": ["30m"] * count,
            "level": ["B50"] * count,
            "daily_pressure_score": rng.normal(size=count),
            "daily_underreaction_score": rng.integers(-4, 5, size=count).astype(float),
            "daily_consensus_score": rng.normal(size=count),
            "late_confirmation_score_z": np.ones(count),
            "target_5m": rng.normal(size=count),
            "target_15m": rng.normal(size=count),
            "pit_audit_pass": [True] * count,
        }
    )
    frame.loc[[1, 5, 9], "daily_pressure_score"] = np.nan
    frame.loc[[3, 7], "daily_underreaction_score"] = np.inf
    frame.loc[[2, 5, 11, 12], "target_5m"] = np.nan
    frame.loc[[0, 3, 7, 18, 21], "target_15m"] = np.nan

    sort_columns = ["score", "target"]
    actual = evaluate_intraday_frame_exact(frame).sort_values(sort_columns).reset_index(drop=True)
    expected = evaluate_intraday_frame(frame).sort_values(sort_columns).reset_index(drop=True)
    assert actual[sort_columns].equals(expected[sort_columns])
    assert actual["sample_count"].tolist() == expected["sample_count"].tolist()
    np.testing.assert_allclose(actual["rank_ic"], expected["rank_ic"], rtol=1e-12, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(actual["top_minus_bottom"], expected["top_minus_bottom"], rtol=1e-12, atol=1e-12, equal_nan=True)


def _write_intraday_partition(root: Path, date: str) -> None:
    path = (
        root
        / f"date={date}"
        / "layer_id=1"
        / "scale=5m"
        / "intraday_relation_features.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    values = list(range(40)) * 2
    pd.DataFrame(
        {
            "date": [date] * 80,
            "decision_time": pd.to_datetime(
                [f"{date}T14:30:00Z"] * 40 + [f"{date}T14:35:00Z"] * 40,
                utc=True,
            ),
            "layer_id": ["1"] * 80,
            "scale": ["5m"] * 80,
            "level": ["B50"] * 80,
            "daily_pressure_score": values,
            "target_5m": [value / 1000 for value in values],
            "pit_audit_pass": [True] * 80,
        }
    ).to_parquet(path, index=False, row_group_size=25)


def test_p2_eval_reuses_partition_shards(tmp_path: Path):
    features = tmp_path / "features"
    _write_intraday_partition(features, "2026-01-02")
    _write_intraday_partition(features, "2026-01-05")
    out = tmp_path / "eval" / "202601"

    first = evaluate_feature_root(features, out, "intraday", workers=2, csv_mode="none")
    metric_dir = out / "intraday_alpha_metrics.parquet"
    parts = sorted(metric_dir.glob("part-*.parquet"))
    mtimes = {path.name: path.stat().st_mtime_ns for path in parts}
    assert first["evaluation_contract_version"] == EVAL_CONTRACT_VERSION
    assert first["metric_shards"] == 2
    assert first["reused_shards"] == 0
    assert not (out / "intraday_alpha_metrics.csv").exists()

    (out / "manifest.json").unlink()
    (out / "intraday_alpha_summary.csv").unlink()
    second = evaluate_feature_root(features, out, "intraday", workers=2, csv_mode="none")
    assert second["reused_shards"] == 2
    assert {path.name: path.stat().st_mtime_ns for path in parts} == mtimes
    assert len(pd.read_parquet(metric_dir)) == 4


def test_global_summary_merges_monthly_states_without_feature_rescan(tmp_path: Path):
    root = tmp_path / "intraday_relation_eval"
    columns = {
        "score": ["daily_pressure_score"],
        "target": ["target_5m"],
        "layer_id": ["1"],
        "scale": ["5m"],
        "level": ["B50"],
        "snapshots": [2],
        "sample_count": [80],
        "rank_ic_sum": [1.5],
        "rank_ic_count": [2],
        "spread_sum": [0.3],
        "spread_count": [2],
        "positive_count": [2],
    }
    for scope, date in (("202601", "2026-01-02"), ("202602", "2026-02-02")):
        directory = root / scope
        directory.mkdir(parents=True)
        frame = pd.DataFrame({"date": [date], **columns})
        frame.to_parquet(directory / "intraday_alpha_summary_state.parquet", index=False)

    out = root / "global"
    result = merge_evaluation_states(root, out, "intraday")
    summary = pd.read_csv(out / "intraday_alpha_summary.csv")
    assert result["global_input_mode"] == "monthly_summary_state_merge_no_raw_feature_rescan"
    assert result["input_states"] == 2
    assert summary.loc[0, "days"] == 2
    assert summary.loc[0, "snapshots"] == 4
    assert summary.loc[0, "mean_rank_ic"] == 0.75
    assert summary.loc[0, "mean_spread"] == 0.15
