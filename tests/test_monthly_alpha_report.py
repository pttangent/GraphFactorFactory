from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from generate_monthly_alpha_report import EvalGates, generate_monthly_report
from generate_p0_eval_report import Gates as P0Gates

MONTH = "2026-01"
DATES = ["2026-01-02", "2026-01-05"]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _partition(root: Path, stage: str, date: str, filename: str, contract: str, frame: pd.DataFrame) -> None:
    directory = root / stage / f"date={date}" / "layer_id=1" / "scale=5m"
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(directory / filename, index=False)
    stage_name = {
        "theme_returns": "theme_returns",
        "relation_spillover": "relation_spillover",
        "intraday_relation_features": "intraday_relation_features",
        "daily_relation_features": "daily_relation_features",
    }[stage]
    _write_json(
        directory / "manifest.json",
        {
            "stage": stage_name,
            "stage_contract_version": contract,
            "status": "complete",
            "date": date,
            "layer_id": "1",
            "scale": "5m",
            "output_rows": len(frame),
        },
    )


def _write_p0_eval(root: Path) -> None:
    eval_dir = root / "p0_alpha" / "202601"
    metric_dir = eval_dir / "p0_alpha_metrics.parquet"
    metric_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for date in DATES:
        for minute in (30, 35):
            rows.append(
                {
                    "date": date,
                    "decision_time": pd.Timestamp(f"{date}T14:{minute}:00Z"),
                    "kind": "node",
                    "layer_id": "1",
                    "scale": "5m",
                    "feature": "p0_total_weight_sum",
                    "target": "label_5m",
                    "sample_count": 40,
                    "rank_ic": 0.02,
                    "top_minus_bottom": 0.001,
                }
            )
    pd.DataFrame(rows).to_parquet(metric_dir / "part-test.parquet", index=False)
    pd.DataFrame(
        [
            {
                "kind": "node",
                "feature": "p0_total_weight_sum",
                "target": "label_5m",
                "layer_id": "1",
                "scale": "5m",
                "days": 2,
                "snapshots": 4,
                "sample_count": 160,
                "mean_rank_ic": 0.02,
                "mean_spread": 0.001,
                "positive_period_rate": 1.0,
            }
        ]
    ).to_csv(eval_dir / "p0_alpha_summary.csv", index=False)
    _write_json(
        eval_dir / "manifest.json",
        {
            "stage": "p0_alpha_eval",
            "status": "complete",
            "evaluation_contract_version": "p0-eval-pairwise-resumable-v3",
            "metric_rows": 4,
        },
    )


def _write_p2_eval(root: Path, mode: str) -> None:
    eval_dir = root / f"{mode}_relation_eval" / "202601"
    eval_dir.mkdir(parents=True, exist_ok=True)
    snapshots = 4 if mode == "intraday" else 1
    target = "target_5m" if mode == "intraday" else "target_1d_open"
    states = []
    for date in DATES:
        states.append(
            {
                "date": date,
                "score": "daily_consensus_score",
                "target": target,
                "layer_id": "1",
                "scale": "5m",
                "level": "B50",
                "snapshots": snapshots,
                "sample_count": 40 * snapshots,
                "rank_ic_sum": 0.02 * snapshots,
                "rank_ic_count": snapshots,
                "spread_sum": 0.001 * snapshots,
                "spread_count": snapshots,
                "positive_count": snapshots,
            }
        )
    pd.DataFrame(states).to_parquet(eval_dir / f"{mode}_alpha_summary_state.parquet", index=False)
    pd.DataFrame(
        [
            {
                "score": "daily_consensus_score",
                "target": target,
                "layer_id": "1",
                "scale": "5m",
                "level": "B50",
                "days": 2,
                "snapshots": snapshots * 2,
                "sample_count": 40 * snapshots * 2,
                "mean_rank_ic": 0.02,
                "mean_spread": 0.001,
                "positive_period_rate": 1.0,
            }
        ]
    ).to_csv(eval_dir / f"{mode}_alpha_summary.csv", index=False)
    (eval_dir / f"{mode}_alpha_metrics.parquet").mkdir()
    _write_json(
        eval_dir / "manifest.json",
        {
            "stage": f"{mode}_feature_eval",
            "status": "complete",
            "mode": mode,
            "evaluation_contract_version": "p2-eval-resumable-partitioned-v3",
            "input_count": 2,
        },
    )


def _write_full_month(root: Path) -> None:
    _write_p0_eval(root)
    for date in DATES:
        theme = pd.DataFrame(
            {
                "decision_time": pd.to_datetime([f"{date}T14:30:00Z"] * 40, utc=True),
                "layer_id": ["1"] * 40,
                "scale": ["5m"] * 40,
                "level": ["B50"] * 40,
                "theme_id": [f"theme-{index}" for index in range(40)],
                "ret_eq_5m": [0.001] * 40,
                "ret_core_5m": [0.0015] * 40,
                "ret_top5_5m": [0.002] * 40,
            }
        )
        _partition(root, "theme_returns", date, "theme_returns.parquet", "theme-returns-stream-v3", theme)

        relation = pd.DataFrame(
            {
                "decision_time": pd.to_datetime([f"{date}T14:30:00Z"] * 40, utc=True),
                "layer_id": ["1"] * 40,
                "scale": ["5m"] * 40,
                "level": ["B50"] * 40,
                "signal": [0.01] * 40,
                "absolute_signal_sum": [0.02] * 40,
                "relation_edge_count": [5] * 40,
                "relation_strength_mean": [0.6] * 40,
                "positive_source_count": [4] * 40,
                "negative_source_count": [1] * 40,
            }
        )
        _partition(root, "relation_spillover", date, "relation_spillover_signals.parquet", "relation-spillover-stream-v3", relation)

        intraday_features = pd.DataFrame(
            {
                "date": [date] * 40,
                "decision_time": pd.to_datetime([f"{date}T14:30:00Z"] * 40, utc=True),
                "layer_id": ["1"] * 40,
                "scale": ["5m"] * 40,
                "level": ["B50"] * 40,
                "pit_audit_pass": [True] * 40,
            }
        )
        _partition(root, "intraday_relation_features", date, "intraday_relation_features.parquet", "intraday-relation-features-v3", intraday_features)

        daily_features = pd.DataFrame(
            {
                "date": [date] * 40,
                "layer_id": ["1"] * 40,
                "scale": ["5m"] * 40,
                "level": ["B50"] * 40,
                "pit_audit_pass": [True] * 40,
            }
        )
        _partition(root, "daily_relation_features", date, "daily_relation_features.parquet", "daily-relation-features-stream-v4", daily_features)
    _write_p2_eval(root, "intraday")
    _write_p2_eval(root, "daily")


def test_monthly_report_covers_p0_theme_relation_and_both_p2_modes(tmp_path: Path):
    root = tmp_path / "p2"
    _write_full_month(root)
    relaxed = {
        "min_abs_ic": 0.001,
        "min_direction_rate": 0.5,
        "min_days": 1,
        "min_day_coverage": 0.5,
        "max_top3_abs_ic_share": 1.0,
    }
    result = generate_monthly_report(
        root,
        MONTH,
        batch_size=20,
        progress_every=0,
        p0_gates=P0Gates(min_periods=1, **relaxed),
        intraday_gates=EvalGates(min_periods=1, **relaxed),
        daily_gates=EvalGates(min_periods=1, **relaxed),
    )
    assert result["status"] == "complete"
    report_dir = Path(result["report_dir"])
    payload = json.loads((report_dir / "monthly_alpha_report.json").read_text(encoding="utf-8"))
    assert payload["counts"]["p0_candidates"] == 1
    assert payload["counts"]["intraday_candidates"] == 1
    assert payload["counts"]["daily_candidates"] == 1
    assert {row["stage"] for row in payload["top_unified_candidates"]} == {"p0_eval", "p2_intraday", "p2_daily"}
    assert payload["scans"]["theme_returns"]["files"] == 2
    assert payload["scans"]["relation_spillover"]["files"] == 2
    assert "Theme Returns" in (report_dir / "monthly_alpha_report.html").read_text(encoding="utf-8")
    with zipfile.ZipFile(result["bundle"]) as archive:
        names = archive.namelist()
    assert "monthly_alpha_report.json" in names
    assert "monthly_alpha_report.html" in names
    assert "p0_eval/p0_eval_combo_scorecard.csv" in names
    assert not any(name.endswith(".parquet") for name in names)
    manifest = json.loads((report_dir / "monthly_alpha_report_manifest.json").read_text(encoding="utf-8"))
    assert manifest["raw_parquet_copied"] is False
