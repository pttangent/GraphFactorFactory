from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from generate_p0_eval_report import Gates, generate_report


def _write_eval_scope(root: Path) -> Path:
    eval_dir = root / "p0_alpha" / "202601"
    metrics_dir = eval_dir / "p0_alpha_metrics.parquet"
    metrics_dir.mkdir(parents=True)
    rows = []
    days = ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"]
    for day_index, date in enumerate(days):
        for snapshot in range(4):
            decision_time = pd.Timestamp(f"{date}T14:{30 + snapshot:02d}:00Z")
            rows.extend(
                [
                    {
                        "date": date,
                        "decision_time": decision_time,
                        "kind": "node",
                        "layer_id": "1",
                        "scale": "15m",
                        "feature": "p0_total_weight_sum",
                        "target": "label_5m",
                        "sample_count": 100,
                        "rank_ic": 0.025 + day_index * 0.001,
                        "top_minus_bottom": 0.010 + snapshot * 0.0001,
                    },
                    {
                        "date": date,
                        "decision_time": decision_time,
                        "kind": "edge",
                        "layer_id": "4",
                        "scale": "15m",
                        "feature": "p0_edge_spillover_signal",
                        "target": "target_15m",
                        "sample_count": 95,
                        "rank_ic": -0.022 - day_index * 0.001,
                        "top_minus_bottom": -0.008 - snapshot * 0.0001,
                    },
                    {
                        "date": date,
                        "decision_time": decision_time,
                        "kind": "node",
                        "layer_id": "2",
                        "scale": "5m",
                        "feature": "p0_total_edge_count",
                        "target": "label_5m",
                        "sample_count": 100,
                        "rank_ic": 0.001 if snapshot % 2 == 0 else -0.001,
                        "top_minus_bottom": 0.0002 if snapshot % 2 == 0 else -0.0002,
                    },
                ]
            )
    metrics = pd.DataFrame(rows)
    metrics.iloc[: len(metrics) // 2].to_parquet(metrics_dir / "part-a.parquet", index=False, row_group_size=7)
    metrics.iloc[len(metrics) // 2 :].to_parquet(metrics_dir / "part-b.parquet", index=False, row_group_size=9)

    summary = (
        metrics.groupby(["kind", "feature", "target", "layer_id", "scale"], sort=False)
        .agg(
            days=("date", "nunique"),
            snapshots=("decision_time", "count"),
            sample_count=("sample_count", "sum"),
            mean_rank_ic=("rank_ic", "mean"),
            mean_spread=("top_minus_bottom", "mean"),
            positive_period_rate=("top_minus_bottom", lambda values: (values > 0).mean()),
        )
        .reset_index()
    )
    summary.to_csv(eval_dir / "p0_alpha_summary.csv", index=False)
    (eval_dir / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "evaluation_contract_version": "p0-eval-pairwise-resumable-v3",
                "missing_data_semantics": "exact_pairwise_complete_by_validity_mask",
                "metric_rows": len(metrics),
            }
        ),
        encoding="utf-8",
    )
    return eval_dir


def test_report_builds_compact_bundle_and_accepts_directional_negative_alpha(tmp_path: Path):
    eval_dir = _write_eval_scope(tmp_path)
    output = tmp_path / "compact_report"
    result = generate_report(
        eval_dir,
        output,
        gates=Gates(
            min_abs_ic=0.015,
            min_direction_rate=0.75,
            min_days=4,
            min_day_coverage=1.0,
            min_periods=12,
            max_top3_abs_ic_share=0.80,
            require_ic_spread_sign_agreement=True,
        ),
        batch_size=11,
        progress_every=0,
    )

    candidates = pd.read_csv(output / "p0_eval_candidate_whitelist.csv")
    assert set(candidates["feature"]) == {"p0_total_weight_sum", "p0_edge_spillover_signal"}
    negative = candidates.loc[candidates["feature"] == "p0_edge_spillover_signal"].iloc[0]
    assert negative["direction"] == "negative"
    assert negative["mean_rank_ic"] < 0
    assert negative["mean_spread"] < 0

    scorecard = pd.read_csv(output / "p0_eval_combo_scorecard.csv")
    weak = scorecard.loc[scorecard["feature"] == "p0_total_edge_count"].iloc[0]
    assert weak["candidate_pass"] in (False, "False")
    assert "abs_ic" in weak["gate_failures"]

    bundle = Path(result["bundle"])
    assert bundle.exists()
    assert bundle.stat().st_size < 1_000_000
    with zipfile.ZipFile(bundle) as archive:
        names = set(archive.namelist())
    assert "p0_eval_report.md" in names
    assert "p0_eval_candidate_whitelist.csv" in names
    assert not any(name.startswith("part-") for name in names)
    assert "p0_alpha_metrics.parquet" not in names

    report = (output / "p0_eval_report.md").read_text(encoding="utf-8")
    assert "p0_graph_state_features.parquet" in report
    manifest = json.loads((output / "p0_eval_report_manifest.json").read_text(encoding="utf-8"))
    assert manifest["raw_metrics_copied"] is False
    assert manifest["source_metric_rows_scanned"] == 48


def test_report_refuses_inconsistent_evaluator_summary(tmp_path: Path):
    eval_dir = _write_eval_scope(tmp_path)
    summary_path = eval_dir / "p0_alpha_summary.csv"
    summary = pd.read_csv(summary_path)
    summary.loc[0, "mean_rank_ic"] += 0.01
    summary.to_csv(summary_path, index=False)

    with pytest.raises(AssertionError, match="does not match evaluator summary"):
        generate_report(
            eval_dir,
            tmp_path / "report",
            gates=Gates(min_days=1, min_periods=1, max_top3_abs_ic_share=1.0),
            progress_every=0,
        )
