from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from generate_monthly_alpha_report_with_risk import enrich


def _base_report(root: Path) -> Path:
    report = root / "monthly_alpha_report" / "202601"
    (report / "p0_eval").mkdir(parents=True)
    (report / "monthly_alpha_report.json").write_text(json.dumps({"month": "2026-01"}), encoding="utf-8")
    (report / "monthly_alpha_report.html").write_text(
        "<html><body><main><script id='monthly-alpha-report' type='application/json'>{}</script></main></body></html>",
        encoding="utf-8",
    )
    rows = []
    for level in ("B50", "B35"):
        for score, ic in (("daily_underreaction_score", 0.03), ("daily_consensus_score", -0.03)):
            rows.append(
                {
                    "layer_id": "1", "layer_name": "return_corr_raw_1m", "layer_family": "return_corr",
                    "scale": "15m", "level": level, "score": score, "target": "target_15m",
                    "mean_rank_ic": ic, "mean_spread": np.sign(ic) * 0.01,
                    "research_score": 90.0, "candidate_pass": True,
                }
            )
    intraday = pd.DataFrame(rows)
    intraday.to_csv(report / "p2_intraday_scorecard.csv", index=False)
    pd.DataFrame(columns=intraday.columns).to_csv(report / "p2_daily_scorecard.csv", index=False)
    p0 = pd.DataFrame(
        [{
            "layer_id": "1", "layer_name": "return_corr_raw_1m", "layer_family": "return_corr",
            "scale": "15m", "kind": "edge", "feature": "p0_edge_spillover_signal",
            "target": "target_15m", "mean_rank_ic": -0.02, "mean_spread": -0.01,
            "candidate_pass": True,
        }]
    )
    p0.to_csv(report / "p0_eval" / "p0_eval_combo_scorecard.csv", index=False)
    pd.concat(
        [intraday.assign(stage="p2_intraday", signal=intraday["score"]),
         p0.assign(stage="p0_eval", signal=p0["feature"], level="symbol")],
        ignore_index=True, sort=False,
    ).to_csv(report / "monthly_alpha_unified_scorecard.csv", index=False)
    return report


def _inputs(root: Path) -> tuple[Path, Path]:
    date = "2026-01-02"
    symbols = np.arange(40)
    own = np.linspace(-1.0, 1.0, 40)
    network = np.sin(np.linspace(-2.0, 2.0, 40))
    rows = []
    for minute in (30, 45):
        target = network - own
        for level in ("B50", "B35"):
            for index in symbols:
                rows.append(
                    {
                        "decision_time": pd.Timestamp(f"{date}T14:{minute}:00Z"), "level": level,
                        "expected_pressure_z": network[index], "target_pre_response_z": own[index],
                        "daily_underreaction_score": target[index], "daily_consensus_score": 2 * network[index],
                        "target_15m": target[index], "pit_audit_pass": True,
                    }
                )
    feature_dir = root / "intraday_relation_features" / f"date={date}" / "layer_id=1" / "scale=15m"
    feature_dir.mkdir(parents=True)
    pd.DataFrame(rows).sort_values("decision_time").to_parquet(feature_dir / "intraday_relation_features.parquet", index=False)

    labels_root = root / "labels"
    label_dir = labels_root / f"date={date}"
    label_dir.mkdir(parents=True)
    t0, t1 = pd.Timestamp(f"{date}T14:15:00Z"), pd.Timestamp(f"{date}T14:30:00Z")
    label_rows = []
    for index in symbols:
        label_rows += [
            {"decision_time": t0, "symbol_id": int(index), "label_15m": float(own[index]),
             "label_entry_time_15m": t0 + pd.Timedelta(minutes=1), "label_exit_time_15m": t1},
            {"decision_time": t1, "symbol_id": int(index), "label_15m": float(network[index] - own[index]),
             "label_entry_time_15m": t1 + pd.Timedelta(minutes=1), "label_exit_time_15m": t1 + pd.Timedelta(minutes=16)},
        ]
    pd.DataFrame(label_rows).sort_values("decision_time").to_parquet(label_dir / "labels.parquet", index=False)

    p0_dir = root / "p0_edge_spillover" / f"date={date}" / "layer_id=1" / "scale=15m"
    p0_dir.mkdir(parents=True)
    pd.DataFrame(
        {"decision_time": [t1] * 40, "dst_id": symbols, "p0_edge_spillover_signal": network,
         "target_15m": network - own, "pit_audit_pass": [True] * 40}
    ).to_parquet(p0_dir / "p0_edge_spillover_features.parquet", index=False)

    p1_root = root / "p1"
    tree_dir = p1_root / f"date={date}" / "layer_id=1" / "scale=15m"
    tree_dir.mkdir(parents=True)
    pd.DataFrame(
        {"parent_level": ["B50"] * 4, "child_level": ["B35"] * 4,
         "split_mode": ["passthrough", "passthrough", "topk_3", "topk_3"],
         "parent_size": [20, 30, 50, 50], "child_size": [20, 30, 25, 25],
         "child_share": [1.0, 1.0, 0.5, 0.5]}
    ).to_parquet(tree_dir / "theme_tree_edges.parquet", index=False)
    return labels_root, p1_root


def test_four_risk_audits_are_embedded_without_raw_parquet(tmp_path: Path):
    report = _base_report(tmp_path)
    labels_root, p1_root = _inputs(tmp_path)
    result = enrich(tmp_path, "2026-01", report_dir=report, labels_root=labels_root, p1_root=p1_root, progress=0)
    payload = json.loads((report / "monthly_alpha_report.json").read_text(encoding="utf-8"))
    audit = payload["alpha_falsification_risk_audit"]
    assert "risk1_underreaction_reversal_proxy" in audit
    assert "risk2_return_corr_own_return_proxy" in audit
    assert audit["risk3_missing_data_median_fill_artifact"]["status"].startswith("unresolved")
    assert audit["risk4_b50_b35_non_independence"]["status"] == "not_independent_confirmed"
    assert audit["risk4_b50_b35_non_independence"]["p1_passthrough_community_redundancy"]["exact_size_passthrough_edges"] == 2
    assert (report / "risk1_underreaction_ablation.csv").exists()
    assert (report / "risk2_return_corr_p0_proxy_audit.csv").exists()
    assert "Alpha 反证与代理风险审计" in (report / "monthly_alpha_report.html").read_text(encoding="utf-8")
    with zipfile.ZipFile(result["bundle"]) as archive:
        names = archive.namelist()
    assert "alpha_falsification_risk_audit.json" in names
    assert not any(name.endswith(".parquet") for name in names)


def test_b50_launcher_only_changes_public_level_and_reporting_policy():
    source = (ROOT / "scripts" / "run_full_alpha_streaming_6m_b50_primary.py").read_text(encoding="utf-8")
    assert "GFF_RESEARCH_LEVELS" in source
    assert 'command.extend(["--levels", LEVELS])' in source
    assert "generate_monthly_alpha_report_with_risk_parallel.py" in source
    assert "monthly_alpha_report" in source
    for forbidden in ("label_", "target_", "rank_ic", "top_minus_bottom", "pit_audit_pass"):
        assert forbidden not in source
