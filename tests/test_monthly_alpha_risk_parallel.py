from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
TESTS = ROOT / "tests"
for directory in (SCRIPTS, TESTS):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import generate_monthly_alpha_report_with_risk_parallel as parallel
from test_monthly_alpha_risk_report import _base_report, _inputs


def _run(root: Path, report: Path, labels: Path, p1: Path, checkpoints: Path):
    parallel.configure(
        p2_root=root,
        month="2026-01",
        workers=2,
        tasks_per_child=2,
        checkpoint_root=checkpoints,
        read_mode="full",
        max_full_read_gb=1.0,
        memory_budget_gb=0.0,
        memory_expansion_factor=6.0,
        stream_worker_gb=1.0,
        max_in_flight=2,
        progress_every=0,
    )
    return parallel.risk.enrich(
        root,
        "2026-01",
        report_dir=report,
        labels_root=labels,
        p1_root=p1,
        progress=0,
    )


def test_parallel_risk_audit_is_resumable_and_numerically_stable(tmp_path: Path):
    report = _base_report(tmp_path)
    labels, p1 = _inputs(tmp_path)
    checkpoints = tmp_path / "checkpoints"

    first = _run(tmp_path, report, labels, p1, checkpoints)
    first_payload = json.loads((report / "monthly_alpha_report.json").read_text(encoding="utf-8"))
    first_scan = first_payload["alpha_falsification_risk_audit"]["source_scan"]
    first_risk1 = pd.read_csv(report / "risk1_underreaction_ablation.csv")
    first_risk2_p0 = pd.read_csv(report / "risk2_return_corr_p0_proxy_audit.csv")

    assert first_scan["intraday"]["computed"] > 0
    assert first_scan["p0"]["computed"] > 0
    assert first_scan["intraday"]["reused"] == 0
    assert list((checkpoints / "p2").glob("part-*.json"))
    assert list((checkpoints / "p0").glob("part-*.json"))

    second = _run(tmp_path, report, labels, p1, checkpoints)
    second_payload = json.loads((report / "monthly_alpha_report.json").read_text(encoding="utf-8"))
    second_scan = second_payload["alpha_falsification_risk_audit"]["source_scan"]
    second_risk1 = pd.read_csv(report / "risk1_underreaction_ablation.csv")
    second_risk2_p0 = pd.read_csv(report / "risk2_return_corr_p0_proxy_audit.csv")

    assert second_scan["intraday"]["reused"] == second_scan["intraday"]["tasks"]
    assert second_scan["p0"]["reused"] == second_scan["p0"]["tasks"]
    pd.testing.assert_frame_equal(first_risk1, second_risk1, check_dtype=False)
    pd.testing.assert_frame_equal(first_risk2_p0, second_risk2_p0, check_dtype=False)

    with zipfile.ZipFile(second["bundle"]) as archive:
        names = archive.namelist()
    assert not any("risk_audit_checkpoints" in name for name in names)
    assert not any(name.endswith(".parquet") for name in names)


def test_worker_count_is_public_and_memory_budget_can_cap_it(tmp_path: Path):
    tasks = []
    for index in range(24):
        path = tmp_path / f"part-{index}.parquet"
        path.write_bytes(b"0" * 1024)
        tasks.append({"path": str(path)})
    config = {
        "read_mode": "full",
        "max_full_bytes": 1024 * 1024,
        "memory_budget_gb": 4.0,
        "memory_expansion_factor": 1_000_000.0,
        "stream_worker_gb": 1.0,
    }
    effective, plan = parallel._effective_workers(tasks, 24, config)
    assert plan["requested_workers"] == 24
    assert 1 <= effective < 24

    source = (SCRIPTS / "generate_monthly_alpha_report_with_risk_parallel.py").read_text(encoding="utf-8")
    assert 'parser.add_argument("--workers"' in source
    assert 'pass 24 for a 24-core host' in source
    assert "daily_underreaction_score =" not in source
    assert "target_15m =" not in source
