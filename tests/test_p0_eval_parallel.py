from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from p2_p0_eval_streaming import evaluate_p0_streaming


def _write_node_partition(root: Path, date: str) -> None:
    path = (
        root
        / "p0_node_features"
        / f"date={date}"
        / "layer_id=1"
        / "scale=5m"
        / "p0_node_features.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    times = pd.to_datetime(
        [f"{date}T14:30:00Z"] * 40 + [f"{date}T14:35:00Z"] * 40,
        utc=True,
    )
    values = list(range(40)) * 2
    frame = pd.DataFrame(
        {
            "decision_time": times,
            "layer_id": ["1"] * 80,
            "scale": ["5m"] * 80,
            "symbol_id": list(range(40)) * 2,
            "p0_total_weight_sum": values,
            "label_5m": [value / 1000 for value in values],
            "pit_audit_pass": [True] * 80,
        }
    )
    frame.to_parquet(path, index=False, row_group_size=25)


def test_p0_eval_keeps_metrics_partitioned_and_parallel(tmp_path: Path):
    root = tmp_path / "p2"
    _write_node_partition(root, "2026-01-02")
    _write_node_partition(root, "2026-01-05")
    out = root / "p0_alpha" / "202601"

    result = evaluate_p0_streaming(
        root,
        out,
        workers=2,
        month="2026-01",
        csv_mode="none",
    )

    metrics_path = out / "p0_alpha_metrics.parquet"
    assert result["workers"] == 2
    assert result["metric_shards"] == 2
    assert result["parallel_summary_reduction"] is True
    assert result["serial_global_dataframe_concat"] is False
    assert metrics_path.is_dir()
    assert not (out / "p0_alpha_metrics.csv").exists()

    metrics = pd.read_parquet(metrics_path)
    summary = pd.read_csv(out / "p0_alpha_summary.csv")
    assert len(metrics) == 4
    assert summary.loc[0, "days"] == 2
    assert summary.loc[0, "snapshots"] == 4

    resumed = evaluate_p0_streaming(
        root,
        out,
        workers=2,
        month="2026-01",
        csv_mode="none",
        skip_existing=True,
    )
    assert resumed["status"] == "skipped"
