from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from p2_p0_eval_streaming import EVAL_CONTRACT_VERSION, _evaluate_snapshot, evaluate_p0_streaming


def _legacy_snapshot(frame: pd.DataFrame, date: str, kind: str) -> pd.DataFrame:
    frame = frame.replace([np.inf, -np.inf], np.nan)
    features = [column for column in frame if column.startswith("p0_") and pd.api.types.is_numeric_dtype(frame[column])]
    targets = [column for column in frame if column.startswith(("label_", "target_")) and column.endswith("m")]
    rows = []
    for keys, subset in frame.groupby(["decision_time", "layer_id", "scale"], dropna=False, sort=False):
        for feature in features:
            for target in targets:
                values = subset[[feature, target]].dropna()
                if len(values) < 30:
                    continue
                q80 = values[feature].quantile(0.8)
                q20 = values[feature].quantile(0.2)
                rank_ic = (
                    np.nan
                    if values[feature].nunique() < 2 or values[target].nunique() < 2
                    else values[feature].rank().corr(values[target].rank())
                )
                rows.append(
                    {
                        "date": date,
                        "decision_time": keys[0],
                        "kind": kind,
                        "layer_id": keys[1],
                        "scale": keys[2],
                        "feature": feature,
                        "target": target,
                        "sample_count": len(values),
                        "rank_ic": rank_ic,
                        "top_minus_bottom": (
                            values.loc[values[feature] >= q80, target].mean()
                            - values.loc[values[feature] <= q20, target].mean()
                        ),
                    }
                )
    return pd.DataFrame(rows)


def test_pairwise_mask_vectorization_matches_legacy_with_missing_inf_ties_and_constant():
    rng = np.random.default_rng(19)
    count = 48
    frame = pd.DataFrame(
        {
            "decision_time": [pd.Timestamp("2026-01-02T14:30:00Z")] * count,
            "layer_id": ["1"] * count,
            "scale": ["5m"] * count,
            "p0_a": rng.normal(size=count),
            "p0_b": rng.integers(0, 6, size=count).astype(float),
            "p0_constant": np.ones(count),
            "label_5m": rng.normal(size=count),
            "target_15m": rng.normal(size=count),
            "pit_audit_pass": [True] * count,
        }
    )
    frame.loc[[1, 5, 9, 17], "p0_a"] = np.nan
    frame.loc[[3, 7], "p0_b"] = np.inf
    frame.loc[[2, 5, 11, 12], "label_5m"] = np.nan
    frame.loc[[0, 3, 7, 18, 21], "target_15m"] = np.nan

    key = ["feature", "target"]
    actual = _evaluate_snapshot(frame, "2026-01-02", "node").sort_values(key).reset_index(drop=True)
    expected = _legacy_snapshot(frame, "2026-01-02", "node").sort_values(key).reset_index(drop=True)
    assert actual[key].equals(expected[key])
    assert actual["sample_count"].tolist() == expected["sample_count"].tolist()
    np.testing.assert_allclose(actual["rank_ic"], expected["rank_ic"], rtol=1e-12, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(actual["top_minus_bottom"], expected["top_minus_bottom"], rtol=1e-12, atol=1e-12, equal_nan=True)


def test_mixed_partition_fails_fast():
    frame = pd.DataFrame(
        {
            "decision_time": [pd.Timestamp("2026-01-02T14:30:00Z")] * 40,
            "layer_id": ["1"] * 39 + ["2"],
            "scale": ["5m"] * 40,
            "p0_x": range(40),
            "label_5m": range(40),
            "pit_audit_pass": [True] * 40,
        }
    )
    with pytest.raises(ValueError, match="mixed or missing layer_id"):
        _evaluate_snapshot(frame, "2026-01-02", "node")


def _write_node_partition(root: Path, date: str) -> None:
    path = root / "p0_node_features" / f"date={date}" / "layer_id=1" / "scale=5m" / "p0_node_features.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    values = list(range(40)) * 2
    pd.DataFrame(
        {
            "decision_time": pd.to_datetime([f"{date}T14:30:00Z"] * 40 + [f"{date}T14:35:00Z"] * 40, utc=True),
            "layer_id": ["1"] * 80,
            "scale": ["5m"] * 80,
            "symbol_id": list(range(40)) * 2,
            "p0_total_weight_sum": values,
            "label_5m": [value / 1000 for value in values],
            "pit_audit_pass": [True] * 80,
        }
    ).to_parquet(path, index=False, row_group_size=25)


def test_p0_eval_is_partitioned_and_reuses_completed_shards(tmp_path: Path):
    root = tmp_path / "p2"
    _write_node_partition(root, "2026-01-02")
    _write_node_partition(root, "2026-01-05")
    out = root / "p0_alpha" / "202601"

    first = evaluate_p0_streaming(root, out, workers=2, month="2026-01", csv_mode="none")
    metrics_path = out / "p0_alpha_metrics.parquet"
    parts = sorted(metrics_path.glob("part-*.parquet"))
    mtimes = {path.name: path.stat().st_mtime_ns for path in parts}
    assert first["evaluation_contract_version"] == EVAL_CONTRACT_VERSION
    assert first["metric_shards"] == 2
    assert first["reused_shards"] == 0
    assert not (out / "p0_alpha_metrics.csv").exists()

    (out / "manifest.json").unlink()
    (out / "p0_alpha_summary.csv").unlink()
    second = evaluate_p0_streaming(root, out, workers=2, month="2026-01", csv_mode="none")
    assert second["reused_shards"] == 2
    assert {path.name: path.stat().st_mtime_ns for path in parts} == mtimes

    metrics = pd.read_parquet(metrics_path)
    summary = pd.read_csv(out / "p0_alpha_summary.csv")
    assert len(metrics) == 4
    assert summary.loc[0, "days"] == 2
    assert summary.loc[0, "snapshots"] == 4
