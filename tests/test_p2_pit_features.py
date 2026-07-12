from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import p2_alpha_pit_features as pit


def test_canonical_theme_path_removes_complete_timestamp():
    raw = pd.Series(["ts=2026-01-05T144200_0000|layer=3|scale=30m|root.b50_001"])
    assert pit.canonical_theme_path(raw).iloc[0] == "layer=3|scale=30m|root.b50_001"


def test_load_labels_uses_actual_exit_time_not_nominal_shift(tmp_path: Path):
    t0 = pd.Timestamp("2026-01-05 14:31", tz="UTC")
    times = pd.date_range(t0, periods=30, freq="min")
    rows = []
    for t in times:
        rows.append({"decision_time": t, "label_entry_time": t + pd.Timedelta(minutes=5), "label_exit_time_15m": t + pd.Timedelta(minutes=20), "label_15m": float((t - t0).total_seconds() / 60), "symbol_id": 1})
    path = tmp_path / "labels.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    loaded = pit.load_labels(path, ["15m"])
    at_exit = loaded.loc[loaded.decision_time.eq(t0 + pd.Timedelta(minutes=20))].iloc[0]
    assert at_exit.past_label_15m == 0.0
    assert at_exit.past_exit_time_15m == t0 + pd.Timedelta(minutes=20)
    too_early = loaded.loc[loaded.decision_time.eq(t0 + pd.Timedelta(minutes=15))].iloc[0]
    assert pd.isna(too_early.past_label_15m)


def _signal_rows(decision_time: pd.Timestamp, values: list[float]) -> pd.DataFrame:
    rows = []
    for i, value in enumerate(values):
        rows.append({
            "date": decision_time.strftime("%Y-%m-%d"), "decision_time": decision_time,
            "layer_id": "3", "scale": "30m", "level": "B50",
            "dst_theme_id": f"ts={decision_time.strftime('%Y-%m-%dT%H%M%S')}_0000|layer=3|scale=30m|root.b50_{i:03d}",
            "signal": value, "signal_sum": value * 2, "absolute_signal_sum": abs(value) * 2,
            "positive_signal_sum": max(value, 0) * 2, "negative_signal_sum": min(value, 0) * 2,
            "positive_source_count": 2 if value > 0 else 0, "negative_source_count": 2 if value < 0 else 0,
            "relation_edge_count": 2, "relation_strength_mean": 0.5,
            "dst_past_eq_15m": value / 3, "dst_past_available_time_15m": decision_time,
            "target_5m": value / 10, "target_entry_time_5m": decision_time + pd.Timedelta(minutes=5),
            "target_exit_time_5m": decision_time + pd.Timedelta(minutes=10),
        })
    return pd.DataFrame(rows)


def test_intraday_normalization_is_snapshot_local_and_future_invariant():
    t1 = pd.Timestamp("2026-01-05 15:00", tz="UTC")
    t2 = pd.Timestamp("2026-01-05 15:01", tz="UTC")
    first = _signal_rows(t1, [-2, -1, 1, 2])
    base = pit.build_intraday_feature_frame(first, "15m")
    extended = pit.build_intraday_feature_frame(pd.concat([first, _signal_rows(t2, [-100, -50, 50, 100])]), "15m")
    columns = ["dst_theme_id", "daily_pressure_z", "daily_underreaction_score", "late_confirmation_score_z"]
    left = base[columns].sort_values("dst_theme_id").reset_index(drop=True)
    right = extended.loc[extended.decision_time.eq(t1), columns].sort_values("dst_theme_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)
    assert base.pit_audit_pass.all()


def test_intraday_rejects_target_entry_at_feature_time():
    t = pd.Timestamp("2026-01-05 15:00", tz="UTC")
    frame = _signal_rows(t, [-2, -1, 1, 2])
    frame["target_entry_time_5m"] = t
    output = pit.build_intraday_feature_frame(frame, "15m")
    assert not output.pit_audit_pass.all()


def test_evaluate_intraday_never_pools_decision_times():
    t1 = pd.Timestamp("2026-01-05 15:00", tz="UTC")
    t2 = pd.Timestamp("2026-01-05 15:01", tz="UTC")
    frame = pit.build_intraday_feature_frame(pd.concat([_signal_rows(t1, list(np.linspace(-2, 2, 40))), _signal_rows(t2, list(np.linspace(2, -2, 40)))]), "15m")
    metrics = pit.evaluate_intraday_frame(frame)
    assert set(metrics.decision_time) == {t1, t2}
    assert metrics.groupby("decision_time").size().min() > 0


def test_symmetric_relation_expansion_has_both_directions():
    t = pd.Timestamp("2026-01-05 15:00", tz="UTC")
    edges = pd.DataFrame([{"decision_time": t, "layer_id": "3", "scale": "30m", "level": "B50", "src_theme_id": "A", "dst_theme_id": "B", "relation_strength": 0.4}])
    expanded = pit.expand_symmetric_relations(edges)
    assert set(zip(expanded.src_theme_id, expanded.dst_theme_id)) == {("A", "B"), ("B", "A")}


def test_daily_full_session_aggregation_accepts_only_next_open_targets():
    t1 = pd.Timestamp("2026-01-05 15:00", tz="UTC")
    t2 = pd.Timestamp("2026-01-05 21:00", tz="UTC")
    frame = pd.concat([_signal_rows(t1, list(np.linspace(-2, 2, 40))), _signal_rows(t2, list(np.linspace(-1, 3, 40)))])
    frame["target_1d_open"] = np.linspace(-0.02, 0.02, len(frame))
    daily = pit.build_daily_feature_frame(frame, "15m", late_minutes=60)
    assert daily.feature_time.max() == t2
    assert daily.feature_contract.eq("end_of_day_next_open_execution").all()
    assert daily.pit_audit_pass.all()


def test_daily_rejects_close_start_label():
    t = pd.Timestamp("2026-01-05 21:00", tz="UTC")
    frame = _signal_rows(t, list(np.linspace(-2, 2, 40)))
    frame["target_1d"] = np.linspace(-0.02, 0.02, len(frame))
    with pytest.raises(ValueError, match="close-start"):
        pit.build_daily_feature_frame(frame, "15m", late_minutes=60)


def test_daily_label_builder_emits_next_open_executable_targets():
    import build_daily_labels as daily
    prices = pd.DataFrame({"date": pd.date_range("2026-01-05", periods=4, freq="B"), "stable_symbol_id": ["A"] * 4, "symbol": ["A"] * 4, "open": [100.0, 101.0, 102.0, 103.0], "close": [100.5, 101.5, 102.5, 103.5]})
    labels = daily.build_daily_labels(prices, max_horizon=3)
    first = labels.iloc[0]
    assert first.next_open_to_t1_close == pytest.approx(101.5 / 101.0 - 1)
    assert first.next_open_to_t2_close == pytest.approx(102.5 / 101.0 - 1)
    assert first.daily_execution_policy == "next_session_open"


def test_daily_label_integration_rejects_close_start_and_maps_next_open():
    import daily_label_integration as integration
    daily = pd.DataFrame({"date": ["2026-01-05"], "stable_symbol_id": ["A"], "next_trade_date": ["2026-01-06"], "t2_trade_date": ["2026-01-07"], "next_open_to_t1_close": [0.01], "next_open_to_t2_close": [0.02], "close_to_next_close": [0.03]})
    mapping = pd.DataFrame({"symbol": ["A"], "symbol_id": [1]})
    result = integration.prepare_daily_labels(daily, mapping)
    assert result.label_1d_open.iloc[0] == pytest.approx(0.01)
    assert result.label_2d_open.iloc[0] == pytest.approx(0.02)
    assert "label_1d" not in result
    assert result.label_entry_date_2d_open.iloc[0] == "2026-01-06"
    assert result.label_exit_date_2d_open.iloc[0] == "2026-01-07"
