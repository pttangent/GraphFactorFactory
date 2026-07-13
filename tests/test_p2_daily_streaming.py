from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from p2_daily_streaming import build_daily_feature_frame_streaming
from p2_pit_features import build_daily_feature_frame


def _theme(timestamp: str, name: str) -> str:
    compact = pd.Timestamp(timestamp).strftime("%Y-%m-%dT%H%M%SZ")
    return f"ts={compact}|B50|{name}"


def test_streaming_daily_features_match_legacy_full_session(tmp_path: Path):
    date = "2026-01-02"
    times = pd.to_datetime(
        [
            "2026-01-02T14:30:00Z",
            "2026-01-02T15:00:00Z",
            "2026-01-02T15:30:00Z",
            "2026-01-02T16:00:00Z",
        ],
        utc=True,
    )
    rows = []
    themes: dict[tuple[int, str], str] = {}
    for time_index, decision_time in enumerate(times):
        for episode_index, name in enumerate(("alpha", "beta")):
            theme_id = _theme(str(decision_time), name)
            themes[(time_index, name)] = theme_id
            signal = float((episode_index + 1) * (time_index + 1))
            rows.append(
                {
                    "date": date,
                    "decision_time": decision_time,
                    "layer_id": "3",
                    "scale": "30m",
                    "level": "B50",
                    "dst_theme_id": theme_id,
                    "signal": signal,
                    "signal_sum": signal * 2.0,
                    "absolute_signal_sum": abs(signal) * 2.0,
                    "positive_signal_sum": max(signal * 2.0, 0.0),
                    "negative_signal_sum": min(signal * 2.0, 0.0),
                    "positive_source_count": 2,
                    "negative_source_count": 0,
                    "relation_edge_count": 2,
                    "relation_strength_mean": 0.25 + 0.05 * time_index,
                    "dst_past_eq_15m": 0.01 * (episode_index + time_index),
                    "dst_past_available_time_15m": decision_time - pd.Timedelta(minutes=1),
                    "target_1d_open": 0.02 * (episode_index + 1),
                    "target_entry_date_1d_open": "2026-01-05",
                    "target_exit_date_1d_open": "2026-01-05",
                }
            )
    source = pd.DataFrame(rows).sort_values("decision_time", kind="mergesort").reset_index(drop=True)
    source_path = tmp_path / "relation_spillover_signals.parquet"
    source.to_parquet(source_path, index=False, row_group_size=3)

    temporal_rows = []
    for name in ("alpha", "beta"):
        for index in range(len(times) - 1):
            temporal_rows.append(
                {
                    "level": "B50",
                    "src_theme_id": themes[(index, name)],
                    "dst_theme_id": themes[(index + 1, name)],
                    "src_time": times[index],
                    "dst_time": times[index + 1],
                }
            )
    temporal = pd.DataFrame(temporal_rows)

    legacy = build_daily_feature_frame(source.copy(), "15m", 60, temporal.copy())
    streaming = build_daily_feature_frame_streaming(source_path, "15m", 60, temporal.copy())

    sort_columns = ["level", "theme_episode_id"]
    legacy = legacy.sort_values(sort_columns).reset_index(drop=True)
    streaming = streaming.sort_values(sort_columns).reset_index(drop=True)
    assert streaming[sort_columns].equals(legacy[sort_columns])

    numeric = [
        "signal_sum",
        "absolute_signal_sum",
        "positive_signal_sum",
        "negative_signal_sum",
        "positive_source_count",
        "negative_source_count",
        "relation_edge_count",
        "relation_strength_mean",
        "late_signal_sum",
        "late_abs_signal_sum",
        "late_absolute_share",
        "daily_pressure_score",
        "daily_consensus_score",
        "late_confirmation_score",
        "daily_underreaction_score",
        "target_1d_open",
    ]
    for column in numeric:
        np.testing.assert_allclose(
            pd.to_numeric(streaming[column], errors="coerce"),
            pd.to_numeric(legacy[column], errors="coerce"),
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
            err_msg=column,
        )

    for column in [
        "first_time",
        "last_time",
        "session_close",
        "feature_time",
        "target_entry_date_1d_open",
        "target_exit_date_1d_open",
        "feature_contract",
        "daily_underreaction_status",
        "pit_audit_pass",
    ]:
        assert streaming[column].astype(str).tolist() == legacy[column].astype(str).tolist(), column
