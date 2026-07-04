import pandas as pd
import pytest

from graphfactorfactory.application.labels import build_forward_labels


def test_labels_use_next_bar_entry_and_exact_exit():
    panel = pd.DataFrame({
        "decision_time": pd.date_range("2025-01-01 14:30", periods=5, freq="5min", tz="UTC"),
        "symbol": ["A"] * 5,
        "close": [100.0, 101.0, 103.0, 104.0, 108.0],
    })
    labels = build_forward_labels(panel, (5,))
    assert labels.loc[0, "label_5m"] == pytest.approx(103.0 / 101.0 - 1.0)
    assert pd.isna(labels.loc[3, "label_5m"])
