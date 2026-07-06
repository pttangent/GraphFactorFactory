import numpy as np
import pandas as pd

from graphfactorfactory.application.math_utils import trajectory
from graphfactorfactory.domain.layers import LAYER_BY_NAME


def _panel(minutes=30):
    timestamps = pd.date_range("2026-06-01 13:30:00+00:00", periods=minutes, freq="1min")
    rows = []
    for symbol, slope in (("AAA", 1.0), ("BBB", 1.1), ("CCC", -0.8)):
        for index, timestamp in enumerate(timestamps):
            value = slope * (index + 1) / 10000.0
            rows.append({
                "timestamp": timestamp,
                "available_time": timestamp,
                "symbol": symbol,
                "ret_1m": value,
                "log_ret_1m": np.log1p(value),
            })
    return pd.DataFrame(rows)


def test_raw_and_cross_sectional_return_corr_use_ret_1m():
    panel = _panel(30)
    universe = ["AAA", "BBB", "CCC"]
    raw, raw_points, raw_columns = trajectory(panel, LAYER_BY_NAME["return_corr_raw_1m"], universe, 20)
    residual, residual_points, residual_columns = trajectory(panel, LAYER_BY_NAME["return_corr_cross_sectional_1m"], universe, 20)
    assert raw.shape == residual.shape == (3, 30)
    assert raw_points == residual_points == 30
    assert raw_columns == residual_columns == ("ret_1m",)
    assert not np.allclose(raw, residual)


def test_rolling_5m_return_corr_is_derived_inside_gff():
    panel = _panel(30)
    vectors, points, columns = trajectory(
        panel,
        LAYER_BY_NAME["return_corr_cross_sectional_rolling_5m"],
        ["AAA", "BBB", "CCC"],
        20,
    )
    assert vectors.shape == (3, 26)
    assert points == 26
    assert columns == ("log_ret_1m",)
