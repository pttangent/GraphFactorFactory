import pandas as pd

from graphfactorfactory.application.pit import build_point_in_time_panel
from graphfactorfactory.domain.config import BuildConfig


def test_late_row_never_moves_backward():
    events = pd.DataFrame({
        "trade_date": ["2025-01-02", "2025-01-02"],
        "symbol": ["A", "A"],
        "timestamp": pd.to_datetime(["2025-01-02 14:30Z", "2025-01-02 14:35Z"]),
        "available_time": pd.to_datetime(["2025-01-02 14:35Z", "2025-01-02 14:45Z"]),
        "close": [100.0, 101.0],
    })
    panel = build_point_in_time_panel(events, BuildConfig(), stale_tolerance="20min")
    assert (panel.source_available_time <= panel.decision_time).all()
    assert not panel.duplicated(["decision_time", "symbol"]).any()
