from __future__ import annotations

import pandas as pd


IDENTITY_COLUMNS = {
    "decision_time",
    "symbol",
    "symbol_id",
    "source_timestamp",
    "source_available_time",
    "trade_date",
    "timestamp",
    "available_time",
    "frequency",
    "factor_set_version",
    "factor_semantics_version",
    "trade_semantics_version",
}


def audit_source_events(frame: pd.DataFrame) -> dict:
    required = {"symbol", "timestamp", "available_time"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"NodeFactorFactory input missing columns: {sorted(missing)}")
    timestamps = pd.to_datetime(frame["timestamp"], utc=True)
    available = pd.to_datetime(frame["available_time"], utc=True)
    invalid = int((available < timestamps).sum())
    duplicates = int(frame.duplicated(["symbol", "timestamp"]).sum())
    if invalid or duplicates:
        raise AssertionError({"available_before_timestamp": invalid, "duplicate_symbol_timestamp": duplicates})
    return {"rows": int(len(frame)), "available_before_timestamp": 0, "duplicate_symbol_timestamp": 0}


def audit_pit_panel(frame: pd.DataFrame) -> dict:
    required = {"decision_time", "symbol", "source_timestamp", "source_available_time"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"PIT frame missing columns: {sorted(missing)}")
    decision = pd.to_datetime(frame["decision_time"], utc=True)
    source_time = pd.to_datetime(frame["source_timestamp"], utc=True)
    available = pd.to_datetime(frame["source_available_time"], utc=True)
    violations = int((source_time > decision).sum() + (available > decision).sum())
    duplicates = int(frame.duplicated(["decision_time", "symbol"]).sum())
    if violations or duplicates:
        raise AssertionError({"lookahead_violations": violations, "duplicate_keys": duplicates})
    return {"rows": int(len(frame)), "lookahead_violations": 0, "duplicate_keys": 0}


def assert_prefix_invariance(builder, events: pd.DataFrame, decision_time) -> bool:
    decision_time = pd.Timestamp(decision_time)
    full = builder.build_snapshot(events, decision_time)
    prefix = builder.build_snapshot(events[pd.to_datetime(events["available_time"], utc=True) <= decision_time], decision_time)
    for key in ("edges", "node_features", "snapshots"):
        left = getattr(full, key).copy()
        right = getattr(prefix, key).copy()
        ignored = [column for column in ("elapsed_ms_total_snapshot",) if column in left.columns]
        if ignored:
            left = left.drop(columns=ignored)
            right = right.drop(columns=ignored)
        sort_columns = list(left.columns)
        left = left.sort_values(sort_columns).reset_index(drop=True)
        right = right.sort_values(sort_columns).reset_index(drop=True)
        pd.testing.assert_frame_equal(left, right, check_dtype=False, check_exact=False, rtol=1e-6, atol=1e-7)
    return True
