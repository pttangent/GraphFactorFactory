from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from graphfactorfactory.infrastructure.nodefactorfactory.parquet_source import ParquetNodeFactorSource

TIMEZONE = "America/New_York"
QA_DATE = "2026-06-16"
QA_START_ET = "10:00"
QA_END_ET = "10:15"
WARMUP_START_ET = "09:30"
EXPECTED_FRAMES = 16
DECISION_STEP_MINUTES = 1
MAX_LOOKBACK_MINUTES = 30


def build_time_pit_qa(source_rows: pd.DataFrame):
    timestamps = pd.to_datetime(source_rows["timestamp"], utc=True, errors="coerce")
    available = pd.to_datetime(source_rows["available_time"], utc=True, errors="coerce")
    conversion_errors = int(timestamps.isna().sum() + available.isna().sum())
    timestamps = timestamps.astype("datetime64[ns, UTC]")
    available = available.astype("datetime64[ns, UTC]")
    valid = timestamps.notna() & available.notna()

    rows = source_rows.loc[valid, ["symbol"]].copy()
    rows["timestamp_utc"] = timestamps.loc[valid]
    rows["available_time_utc"] = available.loc[valid]

    warmup_start = pd.Timestamp(f"{QA_DATE} {WARMUP_START_ET}", tz=TIMEZONE).tz_convert("UTC")
    qa_start = pd.Timestamp(f"{QA_DATE} {QA_START_ET}", tz=TIMEZONE).tz_convert("UTC")
    qa_end = pd.Timestamp(f"{QA_DATE} {QA_END_ET}", tz=TIMEZONE).tz_convert("UTC")
    decisions = pd.date_range(qa_start, qa_end, freq="1min")

    warmup_total = (rows.timestamp_utc >= warmup_start) & (rows.timestamp_utc < qa_start)
    warmup_visible = warmup_total & (rows.available_time_utc <= qa_start)

    records = []
    unique_filtered = set()
    for decision in decisions:
        timestamp_eligible = (rows.timestamp_utc >= warmup_start) & (rows.timestamp_utc <= decision)
        visible = timestamp_eligible & (rows.available_time_utc <= decision)
        filtered = timestamp_eligible & (rows.available_time_utc > decision)
        unique_filtered.update(rows.index[filtered].tolist())
        records.append({
            "decision_time_et": decision.tz_convert(TIMEZONE).isoformat(),
            "decision_time_utc": decision.isoformat(),
            "timestamp_eligible_rows": int(timestamp_eligible.sum()),
            "pit_visible_rows": int(visible.sum()),
            "pit_filtered_rows": int(filtered.sum()),
            "pit_visible_symbols": int(rows.loc[visible, "symbol"].astype(str).nunique()),
            "future_timestamp_violations": int((rows.loc[visible, "timestamp_utc"] > decision).sum()),
            "future_available_time_violations": int((rows.loc[visible, "available_time_utc"] > decision).sum()),
        })

    detail = pd.DataFrame(records)
    spacing = decisions.to_series().diff().dropna().dt.total_seconds()
    report = {
        "timezone": TIMEZONE,
        "qa_date": QA_DATE,
        "qa_start_et": QA_START_ET,
        "qa_end_et": QA_END_ET,
        "qa_frames_expected": EXPECTED_FRAMES,
        "decision_step_minutes": DECISION_STEP_MINUTES,
        "warmup_start_et": WARMUP_START_ET,
        "max_lookback_minutes": MAX_LOOKBACK_MINUTES,
        "decision_frames_expected": EXPECTED_FRAMES,
        "decision_frames_actual": int(len(decisions)),
        "decision_times": [d.tz_convert(TIMEZONE).isoformat() for d in decisions],
        "decision_times_utc": [d.isoformat() for d in decisions],
        "frame_spacing_seconds_min": float(spacing.min()),
        "frame_spacing_seconds_max": float(spacing.max()),
        "warmup_rows_total": int(warmup_total.sum()),
        "warmup_rows_used": int(warmup_visible.sum()),
        "warmup_rows_late_at_qa_start": int((warmup_total & ~warmup_visible).sum()),
        "future_timestamp_violations": int(detail.future_timestamp_violations.sum()),
        "future_available_time_violations": int(detail.future_available_time_violations.sum()),
        "pit_filtered_rows": int(detail.pit_filtered_rows.sum()),
        "pit_filtered_unique_source_rows": int(len(unique_filtered)),
        "timezone_conversion_errors": conversion_errors,
    }
    report["pass"] = (
        report["decision_frames_actual"] == EXPECTED_FRAMES
        and report["frame_spacing_seconds_min"] == 60.0
        and report["frame_spacing_seconds_max"] == 60.0
        and report["future_timestamp_violations"] == 0
        and report["future_available_time_violations"] == 0
        and report["timezone_conversion_errors"] == 0
    )
    return report, detail


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--month-pack-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    pattern = str(Path(args.month_pack_root) / "month=*" / "node_factors_1m" / "date=*" / "*.parquet")
    source = ParquetNodeFactorSource(pattern)
    source_rows = source.load_date(QA_DATE)
    report, detail = build_time_pit_qa(source_rows)

    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    (output / "time_pit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    detail.to_csv(output / "pit_by_decision_time.csv", index=False)
    pd.DataFrame({
        "decision_time_et": report["decision_times"],
        "decision_time_utc": report["decision_times_utc"],
    }).to_csv(output / "decision_times.csv", index=False)
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
