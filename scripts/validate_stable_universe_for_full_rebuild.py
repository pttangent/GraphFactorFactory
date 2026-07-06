from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from graphfactorfactory.infrastructure.nodefactorfactory import ParquetNodeFactorSource


def main() -> None:
    parser = argparse.ArgumentParser(description="Block a full Phase 0 rebuild when daily source coverage would change symbol_id mapping")
    parser.add_argument("--node-factors", required=True)
    parser.add_argument("--symbols-parquet", required=True)
    parser.add_argument("--source-graph-store", required=True)
    parser.add_argument("--start", default="2026-01-02")
    parser.add_argument("--end", default="2026-02-27")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    expected = pd.read_parquet(args.symbols_parquet).sort_values("symbol_id")
    expected_symbols = expected["symbol"].astype(str).tolist()
    expected_set = set(expected_symbols)
    source = ParquetNodeFactorSource(args.node_factors)
    graph_dates = sorted(
        path.name.split("=", 1)[1]
        for path in (Path(args.source_graph_store).expanduser().resolve() / "canonical").glob("date=*")
        if args.start <= path.name.split("=", 1)[1] <= args.end
    )
    rows = []
    for trade_date in graph_dates:
        frame = source.load_date(trade_date)
        present = set(frame["symbol"].astype(str).unique())
        missing = sorted(expected_set - present)
        unexpected = sorted(present - expected_set)
        rows.append({
            "date": trade_date,
            "expected_symbols": len(expected_set),
            "present_expected_symbols": len(expected_set & present),
            "missing_count": len(missing),
            "unexpected_count": len(unexpected),
            "missing_examples": ",".join(missing[:20]),
            "unexpected_examples": ",".join(unexpected[:20]),
            "safe_for_current_full_rebuild_code": len(missing) == 0,
        })
    report = pd.DataFrame(rows)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output, index=False)
    summary = {
        "dates": len(report),
        "safe_dates": int(report["safe_for_current_full_rebuild_code"].sum()) if len(report) else 0,
        "unsafe_dates": int((~report["safe_for_current_full_rebuild_code"]).sum()) if len(report) else 0,
        "report": str(output),
    }
    print(json.dumps(summary, indent=2))
    if summary["unsafe_dates"]:
        raise SystemExit("Full rebuild is blocked: current build_date would reindex a reduced daily universe. Use the ReturnCorr patch path or fix stable-universe handling first.")


if __name__ == "__main__":
    main()
