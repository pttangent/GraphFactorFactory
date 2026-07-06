from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from graphfactorfactory.application.phase0_diagnostics import (
    StrongEdgeThresholds,
    aggregate_daily_market_diagnostics,
    compute_resonance_diagnostics,
    compute_snapshot_diagnostics,
    compute_temporal_diagnostics,
    load_canonical_edges,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute complete Phase 0 market-state diagnostics")
    parser.add_argument("--graph-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-dates", nargs="*", default=[])
    parser.add_argument("--dates", nargs="*", default=[])
    parser.add_argument("--community-resolution", type=float, default=1.0)
    args = parser.parse_args()

    graph_root = Path(args.graph_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    edges = load_canonical_edges(graph_root, args.dates or None)
    train_edges = edges
    if args.train_dates:
        train_edges = edges[edges["trade_date"].isin(args.train_dates)]
        if train_edges.empty:
            raise ValueError("No edges matched --train-dates")
    thresholds = StrongEdgeThresholds.fit(train_edges)

    symbols = pd.read_parquet(graph_root / "dimensions" / "symbols.parquet")
    layers = pd.read_parquet(graph_root / "dimensions" / "layers.parquet")
    universe_count = len(symbols)

    snapshot = compute_snapshot_diagnostics(
        edges,
        universe_count=universe_count,
        thresholds=thresholds,
        community_resolution=args.community_resolution,
    )
    temporal_parts = [
        compute_temporal_diagnostics(day_edges, thresholds)
        for _, day_edges in edges.groupby("trade_date", sort=True)
    ]
    temporal = pd.concat(temporal_parts, ignore_index=True) if temporal_parts else pd.DataFrame()
    resonance = compute_resonance_diagnostics(edges, layers)
    daily = aggregate_daily_market_diagnostics(snapshot, temporal, resonance)

    thresholds.table.to_csv(output / "strong_edge_thresholds.csv", index=False)
    snapshot.to_parquet(output / "snapshot_market_diagnostics.parquet", index=False)
    temporal.to_parquet(output / "temporal_edge_diagnostics.parquet", index=False)
    resonance.to_parquet(output / "cross_layer_resonance.parquet", index=False)
    daily.to_csv(output / "daily_market_diagnostics.csv", index=False)

    report = {
        "graph_root": str(graph_root),
        "dates": sorted(edges["trade_date"].astype(str).unique().tolist()),
        "train_dates": args.train_dates,
        "universe_count": universe_count,
        "edge_rows": int(len(edges)),
        "snapshot_rows": int(len(snapshot)),
        "temporal_rows": int(len(temporal)),
        "resonance_rows": int(len(resonance)),
        "outputs": {
            "strong_edge_thresholds": "strong_edge_thresholds.csv",
            "snapshot": "snapshot_market_diagnostics.parquet",
            "temporal": "temporal_edge_diagnostics.parquet",
            "resonance": "cross_layer_resonance.parquet",
            "daily": "daily_market_diagnostics.csv",
        },
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
