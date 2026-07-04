from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from graphfactorfactory.themes import ThemeDiscoveryConfig, ThemeDiscoveryPipeline


def main():
    parser = argparse.ArgumentParser(prog="graphfactorfactory-theme")
    parser.add_argument("--graph-store", required=True)
    parser.add_argument("--theme-store", required=True)
    parser.add_argument("--metadata-csv")
    parser.add_argument("--date-start")
    parser.add_argument("--date-end")
    parser.add_argument("--run-id", default="gff_theme_run")
    parser.add_argument("--frame-minutes", type=int, default=15)
    parser.add_argument("--min-consensus-score", type=float, default=0.35)
    parser.add_argument("--min-distinct-families", type=int, default=2)
    parser.add_argument("--min-overlap", type=float, default=0.5)
    args = parser.parse_args()
    metadata = pd.read_csv(args.metadata_csv) if args.metadata_csv else None
    config = ThemeDiscoveryConfig(
        run_id=args.run_id,
        frame_minutes=args.frame_minutes,
        min_consensus_score=args.min_consensus_score,
        min_distinct_families=args.min_distinct_families,
        min_overlap=args.min_overlap,
    )
    outputs = ThemeDiscoveryPipeline(args.graph_store, args.theme_store, config, metadata).run(args.date_start, args.date_end)
    print(json.dumps({"snapshot_outputs": [str(Path(path)) for path in outputs], "count": len(outputs)}, indent=2))


if __name__ == "__main__":
    main()
