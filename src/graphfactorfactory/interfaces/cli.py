from __future__ import annotations

import argparse
import json

import pandas as pd

from graphfactorfactory.application.pipeline import GraphFactorPipeline
from graphfactorfactory.domain.config import BuildConfig
from graphfactorfactory.infrastructure.nodefactorfactory import ParquetNodeFactorSource
from graphfactorfactory.infrastructure.qlib import CanonicalQlibDataLoader, materialize_qlib_cache
from graphfactorfactory.infrastructure.store import CanonicalGraphStore


def main() -> None:
    parser = argparse.ArgumentParser(prog="graphfactorfactory")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-date", help="Build canonical graph factors for one date")
    build.add_argument("--node-factors", required=True)
    build.add_argument("--date", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--config", default="configs/default.yaml")
    build.add_argument("--universe-csv")
    cache = commands.add_parser("materialize-qlib-cache", help="Create an optional disposable Qlib cache")
    cache.add_argument("--node-factors", required=True)
    cache.add_argument("--graph-store", required=True)
    cache.add_argument("--config", default="configs/default.yaml")
    cache.add_argument("--output", required=True)
    cache.add_argument("--start-time")
    cache.add_argument("--end-time")
    args = parser.parse_args()
    if args.command == "build-date":
        config = BuildConfig.from_yaml(args.config)
        source = ParquetNodeFactorSource(args.node_factors)
        universe = pd.read_csv(args.universe_csv)["symbol"].astype(str).tolist() if args.universe_csv else None
        result = GraphFactorPipeline(source, CanonicalGraphStore(args.output, config), config).build_date(args.date, universe)
        payload = {"root": str(result.root), "manifest": str(result.manifest_path), "catalog": str(result.catalog_path), "edge_rows": result.edge_rows, "node_feature_rows": result.node_feature_rows, "snapshot_rows": result.snapshot_rows, "label_rows": result.label_rows}
    else:
        loader = CanonicalQlibDataLoader(args.node_factors, args.graph_store, args.config)
        output = materialize_qlib_cache(loader, args.output, args.start_time, args.end_time)
        payload = {"output": str(output)}
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
