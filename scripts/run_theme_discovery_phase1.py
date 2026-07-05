import argparse
import logging
import multiprocessing
import sys
from pathlib import Path

from graphfactorfactory.application.config import BuildConfig
from graphfactorfactory.themes.phase1_pipeline import ThemeDiscoveryPhase1Pipeline, ThemeDiscoveryConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run Phase 1: Snapshot-local Theme Discovery")
    parser.add_argument("--date", type=str, help="YYYY-MM-DD to run a single day")
    parser.add_argument("--date-start", type=str, help="YYYY-MM-DD start date for range")
    parser.add_argument("--date-end", type=str, help="YYYY-MM-DD end date for range")
    parser.add_argument("--max-snapshot-workers", type=int, default=26, help="Workers for parallel snapshots")
    parser.add_argument("--graph-root", type=str, default="data/graph_store", help="Path to Graph Store")
    parser.add_argument("--out-root", type=str, default="outputs/theme_discovery_phase1", help="Output path")
    args = parser.parse_args()

    graph_root = Path(args.graph_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    config = ThemeDiscoveryConfig()
    
    # We do not need metadata for Phase 1 because metadata labeler handles None gracefully 
    # (or we could load metadata if we had it, but usually it's None in old pipeline unless specified)
    
    pipeline = ThemeDiscoveryPhase1Pipeline(
        graph_store_root=graph_root,
        theme_store_root=out_root,
        config=config,
        metadata=None
    )

    if args.date:
        date_start = args.date
        date_end = args.date
    else:
        date_start = args.date_start
        date_end = args.date_end

    logger.info(f"Starting Phase 1 Theme Discovery for {date_start} to {date_end} using {args.max_snapshot_workers} workers.")
    outputs = pipeline.run(date_start=date_start, date_end=date_end, max_workers=args.max_snapshot_workers)
    
    if outputs:
        logger.info(f"Phase 1 successfully completed for {len(outputs)} days. Outputs saved to {out_root}")
    else:
        logger.info("No days were processed (either skipped or no graph data).")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
