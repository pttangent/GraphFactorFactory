import argparse
import logging
import multiprocessing
import sys
from pathlib import Path

from graphfactorfactory.themes.phase2_pipeline import ThemeTemporalPhase2Pipeline
from graphfactorfactory.themes.pipeline import ThemeDiscoveryConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run Phase 2: Temporal Theme Tracking")
    parser.add_argument("--date", type=str, help="YYYY-MM-DD to run a single day")
    parser.add_argument("--date-start", type=str, help="YYYY-MM-DD start date for range")
    parser.add_argument("--date-end", type=str, help="YYYY-MM-DD end date for range")
    parser.add_argument("--max-layer-workers", type=int, default=6, help="Workers for parallel layers (snapshots are sequential)")
    parser.add_argument("--phase1-root", type=str, default="outputs/theme_discovery_phase1", help="Path to Phase 1 Output")
    parser.add_argument("--out-root", type=str, default="outputs/theme_temporal_phase2", help="Output path")
    args = parser.parse_args()

    phase1_root = Path(args.phase1_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    config = ThemeDiscoveryConfig()
    
    pipeline = ThemeTemporalPhase2Pipeline(
        phase1_root=phase1_root,
        phase2_root=out_root,
        config=config
    )

    if args.date:
        date_start = args.date
        date_end = args.date
    else:
        date_start = args.date_start
        date_end = args.date_end

    logger.info(f"Starting Phase 2 Temporal Tracking for {date_start} to {date_end} using {args.max_layer_workers} layer workers.")
    outputs = pipeline.run(date_start=date_start, date_end=date_end, max_layer_workers=args.max_layer_workers)
    
    if outputs:
        logger.info(f"Phase 2 successfully completed for {len(outputs)} layer-days. Outputs saved to {out_root}")
    else:
        logger.info("No days were processed.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
