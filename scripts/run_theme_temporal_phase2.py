import argparse
import logging
import multiprocessing
from pathlib import Path

from graphfactorfactory.themes.phase2_pipeline import ThemeTemporalPhase2Pipeline
from graphfactorfactory.themes.pipeline import ThemeDiscoveryConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run stateful Phase 2 temporal tracking")
    parser.add_argument("--date")
    parser.add_argument("--date-start")
    parser.add_argument("--date-end")
    parser.add_argument("--rebuild-from")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--phase1-root", default="outputs/theme_discovery_phase1")
    parser.add_argument("--out-root", default="outputs/theme_temporal_phase2")
    args = parser.parse_args()

    date_start = args.date if args.date else args.date_start
    date_end = args.date if args.date else args.date_end
    if args.rebuild_from and (not date_start or args.rebuild_from < date_start):
        date_start = args.rebuild_from

    pipeline = ThemeTemporalPhase2Pipeline(
        phase1_root=Path(args.phase1_root),
        phase2_root=Path(args.out_root),
        config=ThemeDiscoveryConfig(),
    )
    outputs = pipeline.run(
        date_start=date_start,
        date_end=date_end,
        rebuild_from=args.rebuild_from,
        resume=not args.no_resume,
    )
    logger.info("Phase 2 wrote %d layer-day outputs", len(outputs))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
