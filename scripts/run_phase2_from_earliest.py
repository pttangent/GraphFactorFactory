from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def phase1_dates(root: Path) -> list[str]:
    return sorted(path.name.split("=", 1)[1] for path in root.glob("date=*") if (path / "layer_communities.parquet").exists())


def earliest_invalid(phase1_root: Path, phase2_root: Path) -> str | None:
    dates = phase1_dates(phase1_root)
    for trade_date in dates:
        source = phase1_root / f"date={trade_date}" / "layer_communities.parquet"
        day_success = phase2_root / f"date={trade_date}" / "_SUCCESS"
        state_success = phase2_root / "_state" / f"date={trade_date}" / "_SUCCESS"
        if not day_success.exists() or not state_success.exists():
            return trade_date
        if source.stat().st_mtime_ns > min(day_success.stat().st_mtime_ns, state_success.stat().st_mtime_ns):
            return trade_date
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild Phase 2 from the earliest missing or stale Phase 1 date")
    parser.add_argument("--phase1-root", type=Path, default=Path("outputs/theme_discovery_phase1"))
    parser.add_argument("--phase2-root", type=Path, default=Path("outputs/theme_temporal_phase2"))
    parser.add_argument("--date-end")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    phase1_root = args.phase1_root.expanduser().resolve()
    phase2_root = args.phase2_root.expanduser().resolve()
    dates = phase1_dates(phase1_root)
    if not dates:
        raise RuntimeError(f"No Phase 1 layer communities found under {phase1_root}")
    rebuild_from = earliest_invalid(phase1_root, phase2_root)
    if rebuild_from is None:
        print("Phase 2 is current for every Phase 1 date.")
        return
    date_end = args.date_end or dates[-1]
    command = [
        sys.executable,
        "scripts/run_theme_temporal_phase2.py",
        "--phase1-root",
        str(phase1_root),
        "--out-root",
        str(phase2_root),
        "--date-start",
        rebuild_from,
        "--date-end",
        date_end,
        "--rebuild-from",
        rebuild_from,
    ]
    print("Earliest invalid Phase 2 date:", rebuild_from)
    print("Command:", " ".join(command))
    if not args.dry_run:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
