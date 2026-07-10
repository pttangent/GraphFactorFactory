#!/usr/bin/env python3
"""Compatibility entrypoint for partition-safe P2 theme returns.

Use the same CLI as the new runner subcommand, for example:
python scripts/p2_build_theme_returns.py --p1-root ... --labels-root ... --out-root ... --workers 20
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p2_alpha_partitioned_lab import main


if __name__ == "__main__":
    sys.argv.insert(1, "theme-returns")
    main()
