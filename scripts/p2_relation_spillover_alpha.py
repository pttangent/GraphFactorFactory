#!/usr/bin/env python3
"""Compatibility entrypoint for partition-safe P2 relation-spillover alpha.

Example:
python scripts/p2_relation_spillover_alpha.py --p1-root ... --theme-returns-root ... --out-root ... --workers 20
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p2_alpha_partitioned_lab import main


if __name__ == "__main__":
    sys.argv.insert(1, "relation-spillover")
    main()
