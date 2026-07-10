#!/usr/bin/env python3
"""Compatibility entrypoint for partition-safe P2 alpha metric reduction.

Example:
python scripts/p2_reduce_alpha_metrics.py --signals-root ... --out-dir ... --horizons 5m,15m,30m,60m
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p2_alpha_partitioned_lab import main


if __name__ == "__main__":
    sys.argv.insert(1, "reduce")
    main()
