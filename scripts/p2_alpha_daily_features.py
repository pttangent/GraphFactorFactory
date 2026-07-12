#!/usr/bin/env python3
"""Compatibility entrypoint for the PIT-safe P2 implementation."""
from p2_alpha_pit_features import *  # noqa: F401,F403
from p2_alpha_pit_features import main

if __name__ == "__main__":
    main()
