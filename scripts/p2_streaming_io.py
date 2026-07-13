#!/usr/bin/env python3
"""Fail-clean streaming I/O wrappers for pipeline stages."""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from p2_pit_core import stream_frames as _stream_frames


def stream_frames(path: str | Path, frames: Iterable[pd.DataFrame | None]) -> tuple[int, int]:
    """Stream frames and remove a stale final file when the new result is empty."""
    output = Path(path)
    rows, batches = _stream_frames(output, frames)
    if rows == 0:
        output.unlink(missing_ok=True)
    return rows, batches
