#!/usr/bin/env python3
"""Date-bundled Theme Returns execution to reuse one daily label table."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from p2_pit_core import Part
from p2_pit_theme_streaming import build_theme_returns_one


def group_parts_by_date(parts: list[Part]) -> list[list[Part]]:
    grouped: dict[str, list[Part]] = defaultdict(list)
    for part in parts:
        grouped[str(part.date)].append(part)
    batches = []
    for date, date_parts in grouped.items():
        date_parts.sort(key=lambda part: (str(part.layer_id), str(part.scale), str(part.base)))
        batches.append(date_parts)
    batches.sort(
        key=lambda batch: sum(Path(part.base).stat().st_size for part in batch),
        reverse=True,
    )
    return batches


def build_theme_returns_date(
    parts: list[Part],
    labels_root: str | Path,
    output_root: str | Path,
    horizons: list[str],
    levels: set[str] | None,
    skip_existing: bool,
    max_row_groups: int | None,
    inner_workers: int,
) -> list[dict]:
    if not parts:
        return []
    dates = {str(part.date) for part in parts}
    if len(dates) != 1:
        raise ValueError(f"theme date bundle contains mixed dates: {sorted(dates)}")
    return [
        build_theme_returns_one(
            part,
            labels_root,
            output_root,
            horizons,
            levels,
            skip_existing,
            max_row_groups,
            inner_workers,
        )
        for part in parts
    ]
