from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from graphfactorfactory.application.return_corr_patch import _atomic_merge_parquet
from graphfactorfactory.infrastructure.schemas import EDGE_SCHEMA


def _edge_row(layer_id: int, src_id: int, dst_id: int, weight: float) -> dict:
    timestamp = pd.Timestamp("2026-01-02 15:00", tz="UTC")
    return {
        "decision_time": timestamp,
        "window_start": timestamp - pd.Timedelta(minutes=60),
        "window_end": timestamp,
        "layer_id": np.int16(layer_id),
        "src_id": np.int32(src_id),
        "dst_id": np.int32(dst_id),
        "weight": np.float32(weight),
        "src_rank": np.int16(1),
        "dst_rank": np.int16(1),
        "directed": False,
        "lag_bars": np.int16(0),
        "window_points": np.int16(12),
        "vector_dimension": np.int16(12),
    }


def test_atomic_merge_replaces_only_return_corr_layers(tmp_path: Path):
    baseline_path = tmp_path / "edges.parquet"
    patch_path = tmp_path / "patch.parquet"
    baseline = pa.Table.from_pylist([
        _edge_row(1, 1, 2, 0.4),
        _edge_row(2, 3, 4, 0.7),
    ], schema=EDGE_SCHEMA)
    patch = pa.Table.from_pylist([
        _edge_row(1, 5, 6, 0.8),
        _edge_row(14, 7, 8, 0.9),
        _edge_row(15, 9, 10, 0.6),
    ], schema=EDGE_SCHEMA)
    pq.write_table(baseline, baseline_path)
    pq.write_table(patch, patch_path)

    stats = _atomic_merge_parquet(
        baseline_path,
        patch_path,
        EDGE_SCHEMA,
        (1, 14, 15),
        ["decision_time", "layer_id", "src_id", "dst_id"],
    )
    merged = pq.read_table(baseline_path).to_pandas().sort_values("layer_id")
    assert merged["layer_id"].tolist() == [1, 2, 14, 15]
    assert merged.loc[merged["layer_id"].eq(2), "weight"].iloc[0] == pytest.approx(0.7)
    assert stats["preserved_rows"] == 1
    assert stats["patch_rows"] == 3
