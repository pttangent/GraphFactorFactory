from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from graphfactorfactory.domain.layers import LayerDefinition


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    mean = np.nanmean(values)
    std = np.nanstd(values)
    if not np.isfinite(std) or std <= 1e-12:
        return np.zeros_like(values)
    return np.nan_to_num((values - mean) / std)


def trajectory(window: pd.DataFrame, layer: LayerDefinition, universe: list[str], minimum_points: int):
    blocks: list[np.ndarray] = []
    timestamps = 0
    used_columns: list[str] = []
    for column in layer.columns:
        if column not in window.columns:
            continue
        pivot = window.pivot_table(index="timestamp", columns="symbol", values=column, aggfunc="last").reindex(columns=universe)
        if len(pivot) < minimum_points:
            continue
        values = pivot.to_numpy(dtype=np.float32).T
        medians = np.nanmedian(values, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        values = np.where(np.isfinite(values), values, medians)
        values -= values.mean(axis=1, keepdims=True)
        scales = values.std(axis=1, keepdims=True)
        values = np.divide(values, scales, out=np.zeros_like(values), where=scales > 1e-12)
        blocks.append(values)
        timestamps = max(timestamps, len(pivot))
        used_columns.append(column)
    if not blocks:
        return None, 0, ()
    return np.concatenate(blocks, axis=1), timestamps, tuple(used_columns)


def neighbor(adjacency: sparse.csr_matrix, signal: np.ndarray) -> np.ndarray:
    denominator = np.asarray(adjacency.sum(axis=1)).ravel()
    numerator = np.asarray(adjacency @ signal).ravel()
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
