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


def _standardize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    values = values - values.mean(axis=1, keepdims=True)
    scales = values.std(axis=1, keepdims=True)
    return np.divide(values, scales, out=np.zeros_like(values), where=scales > 1e-12)


def _raw_pivot(window: pd.DataFrame, column: str, universe: list[str]) -> pd.DataFrame:
    return window.pivot_table(index="timestamp", columns="symbol", values=column, aggfunc="last").reindex(columns=universe)


def _fill_cross_sectional_median(pivot: pd.DataFrame) -> pd.DataFrame:
    values = pivot.to_numpy(dtype=np.float32)
    medians = np.nanmedian(values, axis=1)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    values = np.where(np.isfinite(values), values, medians[:, None])
    return pd.DataFrame(values, index=pivot.index, columns=pivot.columns)


def _filled_pivot(window: pd.DataFrame, column: str, universe: list[str]) -> pd.DataFrame:
    return _fill_cross_sectional_median(_raw_pivot(window, column, universe))


def return_trajectory(
    window: pd.DataFrame,
    layer: LayerDefinition,
    universe: list[str],
    minimum_points: int,
    benchmarks: tuple[str, ...],
    min_benchmark_points: int,
    ridge: float,
):
    if "ret_5m" not in window.columns:
        return None, 0, ()
    raw_pivot = _raw_pivot(window, "ret_5m", universe)
    if len(raw_pivot) < minimum_points:
        return None, 0, ()
    pivot = _fill_cross_sectional_median(raw_pivot)

    if layer.transform == "return_corr_cross_sectional_residual":
        pivot = pivot.sub(pivot.median(axis=1), axis=0)
    elif layer.transform == "return_corr_market_residual":
        available = [
            symbol
            for symbol in benchmarks
            if symbol in raw_pivot.columns
            and int(raw_pivot[symbol].notna().sum()) >= min_benchmark_points
            and float(raw_pivot[symbol].std(ddof=0, skipna=True) or 0.0) > 1e-12
        ]
        if available and len(pivot) >= max(minimum_points, min_benchmark_points):
            x = pivot[available].to_numpy(dtype=np.float64)
            x = np.column_stack([np.ones(len(x)), x])
            xtx = x.T @ x
            penalty = np.eye(xtx.shape[0], dtype=np.float64) * float(ridge)
            penalty[0, 0] = 0.0
            beta = np.linalg.solve(xtx + penalty, x.T @ pivot.to_numpy(dtype=np.float64))
            fitted = x @ beta
            pivot = pd.DataFrame(pivot.to_numpy(dtype=np.float64) - fitted, index=pivot.index, columns=pivot.columns)
        else:
            pivot = pivot.sub(pivot.median(axis=1), axis=0)

    vectors = _standardize_rows(pivot.to_numpy(dtype=np.float32).T)
    return vectors, len(pivot), ("ret_5m",)


def trajectory(
    window: pd.DataFrame,
    layer: LayerDefinition,
    universe: list[str],
    minimum_points: int,
    *,
    return_corr_benchmarks: tuple[str, ...] = ("SPY", "QQQ", "IWM"),
    return_corr_min_benchmark_points: int = 8,
    return_corr_ridge: float = 1e-6,
):
    if layer.transform.startswith("return_corr_"):
        return return_trajectory(
            window,
            layer,
            universe,
            minimum_points,
            return_corr_benchmarks,
            return_corr_min_benchmark_points,
            return_corr_ridge,
        )

    blocks: list[np.ndarray] = []
    timestamps = 0
    used_columns: list[str] = []
    for column in layer.columns:
        if column not in window.columns:
            continue
        pivot = _filled_pivot(window, column, universe)
        if len(pivot) < minimum_points:
            continue
        blocks.append(_standardize_rows(pivot.to_numpy(dtype=np.float32).T))
        timestamps = max(timestamps, len(pivot))
        used_columns.append(column)
    if not blocks:
        return None, 0, ()
    return np.concatenate(blocks, axis=1), timestamps, tuple(used_columns)


def neighbor(adjacency: sparse.csr_matrix, signal: np.ndarray) -> np.ndarray:
    denominator = np.asarray(adjacency.sum(axis=1)).ravel()
    numerator = np.asarray(adjacency @ signal).ravel()
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
