from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from graphfactorfactory.domain.layers import LayerDefinition


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    mean = np.nanmean(values); std = np.nanstd(values)
    if not np.isfinite(std) or std <= 1e-12:
        return np.zeros_like(values)
    return np.nan_to_num((values - mean) / std)


def _standardize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    values = values - values.mean(axis=1, keepdims=True)
    scales = values.std(axis=1, keepdims=True)
    return np.divide(values, scales, out=np.zeros_like(values), where=scales > 1e-12)


def _raw_pivot(window: pd.DataFrame, column: str, universe: list[str]) -> pd.DataFrame:
    return window.pivot_table(index="timestamp", columns="symbol", values=column, aggfunc="last").reindex(columns=universe).sort_index()


def _fill_cross_sectional_median(pivot: pd.DataFrame) -> pd.DataFrame:
    values = pivot.to_numpy(dtype=np.float32)
    medians = np.nanmedian(values, axis=1)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    values = np.where(np.isfinite(values), values, medians[:, None])
    return pd.DataFrame(values, index=pivot.index, columns=pivot.columns)


def _filled_pivot(window: pd.DataFrame, column: str, universe: list[str]) -> pd.DataFrame:
    return _fill_cross_sectional_median(_raw_pivot(window, column, universe))


def _return_source(window: pd.DataFrame, layer: LayerDefinition, universe: list[str]):
    if layer.transform == "return_corr_cross_sectional_rolling_5m":
        source = "log_ret_1m" if "log_ret_1m" in window.columns else "ret_1m"
        if source not in window.columns:
            return None, ()
        raw = _raw_pivot(window, source, universe)
        return raw.rolling(window=5, min_periods=5).sum().dropna(how="all"), (source,)
    for source in ("ret_1m", "log_ret_1m", "ret_5m"):
        if source in window.columns:
            return _raw_pivot(window, source, universe), (source,)
    return None, ()


def return_trajectory(window, layer, universe, minimum_points, benchmarks, min_benchmark_points, ridge):
    raw_pivot, used_columns = _return_source(window, layer, universe)
    if raw_pivot is None or len(raw_pivot) < minimum_points:
        return None, 0, ()
    pivot = _fill_cross_sectional_median(raw_pivot)
    if layer.transform in {"return_corr_cross_sectional_1m", "return_corr_cross_sectional_rolling_5m"}:
        pivot = pivot.sub(pivot.median(axis=1), axis=0)
    elif layer.transform == "return_corr_market_residual":
        available = [s for s in benchmarks if s in raw_pivot.columns and int(raw_pivot[s].notna().sum()) >= min_benchmark_points and float(raw_pivot[s].std(ddof=0, skipna=True) or 0.0) > 1e-12]
        if available and len(pivot) >= max(minimum_points, min_benchmark_points):
            x = pivot[available].to_numpy(dtype=np.float64)
            x = np.column_stack([np.ones(len(x)), x])
            xtx = x.T @ x
            penalty = np.eye(xtx.shape[0], dtype=np.float64) * float(ridge); penalty[0, 0] = 0.0
            beta = np.linalg.solve(xtx + penalty, x.T @ pivot.to_numpy(dtype=np.float64))
            pivot = pd.DataFrame(pivot.to_numpy(dtype=np.float64) - x @ beta, index=pivot.index, columns=pivot.columns)
        else:
            pivot = pivot.sub(pivot.median(axis=1), axis=0)
    vectors = _standardize_rows(pivot.to_numpy(dtype=np.float32).T)
    return vectors, len(pivot), used_columns


def trajectory(window, layer, universe, minimum_points, *, return_corr_benchmarks=("SPY", "QQQ", "IWM"), return_corr_min_benchmark_points=8, return_corr_ridge=1e-6):
    if layer.transform.startswith("return_corr_"):
        return return_trajectory(window, layer, universe, minimum_points, return_corr_benchmarks, return_corr_min_benchmark_points, return_corr_ridge)
    blocks=[]; timestamps=0; used_columns=[]
    for column in layer.columns:
        if column not in window.columns: continue
        pivot = _filled_pivot(window, column, universe)
        if len(pivot) < minimum_points: continue
        blocks.append(_standardize_rows(pivot.to_numpy(dtype=np.float32).T)); timestamps=max(timestamps,len(pivot)); used_columns.append(column)
    if not blocks:return None,0,()
    return np.concatenate(blocks,axis=1),timestamps,tuple(used_columns)


def neighbor(adjacency: sparse.csr_matrix, signal: np.ndarray) -> np.ndarray:
    denominator=np.asarray(adjacency.sum(axis=1)).ravel(); numerator=np.asarray(adjacency@signal).ravel()
    return np.divide(numerator,denominator,out=np.zeros_like(numerator),where=denominator>0)
