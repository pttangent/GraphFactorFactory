#!/usr/bin/env python3
"""Exact pairwise-complete evaluation kernels for P2 scores."""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from p2_pit_core import SCORES


def _validity_groups(frame: pd.DataFrame, columns: list[str]) -> list[tuple[np.ndarray, list[str]]]:
    groups: dict[tuple[int, bytes], tuple[np.ndarray, list[str]]] = {}
    for column in columns:
        mask = frame[column].notna().to_numpy(dtype=np.bool_, copy=False)
        key = (len(mask), np.packbits(mask, bitorder="little").tobytes())
        if key not in groups:
            groups[key] = (mask, [column])
        else:
            groups[key][1].append(column)
    return list(groups.values())


def _rank_correlation_matrix(features: pd.DataFrame, targets: pd.DataFrame) -> np.ndarray:
    feature_ranks = features.rank(axis=0, method="average")
    target_ranks = targets.rank(axis=0, method="average")
    feature_values = (feature_ranks - feature_ranks.mean(axis=0)).to_numpy(dtype=np.float64, copy=False)
    target_values = (target_ranks - target_ranks.mean(axis=0)).to_numpy(dtype=np.float64, copy=False)
    numerator = feature_values.T @ target_values
    denominator = np.outer(
        np.sqrt(np.square(feature_values).sum(axis=0)),
        np.sqrt(np.square(target_values).sum(axis=0)),
    )
    return np.divide(
        numerator,
        denominator,
        out=np.full(numerator.shape, np.nan, dtype=np.float64),
        where=denominator > 0,
    )


def _spread_matrix(features: pd.DataFrame, targets: pd.DataFrame) -> np.ndarray:
    feature_values = features.to_numpy(dtype=np.float64, copy=False)
    target_values = targets.to_numpy(dtype=np.float64, copy=False)
    quantiles = features.quantile([0.8, 0.2], axis=0)
    q80 = quantiles.loc[0.8].to_numpy(dtype=np.float64, copy=False)
    q20 = quantiles.loc[0.2].to_numpy(dtype=np.float64, copy=False)
    high = feature_values >= q80.reshape(1, -1)
    low = feature_values <= q20.reshape(1, -1)
    high_count = high.sum(axis=0).astype(np.float64)
    low_count = low.sum(axis=0).astype(np.float64)
    high_sum = high.astype(np.float64).T @ target_values
    low_sum = low.astype(np.float64).T @ target_values
    high_mean = np.divide(
        high_sum,
        high_count.reshape(-1, 1),
        out=np.full(high_sum.shape, np.nan, dtype=np.float64),
        where=high_count.reshape(-1, 1) > 0,
    )
    low_mean = np.divide(
        low_sum,
        low_count.reshape(-1, 1),
        out=np.full(low_sum.shape, np.nan, dtype=np.float64),
        where=low_count.reshape(-1, 1) > 0,
    )
    return high_mean - low_mean


def _evaluate_group(
    subset: pd.DataFrame,
    scores: list[str],
    targets: list[str],
    keys: tuple[Any, ...],
    key_names: list[str],
) -> list[dict]:
    rows: list[dict] = []
    if len(subset) < 30 or not scores or not targets:
        return rows
    subset = subset.replace([np.inf, -np.inf], np.nan)
    base = dict(zip(key_names, keys))
    for score_mask, score_columns in _validity_groups(subset, scores):
        for target_mask, target_columns in _validity_groups(subset, targets):
            pair_mask = score_mask & target_mask
            sample_count = int(pair_mask.sum())
            if sample_count < 30:
                continue
            pair_scores = subset.loc[pair_mask, score_columns]
            pair_targets = subset.loc[pair_mask, target_columns]
            correlations = _rank_correlation_matrix(pair_scores, pair_targets)
            spreads = _spread_matrix(pair_scores, pair_targets)
            for score_index, score in enumerate(score_columns):
                for target_index, target in enumerate(target_columns):
                    rows.append(
                        {
                            **base,
                            "score": score,
                            "target": target,
                            "sample_count": sample_count,
                            "rank_ic": correlations[score_index, target_index],
                            "top_minus_bottom": spreads[score_index, target_index],
                        }
                    )
    return rows


def evaluate_frame_exact(frame: pd.DataFrame, mode: str) -> pd.DataFrame:
    if "pit_audit_pass" in frame and not bool(frame["pit_audit_pass"].fillna(False).all()):
        raise AssertionError(f"refusing to evaluate {mode} features with failed PIT rows")
    if mode == "intraday":
        key_names = ["date", "decision_time", "layer_id", "scale", "level"]
        target_pattern = r"target_\d+m"
    elif mode == "daily":
        key_names = ["date", "layer_id", "scale", "level"]
        target_pattern = r"target_\d+d_open"
    else:
        raise ValueError("mode must be intraday or daily")
    if missing := set(key_names) - set(frame):
        raise ValueError(f"{mode} evaluation input missing group columns {sorted(missing)}")

    scores = [column for column in SCORES if column in frame and pd.api.types.is_numeric_dtype(frame[column])]
    targets = [column for column in frame if re.fullmatch(target_pattern, column) and pd.api.types.is_numeric_dtype(frame[column])]
    rows: list[dict] = []
    for keys, subset in frame.groupby(key_names, dropna=False, sort=False):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        rows.extend(_evaluate_group(subset, scores, targets, key_tuple, key_names))
    return pd.DataFrame(rows)


def evaluate_intraday_frame_exact(frame: pd.DataFrame) -> pd.DataFrame:
    return evaluate_frame_exact(frame, "intraday")


def evaluate_daily_frame_exact(frame: pd.DataFrame) -> pd.DataFrame:
    return evaluate_frame_exact(frame, "daily")
