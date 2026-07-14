#!/usr/bin/env python3
"""Build a compact, decision-oriented P0 Eval report without moving raw metrics.

The report consumes the partitioned ``p0_alpha_metrics.parquet`` dataset produced
by ``p2_p0_eval_streaming.py`` and emits a small self-contained report bundle.
Raw metric shards are scanned in bounded Arrow batches and are never copied into
that bundle.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import re
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from scipy.stats import t as student_t
except Exception:  # pragma: no cover - scipy is a project dependency
    student_t = None

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

REPORT_CONTRACT_VERSION = "p0-eval-research-report-v1"
EXPECTED_EVAL_CONTRACT = "p0-eval-pairwise-resumable-v3"
COMBO_KEYS = ["kind", "feature", "target", "layer_id", "scale"]
DAILY_KEYS = ["date", *COMBO_KEYS]
METRIC_COLUMNS = [
    "date",
    "decision_time",
    "kind",
    "layer_id",
    "scale",
    "feature",
    "target",
    "sample_count",
    "rank_ic",
    "top_minus_bottom",
]
SCALE_ROLES = {"5m": "trigger", "15m": "confirm", "30m": "structural"}
FEATURE_SEMANTICS = {
    "p0_total_edge_count": (
        "node_connectivity",
        "节点在当前图快照中的总连接数量，反映关系广度与网络活跃度。",
    ),
    "p0_total_weight_sum": (
        "weighted_centrality",
        "节点全部入边与出边的绝对权重总和，反映加权中心性与网络暴露。",
    ),
    "p0_edge_spillover_signal": (
        "mean_spillover",
        "邻接来源过去已实现收益乘以边权后的均值，衡量平均传播方向。",
    ),
    "p0_edge_spillover_sum": (
        "total_spillover",
        "邻接来源过去已实现收益乘以边权后的总和，衡量累计传播压力。",
    ),
    "p0_edge_count": (
        "incoming_breadth",
        "参与目标节点传播计算的来源边数量，反映传播广度。",
    ),
    "p0_edge_abs_weight": (
        "incoming_strength",
        "目标节点全部来源边绝对权重总和，反映累计关系强度。",
    ),
    "p0_edge_mean_abs_weight": (
        "mean_edge_strength",
        "目标节点来源边平均绝对权重，反映单条关系的平均质量。",
    ),
}


@dataclass(frozen=True)
class Gates:
    min_abs_ic: float = 0.015
    min_direction_rate: float = 0.55
    min_days: int = 15
    min_day_coverage: float = 0.70
    min_periods: int = 100
    max_top3_abs_ic_share: float = 0.50
    require_ic_spread_sign_agreement: bool = True


@dataclass
class ScanStats:
    shards: int = 0
    batches: int = 0
    metric_rows: int = 0
    skipped_empty_batches: int = 0


def _safe_number(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if not isinstance(value, (list, dict, tuple, set)) and pd.isna(value):
        return None
    return value


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return _safe_number(value)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _layer_catalog() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    try:
        from graphfactorfactory.domain.layers import LAYERS

        for layer in LAYERS:
            rows.append(
                {
                    "layer_id": str(layer.layer_id),
                    "layer_name": str(layer.name),
                    "layer_family": str(layer.family),
                    "layer_directed": bool(layer.directed),
                    "layer_transform": str(layer.transform),
                    "configured_scales": ",".join(f"{value}m" for value in layer.lookbacks_minutes),
                }
            )
    except Exception:
        pass
    return pd.DataFrame.from_records(
        rows,
        columns=[
            "layer_id",
            "layer_name",
            "layer_family",
            "layer_directed",
            "layer_transform",
            "configured_scales",
        ],
    )


def _resolve_eval_dir(path: str | Path) -> Path:
    root = Path(path)
    if (root / "p0_alpha_metrics.parquet").exists():
        return root
    candidates = sorted(root.rglob("p0_alpha_metrics.parquet")) if root.exists() else []
    candidate_dirs = [item.parent for item in candidates]
    if len(candidate_dirs) == 1:
        return candidate_dirs[0]
    if not candidate_dirs:
        raise FileNotFoundError(f"p0_alpha_metrics.parquet not found under {root}")
    raise ValueError(
        "multiple P0 Eval scopes found; pass the exact scope directory: "
        + ", ".join(str(item) for item in candidate_dirs[:10])
    )


def _metric_shards(eval_dir: Path) -> list[Path]:
    dataset = eval_dir / "p0_alpha_metrics.parquet"
    if not dataset.is_dir():
        raise ValueError(
            f"expected partitioned Parquet dataset directory at {dataset}; "
            "legacy single-file layouts must be regenerated by the current evaluator"
        )
    shards = sorted(dataset.glob("part-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no metric shards found under {dataset}")
    return shards


def _batch_daily_partial(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(METRIC_COLUMNS) - set(frame)
    if missing:
        raise ValueError(f"P0 metric batch missing columns: {sorted(missing)}")
    frame = frame[METRIC_COLUMNS].copy()
    frame = frame.dropna(subset=["date", "decision_time", *COMBO_KEYS])
    if frame.empty:
        return pd.DataFrame()
    frame["date"] = frame["date"].astype(str)
    for column in ("kind", "feature", "target", "layer_id", "scale"):
        frame[column] = frame[column].astype(str)
    frame["decision_time"] = pd.to_datetime(frame["decision_time"], utc=True, errors="coerce")
    for column in ("sample_count", "rank_ic", "top_minus_bottom"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["decision_time"])
    if frame.empty:
        return pd.DataFrame()

    frame["periods"] = 1
    frame["ic_count"] = frame["rank_ic"].notna().astype("int64")
    frame["ic_sum"] = frame["rank_ic"].fillna(0.0)
    frame["ic_sumsq"] = frame["rank_ic"].fillna(0.0).pow(2)
    frame["ic_positive_count"] = frame["rank_ic"].gt(0).astype("int64")
    frame["spread_count"] = frame["top_minus_bottom"].notna().astype("int64")
    frame["spread_sum"] = frame["top_minus_bottom"].fillna(0.0)
    frame["spread_sumsq"] = frame["top_minus_bottom"].fillna(0.0).pow(2)
    frame["spread_positive_count"] = frame["top_minus_bottom"].gt(0).astype("int64")
    frame["sample_count_sum"] = frame["sample_count"].fillna(0.0)

    sum_columns = [
        "periods",
        "ic_count",
        "ic_sum",
        "ic_sumsq",
        "ic_positive_count",
        "spread_count",
        "spread_sum",
        "spread_sumsq",
        "spread_positive_count",
        "sample_count_sum",
    ]
    return frame.groupby(DAILY_KEYS, sort=False, dropna=False)[sum_columns].sum().reset_index()


def scan_metrics(
    shards: list[Path],
    *,
    batch_size: int = 250_000,
    progress_every: int = 25,
) -> tuple[pd.DataFrame, ScanStats]:
    partials: list[pd.DataFrame] = []
    stats = ScanStats(shards=len(shards))
    started = time.time()
    for shard_index, shard in enumerate(shards, start=1):
        parquet = pq.ParquetFile(shard)
        try:
            available = set(parquet.schema.names)
            missing = set(METRIC_COLUMNS) - available
            if missing:
                raise ValueError(f"{shard} missing metric columns {sorted(missing)}")
            for batch in parquet.iter_batches(
                columns=METRIC_COLUMNS,
                batch_size=max(1, int(batch_size)),
                use_threads=False,
            ):
                stats.batches += 1
                table = pa.Table.from_batches([batch])
                frame = table.to_pandas(split_blocks=True, self_destruct=True)
                stats.metric_rows += len(frame)
                partial = _batch_daily_partial(frame)
                if partial.empty:
                    stats.skipped_empty_batches += 1
                else:
                    partials.append(partial)
        finally:
            parquet.close()
        if progress_every > 0 and (shard_index % progress_every == 0 or shard_index == len(shards)):
            elapsed = max(time.time() - started, 1e-9)
            print(
                f"[p0-report] shards={shard_index}/{len(shards)} "
                f"metric_rows={stats.metric_rows:,} rate={stats.metric_rows / elapsed:,.0f} rows/s",
                flush=True,
            )

    if not partials:
        raise ValueError("P0 metric dataset contains no usable rows")
    combined = pd.concat(partials, ignore_index=True)
    numeric = [column for column in combined if column not in DAILY_KEYS]
    daily = combined.groupby(DAILY_KEYS, sort=False, dropna=False)[numeric].sum().reset_index()
    daily["daily_mean_ic"] = daily["ic_sum"] / daily["ic_count"].replace(0, np.nan)
    daily["daily_mean_spread"] = daily["spread_sum"] / daily["spread_count"].replace(0, np.nan)
    return daily, stats


def _sample_std(sum_value: float, sumsq: float, count: int) -> float:
    if count < 2:
        return float("nan")
    numerator = float(sumsq) - float(sum_value) ** 2 / count
    return math.sqrt(max(numerator, 0.0) / (count - 1))


def _two_sided_t_pvalue(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    count = len(numeric)
    if count < 2:
        return float("nan")
    std = numeric.std(ddof=1)
    if not np.isfinite(std) or std == 0:
        return 0.0 if numeric.mean() != 0 else 1.0
    statistic = numeric.mean() / (std / math.sqrt(count))
    if student_t is None:
        return math.erfc(abs(statistic) / math.sqrt(2.0))
    return float(2.0 * student_t.sf(abs(statistic), df=count - 1))


def _bh_qvalues(pvalues: pd.Series) -> pd.Series:
    values = pd.to_numeric(pvalues, errors="coerce").to_numpy(dtype=float)
    result = np.full(len(values), np.nan, dtype=float)
    valid_indices = np.flatnonzero(np.isfinite(values))
    if len(valid_indices) == 0:
        return pd.Series(result, index=pvalues.index)
    order = valid_indices[np.argsort(values[valid_indices])]
    ranked = values[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    result[order] = np.minimum(ranked, 1.0)
    return pd.Series(result, index=pvalues.index)


def _concentration(values: pd.Series, top_n: int) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna().abs().sort_values(ascending=False)
    total = numeric.sum()
    if len(numeric) == 0 or not np.isfinite(total) or total == 0:
        return float("nan")
    return float(numeric.head(top_n).sum() / total)


def _leave_one_out_sign_rate(values: pd.Series, reference_sign: float) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if len(numeric) < 2 or reference_sign == 0 or not np.isfinite(reference_sign):
        return float("nan")
    total = numeric.sum()
    leave_one = (total - numeric) / (len(numeric) - 1)
    return float((np.sign(leave_one) == reference_sign).mean())


def _target_minutes(value: str) -> int | None:
    match = re.search(r"(\d+)m$", str(value))
    return int(match.group(1)) if match else None


def _add_term_structure(scorecard: pd.DataFrame) -> pd.DataFrame:
    base_keys = ["kind", "feature", "layer_id", "scale"]
    records: list[dict[str, Any]] = []
    for key, subset in scorecard.groupby(base_keys, sort=False, dropna=False):
        ordered = subset.sort_values("target_minutes", na_position="last")
        signs = np.sign(pd.to_numeric(ordered["mean_rank_ic"], errors="coerce")).to_numpy(dtype=float)
        nonzero = signs[np.isfinite(signs) & (signs != 0)]
        sign_changes = int(np.sum(nonzero[1:] != nonzero[:-1])) if len(nonzero) > 1 else 0
        if len(nonzero) == 0:
            pattern = "flat_or_missing"
        elif (nonzero > 0).all():
            pattern = "all_positive"
        elif (nonzero < 0).all():
            pattern = "all_negative"
        elif nonzero[0] > 0 and nonzero[-1] < 0:
            pattern = "positive_to_negative"
        elif nonzero[0] < 0 and nonzero[-1] > 0:
            pattern = "negative_to_positive"
        else:
            pattern = "mixed"
        records.append(
            {
                **dict(zip(base_keys, key)),
                "horizon_count": int(len(ordered)),
                "horizon_candidate_count": int(ordered.get("candidate_pass", pd.Series(False, index=ordered.index)).sum()),
                "horizon_sign_consistency": float(max((nonzero > 0).mean(), (nonzero < 0).mean())) if len(nonzero) else float("nan"),
                "horizon_sign_changes": sign_changes,
                "term_structure_pattern": pattern,
                "target_sequence": ",".join(ordered["target"].astype(str)),
            }
        )
    terms = pd.DataFrame.from_records(records)
    return scorecard.merge(terms, on=base_keys, how="left")


def build_scorecard(daily: pd.DataFrame, gates: Gates) -> tuple[pd.DataFrame, pd.DataFrame]:
    total_days = int(daily["date"].nunique())
    required_days = min(max(1, gates.min_days), max(1, total_days))
    rows: list[dict[str, Any]] = []

    for key, subset in daily.groupby(COMBO_KEYS, sort=False, dropna=False):
        kind, feature, target, layer_id, scale = key
        periods = int(subset["periods"].sum())
        ic_count = int(subset["ic_count"].sum())
        spread_count = int(subset["spread_count"].sum())
        ic_sum = float(subset["ic_sum"].sum())
        spread_sum = float(subset["spread_sum"].sum())
        mean_ic = ic_sum / ic_count if ic_count else float("nan")
        mean_spread = spread_sum / spread_count if spread_count else float("nan")
        ic_sign = float(np.sign(mean_ic)) if np.isfinite(mean_ic) else 0.0
        spread_sign = float(np.sign(mean_spread)) if np.isfinite(mean_spread) else 0.0
        daily_ic = pd.to_numeric(subset["daily_mean_ic"], errors="coerce").dropna()
        daily_spread = pd.to_numeric(subset["daily_mean_spread"], errors="coerce").dropna()
        daily_ic_std = float(daily_ic.std(ddof=1)) if len(daily_ic) > 1 else float("nan")
        daily_spread_std = float(daily_spread.std(ddof=1)) if len(daily_spread) > 1 else float("nan")
        daily_ic_tstat = (
            float(daily_ic.mean() / (daily_ic_std / math.sqrt(len(daily_ic))))
            if len(daily_ic) > 1 and np.isfinite(daily_ic_std) and daily_ic_std > 0
            else float("nan")
        )
        daily_spread_tstat = (
            float(daily_spread.mean() / (daily_spread_std / math.sqrt(len(daily_spread))))
            if len(daily_spread) > 1 and np.isfinite(daily_spread_std) and daily_spread_std > 0
            else float("nan")
        )
        days = int(subset["date"].nunique())
        rows.append(
            {
                "kind": str(kind),
                "feature": str(feature),
                "target": str(target),
                "target_minutes": _target_minutes(str(target)),
                "layer_id": str(layer_id),
                "scale": str(scale),
                "scale_role": SCALE_ROLES.get(str(scale), "other"),
                "days": days,
                "total_eval_days": total_days,
                "day_coverage": days / total_days if total_days else float("nan"),
                "periods": periods,
                "periods_per_day": periods / days if days else float("nan"),
                "sample_count": int(subset["sample_count_sum"].sum()),
                "mean_sample_count_per_period": float(subset["sample_count_sum"].sum() / periods) if periods else float("nan"),
                "valid_ic_rate": ic_count / periods if periods else float("nan"),
                "valid_spread_rate": spread_count / periods if periods else float("nan"),
                "mean_rank_ic": mean_ic,
                "snapshot_ic_std": _sample_std(ic_sum, float(subset["ic_sumsq"].sum()), ic_count),
                "snapshot_ic_positive_rate": int(subset["ic_positive_count"].sum()) / ic_count if ic_count else float("nan"),
                "mean_spread": mean_spread,
                "snapshot_spread_std": _sample_std(spread_sum, float(subset["spread_sumsq"].sum()), spread_count),
                "snapshot_spread_positive_rate": int(subset["spread_positive_count"].sum()) / spread_count if spread_count else float("nan"),
                "daily_mean_ic": float(daily_ic.mean()) if len(daily_ic) else float("nan"),
                "daily_median_ic": float(daily_ic.median()) if len(daily_ic) else float("nan"),
                "daily_ic_std": daily_ic_std,
                "daily_ic_tstat": daily_ic_tstat,
                "daily_ic_pvalue": _two_sided_t_pvalue(daily_ic),
                "daily_ic_direction_rate": float((np.sign(daily_ic) == ic_sign).mean()) if len(daily_ic) and ic_sign else float("nan"),
                "daily_mean_spread": float(daily_spread.mean()) if len(daily_spread) else float("nan"),
                "daily_median_spread": float(daily_spread.median()) if len(daily_spread) else float("nan"),
                "daily_spread_std": daily_spread_std,
                "daily_spread_tstat": daily_spread_tstat,
                "daily_spread_direction_rate": float((np.sign(daily_spread) == spread_sign).mean()) if len(daily_spread) and spread_sign else float("nan"),
                "ic_spread_sign_agreement": bool(ic_sign != 0 and ic_sign == spread_sign),
                "largest_day_abs_ic_share": _concentration(daily_ic, 1),
                "top3_days_abs_ic_share": _concentration(daily_ic, 3),
                "leave_one_day_out_sign_rate": _leave_one_out_sign_rate(daily_ic, ic_sign),
                "best_day_ic": float(daily_ic.max()) if len(daily_ic) else float("nan"),
                "worst_day_ic": float(daily_ic.min()) if len(daily_ic) else float("nan"),
                "required_days": required_days,
            }
        )

    scorecard = pd.DataFrame.from_records(rows)
    if scorecard.empty:
        raise ValueError("no P0 Eval combinations could be summarized")
    scorecard["daily_ic_qvalue_bh"] = _bh_qvalues(scorecard["daily_ic_pvalue"])

    catalog = _layer_catalog()
    if not catalog.empty:
        scorecard = scorecard.merge(catalog, on="layer_id", how="left")
    for column, default in (
        ("layer_name", "unknown_layer"),
        ("layer_family", "unknown"),
        ("layer_directed", False),
        ("layer_transform", "unknown"),
        ("configured_scales", "unknown"),
    ):
        if column not in scorecard:
            scorecard[column] = default
        else:
            scorecard[column] = scorecard[column].fillna(default)

    scorecard["feature_family"] = scorecard["feature"].map(
        lambda value: FEATURE_SEMANTICS.get(str(value), ("other_p0", "未登记的 P0 特征；需要结合生成代码复核金融语义。"))[0]
    )
    scorecard["feature_interpretation"] = scorecard["feature"].map(
        lambda value: FEATURE_SEMANTICS.get(str(value), ("other_p0", "未登记的 P0 特征；需要结合生成代码复核金融语义。"))[1]
    )

    conditions = {
        "abs_ic": scorecard["mean_rank_ic"].abs() >= gates.min_abs_ic,
        "direction": scorecard["daily_ic_direction_rate"] >= gates.min_direction_rate,
        "days": scorecard["days"] >= scorecard["required_days"],
        "coverage": scorecard["day_coverage"] >= gates.min_day_coverage,
        "periods": scorecard["periods"] >= gates.min_periods,
        "concentration": scorecard["top3_days_abs_ic_share"] <= gates.max_top3_abs_ic_share,
        "sign_agreement": scorecard["ic_spread_sign_agreement"] if gates.require_ic_spread_sign_agreement else pd.Series(True, index=scorecard.index),
    }
    for name, values in conditions.items():
        scorecard[f"gate_{name}"] = values.fillna(False)
    gate_columns = [f"gate_{name}" for name in conditions]
    scorecard["candidate_pass"] = scorecard[gate_columns].all(axis=1)
    scorecard["gate_failures"] = scorecard.apply(
        lambda row: ";".join(name for name in conditions if not bool(row[f"gate_{name}"])) or "none",
        axis=1,
    )
    scorecard["direction"] = np.where(scorecard["mean_rank_ic"] >= 0, "positive", "negative")

    strength = np.minimum(scorecard["mean_rank_ic"].abs() / max(gates.min_abs_ic * 2.0, 1e-9), 1.0)
    direction = np.clip((scorecard["daily_ic_direction_rate"].fillna(0.0) - 0.5) / 0.2, 0.0, 1.0)
    coverage = scorecard["day_coverage"].fillna(0.0).clip(0.0, 1.0)
    concentration = (1.0 - scorecard["top3_days_abs_ic_share"].fillna(1.0)).clip(0.0, 1.0)
    loo = scorecard["leave_one_day_out_sign_rate"].fillna(0.0).clip(0.0, 1.0)
    multiple_testing = (1.0 - scorecard["daily_ic_qvalue_bh"].fillna(1.0)).clip(0.0, 1.0)
    sign_alignment = scorecard["ic_spread_sign_agreement"].astype(float)
    scorecard["research_score"] = (
        25.0 * strength
        + 20.0 * direction
        + 15.0 * coverage
        + 15.0 * concentration
        + 10.0 * loo
        + 10.0 * sign_alignment
        + 5.0 * multiple_testing
    ).round(3)

    scorecard = _add_term_structure(scorecard)
    scorecard = scorecard.sort_values(
        ["candidate_pass", "research_score", "mean_rank_ic"],
        ascending=[False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)

    candidate_keys = scorecard.loc[scorecard["candidate_pass"], COMBO_KEYS]
    if candidate_keys.empty:
        candidate_daily = daily.iloc[0:0].copy()
    else:
        candidate_daily = daily.merge(candidate_keys.drop_duplicates(), on=COMBO_KEYS, how="inner")
        candidate_daily = candidate_daily.merge(
            scorecard[COMBO_KEYS + ["layer_name", "feature_family", "direction", "research_score"]],
            on=COMBO_KEYS,
            how="left",
        )
    return scorecard, candidate_daily


def _overview(scorecard: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, subset in scorecard.groupby(group_columns, sort=False, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        best = subset.sort_values("research_score", ascending=False).iloc[0]
        rows.append(
            {
                **dict(zip(group_columns, key_values)),
                "tested_combinations": int(len(subset)),
                "candidate_combinations": int(subset["candidate_pass"].sum()),
                "candidate_rate": float(subset["candidate_pass"].mean()),
                "median_abs_ic": float(subset["mean_rank_ic"].abs().median()),
                "best_abs_ic": float(subset["mean_rank_ic"].abs().max()),
                "median_direction_rate": float(subset["daily_ic_direction_rate"].median()),
                "median_top3_concentration": float(subset["top3_days_abs_ic_share"].median()),
                "best_feature": str(best["feature"]),
                "best_target": str(best["target"]),
                "best_research_score": float(best["research_score"]),
            }
        )
    return pd.DataFrame.from_records(rows).sort_values(
        ["candidate_combinations", "best_research_score"], ascending=[False, False]
    )


def _decision(scorecard: pd.DataFrame) -> dict[str, Any]:
    candidates = scorecard[scorecard["candidate_pass"]].copy()
    tested = len(scorecard)
    tested_layers = int(scorecard["layer_id"].nunique())
    candidate_layers = int(candidates["layer_id"].nunique()) if not candidates.empty else 0
    candidate_features = int(candidates["feature"].nunique()) if not candidates.empty else 0
    candidate_horizons = int(candidates["target"].nunique()) if not candidates.empty else 0
    candidate_rate = len(candidates) / tested if tested else 0.0

    if candidates.empty:
        action = "stop_full_p0_eval_after_discovery_month"
        rationale = "没有组合通过强度、稳定性、覆盖率、极端集中度与 IC/Spread 方向一致性闸门。"
    elif (
        candidate_rate >= 0.20
        and candidate_layers >= min(3, max(1, tested_layers))
        and candidate_features >= 3
        and candidate_horizons >= 3
        and float(candidates["top3_days_abs_ic_share"].median()) <= 0.50
    ):
        action = "continue_broad_p0_eval_validation"
        rationale = "候选覆盖多个 Layer、特征与 horizon，且不是由少数极端交易日主导，可继续广泛样本外验证。"
    else:
        action = "validate_whitelist_only"
        rationale = "有效组合较稀疏；保留候选白名单，在两个不同市场环境月份验证，不继续全量 P0 Eval。"

    return {
        "recommended_action": action,
        "rationale": rationale,
        "tested_combinations": tested,
        "candidate_combinations": int(len(candidates)),
        "candidate_rate": candidate_rate,
        "tested_layers": tested_layers,
        "candidate_layers": candidate_layers,
        "candidate_features": candidate_features,
        "candidate_horizons": candidate_horizons,
        "discovery_only_warning": (
            "首月结果属于候选发现，不能视为样本外确认；应控制多重检验并在不同市场环境月份复核。"
        ),
    }


def _summary_audit(eval_dir: Path, scorecard: pd.DataFrame) -> pd.DataFrame:
    source = eval_dir / "p0_alpha_summary.csv"
    columns = [
        *COMBO_KEYS,
        "summary_present",
        "days_diff",
        "snapshots_diff",
        "sample_count_diff",
        "mean_rank_ic_diff",
        "mean_spread_diff",
        "positive_period_rate_diff",
    ]
    if not source.exists():
        return pd.DataFrame(columns=columns)
    summary = pd.read_csv(source, dtype={"layer_id": "string", "scale": "string"})
    for column in COMBO_KEYS:
        summary[column] = summary[column].astype(str)
    current = scorecard.rename(
        columns={
            "periods": "snapshots",
            "snapshot_spread_positive_rate": "positive_period_rate",
        }
    )
    merged = current[
        COMBO_KEYS + ["days", "snapshots", "sample_count", "mean_rank_ic", "mean_spread", "positive_period_rate"]
    ].merge(
        summary[COMBO_KEYS + ["days", "snapshots", "sample_count", "mean_rank_ic", "mean_spread", "positive_period_rate"]],
        on=COMBO_KEYS,
        how="outer",
        suffixes=("_report", "_evaluator"),
        indicator=True,
    )
    result = merged[COMBO_KEYS].copy()
    result["summary_present"] = merged["_merge"].eq("both")
    for column in ("days", "snapshots", "sample_count", "mean_rank_ic", "mean_spread", "positive_period_rate"):
        result[f"{column}_diff"] = pd.to_numeric(merged[f"{column}_report"], errors="coerce") - pd.to_numeric(
            merged[f"{column}_evaluator"], errors="coerce"
        )
    return result[columns]


def _format_frame(frame: pd.DataFrame, columns: list[str], limit: int | None = None) -> pd.DataFrame:
    selected = frame.loc[:, [column for column in columns if column in frame]].copy()
    if limit is not None:
        selected = selected.head(limit)
    for column in selected:
        if pd.api.types.is_float_dtype(selected[column]):
            selected[column] = selected[column].map(lambda value: "" if pd.isna(value) else f"{value:.6g}")
    return selected


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_无数据_"
    values = frame.astype(str).replace({"nan": "", "<NA>": ""})
    headers = [str(column) for column in values.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in values.itertuples(index=False, name=None):
        escaped = [str(value).replace("|", "\\|").replace("\n", " ") for value in row]
        lines.append("| " + " | ".join(escaped) + " |")
    return "\n".join(lines)


def _write_atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _write_atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _report_markdown(
    *,
    decision: dict[str, Any],
    manifest: dict[str, Any],
    gates: Gates,
    stats: ScanStats,
    scorecard: pd.DataFrame,
    layer_overview: pd.DataFrame,
    feature_overview: pd.DataFrame,
    horizon_overview: pd.DataFrame,
    audit: pd.DataFrame,
    top_n: int,
) -> str:
    top_candidates = _format_frame(
        scorecard[scorecard["candidate_pass"]],
        [
            "layer_id",
            "layer_name",
            "layer_family",
            "scale",
            "scale_role",
            "kind",
            "feature",
            "feature_family",
            "target",
            "direction",
            "mean_rank_ic",
            "daily_ic_direction_rate",
            "mean_spread",
            "days",
            "day_coverage",
            "top3_days_abs_ic_share",
            "daily_ic_qvalue_bh",
            "research_score",
            "term_structure_pattern",
        ],
        top_n,
    )
    top_all = _format_frame(
        scorecard,
        [
            "layer_id",
            "layer_name",
            "scale",
            "kind",
            "feature",
            "target",
            "mean_rank_ic",
            "daily_ic_direction_rate",
            "mean_spread",
            "top3_days_abs_ic_share",
            "candidate_pass",
            "gate_failures",
            "research_score",
        ],
        top_n,
    )
    audit_failures = (
        audit[
            (~audit["summary_present"])
            | audit[[column for column in audit if column.endswith("_diff")]].abs().gt(1e-10).any(axis=1)
        ]
        if not audit.empty
        else audit
    )
    graph_scope_note = (
        "当前 P0 Eval 只发现并评估 `p0_node_features.parquet` 与 "
        "`p0_edge_spillover_features.parquet`；`p0_graph_state_features.parquet` "
        "没有 symbol-level 横截面，因此不在本报告的 IC/Spread 范围内。"
    )
    return f"""# P0 Eval 首月研究报告

## 执行结论

- **建议：** `{decision['recommended_action']}`
- **理由：** {decision['rationale']}
- 测试组合：{decision['tested_combinations']:,}
- 通过候选：{decision['candidate_combinations']:,}（{decision['candidate_rate']:.2%}）
- 候选覆盖：{decision['candidate_layers']} 个 Layer、{decision['candidate_features']} 个特征、{decision['candidate_horizons']} 个 horizon
- **研究边界：** {decision['discovery_only_warning']}

## 数据与审计

- Eval contract：`{manifest.get('evaluation_contract_version', 'unknown')}`
- Eval 状态：`{manifest.get('status', 'unknown')}`
- Metric shards：{stats.shards:,}
- 扫描 metric rows：{stats.metric_rows:,}
- 批次数：{stats.batches:,}
- 缺失值语义：`{manifest.get('missing_data_semantics', 'unknown')}`
- 报告 contract：`{REPORT_CONTRACT_VERSION}`
- Scope：{graph_scope_note}
- Summary 一致性异常：{len(audit_failures):,}

## 候选闸门

| 条件 | 阈值 |
| --- | ---: |
| `abs(mean_rank_ic)` | ≥ {gates.min_abs_ic:.4f} |
| 日级方向一致率 | ≥ {gates.min_direction_rate:.2%} |
| 最低交易日 | ≥ {gates.min_days}（若样本不足则取实际总日数） |
| 日期覆盖率 | ≥ {gates.min_day_coverage:.2%} |
| 最低 snapshot 指标数 | ≥ {gates.min_periods:,} |
| Top-3 日绝对 IC 贡献 | ≤ {gates.max_top3_abs_ic_share:.2%} |
| IC 与 Spread 方向一致 | {'必须' if gates.require_ic_spread_sign_agreement else '不强制'} |

`research_score` 只是透明排序工具，不替代上述硬闸门，也不代表样本外收益。

## 通过候选 Top {top_n}

{_markdown_table(top_candidates)}

## 全组合研究排序 Top {top_n}

{_markdown_table(top_all)}

## Layer × Scale 概览

{_markdown_table(_format_frame(layer_overview, list(layer_overview.columns), top_n))}

## Feature 概览

{_markdown_table(_format_frame(feature_overview, list(feature_overview.columns), top_n))}

## Horizon 概览

{_markdown_table(_format_frame(horizon_overview, list(horizon_overview.columns), top_n))}

## 金融语义与解读原则

1. `p0_total_edge_count`：关系广度，不等同于方向性 Alpha。
2. `p0_total_weight_sum`：加权中心性或网络暴露，需要检查是否只是规模／流动性代理。
3. `p0_edge_spillover_signal`：平均传播方向，对 degree 的敏感度较低。
4. `p0_edge_spillover_sum`：累计传播压力，可能同时暴露于边数量与关系强度。
5. `p0_edge_count`、`p0_edge_abs_weight`、`p0_edge_mean_abs_weight`：分别对应传播广度、总强度与平均关系质量。
6. 负 IC 并不自动无效；只要 IC、Spread 与日级方向稳定一致，它可以作为反向因子。
7. 首月只负责发现。候选应优先在两个不同市场环境月份验证，而不是直接全量跑剩余月份。

## 下一步规则

- `stop_full_p0_eval_after_discovery_month`：保留 P0 原始因子，停止后续全量 Eval。
- `validate_whitelist_only`：只对 `p0_eval_candidate_whitelist.csv` 中的组合跑两个验证月。
- `continue_broad_p0_eval_validation`：候选覆盖面足够广，可继续较完整的跨月验证，但仍不能把首月视为样本外证据。

## 输出文件

- `p0_eval_combo_scorecard.csv`：全部组合的完整评分与闸门结果。
- `p0_eval_candidate_whitelist.csv`：后续跨月验证白名单。
- `p0_eval_candidate_daily_stability.csv`：仅候选组合的日级稳定性。
- `p0_eval_layer_scale_overview.csv`
- `p0_eval_feature_overview.csv`
- `p0_eval_horizon_overview.csv`
- `p0_eval_term_structure.csv`
- `p0_eval_summary_consistency_audit.csv`
- `p0_eval_decision.json`
- `p0_eval_report_bundle.zip`
"""


def _report_html(markdown_text: str, scorecard: pd.DataFrame, decision: dict[str, Any], top_n: int) -> str:
    candidates = _format_frame(
        scorecard[scorecard["candidate_pass"]],
        [
            "layer_id",
            "layer_name",
            "scale",
            "kind",
            "feature",
            "target",
            "direction",
            "mean_rank_ic",
            "daily_ic_direction_rate",
            "mean_spread",
            "top3_days_abs_ic_share",
            "daily_ic_qvalue_bh",
            "research_score",
        ],
        top_n,
    )
    narrative = html.escape(markdown_text).replace("\n", "<br>\n")
    return f"""<!doctype html>
<html lang="zh-Hans">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>P0 Eval 首月研究报告</title>
<style>
body {{ font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; margin: 32px auto; max-width: 1500px; line-height: 1.55; color: #1f2937; }}
h1,h2 {{ color: #111827; }}
.badge {{ display:inline-block; padding:6px 10px; border-radius:8px; background:#eef2ff; font-weight:700; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; margin: 16px 0 32px; }}
th,td {{ border:1px solid #d1d5db; padding:6px 8px; text-align:right; }}
th:first-child,td:first-child {{ text-align:left; }}
th {{ background:#f3f4f6; position:sticky; top:0; }}
pre {{ white-space:pre-wrap; background:#f9fafb; padding:16px; border:1px solid #e5e7eb; border-radius:8px; }}
</style>
</head>
<body>
<h1>P0 Eval 首月研究报告</h1>
<p class="badge">建议：{html.escape(str(decision['recommended_action']))}</p>
<p>{html.escape(str(decision['rationale']))}</p>
<h2>候选白名单 Top {top_n}</h2>
{candidates.to_html(index=False, escape=True, border=0)}
<h2>完整 Markdown 报告</h2>
<pre>{narrative}</pre>
</body>
</html>
"""


def generate_report(
    eval_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    gates: Gates | None = None,
    batch_size: int = 250_000,
    top_n: int = 30,
    progress_every: int = 25,
    allow_incomplete: bool = False,
    allow_unknown_contract: bool = False,
    allow_summary_mismatch: bool = False,
    include_all_daily: bool = False,
) -> dict[str, Any]:
    started = time.time()
    gates = gates or Gates()
    eval_path = _resolve_eval_dir(eval_dir)
    report_root = Path(output_dir) if output_dir is not None else eval_path / "report"
    report_root.mkdir(parents=True, exist_ok=True)
    manifest = _read_json(eval_path / "manifest.json")
    status = str(manifest.get("status", "unknown"))
    contract = str(manifest.get("evaluation_contract_version", "unknown"))
    if not allow_incomplete and status != "complete":
        raise ValueError(f"P0 Eval manifest status must be complete, found {status!r}")
    if not allow_unknown_contract and contract != EXPECTED_EVAL_CONTRACT:
        raise ValueError(
            f"unsupported P0 Eval contract {contract!r}; expected {EXPECTED_EVAL_CONTRACT!r}. "
            "Use --allow-unknown-contract only for explicit forensic review."
        )

    shards = _metric_shards(eval_path)
    daily, stats = scan_metrics(shards, batch_size=batch_size, progress_every=progress_every)
    scorecard, candidate_daily = build_scorecard(daily, gates)
    layer_overview = _overview(scorecard, ["layer_id", "layer_name", "layer_family", "scale", "scale_role", "kind"])
    feature_overview = _overview(scorecard, ["kind", "feature", "feature_family"])
    horizon_overview = _overview(scorecard, ["target", "target_minutes"])
    term_structure_columns = [
        "kind",
        "layer_id",
        "layer_name",
        "scale",
        "feature",
        "feature_family",
        "horizon_count",
        "horizon_candidate_count",
        "horizon_sign_consistency",
        "horizon_sign_changes",
        "term_structure_pattern",
        "target_sequence",
    ]
    term_structure = scorecard[term_structure_columns].drop_duplicates().sort_values(
        ["horizon_candidate_count", "horizon_sign_consistency"], ascending=[False, False]
    )
    audit = _summary_audit(eval_path, scorecard)
    audit_diff_columns = [column for column in audit if column.endswith("_diff")]
    audit_failed = False
    if not audit.empty:
        audit_failed = bool((~audit["summary_present"]).any()) or bool(
            audit[audit_diff_columns].abs().gt(1e-10).to_numpy().any()
        )
    if audit_failed and not allow_summary_mismatch:
        worst = audit[audit_diff_columns].abs().max().max()
        raise AssertionError(
            f"generated report does not match evaluator summary; max absolute difference={worst}. "
            "Use --allow-summary-mismatch only to inspect inconsistent artifacts."
        )

    decision = _decision(scorecard)
    decision.update(
        {
            "report_contract_version": REPORT_CONTRACT_VERSION,
            "eval_contract_version": contract,
            "eval_status": status,
            "eval_dir": str(eval_path.resolve()),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "gates": asdict(gates),
            "scan": asdict(stats),
            "graph_state_eval_scope": "excluded_non_symbol_level_cross_section",
            "summary_audit_failed": audit_failed,
        }
    )

    candidate_columns = [
        "priority",
        "layer_id",
        "layer_name",
        "layer_family",
        "scale",
        "scale_role",
        "kind",
        "feature",
        "feature_family",
        "feature_interpretation",
        "target",
        "target_minutes",
        "direction",
        "mean_rank_ic",
        "daily_ic_direction_rate",
        "mean_spread",
        "daily_spread_direction_rate",
        "days",
        "day_coverage",
        "periods",
        "mean_sample_count_per_period",
        "top3_days_abs_ic_share",
        "leave_one_day_out_sign_rate",
        "daily_ic_pvalue",
        "daily_ic_qvalue_bh",
        "research_score",
        "term_structure_pattern",
    ]
    candidates = scorecard[scorecard["candidate_pass"]].copy()
    candidates.insert(0, "priority", np.arange(1, len(candidates) + 1))
    candidates = candidates[candidate_columns]

    markdown = _report_markdown(
        decision=decision,
        manifest=manifest,
        gates=gates,
        stats=stats,
        scorecard=scorecard,
        layer_overview=layer_overview,
        feature_overview=feature_overview,
        horizon_overview=horizon_overview,
        audit=audit,
        top_n=top_n,
    )
    html_report = _report_html(markdown, scorecard, decision, top_n)

    outputs = {
        "p0_eval_report.md": markdown,
        "p0_eval_report.html": html_report,
        "p0_eval_decision.json": json.dumps(_json_ready(decision), indent=2, ensure_ascii=False),
    }
    for name, content in outputs.items():
        _write_atomic_text(report_root / name, content)

    _write_atomic_csv(scorecard, report_root / "p0_eval_combo_scorecard.csv")
    _write_atomic_csv(candidates, report_root / "p0_eval_candidate_whitelist.csv")
    _write_atomic_csv(candidate_daily, report_root / "p0_eval_candidate_daily_stability.csv")
    _write_atomic_csv(layer_overview, report_root / "p0_eval_layer_scale_overview.csv")
    _write_atomic_csv(feature_overview, report_root / "p0_eval_feature_overview.csv")
    _write_atomic_csv(horizon_overview, report_root / "p0_eval_horizon_overview.csv")
    _write_atomic_csv(term_structure, report_root / "p0_eval_term_structure.csv")
    _write_atomic_csv(audit, report_root / "p0_eval_summary_consistency_audit.csv")
    if include_all_daily:
        _write_atomic_csv(daily, report_root / "p0_eval_all_daily_stability.csv")

    file_records = []
    for path in sorted(report_root.iterdir()):
        if path.is_file() and path.suffix != ".zip":
            file_records.append(
                {
                    "name": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
    report_manifest = {
        "report_contract_version": REPORT_CONTRACT_VERSION,
        "source_eval_dir": str(eval_path.resolve()),
        "source_manifest": manifest,
        "source_metric_shards": len(shards),
        "source_metric_rows_scanned": stats.metric_rows,
        "raw_metrics_copied": False,
        "files": file_records,
        "elapsed_sec": round(time.time() - started, 3),
    }
    manifest_path = report_root / "p0_eval_report_manifest.json"
    _write_atomic_text(manifest_path, json.dumps(_json_ready(report_manifest), indent=2, ensure_ascii=False))

    bundle = report_root / "p0_eval_report_bundle.zip"
    temporary_bundle = Path(str(bundle) + ".tmp")
    temporary_bundle.unlink(missing_ok=True)
    with zipfile.ZipFile(temporary_bundle, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(report_root.iterdir()):
            if path.is_file() and path not in {bundle, temporary_bundle}:
                archive.write(path, arcname=path.name)
    os.replace(temporary_bundle, bundle)

    result = {
        **decision,
        "report_dir": str(report_root),
        "bundle": str(bundle),
        "bundle_bytes": bundle.stat().st_size,
        "elapsed_sec": round(time.time() - started, 3),
    }
    print(json.dumps(_json_ready(result), indent=2, ensure_ascii=False), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a compact P0 Eval research report from local partitioned metric shards"
    )
    parser.add_argument("--eval-dir", required=True, help="Exact p0_alpha/<scope> directory or a root containing exactly one scope")
    parser.add_argument("--output-dir", help="Defaults to <eval-dir>/report")
    parser.add_argument("--batch-size", type=int, default=250_000)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--min-abs-ic", type=float, default=0.015)
    parser.add_argument("--min-direction-rate", type=float, default=0.55)
    parser.add_argument("--min-days", type=int, default=15)
    parser.add_argument("--min-day-coverage", type=float, default=0.70)
    parser.add_argument("--min-periods", type=int, default=100)
    parser.add_argument("--max-top3-abs-ic-share", type=float, default=0.50)
    parser.add_argument("--no-require-sign-agreement", action="store_true")
    parser.add_argument("--include-all-daily", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--allow-unknown-contract", action="store_true")
    parser.add_argument("--allow-summary-mismatch", action="store_true")
    args = parser.parse_args()

    gates = Gates(
        min_abs_ic=args.min_abs_ic,
        min_direction_rate=args.min_direction_rate,
        min_days=args.min_days,
        min_day_coverage=args.min_day_coverage,
        min_periods=args.min_periods,
        max_top3_abs_ic_share=args.max_top3_abs_ic_share,
        require_ic_spread_sign_agreement=not args.no_require_sign_agreement,
    )
    generate_report(
        args.eval_dir,
        args.output_dir,
        gates=gates,
        batch_size=args.batch_size,
        top_n=args.top_n,
        progress_every=args.progress_every,
        allow_incomplete=args.allow_incomplete,
        allow_unknown_contract=args.allow_unknown_contract,
        allow_summary_mismatch=args.allow_summary_mismatch,
        include_all_daily=args.include_all_daily,
    )


if __name__ == "__main__":
    main()
