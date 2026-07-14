#!/usr/bin/env python3
"""Generate one compact, auditable monthly Alpha report for the full P0/P2 chain.

The report reads local monthly outputs in place and writes only small JSON/HTML/CSV
artifacts. It never copies raw P0 metric shards, Theme Returns, relation signals,
or feature Parquet files into the report bundle.
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
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from scipy.stats import t as student_t
except Exception:  # pragma: no cover
    student_t = None

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from generate_p0_eval_report import Gates as P0Gates
from generate_p0_eval_report import generate_report as generate_p0_report

REPORT_CONTRACT_VERSION = "monthly-full-alpha-report-v1"
EXPECTED_CONTRACTS = {
    "p0_eval": "p0-eval-pairwise-resumable-v3",
    "theme_returns": "theme-returns-stream-v3",
    "relation_spillover": "relation-spillover-stream-v3",
    "intraday_features": "intraday-relation-features-v3",
    "daily_features": "daily-relation-features-stream-v4",
    "p2_eval": "p2-eval-resumable-partitioned-v3",
}
P2_KEYS = ["score", "target", "layer_id", "scale", "level"]
P2_STATE_COLUMNS = [
    "date", *P2_KEYS, "snapshots", "sample_count", "rank_ic_sum",
    "rank_ic_count", "spread_sum", "spread_count", "positive_count",
]
SCALE_ROLES = {"5m": "trigger", "15m": "confirm", "30m": "structural"}
STAGE_LAYOUT = {
    "theme_returns": ("theme_returns", "theme_returns.parquet", "theme-returns-stream-v3"),
    "relation_spillover": ("relation_spillover", "relation_spillover_signals.parquet", "relation-spillover-stream-v3"),
    "intraday_features": ("intraday_relation_features", "intraday_relation_features.parquet", "intraday-relation-features-v3"),
    "daily_features": ("daily_relation_features", "daily_relation_features.parquet", "daily-relation-features-stream-v4"),
}


@dataclass(frozen=True)
class EvalGates:
    min_abs_ic: float = 0.015
    min_direction_rate: float = 0.55
    min_days: int = 15
    min_day_coverage: float = 0.70
    min_periods: int = 100
    max_top3_abs_ic_share: float = 0.50
    require_ic_spread_sign_agreement: bool = True


@dataclass
class ScanStats:
    files: int = 0
    batches: int = 0
    rows: int = 0


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if value is pd.NA:
        return None
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _partition_value(path: Path, key: str) -> str | None:
    prefix = key + "="
    for part in path.parts:
        if part.startswith(prefix):
            return part[len(prefix):]
    return None


def _month_scope(month: str) -> str:
    if not re.fullmatch(r"\d{4}-\d{2}", month):
        raise ValueError("--month must use YYYY-MM")
    return month.replace("-", "")


def _target_period(value: str) -> int | None:
    match = re.search(r"(\d+)([md])(?:_open)?$", str(value))
    if not match:
        return None
    amount = int(match.group(1))
    return amount if match.group(2) == "m" else amount * 390


def _sample_std(sum_value: float, sumsq: float, count: int) -> float:
    if count < 2:
        return float("nan")
    return math.sqrt(max(float(sumsq) - float(sum_value) ** 2 / count, 0.0) / (count - 1))


def _two_sided_t_pvalue(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if len(numeric) < 2:
        return float("nan")
    std = float(numeric.std(ddof=1))
    if not np.isfinite(std) or std == 0:
        return 0.0 if float(numeric.mean()) != 0 else 1.0
    statistic = float(numeric.mean()) / (std / math.sqrt(len(numeric)))
    if student_t is None:
        return math.erfc(abs(statistic) / math.sqrt(2.0))
    return float(2.0 * student_t.sf(abs(statistic), df=len(numeric) - 1))


def _bh_qvalues(pvalues: pd.Series) -> pd.Series:
    values = pd.to_numeric(pvalues, errors="coerce").to_numpy(dtype=float)
    result = np.full(len(values), np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(values))
    if not len(valid):
        return pd.Series(result, index=pvalues.index)
    order = valid[np.argsort(values[valid])]
    ranked = values[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    result[order] = np.minimum(ranked, 1.0)
    return pd.Series(result, index=pvalues.index)


def _concentration(values: pd.Series, top_n: int = 3) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna().abs().sort_values(ascending=False)
    total = float(numeric.sum())
    return float(numeric.head(top_n).sum() / total) if len(numeric) and total else float("nan")


def _leave_one_out_sign_rate(values: pd.Series, reference_sign: float) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if len(numeric) < 2 or reference_sign == 0 or not np.isfinite(reference_sign):
        return float("nan")
    leave_one = (numeric.sum() - numeric) / (len(numeric) - 1)
    return float((np.sign(leave_one) == reference_sign).mean())


def _layer_catalog() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    try:
        from graphfactorfactory.domain.layers import LAYERS
        for layer in LAYERS:
            rows.append({
                "layer_id": str(layer.layer_id), "layer_name": str(layer.name),
                "layer_family": str(layer.family), "layer_directed": bool(layer.directed),
                "layer_transform": str(layer.transform),
                "configured_scales": ",".join(f"{value}m" for value in layer.lookbacks_minutes),
            })
    except Exception:
        pass
    return pd.DataFrame(rows)


def _add_layer_catalog(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if result.empty:
        return result
    result["layer_id"] = result["layer_id"].astype(str)
    catalog = _layer_catalog()
    if not catalog.empty:
        result = result.merge(catalog, on="layer_id", how="left")
    for column, default in (("layer_name", "unknown_layer"), ("layer_family", "unknown"),
                            ("layer_directed", False), ("layer_transform", "unknown"),
                            ("configured_scales", "unknown")):
        result[column] = result[column].fillna(default) if column in result else default
    return result


def _manifest_audit(p2_root: Path, month: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for stage_name, (root_name, filename, expected_contract) in STAGE_LAYOUT.items():
        for manifest_path in sorted((p2_root / root_name).glob(f"date={month}-*/layer_id=*/scale=*/manifest.json")):
            payload = _read_json(manifest_path)
            status = str(payload.get("status", "missing"))
            contract = str(payload.get("stage_contract_version", "unknown"))
            output = manifest_path.parent / filename
            output_consistent = ((status == "empty" and not output.exists()) or
                                 (status == "complete" and output.exists()))
            rows.append({
                "stage": stage_name,
                "date": str(payload.get("date") or _partition_value(manifest_path, "date") or ""),
                "layer_id": str(payload.get("layer_id") or _partition_value(manifest_path, "layer_id") or ""),
                "scale": str(payload.get("scale") or _partition_value(manifest_path, "scale") or ""),
                "status": status, "contract": contract, "expected_contract": expected_contract,
                "contract_ok": contract == expected_contract,
                "output_rows": int(payload.get("output_rows", 0) or 0),
                "output_exists": output.exists(), "output_consistent": output_consistent,
                "healthy": status in {"complete", "empty"} and contract == expected_contract and output_consistent,
                "manifest": str(manifest_path), "output": str(output),
            })
    return pd.DataFrame(rows)


def _stage_health(audit: pd.DataFrame) -> pd.DataFrame:
    if audit.empty:
        return pd.DataFrame(columns=["stage", "partitions", "complete", "empty", "unhealthy", "dates", "layers", "scales", "output_rows"])
    rows = []
    for stage, subset in audit.groupby("stage", sort=False):
        rows.append({
            "stage": stage, "partitions": len(subset),
            "complete": int(subset["status"].eq("complete").sum()),
            "empty": int(subset["status"].eq("empty").sum()),
            "unhealthy": int((~subset["healthy"]).sum()),
            "dates": int(subset["date"].nunique()), "layers": int(subset["layer_id"].nunique()),
            "scales": int(subset["scale"].nunique()), "output_rows": int(subset["output_rows"].sum()),
        })
    return pd.DataFrame(rows)


def _partition_coverage(audit: pd.DataFrame) -> pd.DataFrame:
    columns = ["check", "status", "missing_count", "orphan_count", "details"]
    if audit.empty:
        return pd.DataFrame([["stage_manifests", "failed", 0, 0, "no manifests found"]], columns=columns)

    def keys(stage: str, statuses: set[str]) -> set[tuple[str, str, str]]:
        subset = audit[audit["stage"].eq(stage) & audit["status"].isin(statuses)]
        return set(subset[["date", "layer_id", "scale"]].itertuples(index=False, name=None))

    theme = keys("theme_returns", {"complete", "empty"})
    relation = keys("relation_spillover", {"complete", "empty"})
    relation_complete = keys("relation_spillover", {"complete"})
    intraday = keys("intraday_features", {"complete", "empty"})
    daily = keys("daily_features", {"complete", "empty"})
    records = []

    def record(name: str, expected: set, actual: set, allow_missing: bool = False) -> None:
        missing = set() if allow_missing else expected - actual
        orphan = actual - expected
        details = []
        if missing:
            details.append("missing=" + ",".join("/".join(x) for x in sorted(missing)[:10]))
        if orphan:
            details.append("orphan=" + ",".join("/".join(x) for x in sorted(orphan)[:10]))
        records.append({"check": name, "status": "pass" if not missing and not orphan else "fail",
                        "missing_count": len(missing), "orphan_count": len(orphan),
                        "details": "; ".join(details) or "aligned"})

    record("relation_has_theme_returns", theme, relation, allow_missing=True)
    record("intraday_features_cover_complete_relations", relation_complete, intraday)
    record("daily_features_cover_complete_relations", relation_complete, daily)
    return pd.DataFrame(records, columns=columns)


def _pit_audit_stage(stage_root: Path, month: str, filename: str, stage: str) -> dict[str, Any]:
    files = sorted(stage_root.glob(f"date={month}-*/layer_id=*/scale=*/{filename}"))
    rows = failed = missing = 0
    for path in files:
        parquet = pq.ParquetFile(path)
        try:
            if "pit_audit_pass" not in parquet.schema.names:
                missing += 1
                continue
            for batch in parquet.iter_batches(columns=["pit_audit_pass"], batch_size=500_000, use_threads=False):
                values = pa.Table.from_batches([batch]).column("pit_audit_pass").to_pandas()
                rows += len(values)
                failed += int((~values.fillna(False).astype(bool)).sum())
        finally:
            parquet.close()
    return {"stage": stage, "files": len(files), "rows_checked": rows, "failed_rows": failed,
            "missing_pit_column_files": missing,
            "status": "pass" if files and failed == 0 and missing == 0 else "fail"}


def _evaluation_dir(root: Path, name: str, month: str) -> Path:
    return root / name / _month_scope(month)


def _evaluation_manifest(eval_dir: Path, mode: str, allow_partial: bool) -> dict[str, Any]:
    manifest = _read_json(eval_dir / "manifest.json")
    errors = []
    if manifest.get("status") != "complete":
        errors.append(f"{mode} eval status={manifest.get('status', 'missing')}")
    if manifest.get("evaluation_contract_version") != EXPECTED_CONTRACTS["p2_eval"]:
        errors.append(f"{mode} eval contract={manifest.get('evaluation_contract_version', 'missing')}")
    for filename in (f"{mode}_alpha_summary_state.parquet", f"{mode}_alpha_summary.csv"):
        if not (eval_dir / filename).exists():
            errors.append(f"missing {eval_dir / filename}")
    if errors and not allow_partial:
        raise ValueError("; ".join(errors))
    return manifest


def _p2_scorecard(state: pd.DataFrame, mode: str, gates: EvalGates) -> pd.DataFrame:
    missing = set(P2_STATE_COLUMNS) - set(state)
    if missing:
        raise ValueError(f"{mode} summary state missing {sorted(missing)}")
    state = state[P2_STATE_COLUMNS].copy()
    for column in ("date", *P2_KEYS):
        state[column] = state[column].astype(str)
    for column in set(P2_STATE_COLUMNS) - {"date", *P2_KEYS}:
        state[column] = pd.to_numeric(state[column], errors="coerce").fillna(0.0)
    state["daily_mean_ic"] = state["rank_ic_sum"] / state["rank_ic_count"].replace(0, np.nan)
    state["daily_mean_spread"] = state["spread_sum"] / state["spread_count"].replace(0, np.nan)
    total_days = int(state["date"].nunique())
    required_days = min(max(1, gates.min_days), max(1, total_days))
    rows = []
    for key, subset in state.groupby(P2_KEYS, sort=False, dropna=False):
        score, target, layer_id, scale, level = key
        rank_count = int(subset["rank_ic_count"].sum())
        spread_count = int(subset["spread_count"].sum())
        periods = int(subset["snapshots"].sum())
        mean_ic = float(subset["rank_ic_sum"].sum()) / rank_count if rank_count else float("nan")
        mean_spread = float(subset["spread_sum"].sum()) / spread_count if spread_count else float("nan")
        ic_sign = float(np.sign(mean_ic)) if np.isfinite(mean_ic) else 0.0
        spread_sign = float(np.sign(mean_spread)) if np.isfinite(mean_spread) else 0.0
        daily_ic = pd.to_numeric(subset["daily_mean_ic"], errors="coerce").dropna()
        daily_spread = pd.to_numeric(subset["daily_mean_spread"], errors="coerce").dropna()
        days = int(subset["date"].nunique())
        daily_std = float(daily_ic.std(ddof=1)) if len(daily_ic) > 1 else float("nan")
        rows.append({
            "mode": mode, "score": score, "target": target, "target_period": _target_period(target),
            "layer_id": layer_id, "scale": scale, "scale_role": SCALE_ROLES.get(scale, "other"),
            "level": level, "days": days, "total_eval_days": total_days,
            "day_coverage": days / total_days if total_days else float("nan"), "periods": periods,
            "sample_count": int(subset["sample_count"].sum()),
            "mean_sample_count_per_period": float(subset["sample_count"].sum() / periods) if periods else float("nan"),
            "mean_rank_ic": mean_ic, "mean_spread": mean_spread,
            "daily_mean_ic": float(daily_ic.mean()) if len(daily_ic) else float("nan"),
            "daily_median_ic": float(daily_ic.median()) if len(daily_ic) else float("nan"),
            "daily_ic_std": daily_std,
            "daily_ic_tstat": float(daily_ic.mean() / (daily_std / math.sqrt(len(daily_ic)))) if len(daily_ic) > 1 and daily_std > 0 else float("nan"),
            "daily_ic_pvalue": _two_sided_t_pvalue(daily_ic),
            "daily_ic_direction_rate": float((np.sign(daily_ic) == ic_sign).mean()) if len(daily_ic) and ic_sign else float("nan"),
            "daily_mean_spread": float(daily_spread.mean()) if len(daily_spread) else float("nan"),
            "daily_spread_direction_rate": float((np.sign(daily_spread) == spread_sign).mean()) if len(daily_spread) and spread_sign else float("nan"),
            "positive_period_rate": int(subset["positive_count"].sum()) / spread_count if spread_count else float("nan"),
            "ic_spread_sign_agreement": bool(ic_sign != 0 and ic_sign == spread_sign),
            "largest_day_abs_ic_share": _concentration(daily_ic, 1),
            "top3_days_abs_ic_share": _concentration(daily_ic, 3),
            "leave_one_day_out_sign_rate": _leave_one_out_sign_rate(daily_ic, ic_sign),
            "required_days": required_days, "direction": "positive" if mean_ic >= 0 else "negative",
        })
    scorecard = pd.DataFrame(rows)
    if scorecard.empty:
        return scorecard
    scorecard["daily_ic_qvalue_bh"] = _bh_qvalues(scorecard["daily_ic_pvalue"])
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
    scorecard["candidate_pass"] = scorecard[[f"gate_{name}" for name in conditions]].all(axis=1)
    scorecard["gate_failures"] = scorecard.apply(lambda row: ";".join(name for name in conditions if not bool(row[f"gate_{name}"])) or "none", axis=1)
    strength = np.minimum(scorecard["mean_rank_ic"].abs() / max(gates.min_abs_ic * 2.0, 1e-9), 1.0)
    direction = np.clip((scorecard["daily_ic_direction_rate"].fillna(0.0) - 0.5) / 0.2, 0.0, 1.0)
    coverage = scorecard["day_coverage"].fillna(0.0).clip(0.0, 1.0)
    concentration = (1.0 - scorecard["top3_days_abs_ic_share"].fillna(1.0)).clip(0.0, 1.0)
    loo = scorecard["leave_one_day_out_sign_rate"].fillna(0.0).clip(0.0, 1.0)
    alignment = scorecard["ic_spread_sign_agreement"].astype(float)
    testing = (1.0 - scorecard["daily_ic_qvalue_bh"].fillna(1.0)).clip(0.0, 1.0)
    scorecard["research_score"] = (25 * strength + 20 * direction + 15 * coverage + 15 * concentration + 10 * loo + 10 * alignment + 5 * testing).round(3)
    return _add_layer_catalog(scorecard).sort_values(["candidate_pass", "research_score", "mean_rank_ic"], ascending=[False, False, False], kind="mergesort").reset_index(drop=True)


def _eval_summary_audit(eval_dir: Path, mode: str, scorecard: pd.DataFrame) -> pd.DataFrame:
    source = eval_dir / f"{mode}_alpha_summary.csv"
    columns = [*P2_KEYS, "summary_present", "days_diff", "snapshots_diff", "sample_count_diff", "mean_rank_ic_diff", "mean_spread_diff", "positive_period_rate_diff"]
    if scorecard.empty or not source.exists():
        return pd.DataFrame(columns=columns)
    summary = pd.read_csv(source, dtype={"layer_id": "string", "scale": "string", "level": "string"})
    for column in P2_KEYS:
        summary[column] = summary[column].astype(str)
    current = scorecard.rename(columns={"periods": "snapshots"})
    merged = current[P2_KEYS + ["days", "snapshots", "sample_count", "mean_rank_ic", "mean_spread", "positive_period_rate"]].merge(
        summary[P2_KEYS + ["days", "snapshots", "sample_count", "mean_rank_ic", "mean_spread", "positive_period_rate"]],
        on=P2_KEYS, how="outer", suffixes=("_report", "_evaluator"), indicator=True)
    result = merged[P2_KEYS].copy()
    result["summary_present"] = merged["_merge"].eq("both")
    for column in ("days", "snapshots", "sample_count", "mean_rank_ic", "mean_spread", "positive_period_rate"):
        result[f"{column}_diff"] = pd.to_numeric(merged[f"{column}_report"], errors="coerce") - pd.to_numeric(merged[f"{column}_evaluator"], errors="coerce")
    return result[columns]


def _eval_decision(scorecard: pd.DataFrame, mode: str) -> dict[str, Any]:
    if scorecard.empty:
        return {"stage": f"p2_{mode}", "recommended_action": "missing_or_empty", "candidate_combinations": 0, "tested_combinations": 0, "rationale": "没有可评估组合。"}
    candidates = scorecard[scorecard["candidate_pass"]]
    tested = len(scorecard)
    rate = len(candidates) / tested if tested else 0.0
    layers = int(candidates["layer_id"].nunique()) if not candidates.empty else 0
    scores = int(candidates["score"].nunique()) if not candidates.empty else 0
    targets = int(candidates["target"].nunique()) if not candidates.empty else 0
    if candidates.empty:
        action, rationale = "stop_or_rework_stage", "没有组合通过强度、日级稳定性、覆盖率、集中度和 IC/Spread 一致性闸门。"
    elif rate >= 0.20 and layers >= 3 and scores >= 2 and targets >= 2:
        action, rationale = "continue_broad_validation", "候选跨多个 Layer、分数和 horizon，适合继续较广泛的样本外验证。"
    else:
        action, rationale = "validate_whitelist_only", "候选较稀疏，应仅对白名单在不同市场环境月份复核。"
    return {"stage": f"p2_{mode}", "recommended_action": action, "rationale": rationale,
            "tested_combinations": tested, "candidate_combinations": int(len(candidates)),
            "candidate_rate": rate, "candidate_layers": layers, "candidate_scores": scores,
            "candidate_targets": targets, "discovery_only_warning": "单月结果用于发现，不构成样本外确认。"}


def _load_p2_eval(root: Path, month: str, mode: str, gates: EvalGates, allow_partial: bool) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    eval_dir = _evaluation_dir(root, f"{mode}_relation_eval", month)
    manifest = _evaluation_manifest(eval_dir, mode, allow_partial)
    state_path = eval_dir / f"{mode}_alpha_summary_state.parquet"
    if not state_path.exists():
        return pd.DataFrame(), pd.DataFrame(), manifest, _eval_decision(pd.DataFrame(), mode)
    scorecard = _p2_scorecard(pd.read_parquet(state_path), mode, gates)
    audit = _eval_summary_audit(eval_dir, mode, scorecard)
    diff_columns = [column for column in audit if column.endswith("_diff")]
    mismatch = not audit.empty and (bool((~audit["summary_present"]).any()) or bool(audit[diff_columns].abs().gt(1e-10).to_numpy().any()))
    if mismatch and not allow_partial:
        raise AssertionError(f"{mode} report aggregation does not match evaluator summary")
    return scorecard, audit, manifest, _eval_decision(scorecard, mode)


def _return_specs(columns: Iterable[str]) -> dict[str, tuple[str, str]]:
    result = {}
    for column in columns:
        match = re.fullmatch(r"ret_(eq|core|top5)_(\d+[md])", str(column))
        if match:
            result[str(column)] = (match.group(1), match.group(2))
    return result


def _scan_theme_returns(root: Path, month: str, batch_size: int, progress_every: int) -> tuple[pd.DataFrame, ScanStats]:
    files = sorted(root.glob(f"date={month}-*/layer_id=*/scale=*/theme_returns.parquet"))
    partials, stats, started = [], ScanStats(files=len(files)), time.time()
    for file_index, path in enumerate(files, start=1):
        date, layer, scale = (_partition_value(path, key) or "" for key in ("date", "layer_id", "scale"))
        parquet = pq.ParquetFile(path)
        try:
            specs = _return_specs(parquet.schema.names)
            columns = [column for column in ["level", *specs] if column in parquet.schema.names]
            for batch in parquet.iter_batches(columns=columns, batch_size=batch_size, use_threads=False):
                stats.batches += 1
                frame = pa.Table.from_batches([batch]).to_pandas(split_blocks=True, self_destruct=True)
                stats.rows += len(frame)
                if frame.empty or not specs:
                    continue
                if "level" not in frame:
                    frame["level"] = "UNKNOWN"
                derived = {}
                for col, (variant, horizon) in specs.items():
                    derived[col] = (variant, horizon, pd.to_numeric(frame[col], errors="coerce"))
                for horizon in sorted({item[1] for item in specs.values()}):
                    eq, core, top5 = f"ret_eq_{horizon}", f"ret_core_{horizon}", f"ret_top5_{horizon}"
                    if eq in frame and core in frame:
                        derived[f"core_minus_eq_{horizon}"] = ("core_minus_eq", horizon, pd.to_numeric(frame[core], errors="coerce") - pd.to_numeric(frame[eq], errors="coerce"))
                    if eq in frame and top5 in frame:
                        derived[f"top5_minus_eq_{horizon}"] = ("top5_minus_eq", horizon, pd.to_numeric(frame[top5], errors="coerce") - pd.to_numeric(frame[eq], errors="coerce"))
                for level, sub in frame.groupby("level", sort=False, dropna=False):
                    level_str = str(level)
                    sub_idx = sub.index
                    for col, (variant, horizon, values) in derived.items():
                        vals = values.loc[sub_idx].dropna()
                        if vals.empty:
                            continue
                        count = len(vals)
                        sum_val = float(vals.sum())
                        sumsq_val = float(vals.pow(2).sum())
                        pos_count = int(vals.gt(0).sum())
                        partials.append(pd.DataFrame([{
                            "date": date, "layer_id": layer, "scale": scale, "level": level_str,
                            "observations": count, "return_sum": sum_val, "return_sumsq": sumsq_val,
                            "positive_count": pos_count, "variant": variant, "horizon": horizon
                        }]))
        finally:
            parquet.close()
        if progress_every and (file_index % progress_every == 0 or file_index == len(files)):
            elapsed = max(time.time() - started, 1e-9)
            print(f"[monthly-report/theme] files={file_index}/{len(files)} rows={stats.rows:,} rate={stats.rows / elapsed:,.0f}/s", flush=True)
    if not partials:
        return pd.DataFrame(), stats
    daily = pd.concat(partials, ignore_index=True)
    keys = ["date", "layer_id", "scale", "level", "variant", "horizon"]
    numeric = ["observations", "return_sum", "return_sumsq", "positive_count"]
    daily = daily.groupby(keys, sort=False, dropna=False)[numeric].sum().reset_index()
    daily["daily_mean_return"] = daily["return_sum"] / daily["observations"].replace(0, np.nan)
    return daily, stats


def _theme_scorecard(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return daily
    rows = []
    for key, subset in daily.groupby(["layer_id", "scale", "level", "variant", "horizon"], sort=False, dropna=False):
        layer, scale, level, variant, horizon = key
        count, total = int(subset["observations"].sum()), float(subset["return_sum"].sum())
        mean = total / count if count else float("nan")
        values = pd.to_numeric(subset["daily_mean_return"], errors="coerce").dropna()
        sign = float(np.sign(mean)) if np.isfinite(mean) else 0.0
        std = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
        rows.append({
            "layer_id": str(layer), "scale": str(scale), "scale_role": SCALE_ROLES.get(str(scale), "other"),
            "level": str(level), "variant": str(variant), "horizon": str(horizon),
            "target_period": _target_period(str(horizon)), "days": int(subset["date"].nunique()),
            "observations": count, "mean_return": mean,
            "return_std": _sample_std(total, float(subset["return_sumsq"].sum()), count),
            "positive_observation_rate": int(subset["positive_count"].sum()) / count if count else float("nan"),
            "daily_mean_return": float(values.mean()) if len(values) else float("nan"),
            "daily_median_return": float(values.median()) if len(values) else float("nan"),
            "daily_direction_rate": float((np.sign(values) == sign).mean()) if len(values) and sign else float("nan"),
            "daily_tstat": float(values.mean() / (std / math.sqrt(len(values)))) if len(values) > 1 and std > 0 else float("nan"),
            "daily_pvalue": _two_sided_t_pvalue(values), "top3_days_abs_share": _concentration(values, 3),
            "direction": "positive" if mean >= 0 else "negative",
            "interpretation": "核心成员相对等权增量" if variant == "core_minus_eq" else "Top5 相对等权增量" if variant == "top5_minus_eq" else f"{variant} 主题组合收益",
        })
    result = pd.DataFrame(rows)
    result["daily_qvalue_bh"] = _bh_qvalues(result["daily_pvalue"])
    result = _add_layer_catalog(result)
    result["descriptive_score"] = (45 * (result["daily_tstat"].abs().fillna(0).clip(upper=5) / 5) +
                                     35 * np.clip((result["daily_direction_rate"].fillna(0.5) - 0.5) / 0.25, 0, 1) +
                                     20 * (1 - result["top3_days_abs_share"].fillna(1)).clip(0, 1)).round(3)
    return result.sort_values(["descriptive_score", "mean_return"], ascending=[False, False]).reset_index(drop=True)


def _scan_relation_structure(root: Path, month: str, batch_size: int, progress_every: int) -> tuple[pd.DataFrame, ScanStats]:
    files = sorted(root.glob(f"date={month}-*/layer_id=*/scale=*/relation_spillover_signals.parquet"))
    partials, stats, started = [], ScanStats(files=len(files)), time.time()
    desired = ["level", "signal", "absolute_signal_sum", "relation_edge_count", "relation_strength_mean", "positive_source_count", "negative_source_count"]
    for file_index, path in enumerate(files, start=1):
        date, layer, scale = (_partition_value(path, key) or "" for key in ("date", "layer_id", "scale"))
        parquet = pq.ParquetFile(path)
        try:
            columns = [column for column in desired if column in parquet.schema.names]
            for batch in parquet.iter_batches(columns=columns, batch_size=batch_size, use_threads=False):
                stats.batches += 1
                frame = pa.Table.from_batches([batch]).to_pandas(split_blocks=True, self_destruct=True)
                stats.rows += len(frame)
                if frame.empty or "signal" not in frame:
                    continue
                if "level" not in frame:
                    frame["level"] = "UNKNOWN"
                frame["date"], frame["layer_id"], frame["scale"] = date, layer, scale
                frame["signal"] = pd.to_numeric(frame["signal"], errors="coerce")
                frame = frame.dropna(subset=["signal"])
                if frame.empty:
                    continue
                frame["signal_sq"], frame["signal_positive"] = frame["signal"].pow(2), frame["signal"].gt(0).astype("int64")
                for column in desired[2:]:
                    frame[column] = pd.to_numeric(frame[column], errors="coerce") if column in frame else np.nan
                grouped = frame.groupby(["date", "layer_id", "scale", "level"], sort=False, dropna=False).agg(
                    rows=("signal", "size"), signal_sum=("signal", "sum"), signal_sumsq=("signal_sq", "sum"),
                    positive_rows=("signal_positive", "sum"), absolute_signal_sum=("absolute_signal_sum", "sum"),
                    relation_edge_count_sum=("relation_edge_count", "sum"), relation_strength_sum=("relation_strength_mean", "sum"),
                    relation_strength_count=("relation_strength_mean", "count"), positive_source_count=("positive_source_count", "sum"),
                    negative_source_count=("negative_source_count", "sum")).reset_index()
                partials.append(grouped)
        finally:
            parquet.close()
        if progress_every and (file_index % progress_every == 0 or file_index == len(files)):
            elapsed = max(time.time() - started, 1e-9)
            print(f"[monthly-report/relation] files={file_index}/{len(files)} rows={stats.rows:,} rate={stats.rows / elapsed:,.0f}/s", flush=True)
    if not partials:
        return pd.DataFrame(), stats
    daily = pd.concat(partials, ignore_index=True)
    keys = ["date", "layer_id", "scale", "level"]
    daily = daily.groupby(keys, sort=False, dropna=False)[[column for column in daily if column not in keys]].sum().reset_index()
    daily["daily_mean_signal"] = daily["signal_sum"] / daily["rows"].replace(0, np.nan)
    return daily, stats


def _relation_scorecard(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return daily
    rows = []
    for key, subset in daily.groupby(["layer_id", "scale", "level"], sort=False, dropna=False):
        layer, scale, level = key
        count, signal_sum = int(subset["rows"].sum()), float(subset["signal_sum"].sum())
        mean = signal_sum / count if count else float("nan")
        values = pd.to_numeric(subset["daily_mean_signal"], errors="coerce").dropna()
        sign = float(np.sign(mean)) if np.isfinite(mean) else 0.0
        source_total = float(subset["positive_source_count"].sum() + subset["negative_source_count"].sum())
        rows.append({
            "layer_id": str(layer), "scale": str(scale), "scale_role": SCALE_ROLES.get(str(scale), "other"),
            "level": str(level), "days": int(subset["date"].nunique()), "rows": count,
            "mean_signal": mean, "signal_std": _sample_std(signal_sum, float(subset["signal_sumsq"].sum()), count),
            "positive_signal_rate": int(subset["positive_rows"].sum()) / count if count else float("nan"),
            "mean_absolute_signal_sum": float(subset["absolute_signal_sum"].sum() / count) if count else float("nan"),
            "mean_relation_edges": float(subset["relation_edge_count_sum"].sum() / count) if count else float("nan"),
            "mean_relation_strength": float(subset["relation_strength_sum"].sum() / subset["relation_strength_count"].sum()) if subset["relation_strength_count"].sum() else float("nan"),
            "positive_source_share": float(subset["positive_source_count"].sum() / source_total) if source_total else float("nan"),
            "daily_signal_direction_rate": float((np.sign(values) == sign).mean()) if len(values) and sign else float("nan"),
            "top3_days_abs_signal_share": _concentration(values, 3),
        })
    result = _add_layer_catalog(pd.DataFrame(rows))
    result["structure_score"] = (40 * np.clip((result["daily_signal_direction_rate"].fillna(0.5) - 0.5) / 0.25, 0, 1) +
                                  30 * (1 - result["top3_days_abs_signal_share"].fillna(1)).clip(0, 1) +
                                  30 * np.clip(np.log1p(result["mean_relation_edges"].fillna(0)) / 5, 0, 1)).round(3)
    return result.sort_values("structure_score", ascending=False).reset_index(drop=True)


def _unified_candidates(p0: pd.DataFrame, intraday: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    frames = []
    if not p0.empty:
        x = p0.copy(); x["stage"], x["signal"], x["level"], x["direction_rate"] = "p0_eval", x["feature"], "symbol", x["daily_ic_direction_rate"]
        frames.append(x)
    for stage, source in (("p2_intraday", intraday), ("p2_daily", daily)):
        if source.empty:
            continue
        x = source.copy(); x["stage"], x["signal"], x["direction_rate"] = stage, x["score"], x["daily_ic_direction_rate"]
        frames.append(x)
    if not frames:
        return pd.DataFrame()
    common = ["stage", "signal", "target", "layer_id", "layer_name", "layer_family", "scale", "scale_role", "level", "direction", "mean_rank_ic", "mean_spread", "direction_rate", "days", "day_coverage", "periods", "sample_count", "top3_days_abs_ic_share", "leave_one_day_out_sign_rate", "daily_ic_qvalue_bh", "research_score", "candidate_pass", "gate_failures"]
    normalized = []
    for frame in frames:
        for column in common:
            if column not in frame:
                frame[column] = np.nan
        normalized.append(frame[common])
    return pd.concat(normalized, ignore_index=True).sort_values(["candidate_pass", "research_score", "mean_rank_ic"], ascending=[False, False, False], kind="mergesort").reset_index(drop=True)


def _layer_stage_overview(unified: pd.DataFrame) -> pd.DataFrame:
    if unified.empty:
        return pd.DataFrame()
    rows = []
    for key, subset in unified.groupby(["stage", "layer_id", "layer_name", "scale"], sort=False, dropna=False):
        best = subset.sort_values("research_score", ascending=False).iloc[0]
        rows.append({"stage": key[0], "layer_id": key[1], "layer_name": key[2], "scale": key[3],
                     "tested_combinations": len(subset), "candidate_combinations": int(subset["candidate_pass"].sum()),
                     "median_abs_ic": float(subset["mean_rank_ic"].abs().median()), "best_signal": str(best["signal"]),
                     "best_target": str(best["target"]), "best_abs_ic": float(subset["mean_rank_ic"].abs().max()),
                     "best_research_score": float(best["research_score"])})
    return pd.DataFrame(rows).sort_values(["candidate_combinations", "best_research_score"], ascending=[False, False])


def _overall_decision(p0: dict[str, Any], intraday: dict[str, Any], daily: dict[str, Any], unified: pd.DataFrame) -> dict[str, Any]:
    candidates = unified[unified["candidate_pass"]] if not unified.empty else pd.DataFrame()
    counts = candidates.groupby("stage").size().astype(int).to_dict() if not candidates.empty else {}
    if counts.get("p2_daily", 0):
        priority, rationale = "p2_daily", "日级 P2 存在通过稳定性闸门的候选，应优先做跨月和执行成本验证。"
    elif counts.get("p2_intraday", 0):
        priority, rationale = "p2_intraday", "日内 P2 存在候选，但日级尚未通过，应优先验证日内持续性与可交易性。"
    elif counts.get("p0_eval", 0):
        priority, rationale = "p0_whitelist", "只有 P0 层出现候选，应保留白名单验证，但不应让 P0 Eval 阻塞 Theme/P2 主链路。"
    else:
        priority, rationale = "rework_or_stop_alpha_validation", "本月没有组合通过预设的强度与稳定性闸门，需要复核覆盖率、因子定义或市场窗口。"
    return {"overall_priority": priority, "rationale": rationale, "candidate_counts_by_stage": counts,
            "p0_action": p0.get("recommended_action"), "p2_intraday_action": intraday.get("recommended_action"),
            "p2_daily_action": daily.get("recommended_action"),
            "discovery_only_warning": "单月属于候选发现；不得据此宣称样本外有效或直接实盘。",
            "recommended_validation_design": "候选白名单在至少两个不同市场环境月份复核，再决定是否补齐全部月份。"}


def _records(frame: pd.DataFrame, limit: int | None = None) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    selected = frame.head(limit) if limit is not None else frame
    return [_json_ready(record) for record in selected.to_dict(orient="records")]


def _table(frame: pd.DataFrame, columns: list[str], limit: int = 30) -> str:
    if frame.empty:
        return "<p><em>无数据</em></p>"
    selected = frame[[column for column in columns if column in frame]].head(limit).copy()
    for column in selected:
        if pd.api.types.is_float_dtype(selected[column]):
            selected[column] = selected[column].map(lambda value: "" if pd.isna(value) else f"{value:.6g}")
    return selected.to_html(index=False, escape=True, border=0, classes="data")


def _html_report(report: dict[str, Any], frames: dict[str, pd.DataFrame], top_n: int) -> str:
    d = report["executive_decision"]
    errors = report.get("errors", [])
    embedded = json.dumps(_json_ready(report), ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html><html lang='zh-Hant'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>{html.escape(report['month'])} 全鏈路 Alpha 報告</title><style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:0;background:#f5f7fb;color:#172033}}main{{max-width:1600px;margin:auto;padding:28px}}h1,h2,h3{{color:#0f172a}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}}.card{{background:white;border:1px solid #dbe3ef;border-radius:12px;padding:16px}}.badge{{display:inline-block;padding:6px 10px;border-radius:999px;background:#e0e7ff;font-weight:700}}table.data{{border-collapse:collapse;width:100%;font-size:12px;background:white}}table.data th,table.data td{{border:1px solid #d8dee9;padding:6px 8px;text-align:right;white-space:nowrap}}table.data th:first-child,table.data td:first-child{{text-align:left}}table.data th{{background:#eef2f7;position:sticky;top:0}}.section{{margin:28px 0}}.scroll{{overflow:auto;max-height:620px}}.warn{{color:#991b1b}}</style></head><body><main>
<h1>{html.escape(report['month'])} GraphFactorFactory 全鏈路 Alpha 報告</h1><p class='badge'>{html.escape(str(d['overall_priority']))}</p><p>{html.escape(str(d['rationale']))}</p>
<div class='grid'><div class='card'><b>統一候選</b><div>{report['counts']['unified_candidates']}</div></div><div class='card'><b>P0 候選</b><div>{report['counts']['p0_candidates']}</div></div><div class='card'><b>P2 日內候選</b><div>{report['counts']['intraday_candidates']}</div></div><div class='card'><b>P2 日級候選</b><div>{report['counts']['daily_candidates']}</div></div><div class='card'><b>Theme 檔案</b><div>{report['scans']['theme_returns']['files']}</div></div><div class='card'><b>Relation 檔案</b><div>{report['scans']['relation_spillover']['files']}</div></div></div>
{"<p class='warn'><b>錯誤：</b>" + html.escape('; '.join(errors)) + "</p>" if errors else ""}
<div class='section'><h2>1. 階段健康與覆蓋</h2><div class='scroll'>{_table(frames['stage_health'], list(frames['stage_health'].columns), 100)}</div><h3>Partition 銜接</h3>{_table(frames['coverage'], list(frames['coverage'].columns), 30)}<h3>PIT 審計</h3>{_table(frames['pit'], list(frames['pit'].columns), 30)}</div>
<div class='section'><h2>2. 全階段統一 Alpha 候選 Top {top_n}</h2><div class='scroll'>{_table(frames['unified'], ['stage','layer_id','layer_name','scale','level','signal','target','direction','mean_rank_ic','mean_spread','direction_rate','days','top3_days_abs_ic_share','daily_ic_qvalue_bh','research_score','candidate_pass'], top_n)}</div></div>
<div class='section'><h2>3. P0 Eval</h2><p>{html.escape(str(report['decisions']['p0'].get('rationale','')))}</p><div class='scroll'>{_table(frames['p0'], ['layer_id','layer_name','scale','kind','feature','target','direction','mean_rank_ic','mean_spread','daily_ic_direction_rate','top3_days_abs_ic_share','research_score','candidate_pass'], top_n)}</div></div>
<div class='section'><h2>4. Theme Returns</h2><p>主題組合實現收益與 Core/Top5 相對等權增量；屬描述性證據，不等同橫截面 IC。</p><div class='scroll'>{_table(frames['theme'], ['layer_id','layer_name','scale','level','variant','horizon','mean_return','daily_direction_rate','daily_tstat','daily_qvalue_bh','top3_days_abs_share','descriptive_score'], top_n)}</div></div>
<div class='section'><h2>5. Relation Spillover 結構</h2><div class='scroll'>{_table(frames['relation'], ['layer_id','layer_name','scale','level','days','rows','mean_signal','mean_absolute_signal_sum','mean_relation_edges','positive_source_share','daily_signal_direction_rate','structure_score'], top_n)}</div></div>
<div class='section'><h2>6. P2 日內 Alpha</h2><p>{html.escape(str(report['decisions']['p2_intraday'].get('rationale','')))}</p><div class='scroll'>{_table(frames['intraday'], ['layer_id','layer_name','scale','level','score','target','direction','mean_rank_ic','mean_spread','daily_ic_direction_rate','top3_days_abs_ic_share','daily_ic_qvalue_bh','research_score','candidate_pass'], top_n)}</div></div>
<div class='section'><h2>7. P2 日級 Alpha</h2><p>{html.escape(str(report['decisions']['p2_daily'].get('rationale','')))}</p><div class='scroll'>{_table(frames['daily'], ['layer_id','layer_name','scale','level','score','target','direction','mean_rank_ic','mean_spread','daily_ic_direction_rate','top3_days_abs_ic_share','daily_ic_qvalue_bh','research_score','candidate_pass'], top_n)}</div></div>
<div class='section'><h2>8. 解讀邊界</h2><ul><li>單月只用於候選發現，不是樣本外確認。</li><li>負 IC 若與 Spread、日級方向一致，仍可作為反向因子。</li><li>Theme Return 平均收益與 Relation 結構分數不能替代 Alpha IC。</li><li>正式候選需在不同市場環境月份做白名單驗證。</li></ul></div>
<script id='monthly-alpha-report' type='application/json'>{embedded}</script></main></body></html>"""


def generate_monthly_report(p2_root: str | Path, month: str, output_dir: str | Path | None = None, *, batch_size: int = 250_000, top_n: int = 50, json_top_n: int = 200, allow_partial: bool = False, p0_gates: P0Gates | None = None, intraday_gates: EvalGates | None = None, daily_gates: EvalGates | None = None, progress_every: int = 25) -> dict[str, Any]:
    started, root, scope = time.time(), Path(p2_root), _month_scope(month)
    report_root = Path(output_dir) if output_dir is not None else root / "monthly_alpha_report" / scope
    report_root.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []

    manifest_audit = _manifest_audit(root, month)
    stage_health, coverage = _stage_health(manifest_audit), _partition_coverage(manifest_audit)
    if manifest_audit.empty:
        errors.append("no monthly stage manifests found")
    elif bool((~manifest_audit["healthy"]).any()):
        errors.append(f"{int((~manifest_audit['healthy']).sum())} unhealthy stage partitions")
    if not coverage.empty and bool(coverage["status"].eq("fail").any()):
        errors.append("partition coverage mismatch")

    pit_audit = pd.DataFrame([
        _pit_audit_stage(root / "intraday_relation_features", month, "intraday_relation_features.parquet", "intraday_features"),
        _pit_audit_stage(root / "daily_relation_features", month, "daily_relation_features.parquet", "daily_features"),
    ])
    if bool(pit_audit["status"].eq("fail").any()):
        errors.append("PIT feature audit failed")

    p0_report_dir = report_root / "p0_eval"
    try:
        p0_result = generate_p0_report(_evaluation_dir(root, "p0_alpha", month), p0_report_dir,
            gates=p0_gates or P0Gates(), batch_size=batch_size, top_n=top_n,
            progress_every=progress_every, allow_incomplete=allow_partial,
            allow_unknown_contract=allow_partial, allow_summary_mismatch=allow_partial)
        p0_scorecard = pd.read_csv(p0_report_dir / "p0_eval_combo_scorecard.csv", dtype={"layer_id": "string", "scale": "string"})
        p0_decision = _read_json(p0_report_dir / "p0_eval_decision.json")
    except Exception as exc:
        if not allow_partial:
            raise
        errors.append(f"P0 Eval report: {exc}")
        p0_result, p0_scorecard = {}, pd.DataFrame()
        p0_decision = {"recommended_action": "missing_or_invalid", "rationale": str(exc)}

    theme_daily, theme_stats = _scan_theme_returns(root / "theme_returns", month, batch_size, progress_every)
    theme_scorecard = _theme_scorecard(theme_daily)
    if theme_scorecard.empty:
        errors.append("Theme Returns are missing or empty")
    relation_daily, relation_stats = _scan_relation_structure(root / "relation_spillover", month, batch_size, progress_every)
    relation_scorecard = _relation_scorecard(relation_daily)
    if relation_scorecard.empty:
        errors.append("Relation Spillover is missing or empty")

    intraday_gates, daily_gates = intraday_gates or EvalGates(min_periods=100), daily_gates or EvalGates(min_periods=10)
    try:
        intraday_scorecard, intraday_summary_audit, intraday_manifest, intraday_decision = _load_p2_eval(root, month, "intraday", intraday_gates, allow_partial)
    except Exception as exc:
        if not allow_partial:
            raise
        errors.append(f"intraday eval: {exc}")
        intraday_scorecard, intraday_summary_audit, intraday_manifest = pd.DataFrame(), pd.DataFrame(), {}
        intraday_decision = _eval_decision(pd.DataFrame(), "intraday")
    try:
        daily_scorecard, daily_summary_audit, daily_manifest, daily_decision = _load_p2_eval(root, month, "daily", daily_gates, allow_partial)
    except Exception as exc:
        if not allow_partial:
            raise
        errors.append(f"daily eval: {exc}")
        daily_scorecard, daily_summary_audit, daily_manifest = pd.DataFrame(), pd.DataFrame(), {}
        daily_decision = _eval_decision(pd.DataFrame(), "daily")

    unified = _unified_candidates(p0_scorecard, intraday_scorecard, daily_scorecard)
    layer_stage = _layer_stage_overview(unified)
    overall = _overall_decision(p0_decision, intraday_decision, daily_decision, unified)
    if errors and not allow_partial:
        raise RuntimeError("; ".join(errors))

    counts = {
        "unified_candidates": int(unified["candidate_pass"].sum()) if not unified.empty else 0,
        "p0_candidates": int(p0_scorecard["candidate_pass"].sum()) if not p0_scorecard.empty else 0,
        "intraday_candidates": int(intraday_scorecard["candidate_pass"].sum()) if not intraday_scorecard.empty else 0,
        "daily_candidates": int(daily_scorecard["candidate_pass"].sum()) if not daily_scorecard.empty else 0,
    }
    p0_candidates = p0_scorecard[p0_scorecard["candidate_pass"]] if not p0_scorecard.empty else p0_scorecard
    intraday_candidates = intraday_scorecard[intraday_scorecard["candidate_pass"]] if not intraday_scorecard.empty else intraday_scorecard
    daily_candidates = daily_scorecard[daily_scorecard["candidate_pass"]] if not daily_scorecard.empty else daily_scorecard
    report = {
        "report_contract_version": REPORT_CONTRACT_VERSION, "status": "complete" if not errors else "partial_or_invalid",
        "month": month, "scope": scope, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_root": str(root.resolve()), "executive_decision": overall,
        "decisions": {"p0": p0_decision, "p2_intraday": intraday_decision, "p2_daily": daily_decision},
        "counts": counts, "stage_health": _records(stage_health), "coverage_audit": _records(coverage),
        "pit_audit": _records(pit_audit),
        "scans": {"theme_returns": asdict(theme_stats), "relation_spillover": asdict(relation_stats)},
        "contracts": {"expected": EXPECTED_CONTRACTS, "intraday_eval_manifest": intraday_manifest, "daily_eval_manifest": daily_manifest},
        "top_unified_candidates": _records(unified, json_top_n), "top_p0_candidates": _records(p0_candidates, json_top_n),
        "top_theme_return_tendencies": _records(theme_scorecard, json_top_n),
        "top_relation_structures": _records(relation_scorecard, json_top_n),
        "top_p2_intraday_candidates": _records(intraday_candidates, json_top_n),
        "top_p2_daily_candidates": _records(daily_candidates, json_top_n),
        "layer_stage_overview": _records(layer_stage),
        "summary_audits": {"intraday": _records(intraday_summary_audit), "daily": _records(daily_summary_audit)},
        "run_metadata": {"schedule_plan": _read_json(root / "p2_24core_schedule_plan.json"),
                         "p0_direct_summary": _read_json(root / "p0_direct_run_summary.json"), "p0_report": p0_result},
        "interpretation_contract": {
            "p0_eval": "symbol-level cross-sectional Rank IC and top-bottom spread",
            "theme_returns": "descriptive theme portfolio returns and core/top5 uplift; not an IC substitute",
            "relation_spillover": "network propagation structure; not directly a tradable Alpha",
            "p2_intraday": "snapshot-level cross-sectional Alpha evaluation",
            "p2_daily": "end-of-day episode-level cross-sectional Alpha evaluation",
            "discovery_only": True,
        },
        "errors": errors,
    }
    frames = {"stage_health": stage_health, "manifest_audit": manifest_audit, "coverage": coverage, "pit": pit_audit,
              "p0": p0_scorecard, "theme_daily": theme_daily, "theme": theme_scorecard,
              "relation_daily": relation_daily, "relation": relation_scorecard,
              "intraday": intraday_scorecard, "daily": daily_scorecard, "unified": unified,
              "layer_stage": layer_stage, "intraday_summary_audit": intraday_summary_audit,
              "daily_summary_audit": daily_summary_audit}
    _write_text(report_root / "monthly_alpha_report.json", json.dumps(_json_ready(report), indent=2, ensure_ascii=False))
    _write_text(report_root / "monthly_alpha_report.html", _html_report(report, frames, top_n))
    for name, frame in (("monthly_stage_health.csv", stage_health), ("monthly_partition_manifest_audit.csv", manifest_audit),
                        ("monthly_partition_coverage_audit.csv", coverage), ("monthly_pit_audit.csv", pit_audit),
                        ("monthly_alpha_unified_scorecard.csv", unified),
                        ("monthly_alpha_candidate_whitelist.csv", unified[unified["candidate_pass"]] if not unified.empty else unified),
                        ("monthly_layer_stage_overview.csv", layer_stage), ("theme_return_daily_stats.csv", theme_daily),
                        ("theme_return_scorecard.csv", theme_scorecard), ("relation_structure_daily_stats.csv", relation_daily),
                        ("relation_structure_scorecard.csv", relation_scorecard), ("p2_intraday_scorecard.csv", intraday_scorecard),
                        ("p2_daily_scorecard.csv", daily_scorecard), ("p2_intraday_summary_audit.csv", intraday_summary_audit),
                        ("p2_daily_summary_audit.csv", daily_summary_audit)):
        _write_csv(frame, report_root / name)

    files = [{"path": str(path.relative_to(report_root)), "bytes": path.stat().st_size, "sha256": _sha256(path)}
             for path in sorted(report_root.rglob("*")) if path.is_file() and path.suffix != ".zip"]
    bundle_manifest = {"report_contract_version": REPORT_CONTRACT_VERSION, "month": month,
                       "raw_parquet_copied": False,
                       "source_roots_referenced_only": ["p0_alpha", "theme_returns", "relation_spillover",
                                                         "intraday_relation_features", "daily_relation_features",
                                                         "intraday_relation_eval", "daily_relation_eval"],
                       "files": files, "elapsed_sec": round(time.time() - started, 3)}
    _write_text(report_root / "monthly_alpha_report_manifest.json", json.dumps(_json_ready(bundle_manifest), indent=2, ensure_ascii=False))
    bundle, temporary = report_root / "monthly_alpha_report_bundle.zip", report_root / "monthly_alpha_report_bundle.zip.tmp"
    temporary.unlink(missing_ok=True)
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(report_root.rglob("*")):
            if path.is_file() and path not in {bundle, temporary} and path.suffix != ".zip":
                archive.write(path, arcname=str(path.relative_to(report_root)))
    os.replace(temporary, bundle)
    result = {"status": report["status"], "month": month, "report_dir": str(report_root),
              "json": str(report_root / "monthly_alpha_report.json"),
              "html": str(report_root / "monthly_alpha_report.html"), "bundle": str(bundle),
              "bundle_bytes": bundle.stat().st_size, "counts": counts, "executive_decision": overall,
              "elapsed_sec": round(time.time() - started, 3)}
    print(json.dumps(_json_ready(result), indent=2, ensure_ascii=False), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a compact full-month P0/Theme/P2 Alpha report")
    parser.add_argument("--p2-root", required=True)
    parser.add_argument("--month", required=True, help="YYYY-MM")
    parser.add_argument("--output-dir")
    parser.add_argument("--batch-size", type=int, default=250_000)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--json-top-n", type=int, default=200)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--p0-min-abs-ic", type=float, default=0.015)
    parser.add_argument("--intraday-min-abs-ic", type=float, default=0.015)
    parser.add_argument("--daily-min-abs-ic", type=float, default=0.015)
    parser.add_argument("--min-direction-rate", type=float, default=0.55)
    parser.add_argument("--min-days", type=int, default=15)
    parser.add_argument("--min-day-coverage", type=float, default=0.70)
    parser.add_argument("--max-top3-abs-ic-share", type=float, default=0.50)
    args = parser.parse_args()
    shared = {"min_direction_rate": args.min_direction_rate, "min_days": args.min_days,
              "min_day_coverage": args.min_day_coverage, "max_top3_abs_ic_share": args.max_top3_abs_ic_share}
    generate_monthly_report(args.p2_root, args.month, args.output_dir, batch_size=args.batch_size,
        top_n=args.top_n, json_top_n=args.json_top_n, allow_partial=args.allow_partial,
        progress_every=args.progress_every, p0_gates=P0Gates(min_abs_ic=args.p0_min_abs_ic, **shared),
        intraday_gates=EvalGates(min_abs_ic=args.intraday_min_abs_ic, min_periods=100, **shared),
        daily_gates=EvalGates(min_abs_ic=args.daily_min_abs_ic, min_periods=10, **shared))


if __name__ == "__main__":
    main()
