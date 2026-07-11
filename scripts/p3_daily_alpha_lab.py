#!/usr/bin/env python3
"""P3 daily alpha lab: convert intraday relation-spillover pulses into daily features.

This module is intentionally independent from P2. P2 produces intraday signal shards;
P3 aggregates those shards into daily path-level features that can later be evaluated
against next-day / multi-day labels.

Design rules:
- scan partitioned parquet shards incrementally;
- write atomic parquet outputs plus manifest.json;
- do not assume snapshot-local theme_id is a persistent daily ID;
- derive a stable-ish target_path_id by stripping the timestamp from dst_theme_id when
  no explicit path id is present;
- support new alpha feature families by adding feature builders without changing P2.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Iterable

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("ARROW_NUM_THREADS", "2")

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import stats

DEFAULT_TARGETS = ["target_5m", "target_15m", "target_30m", "target_60m", "target_120m"]


def csvset(s: str | None) -> set[str] | None:
    return None if not s else {x.strip() for x in s.split(",") if x.strip()}


def csvlist(s: str | None) -> list[str] | None:
    return None if not s else [x.strip() for x in s.split(",") if x.strip()]


def extract_part(path: Path, key: str) -> str | None:
    for piece in path.parts:
        if piece.startswith(key + "="):
            return piece.split("=", 1)[1]
    return None


def discover_signal_files(root: Path, dates=None, layers=None, scales=None, levels=None, filename="relation_spillover_signals.parquet") -> list[Path]:
    out: list[Path] = []
    for fp in root.rglob(filename):
        d = extract_part(fp, "date")
        layer = extract_part(fp, "layer_id")
        scale = extract_part(fp, "scale")
        if dates and d not in dates:
            continue
        if layers and layer not in layers:
            continue
        if scales and scale not in scales:
            continue
        # level is inside file, not in path for current P2 outputs.
        out.append(fp)
    out.sort(key=lambda p: (extract_part(p, "date") or "", extract_part(p, "layer_id") or "", extract_part(p, "scale") or ""))
    return out


def stable_target_path_id(x: object) -> str:
    """Convert snapshot-local dst_theme_id into a daily-comparable path signature.

    Current P1 theme ids usually look like:
      ts=2026-01-02_14_38_00_00_00|layer=9|scale=15m|root.b50_...
    We strip only the timestamp token. This is not a perfect persistent theme path;
    it is a deterministic fallback until temporal_theme_edges/path ids are available.
    """
    if pd.isna(x):
        return "UNKNOWN"
    s = str(x)
    parts = [p for p in s.split("|") if not p.startswith("ts=")]
    return "|".join(parts) if parts else s


class AtomicParquetSink:
    def __init__(self, path: Path, compression: str = "zstd") -> None:
        self.path = path
        self.tmp = path.with_suffix(path.suffix + ".tmp")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.tmp.exists():
            self.tmp.unlink()
        self.writer: pq.ParquetWriter | None = None
        self.rows = 0
        self.compression = compression

    def write(self, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return
        table = pa.Table.from_pandas(df, preserve_index=False)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.tmp, table.schema, compression=self.compression)
        else:
            table = table.cast(self.writer.schema)
        self.writer.write_table(table)
        self.rows += len(df)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        if self.tmp.exists():
            self.tmp.replace(self.path)


def write_manifest(out_dir: Path, meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def get_partition_max_time(fp: Path, max_row_groups: int | None = None) -> pd.Timestamp | None:
    pf = pq.ParquetFile(fp)
    names = set(pf.schema.names)
    if "decision_time" not in names:
        return None
    if max_row_groups is None:
        ser = pd.read_parquet(fp, columns=["decision_time"])["decision_time"]
        return pd.to_datetime(ser, utc=True).max() if len(ser) else None
    n = min(max_row_groups, pf.metadata.num_row_groups)
    mx = None
    for rg in range(n):
        ser = pf.read_row_group(rg, columns=["decision_time"]).to_pandas()["decision_time"]
        if ser.empty:
            continue
        val = pd.to_datetime(ser, utc=True).max()
        mx = val if mx is None or val > mx else mx
    return mx


def iter_parquet_frames(fp: Path, cols: list[str], max_row_groups: int | None = None):
    pf = pq.ParquetFile(fp)
    if max_row_groups is None:
        yield pd.read_parquet(fp, columns=cols)
        return
    n = min(max_row_groups, pf.metadata.num_row_groups)
    for rg in range(n):
        yield pf.read_row_group(rg, columns=cols).to_pandas()


def aggregate_one_file(fp: Path, targets: list[str], levels: set[str] | None, late_minutes: int, max_row_groups: int | None = None) -> pd.DataFrame:
    pf = pq.ParquetFile(fp)
    names = set(pf.schema.names)
    cols = [c for c in [
        "date", "decision_time", "layer_id", "scale", "level", "dst_theme_id", "target_path_id",
        "signal", "relation_strength_mean", "relation_edge_count", *targets
    ] if c in names]
    if "signal" not in cols:
        raise ValueError(f"missing signal column in {fp}")
    max_dt = get_partition_max_time(fp, max_row_groups=max_row_groups)
    late_cutoff = None if max_dt is None else max_dt - pd.Timedelta(minutes=late_minutes)
    chunks: list[pd.DataFrame] = []
    path_date = extract_part(fp, "date")
    path_layer = extract_part(fp, "layer_id")
    path_scale = extract_part(fp, "scale")

    for df in iter_parquet_frames(fp, cols, max_row_groups=max_row_groups):
        if df.empty:
            continue
        if "date" not in df.columns:
            df["date"] = path_date
        if "layer_id" not in df.columns:
            df["layer_id"] = path_layer
        if "scale" not in df.columns:
            df["scale"] = path_scale
        if "level" not in df.columns:
            df["level"] = "UNKNOWN"
        if levels:
            df = df[df["level"].astype(str).isin(levels)]
            if df.empty:
                continue
        df["decision_time"] = pd.to_datetime(df["decision_time"], utc=True)
        if "target_path_id" not in df.columns:
            df["target_path_id"] = df.get("dst_theme_id", pd.Series(["UNKNOWN"] * len(df))).map(stable_target_path_id)
        df["signal"] = pd.to_numeric(df["signal"], errors="coerce").fillna(0.0).astype("float64")
        if "relation_strength_mean" in df.columns:
            df["relation_strength_mean"] = pd.to_numeric(df["relation_strength_mean"], errors="coerce").fillna(0.0)
        else:
            df["relation_strength_mean"] = np.nan
        if "relation_edge_count" in df.columns:
            df["relation_edge_count"] = pd.to_numeric(df["relation_edge_count"], errors="coerce").fillna(0.0)
        else:
            df["relation_edge_count"] = 1.0
        for t in targets:
            if t in df.columns:
                df[t] = pd.to_numeric(df[t], errors="coerce")
        df["pos_signal"] = df["signal"].clip(lower=0.0)
        df["neg_signal"] = df["signal"].clip(upper=0.0)
        df["abs_signal"] = df["signal"].abs()
        df["is_positive"] = (df["signal"] > 0).astype("int64")
        df["is_negative"] = (df["signal"] < 0).astype("int64")
        df["is_late"] = False if late_cutoff is None else df["decision_time"] >= late_cutoff
        group_cols = ["date", "layer_id", "scale", "level", "target_path_id"]
        agg_spec = {
            "decision_time": ["count", "nunique", "min", "max"],
            "signal": ["sum", "mean", "std"],
            "pos_signal": "sum",
            "neg_signal": "sum",
            "abs_signal": "sum",
            "is_positive": "sum",
            "is_negative": "sum",
            "relation_strength_mean": "mean",
            "relation_edge_count": "sum",
        }
        for t in targets:
            if t in df.columns:
                agg_spec[t] = "mean"
        a = df.groupby(group_cols, sort=False).agg(agg_spec)
        a.columns = ["_".join([str(x) for x in c if x]) for c in a.columns.to_flat_index()]
        a = a.reset_index()
        late = df[df["is_late"]]
        if not late.empty:
            la = late.groupby(group_cols, sort=False).agg(
                late_count=("signal", "size"),
                late_signal_sum=("signal", "sum"),
                late_pos_signal_sum=("pos_signal", "sum"),
                late_abs_signal_sum=("abs_signal", "sum"),
                late_strength_mean=("relation_strength_mean", "mean"),
            ).reset_index()
            a = a.merge(la, on=group_cols, how="left")
        else:
            a["late_count"] = 0
            a["late_signal_sum"] = 0.0
            a["late_pos_signal_sum"] = 0.0
            a["late_abs_signal_sum"] = 0.0
            a["late_strength_mean"] = np.nan
        chunks.append(a)

    if not chunks:
        return pd.DataFrame()
    raw = pd.concat(chunks, ignore_index=True)
    group_cols = ["date", "layer_id", "scale", "level", "target_path_id"]
    sum_cols = [c for c in raw.columns if c.endswith("_sum") or c in ["decision_time_count", "decision_time_nunique", "is_positive_sum", "is_negative_sum", "relation_edge_count_sum", "late_count"]]
    mean_cols = [c for c in raw.columns if c.endswith("_mean")]
    min_cols = ["decision_time_min"] if "decision_time_min" in raw.columns else []
    max_cols = ["decision_time_max"] if "decision_time_max" in raw.columns else []
    agg = {c: "sum" for c in sum_cols}
    agg.update({c: "mean" for c in mean_cols})
    agg.update({c: "min" for c in min_cols})
    agg.update({c: "max" for c in max_cols})
    out = raw.groupby(group_cols, sort=False).agg(agg).reset_index()
    out = out.rename(columns={
        "decision_time_count": "signal_observation_count",
        "decision_time_nunique": "active_window_count_proxy",
        "signal_sum": "daily_pressure",
        "signal_mean": "mean_signal",
        "signal_std": "signal_std",
        "pos_signal_sum": "positive_pressure",
        "neg_signal_sum": "negative_pressure",
        "abs_signal_sum": "absolute_pressure",
        "is_positive_sum": "positive_observation_count",
        "is_negative_sum": "negative_observation_count",
        "relation_strength_mean": "avg_relation_strength",
        "relation_edge_count_sum": "relation_edge_count_sum",
    })
    out["positive_observation_rate"] = out["positive_observation_count"] / out["signal_observation_count"].replace(0, np.nan)
    out["negative_observation_rate"] = out["negative_observation_count"] / out["signal_observation_count"].replace(0, np.nan)
    out["persistence_proxy"] = out["positive_observation_rate"] - out["negative_observation_rate"]
    out["pressure_intensity"] = out["daily_pressure"] / np.sqrt(out["signal_observation_count"].clip(lower=1))
    out["absolute_pressure_intensity"] = out["absolute_pressure"] / np.sqrt(out["signal_observation_count"].clip(lower=1))
    out["late_pressure_share"] = out["late_signal_sum"] / out["daily_pressure"].replace(0, np.nan)
    out["late_absolute_share"] = out["late_abs_signal_sum"] / out["absolute_pressure"].replace(0, np.nan)
    out["late_positive_share"] = out["late_pos_signal_sum"] / out["positive_pressure"].replace(0, np.nan)
    for t in targets:
        c = f"{t}_mean"
        if c in out.columns:
            out = out.rename(columns={c: f"{t}_mean_proxy"})
    rank_group = ["date", "layer_id", "scale", "level"]
    def zscore(s: pd.Series) -> pd.Series:
        sd = s.std(ddof=0)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - s.mean()) / sd
    out["pressure_z"] = out.groupby(rank_group, sort=False)["pressure_intensity"].transform(zscore)
    out["persistence_z"] = out.groupby(rank_group, sort=False)["persistence_proxy"].transform(zscore)
    response_col = "target_5m_mean_proxy" if "target_5m_mean_proxy" in out.columns else None
    if response_col:
        out["target_response_z"] = out.groupby(rank_group, sort=False)[response_col].transform(zscore)
        out["target_underreaction_z"] = out["pressure_z"] - out["target_response_z"]
    else:
        out["target_response_z"] = np.nan
        out["target_underreaction_z"] = out["pressure_z"]
    out["late_confirmation_score"] = out["late_absolute_share"].fillna(0.0).clip(lower=0.0, upper=5.0)
    out["daily_pressure_score"] = out["pressure_z"] * out["persistence_proxy"].fillna(0.0) * (1.0 + out["late_confirmation_score"].fillna(0.0))
    out["daily_underreaction_score"] = out["daily_pressure_score"] * out["target_underreaction_z"].clip(lower=0.0).fillna(0.0)
    out["daily_consensus_score"] = out["daily_pressure_score"] * np.log1p(out["relation_edge_count_sum"].clip(lower=0.0))
    return out


def build_reaction_features(args) -> None:
    t0 = time.time()
    root = Path(args.signals_root)
    out_dir = Path(args.out_dir)
    out_file = out_dir / "daily_relation_features.parquet"
    targets = csvlist(args.targets) or DEFAULT_TARGETS
    files = discover_signal_files(root, dates=csvset(args.dates), layers=csvset(args.layers), scales=csvset(args.scales), levels=csvset(args.levels))
    if args.max_files:
        files = files[: args.max_files]
    sink = AtomicParquetSink(out_file)
    metas = []
    for i, fp in enumerate(files, 1):
        df = aggregate_one_file(fp, targets=targets, levels=csvset(args.levels), late_minutes=args.late_minutes, max_row_groups=args.max_row_groups)
        if not df.empty:
            sink.write(df)
        metas.append({"file": str(fp), "rows": int(len(df))})
        if i % 10 == 0 or i == len(files):
            print(json.dumps({"processed_files": i, "total_files": len(files), "last_rows": int(len(df))}, ensure_ascii=False), flush=True)
    sink.close()
    meta = {
        "status": "complete" if sink.rows else "empty",
        "input_files": len(files),
        "output_rows": sink.rows,
        "output": str(out_file),
        "late_minutes": args.late_minutes,
        "targets": targets,
        "elapsed_sec": round(time.time() - t0, 3),
        "sample_inputs": metas[:5],
    }
    write_manifest(out_dir, meta)
    print(json.dumps(meta, indent=2, ensure_ascii=False), flush=True)


def evaluate_features(args) -> None:
    t0 = time.time()
    fp = Path(args.features)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    score_cols = csvlist(args.score_cols) or ["daily_pressure_score", "daily_underreaction_score", "daily_consensus_score"]
    target_cols = csvlist(args.target_cols) or ["target_5m_mean_proxy", "target_15m_mean_proxy", "target_30m_mean_proxy", "target_60m_mean_proxy"]
    pf = pq.ParquetFile(fp)
    rows = []
    frames = [pf.read_row_group(rg).to_pandas() for rg in range(pf.metadata.num_row_groups)]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if df.empty:
        raise ValueError(f"no rows in {fp}")
    keys = ["date", "layer_id", "scale", "level"]
    for score in score_cols:
        if score not in df.columns:
            continue
        for target in target_cols:
            if target not in df.columns:
                continue
            valid = df.dropna(subset=[score, target]).copy()
            if valid.empty:
                continue
            for key_vals, g in valid.groupby(keys, sort=False):
                if len(g) < args.min_samples:
                    continue
                if g[score].nunique() < 2 or g[target].nunique() < 2:
                    ic = np.nan
                else:
                    ic = float(stats.spearmanr(g[score], g[target], nan_policy="omit").statistic)
                top_q = g[score].quantile(0.8)
                bot_q = g[score].quantile(0.2)
                top = g[g[score] >= top_q][target].mean()
                bot = g[g[score] <= bot_q][target].mean()
                rows.append({
                    "score": score,
                    "target": target,
                    "date": key_vals[0],
                    "layer_id": key_vals[1],
                    "scale": key_vals[2],
                    "level": key_vals[3],
                    "sample_count": int(len(g)),
                    "rank_ic": ic,
                    "top_mean": float(top),
                    "bottom_mean": float(bot),
                    "spread": float(top - bot),
                    "hit_rate": float((g[score] * g[target] > 0).mean()),
                })
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "daily_alpha_metrics.csv", index=False)
    summary_keys = ["score", "target", "layer_id", "scale", "level"]
    if metrics.empty:
        summary = metrics
    else:
        summary = metrics.groupby(summary_keys, sort=False).agg(
            days=("date", "nunique"),
            sample_count=("sample_count", "sum"),
            mean_daily_spread=("spread", "mean"),
            daily_spread_std=("spread", "std"),
            positive_day_rate=("spread", lambda x: float((x > 0).mean())),
            mean_daily_rank_ic=("rank_ic", "mean"),
            mean_hit_rate=("hit_rate", "mean"),
        ).reset_index()
        summary["daily_spread_t"] = summary["mean_daily_spread"] / (summary["daily_spread_std"].replace(0, np.nan) / np.sqrt(summary["days"].clip(lower=1)))
        summary = summary.sort_values(["target", "daily_spread_t"], ascending=[True, False])
    summary.to_csv(out_dir / "daily_alpha_summary.csv", index=False)
    meta = {"status": "complete", "features": str(fp), "metric_rows": int(len(metrics)), "summary_rows": int(len(summary)), "elapsed_sec": round(time.time() - t0, 3)}
    write_manifest(out_dir, meta)
    print(json.dumps(meta, indent=2, ensure_ascii=False), flush=True)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="P3 modular daily alpha lab")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-reaction-features", help="Aggregate intraday relation_spillover pulses into daily path features")
    b.add_argument("--signals-root", required=True)
    b.add_argument("--out-dir", required=True)
    b.add_argument("--dates")
    b.add_argument("--layers")
    b.add_argument("--scales")
    b.add_argument("--levels", default="B50,B35")
    b.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    b.add_argument("--late-minutes", type=int, default=60)
    b.add_argument("--max-files", type=int)
    b.add_argument("--max-row-groups", type=int)
    b.set_defaults(func=build_reaction_features)

    e = sub.add_parser("evaluate-features", help="Evaluate daily feature scores against available target proxy columns or future labels")
    e.add_argument("--features", required=True)
    e.add_argument("--out-dir", required=True)
    e.add_argument("--score-cols", default="daily_pressure_score,daily_underreaction_score,daily_consensus_score")
    e.add_argument("--target-cols", default="target_5m_mean_proxy,target_15m_mean_proxy,target_30m_mean_proxy,target_60m_mean_proxy,target_120m_mean_proxy")
    e.add_argument("--min-samples", type=int, default=20)
    e.set_defaults(func=evaluate_features)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
