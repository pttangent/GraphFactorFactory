#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("ARROW_NUM_THREADS", "1")
os.environ.setdefault("POLARS_MAX_THREADS", "1")

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HORIZONS = ["5m", "15m", "30m", "60m", "120m"]


@dataclass(frozen=True)
class Part:
    date: str
    layer_id: str
    scale: str
    base: Path


def csvset(s):
    return None if not s else {x.strip() for x in str(s).split(",") if x.strip()}


def csvlist(s):
    return None if not s else [x.strip() for x in str(s).split(",") if x.strip()]


def mins(h):
    return int(str(h)[:-1]) if str(h).endswith("m") else (_ for _ in ()).throw(ValueError(h))


def ext(p, k):
    for x in Path(p).parts:
        if x.startswith(k + "="):
            return x.split("=", 1)[1]
    return None


def done(m):
    try:
        j = json.loads(Path(m).read_text(encoding="utf-8"))
        return j.get("status") == "complete" and int(j.get("output_rows", 0)) > 0
    except Exception:
        return False


def manifest(d, meta):
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    t = d / "manifest.json.tmp"
    t.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    t.replace(d / "manifest.json")


def label_path(root, date):
    r = Path(root)
    cands = [
        r if r.is_file() else None,
        r / f"date={date}" / "labels.parquet",
        r / "canonical" / f"date={date}" / "labels.parquet",
        r / date / "labels.parquet",
    ]
    for c in cands:
        if c is not None and c.exists():
            return c
    raise FileNotFoundError(f"labels.parquet not found for date={date} under {r}")


def discover(root, filename, dates=None, layers=None, scales=None):
    out = []
    for p in Path(root).rglob(filename):
        d, l, s = ext(p, "date"), ext(p, "layer_id"), ext(p, "scale")
        if not d or not l or not s:
            continue
        if dates and d not in dates:
            continue
        if layers and l not in layers:
            continue
        if scales and s not in scales:
            continue
        out.append(Part(d, l, s, p))
    out.sort(key=lambda x: x.base.stat().st_size, reverse=True)
    return out


def read_partition(path, cols, max_rg=None):
    pf = pq.ParquetFile(path)
    names = set(pf.schema.names)
    cols = [c for c in cols if c in names]
    if not cols:
        return pd.DataFrame()
    if max_rg is None:
        return pd.read_parquet(path, columns=cols)
    tabs = [pf.read_row_group(i, columns=cols) for i in range(min(max_rg, pf.metadata.num_row_groups))]
    return pa.concat_tables(tabs).to_pandas() if tabs else pd.DataFrame()


def write_parquet_atomic(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), tmp, compression="zstd")
    tmp.replace(path)


def parse_theme_ts_series(s: pd.Series) -> pd.Series:
    tok = s.astype(str).str.extract(r"ts=([^|.]+)", expand=False)
    direct = pd.to_datetime(tok, utc=True, errors="coerce")
    miss = direct.isna() & tok.notna()
    if miss.any():
        x = tok[miss].str.extract(r"(\d{4}-\d{2}-\d{2})T(\d{2})(\d{2})(\d{2})", expand=True)
        val = pd.to_datetime(x[0] + " " + x[1] + ":" + x[2] + ":" + x[3], utc=True, errors="coerce")
        direct.loc[miss] = val
    return direct


def load_labels(path, horizons):
    names = set(pq.ParquetFile(path).schema.names)
    labs = [f"label_{h}" for h in horizons if f"label_{h}" in names]
    if not labs:
        raise ValueError(f"no label_* columns for horizons={horizons} in {path}")
    df = pd.read_parquet(path, columns=["decision_time", "symbol_id"] + labs)
    df["decision_time"] = pd.to_datetime(df["decision_time"], utc=True)
    df["symbol_id"] = pd.to_numeric(df["symbol_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["decision_time", "symbol_id"]).copy()
    df["symbol_id"] = df["symbol_id"].astype("int64")
    for h in horizons:
        c = f"label_{h}"
        if c in df:
            p = df[["decision_time", "symbol_id", c]].copy()
            p["decision_time"] = p["decision_time"] + pd.Timedelta(minutes=mins(h))
            p = p.rename(columns={c: f"past_label_{h}"})
            df = df.merge(p, on=["decision_time", "symbol_id"], how="left")
    return df


def labels_by_time(lab):
    return {k: v for k, v in lab.groupby("decision_time", sort=False, dropna=False)}


def zscore_by_group(df: pd.DataFrame, group_cols: list[str], col: str) -> pd.Series:
    def _z(s: pd.Series) -> pd.Series:
        std = s.std(ddof=0)
        if not np.isfinite(std) or std == 0:
            return pd.Series(0.0, index=s.index)
        return (s - s.mean()) / std

    return df.groupby(group_cols, sort=False, dropna=False)[col].transform(_z)


def stream_df(path, frames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    if tmp.exists():
        tmp.unlink()
    writer = None
    schema = None
    rows = 0
    batches = 0
    try:
        for odf in frames:
            if odf is None or odf.empty:
                continue
            table = pa.Table.from_pandas(odf, preserve_index=False)
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(tmp, schema, compression="zstd")
            elif table.schema != schema:
                table = table.cast(schema)
            writer.write_table(table)
            rows += len(odf)
            batches += 1
    finally:
        if writer is not None:
            writer.close()
    if rows:
        os.replace(tmp, path)
    elif tmp.exists():
        tmp.unlink()
    return rows, batches


def build_returns_one(part, labels_root, out_root, horizons, levels, skip, max_rg, inner_workers):
    t = time.time()
    out = Path(out_root) / f"date={part.date}" / f"layer_id={part.layer_id}" / f"scale={part.scale}"
    op = out / "theme_returns.parquet"
    if skip and done(out / "manifest.json"):
        return {"stage": "theme_returns", "status": "skipped", "date": part.date, "layer_id": part.layer_id, "scale": part.scale}

    lab = load_labels(label_path(labels_root, part.date), horizons)
    alpha = [c for c in lab.columns if c.startswith("label_") or c.startswith("past_label_")]
    lbt = labels_by_time(lab)
    mem = read_partition(
        part.base,
        ["decision_time", "layer_id", "scale", "level", "theme_id", "member_id", "core_score", "rank_in_theme"],
        max_rg,
    )

    if mem.empty:
        rows = batches = 0
    else:
        if "decision_time" not in mem or mem["decision_time"].isna().all():
            mem["decision_time"] = parse_theme_ts_series(mem["theme_id"])
        mem["decision_time"] = pd.to_datetime(mem["decision_time"], utc=True, errors="coerce")
        mem["member_id"] = pd.to_numeric(mem["member_id"], errors="coerce").astype("Int64")
        mem["core_score"] = pd.to_numeric(mem.get("core_score", 0), errors="coerce").fillna(0.0)
        mem = mem.dropna(subset=["decision_time", "member_id", "theme_id"]).copy()
        mem["member_id"] = mem["member_id"].astype("int64")
        if "level" not in mem:
            mem["level"] = "UNKNOWN"
        if levels:
            mem = mem[mem["level"].astype(str).isin(levels)]
        if "layer_id" not in mem:
            mem["layer_id"] = part.layer_id
        if "scale" not in mem:
            mem["scale"] = part.scale

        def one(item):
            dt, mc = item
            lc = lbt.get(dt)
            if lc is None or mc.empty:
                return None
            df = mc.merge(lc, left_on=["decision_time", "member_id"], right_on=["decision_time", "symbol_id"], how="inner")
            if df.empty:
                return None
            gcols = ["decision_time", "layer_id", "scale", "level", "theme_id"]
            df = df.sort_values(gcols + ["core_score"], ascending=[1, 1, 1, 1, 1, 0])
            g = df.groupby(gcols, sort=False)
            res = g[alpha].mean()
            res.columns = [
                c.replace("past_label_", "past_eq_") if c.startswith("past_label_") else c.replace("label_", "ret_eq_")
                for c in res.columns
            ]
            w = g["core_score"].sum().replace(0, np.nan)
            for c in alpha:
                wc = df[c] * df["core_score"]
                out_col = c.replace("past_label_", "past_core_") if c.startswith("past_label_") else c.replace("label_", "ret_core_")
                res[out_col] = wc.groupby(g._grouper).sum() / w
            top5 = g.head(5).groupby(gcols, sort=False)[alpha].mean()
            top5.columns = [
                c.replace("past_label_", "past_top5_") if c.startswith("past_label_") else c.replace("label_", "ret_top5_")
                for c in top5.columns
            ]
            return res.join(top5).reset_index()

        groups = list(mem.groupby("decision_time", sort=False, dropna=False))
        if inner_workers and inner_workers > 1:
            def frames():
                with cf.ThreadPoolExecutor(max_workers=inner_workers) as ex:
                    futs = [ex.submit(one, g) for g in groups]
                    for f in cf.as_completed(futs):
                        yield f.result()

            rows, batches = stream_df(op, frames())
        else:
            rows, batches = stream_df(op, (one(g) for g in groups))

    meta = {
        "stage": "theme_returns",
        "status": "complete" if rows else "empty",
        "date": part.date,
        "layer_id": part.layer_id,
        "scale": part.scale,
        "output_rows": int(rows),
        "write_batches": int(batches),
        "input": str(part.base),
        "output": str(op),
        "elapsed_sec": round(time.time() - t, 3),
        "time_aligned_join": True,
        "inner_workers": inner_workers,
    }
    manifest(out, meta)
    return meta


def relation_one(part, returns_root, out_root, horizons, past_h, levels, tiers, skip, max_rg, inner_workers):
    t = time.time()
    rp = Path(returns_root) / f"date={part.date}" / f"layer_id={part.layer_id}" / f"scale={part.scale}" / "theme_returns.parquet"
    out = Path(out_root) / f"date={part.date}" / f"layer_id={part.layer_id}" / f"scale={part.scale}"
    op = out / "relation_spillover_signals.parquet"
    if skip and done(out / "manifest.json"):
        return {"stage": "relation_spillover", "status": "skipped", "date": part.date, "layer_id": part.layer_id, "scale": part.scale}
    if not rp.exists():
        return {"stage": "relation_spillover", "status": "missing_theme_returns", "date": part.date, "layer_id": part.layer_id, "scale": part.scale}

    r = pd.read_parquet(rp)
    r["decision_time"] = pd.to_datetime(r["decision_time"], utc=True)
    pc = f"past_eq_{past_h}"
    if pc not in r:
        raise ValueError(f"missing {pc} in {rp}")
    r["layer_id"] = r["layer_id"].astype(str)
    r["scale"] = r["scale"].astype(str)

    targets = [f"ret_eq_{h}" for h in horizons if f"ret_eq_{h}" in r]
    dst_past_cols = [f"past_eq_{h}" for h in horizons if f"past_eq_{h}" in r]

    past = r[["decision_time", "layer_id", "scale", "level", "theme_id", pc]].rename(
        columns={"theme_id": "src_theme_id", pc: "src_past_return"}
    )
    fut = r[
        ["decision_time", "layer_id", "scale", "level", "theme_id"] + targets + dst_past_cols
    ].rename(
        columns={
            "theme_id": "dst_theme_id",
            **{c: c.replace("ret_eq_", "target_") for c in targets},
            **{c: "dst_" + c for c in dst_past_cols},
        }
    )
    fbt = {k: v for k, v in fut.groupby("decision_time", sort=False, dropna=False)}

    e = read_partition(
        part.base,
        ["decision_time", "layer_id", "scale", "level", "src_theme_id", "dst_theme_id", "relation_strength", "relation_tier", "hard_keep", "edge_count"],
        max_rg,
    )
    if e.empty:
        rows = batches = 0
    else:
        if "decision_time" not in e or e["decision_time"].isna().all():
            e["decision_time"] = parse_theme_ts_series(e["src_theme_id"])
        e["decision_time"] = pd.to_datetime(e["decision_time"], utc=True, errors="coerce")
        e = e.dropna(subset=["decision_time", "src_theme_id", "dst_theme_id"]).copy()
        if levels and "level" in e:
            e = e[e["level"].astype(str).isin(levels)]
        if tiers and "relation_tier" in e:
            e = e[e["relation_tier"].astype(str).isin(tiers)]
        if "layer_id" not in e:
            e["layer_id"] = part.layer_id
        if "scale" not in e:
            e["scale"] = part.scale
        e["layer_id"] = e["layer_id"].astype(str)
        e["scale"] = e["scale"].astype(str)

        def one(item):
            dt, pcu = item
            ecu = e[e["decision_time"].eq(dt)]
            if ecu.empty:
                return None
            m = ecu.merge(pcu, on=["decision_time", "layer_id", "scale", "level", "src_theme_id"], how="inner")
            if m.empty:
                return None
            m["signal"] = (
                pd.to_numeric(m["relation_strength"], errors="coerce").fillna(0)
                * pd.to_numeric(m["src_past_return"], errors="coerce").fillna(0)
            )
            a = (
                m.groupby(["decision_time", "layer_id", "scale", "level", "dst_theme_id"], sort=False)
                .agg(
                    signal=("signal", "mean"),
                    relation_strength_mean=("relation_strength", "mean"),
                    relation_edge_count=("src_theme_id", "size"),
                )
                .reset_index()
            )
            z = a.merge(fbt.get(dt, pd.DataFrame()), on=["decision_time", "layer_id", "scale", "level", "dst_theme_id"], how="inner")
            if z.empty:
                return None
            z.insert(1, "date", part.date)
            return z

        groups = list(past.groupby("decision_time", sort=False, dropna=False))
        if inner_workers and inner_workers > 1:
            def frames():
                with cf.ThreadPoolExecutor(max_workers=inner_workers) as ex:
                    futs = [ex.submit(one, g) for g in groups]
                    for f in cf.as_completed(futs):
                        yield f.result()

            rows, batches = stream_df(op, frames())
        else:
            rows, batches = stream_df(op, (one(g) for g in groups))

    meta = {
        "stage": "relation_spillover",
        "status": "complete" if rows else "empty",
        "date": part.date,
        "layer_id": part.layer_id,
        "scale": part.scale,
        "output_rows": int(rows),
        "write_batches": int(batches),
        "input": str(part.base),
        "theme_returns": str(rp),
        "output": str(op),
        "elapsed_sec": round(time.time() - t, 3),
        "time_aligned_join": True,
        "inner_workers": inner_workers,
        "dst_past_columns": dst_past_cols if "dst_past_cols" in locals() else [],
    }
    manifest(out, meta)
    return meta


def path_id(s):
    return (
        s.astype(str)
        .str.replace(r"([.|_\-])?ts=[^.|_\-]+", "", regex=True)
        .str.replace(r"([.|_\-])?time=[^.|_\-]+", "", regex=True)
    )


def daily_one(part, out_root, late_min, underreaction_past_h, skip, max_rg):
    t = time.time()
    out = Path(out_root) / f"date={part.date}" / f"layer_id={part.layer_id}" / f"scale={part.scale}"
    op = out / "daily_relation_features.parquet"
    if skip and done(out / "manifest.json"):
        return {"stage": "daily_relation_features", "status": "skipped", "date": part.date, "layer_id": part.layer_id, "scale": part.scale}

    dst_past_cols = [f"dst_past_eq_{h}" for h in HORIZONS]
    df = read_partition(
        part.base,
        [
            "date",
            "decision_time",
            "layer_id",
            "scale",
            "level",
            "dst_theme_id",
            "signal",
            "relation_strength_mean",
            "relation_edge_count",
            "target_5m",
            "target_15m",
            "target_30m",
            "target_60m",
            "target_120m",
        ] + dst_past_cols,
        max_rg,
    )
    if df.empty:
        rows = 0
    else:
        if "date" not in df:
            df["date"] = part.date
        df["decision_time"] = pd.to_datetime(df["decision_time"], utc=True)
        df["target_path_id"] = path_id(df["dst_theme_id"])
        df["signal"] = pd.to_numeric(df["signal"], errors="coerce").fillna(0.0)
        df["pos_signal"] = df.signal.clip(lower=0)
        df["neg_signal"] = df.signal.clip(upper=0)
        df["abs_signal"] = df.signal.abs()
        late = df.decision_time >= (df.decision_time.max() - pd.Timedelta(minutes=late_min))
        df["late_signal"] = df.signal.where(late, 0.0)
        df["late_abs_signal"] = df.abs_signal.where(late, 0.0)
        df["obs"] = 1
        df["pos_obs"] = (df.signal > 0).astype("int64")
        df["neg_obs"] = (df.signal < 0).astype("int64")

        gcols = ["date", "layer_id", "scale", "level", "target_path_id"]
        targets = [c for c in ["target_5m", "target_15m", "target_30m", "target_60m", "target_120m"] if c in df]
        known_past_cols = [c for c in dst_past_cols if c in df.columns]

        res = (
            df.groupby(gcols, sort=False)
            .agg(
                first_time=("decision_time", "min"),
                last_time=("decision_time", "max"),
                observation_count=("obs", "sum"),
                daily_pressure=("signal", "sum"),
                positive_pressure=("pos_signal", "sum"),
                negative_pressure=("neg_signal", "sum"),
                absolute_pressure=("abs_signal", "sum"),
                positive_observations=("pos_obs", "sum"),
                negative_observations=("neg_obs", "sum"),
                relation_edge_count_sum=("relation_edge_count", "sum"),
                avg_relation_strength=("relation_strength_mean", "mean"),
                late_signal_sum=("late_signal", "sum"),
                late_abs_signal_sum=("late_abs_signal", "sum"),
            )
            .reset_index()
        )

        if targets:
            res = res.merge(
                df.groupby(gcols, sort=False)[targets]
                .mean()
                .reset_index()
                .rename(columns={c: f"{c}_mean_proxy" for c in targets}),
                on=gcols,
                how="left",
            )

        if known_past_cols:
            res = res.merge(
                df.groupby(gcols, sort=False)[known_past_cols]
                .mean()
                .reset_index()
                .rename(columns={c: f"{c}_mean" for c in known_past_cols}),
                on=gcols,
                how="left",
            )

        res["pressure_intensity"] = res.daily_pressure / res.observation_count.replace(0, np.nan)
        res["positive_observation_rate"] = res.positive_observations / res.observation_count.replace(0, np.nan)
        res["negative_observation_rate"] = res.negative_observations / res.observation_count.replace(0, np.nan)
        res["persistence_proxy"] = res.positive_observation_rate - res.negative_observation_rate
        res["late_absolute_share"] = res.late_abs_signal_sum / res.absolute_pressure.replace(0, np.nan)
        res["late_confirmation_score"] = res.late_signal_sum * res.late_absolute_share.fillna(0)

        for c in ["daily_pressure", "absolute_pressure", "relation_edge_count_sum", "late_confirmation_score"]:
            std = res[c].std(ddof=0)
            res[c + "_z"] = 0.0 if not np.isfinite(std) or std == 0 else (res[c] - res[c].mean()) / std

        res["daily_pressure_score"] = res.daily_pressure_z * res.persistence_proxy.fillna(0)
        res["daily_consensus_score"] = res.daily_pressure_z * np.log1p(res.relation_edge_count_sum.clip(lower=0))

        preferred_past_col = f"dst_past_eq_{underreaction_past_h}_mean"
        group_cols = ["date", "layer_id", "scale", "level"]
        if preferred_past_col in res:
            res["expected_pressure_z"] = zscore_by_group(res, group_cols, "daily_pressure_score")
            res["target_pre_response_z"] = zscore_by_group(res, group_cols, preferred_past_col)
            res["underreaction_gap_z"] = res["expected_pressure_z"] - res["target_pre_response_z"]
            res["daily_underreaction_score"] = res["underreaction_gap_z"] * (1.0 + res.late_absolute_share.fillna(0.0))
            res["daily_underreaction_status"] = "non_leaky_known_past_response"
        else:
            res["expected_pressure_z"] = np.nan
            res["target_pre_response_z"] = np.nan
            res["underreaction_gap_z"] = np.nan
            res["daily_underreaction_score"] = np.nan
            res["daily_underreaction_status"] = f"missing_{preferred_past_col}"

        write_parquet_atomic(res, op)
        rows = len(res)

    meta = {
        "stage": "daily_relation_features",
        "status": "complete" if rows else "empty",
        "date": part.date,
        "layer_id": part.layer_id,
        "scale": part.scale,
        "output_rows": int(rows),
        "input": str(part.base),
        "output": str(op),
        "late_minutes": late_min,
        "underreaction_past_horizon": underreaction_past_h,
        "underreaction_uses_future_target": False,
        "elapsed_sec": round(time.time() - t, 3),
    }
    manifest(out, meta)
    return meta


def eval_daily(root, out_dir):
    t = time.time()
    frames = [pd.read_parquet(p) for p in Path(root).rglob("daily_relation_features.parquet")]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not frames:
        meta = {"stage": "daily_feature_eval", "status": "empty", "input_count": 0}
        manifest(out, meta)
        return meta

    df = pd.concat(frames, ignore_index=True)
    rows = []
    scores = [
        c
        for c in [
            "daily_pressure_score",
            "daily_underreaction_score",
            "daily_consensus_score",
            "late_confirmation_score_z",
        ]
        if c in df
    ]
    targets = [
        c
        for c in [
            "target_5m_mean_proxy",
            "target_15m_mean_proxy",
            "target_30m_mean_proxy",
            "target_60m_mean_proxy",
        ]
        if c in df
    ]
    for sc in scores:
        for ta in targets:
            for keys, sub in df.groupby(["date", "layer_id", "scale", "level"], dropna=False, sort=False):
                v = sub[[sc, ta]].replace([np.inf, -np.inf], np.nan).dropna()
                if len(v) < 30:
                    continue
                q8, q2 = v[sc].quantile(0.8), v[sc].quantile(0.2)
                top = v[v[sc] >= q8][ta].mean()
                bot = v[v[sc] <= q2][ta].mean()
                rows.append(
                    {
                        "date": keys[0],
                        "layer_id": keys[1],
                        "scale": keys[2],
                        "level": keys[3],
                        "score": sc,
                        "target": ta,
                        "sample_count": len(v),
                        "rank_ic": v[sc].rank().corr(v[ta].rank()),
                        "top_minus_bottom": top - bot,
                    }
                )

    m = pd.DataFrame(rows)
    mp = out / "daily_alpha_metrics.csv"
    sp = out / "daily_alpha_summary.csv"
    m.to_csv(mp, index=False)
    s = (
        m.groupby(["score", "target", "layer_id", "scale", "level"], sort=False)
        .agg(
            days=("date", "nunique"),
            sample_count=("sample_count", "sum"),
            mean_rank_ic=("rank_ic", "mean"),
            mean_spread=("top_minus_bottom", "mean"),
            positive_day_rate=("top_minus_bottom", lambda x: float((x > 0).mean())),
        )
        .reset_index()
        if not m.empty
        else pd.DataFrame()
    )
    s.to_csv(sp, index=False)
    meta = {
        "stage": "daily_feature_eval",
        "status": "complete" if len(m) else "empty",
        "input_count": len(frames),
        "metric_rows": int(len(m)),
        "summary_rows": int(len(s)),
        "metrics": str(mp),
        "summary": str(sp),
        "elapsed_sec": round(time.time() - t, 3),
    }
    manifest(out, meta)
    return meta


def pool(parts, workers, fn, *args):
    if not parts:
        return []
    res = []
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(fn, p, *args) for p in parts]
        for f in cf.as_completed(futs):
            res.append(f.result())
    return res


def save_summary(root, res):
    Path(root).mkdir(parents=True, exist_ok=True)
    (Path(root) / "run_summary.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(x):
        x.add_argument("--dates")
        x.add_argument("--layers")
        x.add_argument("--scales")
        x.add_argument("--levels", default="B50,B35")
        x.add_argument("--horizons", default=",".join(HORIZONS))
        x.add_argument("--workers", type=int, default=16)
        x.add_argument("--inner-workers", type=int, default=1)
        x.add_argument("--max-row-groups", type=int)
        x.add_argument("--skip-existing", action="store_true")

    a = sub.add_parser("build-theme-returns")
    common(a)
    a.add_argument("--p1-root", required=True)
    a.add_argument("--labels-root", required=True)
    a.add_argument("--out-root", required=True)

    a = sub.add_parser("relation-spillover")
    common(a)
    a.add_argument("--p1-root", required=True)
    a.add_argument("--theme-returns-root", required=True)
    a.add_argument("--out-root", required=True)
    a.add_argument("--past-horizon", default="15m")
    a.add_argument("--tiers")

    a = sub.add_parser("daily-relation-features")
    common(a)
    a.add_argument("--signals-root", required=True)
    a.add_argument("--out-root", required=True)
    a.add_argument("--late-minutes", type=int, default=60)
    a.add_argument("--underreaction-past-horizon", default="15m")

    a = sub.add_parser("evaluate-daily")
    a.add_argument("--features-root", required=True)
    a.add_argument("--out-dir", required=True)

    args = p.parse_args()
    dates, layers, scales, levels = (
        csvset(getattr(args, "dates", None)),
        csvset(getattr(args, "layers", None)),
        csvset(getattr(args, "scales", None)),
        csvset(getattr(args, "levels", None)),
    )
    horizons = csvlist(getattr(args, "horizons", None)) or HORIZONS

    if args.cmd == "build-theme-returns":
        parts = discover(args.p1_root, "theme_memberships.parquet", dates, layers, scales)
        res = pool(parts, args.workers, build_returns_one, args.labels_root, args.out_root, horizons, levels, args.skip_existing, args.max_row_groups, args.inner_workers)
        save_summary(args.out_root, res)
        print(json.dumps({"stage": args.cmd, "parts": len(parts), "results": len(res), "out_root": args.out_root}, indent=2))
    elif args.cmd == "relation-spillover":
        parts = discover(args.p1_root, "theme_relation_edges.parquet", dates, layers, scales)
        res = pool(
            parts,
            args.workers,
            relation_one,
            args.theme_returns_root,
            args.out_root,
            horizons,
            args.past_horizon,
            levels,
            csvset(args.tiers),
            args.skip_existing,
            args.max_row_groups,
            args.inner_workers,
        )
        save_summary(args.out_root, res)
        print(json.dumps({"stage": args.cmd, "parts": len(parts), "results": len(res), "out_root": args.out_root}, indent=2))
    elif args.cmd == "daily-relation-features":
        parts = discover(args.signals_root, "relation_spillover_signals.parquet", dates, layers, scales)
        res = pool(
            parts,
            args.workers,
            daily_one,
            args.out_root,
            args.late_minutes,
            args.underreaction_past_horizon,
            args.skip_existing,
            args.max_row_groups,
        )
        save_summary(args.out_root, res)
        print(json.dumps({"stage": args.cmd, "parts": len(parts), "results": len(res), "out_root": args.out_root}, indent=2))
    elif args.cmd == "evaluate-daily":
        print(json.dumps(eval_daily(args.features_root, args.out_dir), indent=2))


if __name__ == "__main__":
    main()
