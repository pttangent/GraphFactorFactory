#!/usr/bin/env python3
"""Add four post-hoc falsification audits to the compact monthly Alpha report.

No production Feature, Label, PIT, IC, Spread, or checkpoint logic is changed.
The script reads existing monthly outputs, enriches JSON/HTML, and rebuilds the
small report bundle without copying raw Parquet files.
"""
from __future__ import annotations

import argparse, hashlib, html, json, math, os, re, time, zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from scipy.stats import t as student_t
except Exception:  # pragma: no cover
    student_t = None

from generate_monthly_alpha_report import generate_monthly_report
from p2_pit_core import iter_time_groups, load_labels

CONTRACT = "monthly-alpha-falsification-audit-v1"
MIN_SAMPLE = 30


def jsonable(x: Any) -> Any:
    if isinstance(x, dict): return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [jsonable(v) for v in x]
    if isinstance(x, np.integer): return int(x)
    if isinstance(x, (np.floating, float)): return None if not np.isfinite(x) else float(x)
    if isinstance(x, (Path, pd.Timestamp)): return str(x)
    if x is pd.NA: return None
    return x


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp = Path(str(path) + ".tmp")
    tmp.write_text(text, encoding="utf-8"); os.replace(tmp, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp = Path(str(path) + ".tmp")
    frame.to_csv(tmp, index=False); os.replace(tmp, path)


def read_json(path: Path) -> dict:
    try: return json.loads(path.read_text(encoding="utf-8"))
    except Exception: return {}


def part(path: Path, key: str) -> str | None:
    for token in path.parts:
        if token.startswith(key + "="): return token.split("=", 1)[1]
    return None


def bools(series: pd.Series) -> pd.Series:
    return series.fillna(False) if pd.api.types.is_bool_dtype(series) else series.astype("string").str.lower().isin({"true", "1", "yes"})


def candidates(path: Path) -> pd.DataFrame:
    if not path.exists(): return pd.DataFrame()
    frame = pd.read_csv(path, dtype={"layer_id": "string", "scale": "string", "level": "string"})
    return frame[bools(frame["candidate_pass"])].copy() if "candidate_pass" in frame else frame


def return_corr(frame: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=frame.index)
    for column in ("layer_name", "layer_transform", "layer_family"):
        if column in frame: mask |= frame[column].astype(str).str.contains("return_corr", case=False, na=False)
    return mask


def residual(y, x):
    y, x = np.asarray(y, float), np.asarray(x, float)
    design = np.column_stack([np.ones(len(x)), x]); beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ beta


def metric(score, target):
    values = pd.DataFrame({"score": score, "target": target}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) < MIN_SAMPLE: return 0, np.nan, np.nan
    q80, q20 = values.score.quantile(.8), values.score.quantile(.2)
    spread = values.loc[values.score >= q80, "target"].mean() - values.loc[values.score <= q20, "target"].mean()
    ic = np.nan if values.score.nunique() < 2 or values.target.nunique() < 2 else values.score.rank().corr(values.target.rank())
    return len(values), float(ic), float(spread)


def new_state(): return {"snapshots": 0, "sample_count": 0, "ic_sum": 0., "ic_count": 0, "spread_sum": 0., "spread_count": 0, "positive": 0}


def update(state, count, ic, spread):
    if not count: return
    state["snapshots"] += 1; state["sample_count"] += count
    if np.isfinite(ic): state["ic_sum"] += ic; state["ic_count"] += 1
    if np.isfinite(spread): state["spread_sum"] += spread; state["spread_count"] += 1; state["positive"] += int(spread > 0)


def t_pvalue(values):
    values = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if len(values) < 2: return np.nan
    std = values.std(ddof=1)
    if not np.isfinite(std) or std == 0: return 0. if values.mean() else 1.
    stat = values.mean() / (std / math.sqrt(len(values)))
    return float(2 * student_t.sf(abs(stat), len(values)-1)) if student_t else math.erfc(abs(stat)/math.sqrt(2))


def summarize(states, key_names):
    daily = []
    for key, s in states.items():
        row = dict(zip(key_names, key)); row.update(s)
        row["mean_rank_ic"] = s["ic_sum"] / s["ic_count"] if s["ic_count"] else np.nan
        row["mean_spread"] = s["spread_sum"] / s["spread_count"] if s["spread_count"] else np.nan
        daily.append(row)
    daily = pd.DataFrame(daily)
    if daily.empty: return daily
    ids = [x for x in key_names if x != "date"]; rows = []
    for key, g in daily.groupby(ids, sort=False, dropna=False):
        ic_count, spread_count = int(g.ic_count.sum()), int(g.spread_count.sum())
        daily_ic = pd.to_numeric(g.mean_rank_ic, errors="coerce").dropna()
        mean_ic = g.ic_sum.sum()/ic_count if ic_count else np.nan; sign = np.sign(mean_ic) if np.isfinite(mean_ic) else 0
        row = dict(zip(ids, key if isinstance(key, tuple) else (key,)))
        row.update(days=g.date.nunique(), snapshots=int(g.snapshots.sum()), sample_count=int(g.sample_count.sum()),
                   mean_rank_ic=mean_ic, mean_spread=g.spread_sum.sum()/spread_count if spread_count else np.nan,
                   daily_ic_direction_rate=float((np.sign(daily_ic)==sign).mean()) if len(daily_ic) and sign else np.nan,
                   daily_ic_pvalue=t_pvalue(daily_ic))
        rows.append(row)
    return pd.DataFrame(rows)


def pivot(summary, ids):
    if summary.empty: return summary
    values = ["mean_rank_ic", "mean_spread", "daily_ic_direction_rate", "days", "snapshots", "sample_count", "daily_ic_pvalue"]
    out = summary.pivot_table(index=ids, columns="signal", values=values, aggfunc="first")
    out.columns = [f"{a}__{b}" for a,b in out.columns]
    return out.reset_index()


def classify(full, baseline, resid, own_proxy=False):
    if not all(np.isfinite(v) for v in (full, baseline, resid)) or full == 0: return "insufficient_data"
    retention = abs(resid)/max(abs(full), 1e-12)
    if abs(baseline) >= .8*abs(full) and retention < .5: return "own_return_proxy_likely" if own_proxy else "simple_reversal_proxy_likely"
    if np.sign(full)==np.sign(resid) and abs(resid) >= max(.005, .5*abs(full)) and (own_proxy or abs(full) >= abs(baseline)+.002): return "network_increment_supported"
    return "inconclusive"


def scopes(report: Path):
    frame = candidates(report/"p2_intraday_scorecard.csv")
    if frame.empty: return set(), set()
    under = frame[frame.score.astype(str).eq("daily_underreaction_score")]
    corr = frame[return_corr(frame)]
    cols = ["layer_id","scale","target"]
    return set(under[cols].astype(str).itertuples(index=False,name=None)), set(corr[cols].astype(str).itertuples(index=False,name=None))


def scan_p2(root: Path, month: str, under_scope, corr_scope, progress=25):
    states = defaultdict(new_state); files = rows = groups = 0; started = time.time()
    paths = sorted((root/"intraday_relation_features").glob(f"date={month}-*/layer_id=*/scale=*/intraday_relation_features.parquet"))
    for path in paths:
        date, layer, scale = part(path,"date") or "", part(path,"layer_id") or "", part(path,"scale") or ""
        targets = sorted({t for l,s,t in under_scope|corr_scope if l==layer and s==scale})
        if not targets: continue
        pf = pq.ParquetFile(path)
        try: available=set(pf.schema.names)
        finally: pf.close()
        cols=[c for c in ["decision_time","level","expected_pressure_z","target_pre_response_z","daily_underreaction_score","daily_consensus_score","pit_audit_pass",*targets] if c in available]
        if not {"decision_time","level","expected_pressure_z","target_pre_response_z","pit_audit_pass"}.issubset(cols): continue
        files += 1
        for ts,snap in iter_time_groups(path, cols, time_column="decision_time"):
            groups += 1; rows += len(snap)
            if not snap.pit_audit_pass.fillna(False).all(): raise AssertionError(f"PIT failure: {path} {ts}")
            for level,g in snap.groupby("level",sort=False,dropna=False):
                for target in targets:
                    if target not in g: continue
                    key=(layer,scale,target)
                    if key in under_scope and "daily_underreaction_score" in g:
                        w=g[["expected_pressure_z","target_pre_response_z","daily_underreaction_score",target]].replace([np.inf,-np.inf],np.nan).dropna()
                        if len(w)>=MIN_SAMPLE:
                            a=w.expected_pressure_z.to_numpy(float); b=-w.target_pre_response_z.to_numpy(float); c=w.daily_underreaction_score.to_numpy(float); d=residual(c,b); y=w[target].to_numpy(float)
                            for name,x in (("A_expected_pressure",a),("B_simple_reversal",b),("C_full_underreaction",c),("D_residualized_on_reversal",d)):
                                n,ic,sp=metric(x,y); update(states[("risk1",date,layer,scale,str(level),target,name)],n,ic,sp)
                    if key in corr_scope:
                        needed=["expected_pressure_z","target_pre_response_z",target]+(["daily_consensus_score"] if "daily_consensus_score" in g else [])
                        w=g[needed].replace([np.inf,-np.inf],np.nan).dropna()
                        if len(w)>=MIN_SAMPLE:
                            own=w.target_pre_response_z.to_numpy(float); pressure=w.expected_pressure_z.to_numpy(float); y=w[target].to_numpy(float)
                            sig=[("own_past_return",own),("network_pressure",pressure),("network_pressure_residual",residual(pressure,own))]
                            if "daily_consensus_score" in w:
                                con=w.daily_consensus_score.to_numpy(float); sig += [("network_consensus",con),("network_consensus_residual",residual(con,own))]
                            for name,x in sig:
                                n,ic,sp=metric(x,y); update(states[("risk2_p2",date,layer,scale,str(level),target,name)],n,ic,sp)
        if progress and files%progress==0:
            print(f"[risk-audit] files={files} rows={rows:,} rate={rows/max(time.time()-started,1e-9):,.0f}/s",flush=True)
    summary=summarize(states,["audit","date","layer_id","scale","level","target","signal"])
    r1=pivot(summary[summary.audit.eq("risk1")],["layer_id","scale","level","target"]) if not summary.empty else pd.DataFrame()
    r2=pivot(summary[summary.audit.eq("risk2_p2")],["layer_id","scale","level","target"]) if not summary.empty else pd.DataFrame()
    if not r1.empty:
        c=r1.get("mean_rank_ic__C_full_underreaction",pd.Series(np.nan,index=r1.index)); b=r1.get("mean_rank_ic__B_simple_reversal",pd.Series(np.nan,index=r1.index)); d=r1.get("mean_rank_ic__D_residualized_on_reversal",pd.Series(np.nan,index=r1.index))
        r1["c_minus_b_abs_ic"]=c.abs()-b.abs(); r1["residual_ic_retention"]=d.abs()/c.abs().replace(0,np.nan); r1["risk_status"]=[classify(x,y,z) for x,y,z in zip(c,b,d)]
    if not r2.empty:
        own=r2.get("mean_rank_ic__own_past_return",pd.Series(np.nan,index=r2.index))
        for family in ("pressure","consensus"):
            net=r2.get(f"mean_rank_ic__network_{family}",pd.Series(np.nan,index=r2.index)); res=r2.get(f"mean_rank_ic__network_{family}_residual",pd.Series(np.nan,index=r2.index))
            r2[f"{family}_residual_ic_retention"]=res.abs()/net.abs().replace(0,np.nan); r2[f"{family}_risk_status"]=[classify(x,y,z,True) for x,y,z in zip(net,own,res)]
    return r1,r2,{"files":files,"rows":rows,"groups":groups}


def p0_scope(report: Path):
    frame=candidates(report/"p0_eval"/"p0_eval_combo_scorecard.csv")
    if frame.empty:return set()
    frame=frame[return_corr(frame)]
    frame=frame[frame.feature.astype(str).isin({"p0_edge_spillover_signal","p0_edge_spillover_sum"})]
    return set(frame[["layer_id","scale","feature","target"]].astype(str).itertuples(index=False,name=None))


def scan_p0(root: Path, labels_root: Path|None, month: str, scope):
    if not scope:return pd.DataFrame(),{"status":"no_candidates","files":0}
    if labels_root is None:return pd.DataFrame(),{"status":"labels_root_missing","files":0}
    states=defaultdict(new_state); files=0
    for date_dir in sorted((root/"p0_edge_spillover").glob(f"date={month}-*")):
        date=part(date_dir,"date") or date_dir.name.split("=",1)[-1]; label_file=labels_root/f"date={date}"/"labels.parquet"
        if not label_file.exists():continue
        hs=sorted({t.replace("target_","") for *_,t in scope}|{"15m"}); lab=load_labels(label_file,hs)
        if "past_label_15m" not in lab:continue
        own=lab[["decision_time","symbol_id","past_label_15m"]].rename(columns={"symbol_id":"dst_id","past_label_15m":"own_past_return"})
        for layer,scale,feature,target in scope:
            path=date_dir/f"layer_id={layer}"/f"scale={scale}"/"p0_edge_spillover_features.parquet"
            if not path.exists():continue
            files+=1
            for ts,snap in iter_time_groups(path,["decision_time","dst_id",feature,target,"pit_audit_pass"],time_column="decision_time"):
                if "pit_audit_pass" in snap and not snap.pit_audit_pass.fillna(False).all():raise AssertionError(f"P0 PIT failure: {path} {ts}")
                w=snap.merge(own,on=["decision_time","dst_id"],how="inner",validate="many_to_one")[[feature,"own_past_return",target]].replace([np.inf,-np.inf],np.nan).dropna()
                if len(w)<MIN_SAMPLE:continue
                net=w[feature].to_numpy(float); o=w.own_past_return.to_numpy(float); y=w[target].to_numpy(float)
                for name,x in (("network_spillover",net),("own_past_return",o),("network_spillover_residual",residual(net,o))):
                    n,ic,sp=metric(x,y);update(states[(date,layer,scale,feature,target,name)],n,ic,sp)
    out=pivot(summarize(states,["date","layer_id","scale","feature","target","signal"]),["layer_id","scale","feature","target"])
    if not out.empty:
        net=out.get("mean_rank_ic__network_spillover",pd.Series(np.nan,index=out.index)); own=out.get("mean_rank_ic__own_past_return",pd.Series(np.nan,index=out.index)); res=out.get("mean_rank_ic__network_spillover_residual",pd.Series(np.nan,index=out.index))
        out["residual_ic_retention"]=res.abs()/net.abs().replace(0,np.nan);out["risk_status"]=[classify(x,y,z,True) for x,y,z in zip(net,own,res)]
    return out,{"status":"complete" if files else "no_matching_files","files":files}


def missing_data(report: Path, counterfactual: Path|None):
    frame=candidates(report/"monthly_alpha_unified_scorecard.csv"); corr=frame[return_corr(frame)] if not frame.empty else pd.DataFrame()
    variants=["median_fill_baseline","pairwise_overlap","complete_active_points","higher_minimum_overlap"]
    detail=pd.read_csv(counterfactual) if counterfactual and counterfactual.exists() else pd.DataFrame({"variant":variants,"status":["required"]*4,"notes":["controlled graph/eval result required"]*4})
    present=set(detail.variant.astype(str)) if "variant" in detail else set(); status="counterfactual_results_present" if set(variants).issubset(present) and "mean_rank_ic" in detail else "unresolved_requires_counterfactual_graph_variants"
    payload={"risk":"return_corr_median_fill_artifact","status":status,"pipeline_changed":False,"candidate_count":len(frame),"return_corr_candidate_count":len(corr),"return_corr_candidate_share":len(corr)/len(frame) if len(frame) else np.nan,"required_variants":variants,"pass_condition":"Sign and economically meaningful IC/Spread survive all alternative missing-data definitions.","limitation":"Existing outputs do not retain per-symbol imputation masks, so this cannot be reconstructed post hoc."}
    return payload,detail


def level_redundancy(report: Path, primary="B50", replica="B35"):
    details=[]; counts={}
    for stage,name in (("p2_intraday","p2_intraday_scorecard.csv"),("p2_daily","p2_daily_scorecard.csv")):
        path=report/name
        if not path.exists():continue
        frame=pd.read_csv(path,dtype={"layer_id":"string","scale":"string","level":"string"}); keys=["layer_id","scale","score","target"]
        left=frame[frame.level.astype(str).eq(primary)];right=frame[frame.level.astype(str).eq(replica)];cols=keys+[c for c in ["mean_rank_ic","mean_spread","candidate_pass"] if c in frame]
        merged=left[cols].merge(right[cols],on=keys,how="outer",suffixes=(f"_{primary}",f"_{replica}"),indicator=True);merged.insert(0,"stage",stage)
        merged["abs_ic_diff"]=(pd.to_numeric(merged.get(f"mean_rank_ic_{primary}"),errors="coerce")-pd.to_numeric(merged.get(f"mean_rank_ic_{replica}"),errors="coerce")).abs();merged["abs_spread_diff"]=(pd.to_numeric(merged.get(f"mean_spread_{primary}"),errors="coerce")-pd.to_numeric(merged.get(f"mean_spread_{replica}"),errors="coerce")).abs();merged["near_identical_metrics"]=merged.abs_ic_diff.le(1e-8)&merged.abs_spread_diff.le(1e-10);details.append(merged)
        cand=frame[bools(frame.candidate_pass)] if "candidate_pass" in frame else frame.iloc[0:0];groups=cand.groupby(keys)["level"].nunique() if len(cand) else pd.Series(dtype=int)
        counts[stage]={"raw_candidate_rows":len(cand),"unique_hypotheses_ignoring_level":len(groups),"both_levels_candidate_hypotheses":int((groups>=2).sum()),"duplicate_inflation_rows":len(cand)-len(groups)}
    detail=pd.concat(details,ignore_index=True) if details else pd.DataFrame();paired=detail[detail._merge.eq("both")] if not detail.empty else pd.DataFrame()
    return detail,{"stage_counts":counts,"paired_hypotheses":len(paired),"near_identical_metric_pairs":int(paired.near_identical_metrics.sum()) if len(paired) else 0,"near_identical_metric_rate":float(paired.near_identical_metrics.mean()) if len(paired) else np.nan}


def p1_passthrough(p1_root: Path|None, month: str, primary="B50", replica="B35"):
    if p1_root is None:return pd.DataFrame(),{"status":"p1_root_missing","files":0}
    rows=[];tot={"b50_to_b35_edges":0,"passthrough_edges":0,"exact_size_passthrough_edges":0};files=sorted(p1_root.glob(f"date={month}-*/layer_id=*/scale=*/theme_tree_edges.parquet"))
    for path in files:
        local={k:0 for k in tot};pf=pq.ParquetFile(path)
        try:
            cols=[c for c in ["parent_level","child_level","split_mode","parent_size","child_size","child_share"] if c in pf.schema.names]
            for batch in pf.iter_batches(columns=cols,batch_size=250000,use_threads=False):
                f=pa.Table.from_batches([batch]).to_pandas(split_blocks=True,self_destruct=True);g=f[f.parent_level.astype(str).eq(primary)&f.child_level.astype(str).eq(replica)];pas=g.split_mode.astype(str).eq("passthrough");exact=pas.copy()
                if "parent_size" in g:exact&=pd.to_numeric(g.parent_size,errors="coerce").eq(pd.to_numeric(g.child_size,errors="coerce"))
                if "child_share" in g:exact&=pd.to_numeric(g.child_share,errors="coerce").sub(1).abs().le(1e-12)
                local["b50_to_b35_edges"]+=len(g);local["passthrough_edges"]+=int(pas.sum());local["exact_size_passthrough_edges"]+=int(exact.sum())
        finally:pf.close()
        for k in tot:tot[k]+=local[k]
        rows.append({"date":part(path,"date"),"layer_id":part(path,"layer_id"),"scale":part(path,"scale"),**local,"passthrough_rate":local["passthrough_edges"]/local["b50_to_b35_edges"] if local["b50_to_b35_edges"] else np.nan})
    return pd.DataFrame(rows),{"status":"complete" if files else "no_files","files":len(files),**tot,"passthrough_rate":tot["passthrough_edges"]/tot["b50_to_b35_edges"] if tot["b50_to_b35_edges"] else np.nan,"exact_duplicate_semantics":"split_mode=passthrough preserves the complete B50 member set in the B35 child."}


def table(frame,cols,limit=100):
    if frame.empty:return "<p><em>无可用数据</em></p>"
    f=frame[[c for c in cols if c in frame]].head(limit).copy()
    for c in f:
        if pd.api.types.is_float_dtype(f[c]):f[c]=f[c].map(lambda x:"" if pd.isna(x) else f"{x:.6g}")
    return f.to_html(index=False,escape=True,border=0,classes="data")


def html_section(summary,r1,r2p2,r2p0,r3d,r4d,r4p1):
    r3=summary["risk3_missing_data_median_fill_artifact"];r4=summary["risk4_b50_b35_non_independence"]
    return f"""<!-- ALPHA_RISK_AUDIT_BEGIN --><div class='section' id='alpha-falsification-audits'><h2>9. Alpha 反证与代理风险审计</h2><p>本节只读取正式产物，不修改 Feature、Label、PIT 或 Eval。未通过或 unresolved 的信号不得解释为已确认网络 Alpha。</p><h3>风险1：Underreaction vs 简单反转</h3>{table(r1,['layer_id','scale','level','target','mean_rank_ic__A_expected_pressure','mean_rank_ic__B_simple_reversal','mean_rank_ic__C_full_underreaction','mean_rank_ic__D_residualized_on_reversal','residual_ic_retention','risk_status'])}<h3>风险2：Return-correlation vs 自身过去收益</h3><h4>P2</h4>{table(r2p2,['layer_id','scale','level','target','mean_rank_ic__own_past_return','mean_rank_ic__network_pressure','mean_rank_ic__network_pressure_residual','pressure_risk_status','mean_rank_ic__network_consensus','mean_rank_ic__network_consensus_residual','consensus_risk_status'])}<h4>P0</h4>{table(r2p0,['layer_id','scale','feature','target','mean_rank_ic__own_past_return','mean_rank_ic__network_spillover','mean_rank_ic__network_spillover_residual','residual_ic_retention','risk_status'])}<h3>风险3：中位数填补伪影</h3><p>状态：<b>{html.escape(str(r3['status']))}</b>。现有产物没有 imputation mask，不能在报告中伪装成已经排除；必须做受控图对照。</p>{table(r3d,list(r3d.columns))}<h3>风险4：B50/B35 非独立</h3><p>状态：<b>{html.escape(str(r4['status']))}</b>。B35 passthrough 与 B50 成员完全相同；其余 B35 也是 B50 的嵌套细分。未来默认只跑 B50。</p>{table(r4d,['stage','layer_id','scale','score','target','mean_rank_ic_B50','mean_rank_ic_B35','abs_ic_diff','near_identical_metrics','_merge'])}<h4>P1 passthrough 社区</h4>{table(r4p1,['date','layer_id','scale','b50_to_b35_edges','passthrough_edges','exact_size_passthrough_edges','passthrough_rate'])}</div><!-- ALPHA_RISK_AUDIT_END -->"""


def rebuild_bundle(report:Path,month:str):
    bundle=report/"monthly_alpha_report_bundle.zip";tmp=Path(str(bundle)+".tmp");tmp.unlink(missing_ok=True);files=[]
    for p in sorted(report.rglob("*")):
        if p.is_file() and p not in {bundle,tmp} and p.name!="monthly_alpha_report_manifest.json" and p.suffix!=".zip":
            h=hashlib.sha256(p.read_bytes()).hexdigest();files.append({"path":str(p.relative_to(report)),"bytes":p.stat().st_size,"sha256":h})
    atomic_text(report/"monthly_alpha_report_manifest.json",json.dumps({"report_contract_version":"monthly-full-alpha-report-v1+risk-audit-v1","risk_audit_contract_version":CONTRACT,"month":month,"raw_parquet_copied":False,"files":files},indent=2,ensure_ascii=False))
    with zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(report.rglob("*")):
            if p.is_file() and p not in {bundle,tmp} and p.suffix!=".zip":z.write(p,str(p.relative_to(report)))
    os.replace(tmp,bundle);return bundle


def enrich(p2_root,month,report_dir=None,labels_root=None,p1_root=None,primary="B50",replica="B35",counterfactual=None,progress=25):
    root=Path(p2_root);report=Path(report_dir) if report_dir else root/"monthly_alpha_report"/month.replace("-","");jp=report/"monthly_alpha_report.json";hp=report/"monthly_alpha_report.html"
    if not jp.exists() or not hp.exists():raise FileNotFoundError("base monthly report missing")
    us,cs=scopes(report);r1,r2p2,scan=scan_p2(root,month,us,cs,progress);r2p0,p0stats=scan_p0(root,Path(labels_root) if labels_root else None,month,p0_scope(report));r3,r3d=missing_data(report,Path(counterfactual) if counterfactual else None);r4d,r4m=level_redundancy(report,primary,replica);r4p1,r4pm=p1_passthrough(Path(p1_root) if p1_root else None,month,primary,replica)
    summary={"risk_audit_contract_version":CONTRACT,"risk1_underreaction_reversal_proxy":{"status_counts":r1.risk_status.value_counts().to_dict() if not r1.empty else {}},"risk2_return_corr_own_return_proxy":{"p2_pressure":r2p2.pressure_risk_status.value_counts().to_dict() if not r2p2.empty and "pressure_risk_status" in r2p2 else {},"p2_consensus":r2p2.consensus_risk_status.value_counts().to_dict() if not r2p2.empty and "consensus_risk_status" in r2p2 else {},"p0":r2p0.risk_status.value_counts().to_dict() if not r2p0.empty else {}},"risk3_missing_data_median_fill_artifact":r3,"risk4_b50_b35_non_independence":{"status":"not_independent_confirmed" if r4pm.get("exact_size_passthrough_edges",0) or r4m.get("near_identical_metric_pairs",0) else "replication_unavailable","primary_level":primary,"replication_level":replica,"metric_redundancy":r4m,"p1_passthrough_community_redundancy":r4pm,"compute_policy":f"Default {primary}; use GFF_RESEARCH_LEVELS={primary},{replica} only for explicit replication."},"source_scan":{"intraday":scan,"p0":p0stats},"generated_at":time.strftime("%Y-%m-%dT%H:%M:%S")}
    for name,frame in (("risk1_underreaction_ablation.csv",r1),("risk2_return_corr_p2_proxy_audit.csv",r2p2),("risk2_return_corr_p0_proxy_audit.csv",r2p0),("risk3_missing_data_counterfactual_audit.csv",r3d),("risk4_b50_b35_metric_redundancy.csv",r4d),("risk4_p1_passthrough_communities.csv",r4p1)):atomic_csv(frame,report/name)
    atomic_text(report/"alpha_falsification_risk_audit.json",json.dumps(jsonable(summary),indent=2,ensure_ascii=False));payload=read_json(jp);payload["alpha_falsification_risk_audit"]=summary;payload["risk_adjusted_reporting_policy"]={"primary_theme_level":primary,"replication_theme_level":replica,"count_replication_as_independent_alpha":False,"underreaction_requires_residual_increment":True,"return_corr_requires_own_return_residual_increment":True,"missing_data_artifact_status":r3["status"]};atomic_text(jp,json.dumps(jsonable(payload),indent=2,ensure_ascii=False))
    doc=hp.read_text(encoding="utf-8");doc=re.sub(r"<!-- ALPHA_RISK_AUDIT_BEGIN -->.*?<!-- ALPHA_RISK_AUDIT_END -->","",doc,flags=re.S);section=html_section(summary,r1,r2p2,r2p0,r3d,r4d,r4p1);pos=doc.rfind("<script id='monthly-alpha-report'");pos=doc.rfind("</main>") if pos<0 else pos;doc=doc[:pos]+section+doc[pos:];encoded=json.dumps(jsonable(payload),ensure_ascii=False).replace("</","<\\/");doc=re.sub(r"(<script id='monthly-alpha-report' type='application/json'>).*?(</script>)",lambda m:m.group(1)+encoded+m.group(2),doc,count=1,flags=re.S);atomic_text(hp,doc);bundle=rebuild_bundle(report,month)
    out={"status":"complete","month":month,"report_dir":str(report),"bundle":str(bundle),"risk_audit":str(report/"alpha_falsification_risk_audit.json")};print(json.dumps(out,indent=2,ensure_ascii=False));return out


def generate(p2_root,month,**kwargs):
    report=Path(kwargs.pop("report_dir",None) or Path(p2_root)/"monthly_alpha_report"/month.replace("-",""));generate_monthly_report(p2_root,month,report,batch_size=kwargs.pop("batch_size",250000),top_n=kwargs.pop("top_n",50),json_top_n=kwargs.pop("json_top_n",200),allow_partial=kwargs.pop("allow_partial",False),progress_every=kwargs.get("progress",25));return enrich(p2_root,month,report_dir=report,**kwargs)


def main():
    p=argparse.ArgumentParser();p.add_argument("--p2-root",required=True);p.add_argument("--month",required=True);p.add_argument("--output-dir");p.add_argument("--labels-root");p.add_argument("--p1-root");p.add_argument("--primary-level",default="B50");p.add_argument("--replication-level",default="B35");p.add_argument("--counterfactual-results");p.add_argument("--batch-size",type=int,default=250000);p.add_argument("--top-n",type=int,default=50);p.add_argument("--json-top-n",type=int,default=200);p.add_argument("--progress-every",type=int,default=25);p.add_argument("--allow-partial",action="store_true");p.add_argument("--enrich-existing",action="store_true");a=p.parse_args();kw=dict(report_dir=a.output_dir,labels_root=a.labels_root,p1_root=a.p1_root,primary=a.primary_level,replica=a.replication_level,counterfactual=a.counterfactual_results,progress=a.progress_every)
    enrich(a.p2_root,a.month,**kw) if a.enrich_existing else generate(a.p2_root,a.month,batch_size=a.batch_size,top_n=a.top_n,json_top_n=a.json_top_n,allow_partial=a.allow_partial,**kw)


if __name__=="__main__":main()
