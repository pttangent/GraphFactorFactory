#!/usr/bin/env python3
from __future__ import annotations
import argparse, concurrent.futures as cf, json, os, time
from dataclasses import dataclass
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','1'); os.environ.setdefault('MKL_NUM_THREADS','1'); os.environ.setdefault('OPENBLAS_NUM_THREADS','1'); os.environ.setdefault('NUMEXPR_NUM_THREADS','1'); os.environ.setdefault('ARROW_NUM_THREADS','1'); os.environ.setdefault('POLARS_MAX_THREADS','1')
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
try:
    _orig_unreg=pa.unregister_extension_type
    def _safe_unreg(name):
        try: return _orig_unreg(name)
        except Exception: return None
    pa.unregister_extension_type=_safe_unreg
except Exception: pass
HORIZONS=['5m','15m','30m','60m','120m']
@dataclass(frozen=True)
class Part: date:str; layer_id:str; scale:str; base:Path

def csvset(s): return None if not s else {x.strip() for x in str(s).split(',') if x.strip()}
def csvlist(s): return None if not s else [x.strip() for x in str(s).split(',') if x.strip()]
def mins(h): return int(str(h)[:-1]) if str(h).endswith('m') else (_ for _ in ()).throw(ValueError(h))
def ext(p,k):
    for x in Path(p).parts:
        if x.startswith(k+'='): return x.split('=',1)[1]
    return None
def write_parquet_atomic(df, path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp')
    if tmp.exists(): tmp.unlink()
    pq.write_table(pa.Table.from_pandas(df,preserve_index=False), tmp, compression='zstd'); tmp.replace(path)
def done(m):
    try:
        j=json.loads(Path(m).read_text(encoding='utf-8')); return j.get('status')=='complete' and int(j.get('output_rows',0))>0
    except Exception: return False
def manifest(d,meta):
    d=Path(d); d.mkdir(parents=True,exist_ok=True); t=d/'manifest.json.tmp'; t.write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8'); t.replace(d/'manifest.json')
def label_path(root,date):
    r=Path(root); cands=[r if r.is_file() else None, r/f'date={date}'/'labels.parquet', r/'canonical'/f'date={date}'/'labels.parquet', r/date/'labels.parquet']
    for c in cands:
        if c is not None and c.exists(): return c
    raise FileNotFoundError(f'labels.parquet not found for date={date} under {r}')
def discover(root, filename='edges.parquet', dates=None, layers=None, scales=None):
    out=[]
    for p in Path(root).rglob(filename):
        d,l,s=ext(p,'date'),ext(p,'layer_id'),ext(p,'scale')
        if not d:
            # allow a single flat edges.parquet benchmark file; infer date from parent or filter if only one date is requested
            d=next(iter(dates)) if dates and len(dates)==1 else 'unknown'
        if not l: l='all'
        if not s: s='default'
        if dates and d not in dates: continue
        if layers and l not in layers and l!='all': continue
        if scales and s not in scales and s!='default': continue
        out.append(Part(d,l,s,p))
    out.sort(key=lambda x:x.base.stat().st_size, reverse=True); return out
def read_table(path, cols=None): return pq.ParquetFile(path).read(columns=cols).to_pandas()
def stream_df(path, frames):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=Path(str(path)+'.tmp')
    if tmp.exists(): tmp.unlink()
    writer=None; schema=None; rows=0; batches=0
    try:
        for df in frames:
            if df is None or df.empty: continue
            table=pa.Table.from_pandas(df,preserve_index=False)
            if writer is None:
                schema=table.schema; writer=pq.ParquetWriter(tmp,schema,compression='zstd')
            elif table.schema != schema:
                table=table.cast(schema)
            writer.write_table(table); rows += len(df); batches += 1
    finally:
        if writer is not None: writer.close()
    if rows: os.replace(tmp,path)
    elif tmp.exists(): tmp.unlink()
    return rows,batches
def load_labels(path,horizons):
    names=set(pq.ParquetFile(path).schema.names); labs=[f'label_{h}' for h in horizons if f'label_{h}' in names]
    if not labs: raise ValueError(f'no label columns in {path}')
    df=read_table(path,['decision_time','symbol_id']+labs)
    df['decision_time']=pd.to_datetime(df['decision_time'],utc=True); df['symbol_id']=pd.to_numeric(df['symbol_id'],errors='coerce').astype('Int64')
    df=df.dropna(subset=['decision_time','symbol_id']).copy(); df['symbol_id']=df['symbol_id'].astype('int64')
    for h in horizons:
        c=f'label_{h}'
        if c in df:
            p=df[['decision_time','symbol_id',c]].copy(); p['decision_time']=p['decision_time']+pd.Timedelta(minutes=mins(h)); p=p.rename(columns={c:f'past_label_{h}'})
            df=df.merge(p,on=['decision_time','symbol_id'],how='left')
    return df
def normalize_edges(e,part):
    e=e.copy(); e['decision_time']=pd.to_datetime(e['decision_time'],utc=True,errors='coerce')
    e['src_id']=pd.to_numeric(e['src_id'],errors='coerce').astype('Int64'); e['dst_id']=pd.to_numeric(e['dst_id'],errors='coerce').astype('Int64')
    e=e.dropna(subset=['decision_time','src_id','dst_id']).copy(); e['src_id']=e['src_id'].astype('int64'); e['dst_id']=e['dst_id'].astype('int64')
    if 'layer_id' not in e: e['layer_id']=part.layer_id
    if 'scale' not in e: e['scale']=part.scale
    e['weight']=pd.to_numeric(e.get('weight',0.0),errors='coerce').fillna(0.0); e['abs_weight']=e['weight'].abs()
    return e

def p0_node_one(part, labels_root, out_root, horizons, skip, max_rg):
    t=time.time(); out=Path(out_root)/f'date={part.date}'/f'layer_id={part.layer_id}'/f'scale={part.scale}'; op=out/'p0_node_features.parquet'
    if skip and done(out/'manifest.json'): return {'stage':'p0_node_features','status':'skipped','date':part.date,'layer_id':part.layer_id,'scale':part.scale}
    lab=load_labels(label_path(labels_root,part.date),horizons); lab_by={k:v for k,v in lab.groupby('decision_time',sort=False,dropna=False)}
    pf=pq.ParquetFile(part.base); nrg=pf.metadata.num_row_groups if max_rg is None else min(max_rg,pf.metadata.num_row_groups)
    def frames():
        for i in range(nrg):
            e=normalize_edges(pf.read_row_group(i,columns=[c for c in ['decision_time','layer_id','scale','src_id','dst_id','weight'] if c in pf.schema.names]).to_pandas(),part)
            if e.empty: continue
            src=e.groupby(['decision_time','layer_id','scale','src_id'],sort=False).agg(src_edge_count=('dst_id','size'),src_weight_sum=('abs_weight','sum'),src_weight_mean=('abs_weight','mean'),src_weight_max=('abs_weight','max')).reset_index().rename(columns={'src_id':'symbol_id'})
            dst=e.groupby(['decision_time','layer_id','scale','dst_id'],sort=False).agg(dst_edge_count=('src_id','size'),dst_weight_sum=('abs_weight','sum'),dst_weight_mean=('abs_weight','mean'),dst_weight_max=('abs_weight','max')).reset_index().rename(columns={'dst_id':'symbol_id'})
            z=src.merge(dst,on=['decision_time','layer_id','scale','symbol_id'],how='outer').fillna(0)
            z['p0_total_edge_count']=z.src_edge_count+z.dst_edge_count; z['p0_total_weight_sum']=z.src_weight_sum+z.dst_weight_sum
            dt=z['decision_time'].iloc[0]; y=lab_by.get(dt)
            if y is not None: z=z.merge(y,on=['decision_time','symbol_id'],how='inner')
            if not z.empty: yield z
    rows,batches=stream_df(op,frames()); meta={'stage':'p0_node_features','status':'complete' if rows else 'empty','date':part.date,'layer_id':part.layer_id,'scale':part.scale,'output_rows':int(rows),'write_batches':int(batches),'input':str(part.base),'output':str(op),'row_groups':int(nrg),'elapsed_sec':round(time.time()-t,3)}; manifest(out,meta); return meta

def p0_edge_one(part, labels_root, out_root, horizons, past_h, skip, max_rg):
    t=time.time(); out=Path(out_root)/f'date={part.date}'/f'layer_id={part.layer_id}'/f'scale={part.scale}'; op=out/'p0_edge_spillover_features.parquet'
    if skip and done(out/'manifest.json'): return {'stage':'p0_edge_spillover','status':'skipped','date':part.date,'layer_id':part.layer_id,'scale':part.scale}
    lab=load_labels(label_path(labels_root,part.date),horizons); pc=f'past_label_{past_h}'
    if pc not in lab: raise ValueError(f'missing {pc}')
    src_by={k:v[['decision_time','symbol_id',pc]].rename(columns={'symbol_id':'src_id',pc:'src_past_return'}) for k,v in lab.groupby('decision_time',sort=False,dropna=False)}
    dst_cols=[c for c in lab.columns if c.startswith('label_')]
    dst_by={k:v[['decision_time','symbol_id']+dst_cols].rename(columns={'symbol_id':'dst_id',**{c:'target_'+c.replace('label_','') for c in dst_cols}}) for k,v in lab.groupby('decision_time',sort=False,dropna=False)}
    pf=pq.ParquetFile(part.base); nrg=pf.metadata.num_row_groups if max_rg is None else min(max_rg,pf.metadata.num_row_groups)
    def frames():
        for i in range(nrg):
            e=normalize_edges(pf.read_row_group(i,columns=[c for c in ['decision_time','layer_id','scale','src_id','dst_id','weight'] if c in pf.schema.names]).to_pandas(),part)
            if e.empty: continue
            dt=e['decision_time'].iloc[0]; src=src_by.get(dt); dst=dst_by.get(dt)
            if src is None or dst is None: continue
            m=e.merge(src,on=['decision_time','src_id'],how='inner')
            if m.empty: continue
            m['edge_signal']=m['weight']*pd.to_numeric(m['src_past_return'],errors='coerce').fillna(0.0)
            g=m.groupby(['decision_time','layer_id','scale','dst_id'],sort=False).agg(p0_edge_spillover_signal=('edge_signal','mean'),p0_edge_spillover_sum=('edge_signal','sum'),p0_edge_count=('src_id','size'),p0_edge_abs_weight=('abs_weight','sum'),p0_edge_mean_abs_weight=('abs_weight','mean')).reset_index()
            z=g.merge(dst,on=['decision_time','dst_id'],how='inner')
            if not z.empty: yield z
    rows,batches=stream_df(op,frames()); meta={'stage':'p0_edge_spillover','status':'complete' if rows else 'empty','date':part.date,'layer_id':part.layer_id,'scale':part.scale,'past_horizon':past_h,'output_rows':int(rows),'write_batches':int(batches),'input':str(part.base),'output':str(op),'row_groups':int(nrg),'elapsed_sec':round(time.time()-t,3)}; manifest(out,meta); return meta

def p0_graph_state_one(part,out_root,skip,max_rg):
    t=time.time(); out=Path(out_root)/f'date={part.date}'/f'layer_id={part.layer_id}'/f'scale={part.scale}'; op=out/'p0_graph_state_features.parquet'
    if skip and done(out/'manifest.json'): return {'stage':'p0_graph_state','status':'skipped','date':part.date,'layer_id':part.layer_id,'scale':part.scale}
    pf=pq.ParquetFile(part.base); nrg=pf.metadata.num_row_groups if max_rg is None else min(max_rg,pf.metadata.num_row_groups)
    rows=[]
    for i in range(nrg):
        e=normalize_edges(pf.read_row_group(i,columns=[c for c in ['decision_time','layer_id','scale','src_id','dst_id','weight'] if c in pf.schema.names]).to_pandas(),part)
        if e.empty: continue
        for keys,g in e.groupby(['decision_time','layer_id','scale'],sort=False):
            nodes=pd.unique(pd.concat([g.src_id,g.dst_id],ignore_index=True)); rows.append({'decision_time':keys[0],'layer_id':keys[1],'scale':keys[2],'edge_count':len(g),'active_node_count':len(nodes),'avg_abs_weight':float(g.abs_weight.mean()),'sum_abs_weight':float(g.abs_weight.sum()),'max_abs_weight':float(g.abs_weight.max()),'density_proxy':float(len(g)/max(len(nodes),1))})
    df=pd.DataFrame(rows); write_parquet_atomic(df,op) if not df.empty else None; meta={'stage':'p0_graph_state','status':'complete' if len(df) else 'empty','date':part.date,'layer_id':part.layer_id,'scale':part.scale,'output_rows':int(len(df)),'input':str(part.base),'output':str(op),'row_groups':int(nrg),'elapsed_sec':round(time.time()-t,3)}; manifest(out,meta); return meta

def eval_p0_one(p):
    df=read_table(p); date=ext(p,'date') or 'unknown'; kind='edge' if 'edge_spillover' in p.name else 'node'
    targets=[c for c in df.columns if c.startswith('label_') or c.startswith('target_')]
    feats=[c for c in df.columns if c.startswith('p0_') and pd.api.types.is_numeric_dtype(df[c])]
    rows=[]
    for keys,sub in df.groupby(['layer_id','scale'],dropna=False,sort=False):
        for f in feats:
            for ta in targets:
                v=sub[[f,ta]].replace([np.inf,-np.inf],np.nan).dropna()
                if len(v)<30: continue
                q8,q2=v[f].quantile(.8),v[f].quantile(.2); rows.append({'date':date,'kind':kind,'layer_id':keys[0],'scale':keys[1],'feature':f,'target':ta,'sample_count':len(v),'rank_ic':v[f].rank().corr(v[ta].rank()),'top_minus_bottom':v[v[f]>=q8][ta].mean()-v[v[f]<=q2][ta].mean(),'source':str(p)})
    return rows

def eval_p0(root,out_dir,workers=24):
    t=time.time(); files=list(Path(root).rglob('p0_node_features.parquet'))+list(Path(root).rglob('p0_edge_spillover_features.parquet'))
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    res = pool(files, workers, eval_p0_one)
    rows = [r for sublist in res for r in sublist]
    m=pd.DataFrame(rows); mp=out/'p0_alpha_metrics.csv'; sp=out/'p0_alpha_summary.csv'; m.to_csv(mp,index=False)
    s=m.groupby(['kind','feature','target','layer_id','scale'],sort=False).agg(days=('date','nunique'),sample_count=('sample_count','sum'),mean_rank_ic=('rank_ic','mean'),mean_spread=('top_minus_bottom','mean'),positive_day_rate=('top_minus_bottom',lambda x:float((x>0).mean()))).reset_index() if not m.empty else pd.DataFrame(); s.to_csv(sp,index=False)
    meta={'stage':'p0_alpha_eval','status':'complete' if len(m) else 'empty','input_files':len(files),'metric_rows':int(len(m)),'summary_rows':int(len(s)),'metrics':str(mp),'summary':str(sp),'elapsed_sec':round(time.time()-t,3)}; manifest(out,meta); return meta

def pool(parts,workers,fn,*args):
    if not parts: return []
    res=[]
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        futs=[ex.submit(fn,p,*args) for p in parts]
        for f in cf.as_completed(futs): res.append(f.result())
    return res
def save(root,res): Path(root).mkdir(parents=True,exist_ok=True); (Path(root)/'run_summary.json').write_text(json.dumps(res,indent=2,ensure_ascii=False),encoding='utf-8')

def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='cmd',required=True)
    def common(x):
        x.add_argument('--p0-root',required=True); x.add_argument('--labels-root'); x.add_argument('--out-root',required=True); x.add_argument('--dates'); x.add_argument('--layers'); x.add_argument('--scales'); x.add_argument('--horizons',default=','.join(HORIZONS)); x.add_argument('--workers',type=int,default=16); x.add_argument('--max-row-groups',type=int); x.add_argument('--skip-existing',action='store_true')
    a=sub.add_parser('node-features'); common(a)
    a=sub.add_parser('edge-spillover'); common(a); a.add_argument('--past-horizon',default='15m')
    a=sub.add_parser('graph-state'); common(a)
    a=sub.add_parser('eval-p0'); a.add_argument('--p0-alpha-root',required=True); a.add_argument('--out-dir',required=True)
    args=p.parse_args(); dates,layers,scales=csvset(getattr(args,'dates',None)),csvset(getattr(args,'layers',None)),csvset(getattr(args,'scales',None)); horizons=csvlist(getattr(args,'horizons',None)) or HORIZONS
    if args.cmd=='node-features': parts=discover(args.p0_root,'edges.parquet',dates,layers,scales); res=pool(parts,args.workers,p0_node_one,args.labels_root,args.out_root,horizons,args.skip_existing,args.max_row_groups); save(args.out_root,res); print(json.dumps({'stage':args.cmd,'parts':len(parts),'results':len(res),'out_root':args.out_root},indent=2))
    elif args.cmd=='edge-spillover': parts=discover(args.p0_root,'edges.parquet',dates,layers,scales); res=pool(parts,args.workers,p0_edge_one,args.labels_root,args.out_root,horizons,args.past_horizon,args.skip_existing,args.max_row_groups); save(args.out_root,res); print(json.dumps({'stage':args.cmd,'parts':len(parts),'results':len(res),'out_root':args.out_root},indent=2))
    elif args.cmd=='graph-state': parts=discover(args.p0_root,'edges.parquet',dates,layers,scales); res=pool(parts,args.workers,p0_graph_state_one,args.out_root,args.skip_existing,args.max_row_groups); save(args.out_root,res); print(json.dumps({'stage':args.cmd,'parts':len(parts),'results':len(res),'out_root':args.out_root},indent=2))
    elif args.cmd=='eval-p0': print(json.dumps(eval_p0(args.p0_alpha_root,args.out_dir),indent=2))
if __name__=='__main__': main()
