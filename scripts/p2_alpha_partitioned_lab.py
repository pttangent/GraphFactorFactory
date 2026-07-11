#!/usr/bin/env python3
from __future__ import annotations
import argparse, concurrent.futures as cf, json, math, os, time
from dataclasses import dataclass
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','1'); os.environ.setdefault('MKL_NUM_THREADS','1'); os.environ.setdefault('OPENBLAS_NUM_THREADS','1'); os.environ.setdefault('NUMEXPR_NUM_THREADS','1'); os.environ.setdefault('ARROW_NUM_THREADS','2')
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
from scipy import stats
HORIZONS=['5m','15m','30m','60m','120m']
@dataclass(frozen=True)
class Part:
    date:str; layer_id:str; scale:str; base:Path

def csvset(s): return None if not s else {x.strip() for x in s.split(',') if x.strip()}
def csvlist(s): return None if not s else [x.strip() for x in s.split(',') if x.strip()]
def ext(p,k):
    for x in p.parts:
        if x.startswith(k+'='): return x.split('=',1)[1]
    return None
def complete(m):
    try: return m.exists() and json.loads(m.read_text()).get('status')=='complete' and json.loads(m.read_text()).get('output_rows',0)>0
    except Exception: return False
def labpath(root,date):
    r=Path(root)
    if r.is_file(): return r
    for c in [r/f'date={date}'/'labels.parquet', r/'canonical'/f'date={date}'/'labels.parquet', r/date/'labels.parquet']:
        if c.exists(): return c
    raise FileNotFoundError(f'labels.parquet not found for date={date} under {r}')
def mins(h): return int(h[:-1]) if h.endswith('m') else (_ for _ in ()).throw(ValueError(h))
def discover(root, filename, dates=None, layers=None, scales=None):
    out=[]
    for p in Path(root).rglob(filename):
        d,l,s=ext(p,'date'), ext(p,'layer_id'), ext(p,'scale')
        if not d or not l or not s: continue
        if dates and d not in dates: continue
        if layers and l not in layers: continue
        if scales and s not in scales: continue
        out.append(Part(d,l,s,p))
    out.sort(key=lambda x:x.base.stat().st_size, reverse=True); return out
class Sink:
    def __init__(self,path):
        self.path=Path(path); self.tmp=self.path.with_suffix(self.path.suffix+'.tmp'); self.path.parent.mkdir(parents=True,exist_ok=True)
        if self.tmp.exists(): self.tmp.unlink()
        self.w=None; self.rows=0
    def write(self,df):
        if df is None or df.empty: return
        t=pa.Table.from_pandas(df,preserve_index=False)
        if self.w is None: self.w=pq.ParquetWriter(self.tmp,t.schema,compression='zstd')
        else: t=t.cast(self.w.schema)
        self.w.write_table(t); self.rows+=len(df)
    def close(self):
        if self.w: self.w.close(); self.w=None
        if self.tmp.exists(): self.tmp.replace(self.path)
def labels(path,horizons,with_past=True):
    names=set(pq.ParquetFile(path).schema.names); labs=[f'label_{h}' for h in horizons if f'label_{h}' in names]
    df=pd.read_parquet(path,columns=['decision_time','symbol_id']+labs); df['decision_time']=pd.to_datetime(df['decision_time'],utc=True); df['symbol_id']=pd.to_numeric(df['symbol_id'],errors='coerce').astype('Int64'); df=df.dropna(subset=['symbol_id']).copy(); df['symbol_id']=df['symbol_id'].astype('int64')
    if with_past:
        for h in horizons:
            c=f'label_{h}'
            if c in df.columns:
                p=df[['decision_time','symbol_id',c]].copy(); p['decision_time']=p['decision_time']+pd.Timedelta(minutes=mins(h)); p=p.rename(columns={c:f'past_label_{h}'})
                df=df.merge(p,on=['decision_time','symbol_id'],how='left')
    return df
def manifest(out, meta): Path(out).mkdir(parents=True,exist_ok=True); (Path(out)/'manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')

def build_ret_one(part, labels_root, out_root, horizons, levels, skip, max_rg):
    t0=time.time(); out=Path(out_root)/f'date={part.date}'/f'layer_id={part.layer_id}'/f'scale={part.scale}'; m=out/'manifest.json'; op=out/'theme_returns.parquet'
    if skip and complete(m): return {'status':'skipped','date':part.date,'layer_id':part.layer_id,'scale':part.scale}
    lab=labels(labpath(labels_root,part.date),horizons,True); fut=[c for c in lab.columns if c.startswith('label_')]; past=[c for c in lab.columns if c.startswith('past_label_')]; allc=fut+past
    pf=pq.ParquetFile(part.base); cols=[c for c in ['decision_time','layer_id','scale','level','theme_id','member_id','core_score','rank_in_theme'] if c in pf.schema.names]
    s=Sink(op); n=pf.metadata.num_row_groups if max_rg is None else min(max_rg,pf.metadata.num_row_groups)
    for rg in range(n):
        mem=pf.read_row_group(rg,columns=cols).to_pandas();
        if mem.empty: continue
        mem['decision_time']=pd.to_datetime(mem['decision_time'],utc=True); mem['member_id']=pd.to_numeric(mem['member_id'],errors='coerce').astype('Int64'); mem['core_score']=pd.to_numeric(mem['core_score'],errors='coerce').fillna(0.0); mem=mem.dropna(subset=['member_id','theme_id']).copy(); mem['member_id']=mem['member_id'].astype('int64')
        if 'level' not in mem: mem['level']='UNKNOWN'
        if levels: mem=mem[mem['level'].astype(str).isin(levels)]
        if mem.empty: continue
        if 'layer_id' not in mem: mem['layer_id']=part.layer_id
        if 'scale' not in mem: mem['scale']=part.scale
        df=mem.merge(lab,left_on=['decision_time','member_id'],right_on=['decision_time','symbol_id'],how='inner')
        if df.empty: continue
        gcols=['decision_time','layer_id','scale','level','theme_id']; df=df.sort_values(gcols+['core_score'],ascending=[1,1,1,1,1,0]); g=df.groupby(gcols,sort=False)
        res=g[allc].mean(); res.columns=[c.replace('past_label_','past_eq_') if c.startswith('past_label_') else c.replace('label_','ret_eq_') for c in res.columns]
        w=g['core_score'].sum().replace(0,np.nan)
        for c in allc:
            wc='w_'+c; df[wc]=df[c]*df['core_score']; res[c.replace('past_label_','past_core_') if c.startswith('past_label_') else c.replace('label_','ret_core_')]=g[wc].sum()/w
        top5=g.head(5).groupby(gcols,sort=False)[allc].mean(); top5.columns=[c.replace('past_label_','past_top5_') if c.startswith('past_label_') else c.replace('label_','ret_top5_') for c in top5.columns]
        top10=g.head(10).groupby(gcols,sort=False)[allc].mean(); top10.columns=[c.replace('past_label_','past_top10_') if c.startswith('past_label_') else c.replace('label_','ret_top10_') for c in top10.columns]
        s.write(res.join(top5).join(top10).reset_index())
    s.close(); meta={'status':'complete' if s.rows else 'empty','date':part.date,'layer_id':part.layer_id,'scale':part.scale,'output_rows':s.rows,'output':str(op),'input':str(part.base),'elapsed_sec':round(time.time()-t0,3)}; manifest(out,meta); return meta

def rel_one(part, ret_root, out_root, horizons, past_h, levels, tiers, skip, max_rg):
    t0=time.time(); rp=Path(ret_root)/f'date={part.date}'/f'layer_id={part.layer_id}'/f'scale={part.scale}'/'theme_returns.parquet'; out=Path(out_root)/f'date={part.date}'/f'layer_id={part.layer_id}'/f'scale={part.scale}'; m=out/'manifest.json'; op=out/'relation_spillover_signals.parquet'
    if skip and complete(m): return {'status':'skipped','date':part.date,'layer_id':part.layer_id,'scale':part.scale}
    if not rp.exists(): return {'status':'missing_returns','date':part.date,'layer_id':part.layer_id,'scale':part.scale,'returns':str(rp)}
    rets=pd.read_parquet(rp); rets['decision_time']=pd.to_datetime(rets['decision_time'],utc=True); pc=f'past_eq_{past_h}'
    if pc not in rets: raise ValueError(f'missing {pc}; rebuild theme returns')
    targets=[f'ret_eq_{h}' for h in horizons if f'ret_eq_{h}' in rets]
    past=rets[['decision_time','layer_id','scale','level','theme_id',pc]].rename(columns={'theme_id':'src_theme_id',pc:'src_past_return'})
    fut=rets[['decision_time','layer_id','scale','level','theme_id']+targets].rename(columns={'theme_id':'dst_theme_id',**{c:c.replace('ret_eq_','target_') for c in targets}})
    pf=pq.ParquetFile(part.base); cols=[c for c in ['decision_time','layer_id','scale','level','src_theme_id','dst_theme_id','relation_strength','relation_tier','hard_keep','edge_count'] if c in pf.schema.names]
    s=Sink(op); n=pf.metadata.num_row_groups if max_rg is None else min(max_rg,pf.metadata.num_row_groups)
    for rg in range(n):
        e=pf.read_row_group(rg,columns=cols).to_pandas();
        if e.empty: continue
        e['decision_time']=pd.to_datetime(e['decision_time'],utc=True)
        if levels: e=e[e['level'].astype(str).isin(levels)]
        if tiers and 'relation_tier' in e: e=e[e['relation_tier'].astype(str).isin(tiers)]
        if e.empty: continue
        if 'layer_id' not in e: e['layer_id']=part.layer_id
        if 'scale' not in e: e['scale']=part.scale
        m1=e.merge(past,on=['decision_time','layer_id','scale','level','src_theme_id'],how='inner')
        if m1.empty: continue
        m1['signal']=pd.to_numeric(m1['relation_strength'],errors='coerce').fillna(0)*pd.to_numeric(m1['src_past_return'],errors='coerce').fillna(0)
        a=m1.groupby(['decision_time','layer_id','scale','level','dst_theme_id'],sort=False).agg(signal=('signal','mean'),relation_strength_mean=('relation_strength','mean'),relation_edge_count=('src_theme_id','size')).reset_index()
        z=a.merge(fut,on=['decision_time','layer_id','scale','level','dst_theme_id'],how='inner')
        if not z.empty: z.insert(0,'alpha_name','relation_spillover'); z.insert(1,'date',part.date); s.write(z)
    s.close(); meta={'status':'complete' if s.rows else 'empty','date':part.date,'layer_id':part.layer_id,'scale':part.scale,'output_rows':s.rows,'output':str(op),'input':str(part.base),'elapsed_sec':round(time.time()-t0,3)}; manifest(out,meta); return meta

def core_one(part, labels_root, out_root, horizons, past_h, levels, skip, max_rg):
    t0=time.time(); out=Path(out_root)/f'date={part.date}'/f'layer_id={part.layer_id}'/f'scale={part.scale}'; m=out/'manifest.json'; op=out/'core_peripheral_signals.parquet'
    if skip and complete(m): return {'status':'skipped','date':part.date,'layer_id':part.layer_id,'scale':part.scale}
    lab=labels(labpath(labels_root,part.date),horizons,True); pc=f'past_label_{past_h}'
    if pc not in lab: raise ValueError(f'missing {pc}')
    targets=[c for c in lab.columns if c.startswith('label_')]
    pf=pq.ParquetFile(part.base); cols=[c for c in ['decision_time','layer_id','scale','level','theme_id','member_id','core_score'] if c in pf.schema.names]
    s=Sink(op); n=pf.metadata.num_row_groups if max_rg is None else min(max_rg,pf.metadata.num_row_groups)
    for rg in range(n):
        mem=pf.read_row_group(rg,columns=cols).to_pandas();
        if mem.empty: continue
        mem['decision_time']=pd.to_datetime(mem['decision_time'],utc=True); mem['member_id']=pd.to_numeric(mem['member_id'],errors='coerce').astype('Int64'); mem['core_score']=pd.to_numeric(mem['core_score'],errors='coerce').fillna(0.0); mem=mem.dropna(subset=['member_id','theme_id']).copy(); mem['member_id']=mem['member_id'].astype('int64')
        if 'level' not in mem: mem['level']='UNKNOWN'
        if levels: mem=mem[mem['level'].astype(str).isin(levels)]
        if mem.empty: continue
        if 'layer_id' not in mem: mem['layer_id']=part.layer_id
        if 'scale' not in mem: mem['scale']=part.scale
        df=mem.merge(lab,left_on=['decision_time','member_id'],right_on=['decision_time','symbol_id'],how='inner')
        if df.empty: continue
        gcols=['decision_time','layer_id','scale','level','theme_id']; df=df.sort_values(gcols+['core_score'],ascending=[1,1,1,1,1,0]); df['rank']=df.groupby(gcols,sort=False).cumcount(); df['group_size']=df.groupby(gcols,sort=False)['rank'].transform('size'); df=df[df['group_size']>=5].copy()
        if df.empty: continue
        df['is_core']=df['rank']<np.maximum(1,(df['group_size']*0.2).astype(int)); df['is_peripheral']=df['rank']>=(df['group_size']-np.maximum(1,(df['group_size']*0.5).astype(int)))
        core=df[df.is_core].groupby(gcols,sort=False)[pc].mean().rename('core_past_return'); mx=df.groupby(gcols,sort=False)['core_score'].max().rename('max_core_score')
        peri=df[df.is_peripheral].copy().join(core,on=gcols).join(mx,on=gcols); peri['signal']=peri['core_past_return']*(peri['max_core_score']-peri['core_score'])
        outdf=peri[['decision_time','layer_id','scale','level','theme_id','symbol_id','member_id','signal','core_past_return','core_score','max_core_score']+targets].rename(columns={c:c.replace('label_','target_') for c in targets})
        outdf.insert(0,'alpha_name','core_peripheral'); outdf.insert(1,'date',part.date); s.write(outdf)
    s.close(); meta={'status':'complete' if s.rows else 'empty','date':part.date,'layer_id':part.layer_id,'scale':part.scale,'output_rows':s.rows,'output':str(op),'input':str(part.base),'elapsed_sec':round(time.time()-t0,3)}; manifest(out,meta); return meta


def task_runner(task):
    cmd, part, params = task
    if cmd == 'theme-returns':
        return build_ret_one(part, params['labels_root'], params['out_root'], params['horizons'], params['levels'], params['skip_existing'], params['max_row_groups'])
    if cmd == 'relation-spillover':
        return rel_one(part, params['theme_returns_root'], params['out_root'], params['horizons'], params['past_horizon'], params['levels'], params['relation_tiers'], params['skip_existing'], params['max_row_groups'])
    if cmd == 'core-peripheral':
        return core_one(part, params['labels_root'], params['out_root'], params['horizons'], params['past_horizon'], params['levels'], params['skip_existing'], params['max_row_groups'])
    raise ValueError(cmd)

def run(tasks, workers):
    res=[]
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        futs=[ex.submit(task_runner,t) for t in tasks]
        for f in cf.as_completed(futs):
            r=f.result(); res.append(r); print(json.dumps(r,ensure_ascii=False),flush=True)
    return res
def add_common(ap):
    ap.add_argument('--p1-root',required=True); ap.add_argument('--out-root',required=True); ap.add_argument('--dates'); ap.add_argument('--layers'); ap.add_argument('--scales'); ap.add_argument('--levels',default='B50,B35'); ap.add_argument('--horizons',default=','.join(HORIZONS)); ap.add_argument('--workers',type=int,default=20); ap.add_argument('--max-partitions',type=int); ap.add_argument('--max-row-groups',type=int); ap.add_argument('--skip-existing',action='store_true')
def main(argv=None):
    p=argparse.ArgumentParser(description='Partition-safe P2 alpha lab; workers write shards and return metadata only.'); sp=p.add_subparsers(dest='cmd',required=True)
    a=sp.add_parser('theme-returns'); add_common(a); a.add_argument('--labels-root',required=True)
    a=sp.add_parser('relation-spillover'); add_common(a); a.add_argument('--theme-returns-root',required=True); a.add_argument('--past-horizon',default='15m'); a.add_argument('--relation-tiers')
    a=sp.add_parser('core-peripheral'); add_common(a); a.add_argument('--labels-root',required=True); a.add_argument('--past-horizon',default='15m')
    a=sp.add_parser('reduce'); a.add_argument('--signals-root',required=True); a.add_argument('--out-dir',required=True); a.add_argument('--horizons'); a.add_argument('--pattern',default='*signals.parquet')
    x=p.parse_args(argv); horizons=[h.strip() for h in getattr(x,'horizons','').split(',') if h.strip()] if getattr(x,'horizons',None) else HORIZONS
    if x.cmd=='reduce':
        rows=[]
        for fp in Path(x.signals_root).rglob(x.pattern):
            pf=pq.ParquetFile(fp)
            for rg in range(pf.metadata.num_row_groups):
                df=pf.read_row_group(rg).to_pandas();
                if 'signal' not in df: continue
                tg=[c for c in df.columns if c.startswith('target_') and (not horizons or c.replace('target_','') in horizons)]
                for lvl,sub in (df.groupby('level') if 'level' in df else [('ALL',df)]):
                    for c in tg:
                        v=sub[['signal',c]].dropna();
                        if len(v)<20 or v.signal.nunique()<3: continue
                        ic,_=stats.spearmanr(v.signal,v[c]); qt=v.signal.quantile(.8); qb=v.signal.quantile(.2); top=v[v.signal>=qt][c]; bot=v[v.signal<=qb][c]
                        rows.append({'alpha_name':sub.alpha_name.iloc[0] if 'alpha_name' in sub else fp.parent.name,'date':sub.date.iloc[0] if 'date' in sub else None,'layer_id':sub.layer_id.iloc[0] if 'layer_id' in sub else None,'scale':sub.scale.iloc[0] if 'scale' in sub else None,'level':lvl,'horizon':c.replace('target_',''),'sample_count':len(v),'rank_ic':ic,'target_mean':v[c].mean(),'top_quintile_ret':top.mean(),'bottom_quintile_ret':bot.mean(),'long_short_spread':top.mean()-bot.mean(),'hit_rate':(v[c]>0).mean(),'file':str(fp)})
        od=Path(x.out_dir); od.mkdir(parents=True,exist_ok=True); part=pd.DataFrame(rows); part.to_csv(od/'partition_alpha_metrics.csv',index=False)
        agg=[]
        if not part.empty:
            for k,g in part.groupby(['alpha_name','horizon','level']):
                w=g.sample_count.clip(lower=1); rec=dict(zip(['alpha_name','horizon','level'],k)); rec['partition_count']=len(g); rec['sample_count']=int(g.sample_count.sum())
                for c in ['rank_ic','target_mean','top_quintile_ret','bottom_quintile_ret','long_short_spread','hit_rate']: rec[c]=float(np.average(g[c].fillna(0),weights=w))
                bd=g.groupby('date').long_short_spread.mean().dropna(); rec['positive_date_ratio']=float((bd>0).mean()) if len(bd) else np.nan; rec['spread_tstat_by_date']=float(bd.mean()/(bd.std(ddof=1)/math.sqrt(len(bd)))) if len(bd)>=2 and bd.std(ddof=1)!=0 else np.nan; agg.append(rec)
        sm=pd.DataFrame(agg); sm.to_csv(od/'alpha_metrics_summary.csv',index=False); manifest(od,{'status':'complete','partition_metric_rows':len(part),'summary_rows':len(sm)}); print(json.dumps({'partition_metric_rows':len(part),'summary_rows':len(sm)},indent=2)); return
    filename='theme_relation_edges.parquet' if x.cmd=='relation-spillover' else 'theme_memberships.parquet'
    parts=discover(x.p1_root,filename,csvset(x.dates),csvset(x.layers),csvset(x.scales));
    if x.max_partitions: parts=parts[:x.max_partitions]
    if not parts: raise FileNotFoundError('no partitions found')
    print(json.dumps({'partitions':len(parts),'workers':x.workers,'largest_mb':round(parts[0].base.stat().st_size/1024/1024,2)},indent=2),flush=True)
    levels=csvset(x.levels)
    params={'out_root':x.out_root,'horizons':horizons,'levels':levels,'skip_existing':x.skip_existing,'max_row_groups':x.max_row_groups}
    if x.cmd=='theme-returns': params['labels_root']=x.labels_root
    elif x.cmd=='relation-spillover': params.update({'theme_returns_root':x.theme_returns_root,'past_horizon':x.past_horizon,'relation_tiers':csvset(x.relation_tiers)})
    else: params.update({'labels_root':x.labels_root,'past_horizon':x.past_horizon})
    tasks=[(x.cmd, part, params) for part in parts]
    res=run(tasks,x.workers); Path(x.out_root).mkdir(parents=True,exist_ok=True); (Path(x.out_root)/'run_summary.json').write_text(json.dumps({'total':len(res),'complete':sum(r.get('status')=='complete' for r in res),'empty':sum(r.get('status')=='empty' for r in res),'results':res},indent=2),encoding='utf-8')
if __name__=='__main__': main()
