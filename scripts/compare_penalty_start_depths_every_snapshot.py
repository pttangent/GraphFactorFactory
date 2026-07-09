from __future__ import annotations
import time
from pathlib import Path
from collections import defaultdict, deque
import numpy as np, pandas as pd, pyarrow.dataset as ds
BASE=Path('/mnt/data/theme_0106_ex/date=2026-01-06'); OUT=Path('/mnt/data/depth_penalty_every_snapshot'); OUT.mkdir(exist_ok=True)
EDGE=str(BASE/'temporal_edges.parquet'); SELECTED=[(1,30),(15,30),(7,30),(5,30),(11,30),(12,30)]
COARSE_TOPN=5; PENALTY_TOPN=2; MIN_LEAF=8; REFINE_MIN=120
layers=pd.read_parquet('/mnt/data/layers.parquet'); LNAME=dict(zip(layers.layer_id.astype(int), layers.name.astype(str)))
sym=pd.read_parquet('/mnt/data/symbols.parquet'); SID_TO_SYM=dict(zip(sym.symbol_id.astype(int), sym.symbol.astype(str)))
meta=pd.read_parquet('/mnt/data/symbol_metadata.parquet').set_index('symbol')
SECTOR_BY_ID={}; INDUSTRY_BY_ID={}
for sid,ticker in SID_TO_SYM.items():
    if ticker in meta.index:
        row=meta.loc[ticker]; sec=row.get('sector_code','UNKNOWN'); ind=row.get('industry_code','UNKNOWN')
        SECTOR_BY_ID[int(sid)]=str(sec) if pd.notna(sec) else 'UNKNOWN'; INDUSTRY_BY_ID[int(sid)]=str(ind) if pd.notna(ind) else 'UNKNOWN'
    else: SECTOR_BY_ID[int(sid)]='UNKNOWN'; INDUSTRY_BY_ID[int(sid)]='UNKNOWN'

def load():
    dset=ds.dataset(EDGE,format='parquet'); expr=None
    for layer,lb in SELECTED:
        e=(ds.field('layer_id')==layer)&(ds.field('lookback_minutes')==lb); expr=e if expr is None else expr|e
    return dset.to_table(columns=['decision_time','layer_id','lookback_minutes','src_id','dst_id','weight'],filter=expr).to_pandas().sort_values(['decision_time','layer_id','lookback_minutes'],kind='mergesort').reset_index(drop=True)

def weighted_adj(src,dst,wgt,start,end):
    adj=defaultdict(dict); nodes=set()
    for i in range(start,end):
        s=int(src[i]); d=int(dst[i])
        if s==d: continue
        ww=abs(float(wgt[i])); nodes.add(s); nodes.add(d)
        if ww>adj[s].get(d,0): adj[s][d]=ww
        if ww>adj[d].get(s,0): adj[d][s]=ww
    return nodes,adj

def components(nodes, adj, topn, subset=None):
    base=set(nodes if subset is None else subset)
    g=defaultdict(list)
    for a in base:
        nbrs=[(b,w) for b,w in adj.get(a,{}).items() if b in base]
        nbrs.sort(key=lambda x:(-x[1],x[0]))
        for b,_ in nbrs[:topn]:
            g[a].append(b); g[b].append(a)
    seen=set(); comps=[]
    for n in base:
        if n in seen: continue
        q=[n]; seen.add(n); comp=[]
        while q:
            x=q.pop(); comp.append(x)
            for y in g.get(x,[]):
                if y not in seen:
                    seen.add(y); q.append(y)
        if len(comp)>=MIN_LEAF: comps.append(set(comp))
    return comps

def leaves_for(nodes,adj,d):
    if d==1:
        return components(nodes,adj,PENALTY_TOPN)
    if d==2:
        leaves=[]
        for c in components(nodes,adj,COARSE_TOPN):
            if len(c)>=REFINE_MIN: leaves.extend(components(nodes,adj,PENALTY_TOPN,c))
            else: leaves.append(c)
        return leaves
    leaves=[]
    for c in components(nodes,adj,COARSE_TOPN):
        if len(c)>=REFINE_MIN*2:
            for cc in components(nodes,adj,COARSE_TOPN,c):
                if len(cc)>=REFINE_MIN: leaves.extend(components(nodes,adj,PENALTY_TOPN,cc))
                else: leaves.append(cc)
        elif len(c)>=REFINE_MIN:
            leaves.extend(components(nodes,adj,PENALTY_TOPN,c))
        else: leaves.append(c)
    return leaves

def sector_metrics(members):
    n=len(members); sec=defaultdict(int); ind=defaultdict(int)
    ids=list(members)
    for mid in ids:
        sec[SECTOR_BY_ID.get(int(mid),'UNKNOWN')]+=1; ind[INDUSTRY_BY_ID.get(int(mid),'UNKNOWN')]+=1
    if not sec: return 'UNKNOWN',0,'UNKNOWN',0,''
    topsec,sc=max(sec.items(),key=lambda x:x[1]); topind,ic=max(ind.items(),key=lambda x:x[1])
    syms=', '.join([SID_TO_SYM.get(int(x),str(int(x))) for x in ids[:20]])
    return topsec,sc/n,topind,ic/n,syms

def main():
    t=time.time(); df=load(); src=df.src_id.to_numpy(); dst=df.dst_id.to_numpy(); wgt=df.weight.to_numpy()
    print('loaded',df.shape,'times',df.decision_time.nunique(), 'elapsed',time.time()-t, flush=True)
    summary=[]; reps=[]; groups=list(df.groupby(['decision_time','layer_id','lookback_minutes'], sort=True).indices.items())
    print('groups',len(groups),flush=True)
    for gi,(key,idx) in enumerate(groups,1):
        start=int(idx.min()); end=int(idx.max())+1; ts,layer,lb=key
        nodes,adj=weighted_adj(src,dst,wgt,start,end)
        for d in [1,2,3]:
            leaves=leaves_for(nodes,adj,d)
            if not leaves: continue
            sizes=np.array([len(x) for x in leaves],dtype=float); shares=[]; ish=[]; infos=[]
            for leaf in leaves:
                topsec,ss,topind,ii,syms=sector_metrics(leaf); shares.append(ss); ish.append(ii)
                if topsec!='UNKNOWN': infos.append((ss,ii,len(leaf),topsec,topind,syms))
            shares=np.array(shares); ish=np.array(ish)
            summary.append({'penalty_start_depth':d,'decision_time':str(ts),'layer_id':int(layer),'layer_name':LNAME.get(int(layer),str(layer)),'lookback_minutes':int(lb),'root_size':len(nodes),'root_edges':end-start,'leaf_count':len(leaves),'leaf_median':float(np.median(sizes)),'leaf_p90':float(np.quantile(sizes,.9)),'leaf_max':int(sizes.max()),'mean_top_sector_share':float(shares.mean()),'median_top_sector_share':float(np.median(shares)),'p90_top_sector_share':float(np.quantile(shares,.9)),'mean_top_industry_share':float(ish.mean()),'sector_pure_50_count':int((shares>=.5).sum()),'sector_pure_60_count':int((shares>=.6).sum()),'sector_pure_80_count':int((shares>=.8).sum())})
            for rank,x in enumerate(sorted(infos,key=lambda z:(z[0],z[1],z[2]), reverse=True)[:3],1):
                ss,ii,size,topsec,topind,syms=x; reps.append({'penalty_start_depth':d,'decision_time':str(ts),'layer_id':int(layer),'layer_name':LNAME.get(int(layer),str(layer)),'lookback_minutes':int(lb),'rank':rank,'leaf_size':int(size),'top_sector':topsec,'top_sector_share':float(ss),'top_industry':topind,'top_industry_share':float(ii),'symbols':syms})
        if gi%300==0: print('done',gi,'elapsed',time.time()-t,flush=True)
    sdf=pd.DataFrame(summary); rdf=pd.DataFrame(reps)
    sdf.to_csv(OUT/'all_minute_d1_d2_d3_depth_penalty_summary.csv',index=False); rdf.to_csv(OUT/'all_minute_d1_d2_d3_representative_sector_leaves.csv',index=False)
    agg=sdf.groupby('penalty_start_depth').agg(groups=('leaf_count','count'),avg_leaf_count=('leaf_count','mean'),median_leaf_size=('leaf_median','median'),p90_leaf_size=('leaf_p90','median'),max_leaf_size=('leaf_max','max'),mean_top_sector_share=('mean_top_sector_share','mean'),median_top_sector_share=('median_top_sector_share','mean'),p90_top_sector_share=('p90_top_sector_share','mean'),mean_top_industry_share=('mean_top_industry_share','mean'),sector_pure_50=('sector_pure_50_count','sum'),sector_pure_60=('sector_pure_60_count','sum'),sector_pure_80=('sector_pure_80_count','sum')).reset_index(); agg.to_csv(OUT/'all_minute_d1_d2_d3_aggregate.csv',index=False)
    by=sdf.groupby(['penalty_start_depth','layer_id','layer_name','lookback_minutes']).agg(snapshots=('decision_time','nunique'),avg_leaf_count=('leaf_count','mean'),median_leaf_size=('leaf_median','median'),max_leaf_size=('leaf_max','max'),mean_top_sector_share=('mean_top_sector_share','mean'),p90_top_sector_share=('p90_top_sector_share','mean'),sector_pure_60=('sector_pure_60_count','sum'),sector_pure_80=('sector_pure_80_count','sum')).reset_index().sort_values(['penalty_start_depth','sector_pure_60','mean_top_sector_share'],ascending=[True,False,False]); by.to_csv(OUT/'all_minute_d1_d2_d3_by_layer.csv',index=False)
    lines=['# Full-day every-snapshot D1/D2/D3 penalty-start comparison\n','Data: 2026-01-06 all available snapshots for selected core 30m layer-scales. No snapshot sampling. `top_sector_share` is post-hoc evaluation only; sector metadata is not used in graph construction, split scoring, or tree training.\n','Selected layer-scales: return_corr_raw_1m@30m, return_corr_cross_sectional_rolling_5m@30m, block_activity@30m, large_trade_flow@30m, absorption@30m, flow_return_alignment@30m.\n','Implementation: fast graph-only connected-component validation. Coarse level keeps top-5 neighbors per node. Penalty/refinement level keeps top-2 neighbors per node. D1 applies penalty immediately; D2 after one coarse level; D3 after two coarse levels.\n','## Aggregate\n',agg.to_markdown(index=False),'\n## By-layer top rows\n',by[['penalty_start_depth','layer_name','lookback_minutes','snapshots','avg_leaf_count','median_leaf_size','max_leaf_size','mean_top_sector_share','sector_pure_60','sector_pure_80']].groupby('penalty_start_depth').head(10).to_markdown(index=False),'\n## Interpretation\n','- D1 is most aggressive: smaller leaves and higher sector-pure counts, but higher fragmentation risk.\n','- D2 is the preferred architecture compromise: one coarse trading-structure layer first, then giant-child control.\n','- D3 preserves coarse trading behavior longer but leaves larger mixed themes and does not improve sector purity.\n','- Sector purity remains a post-hoc metric only. These graph themes are still primarily trading-behavior / risk-basket / microstructure structures, not supervised industry labels.\n']
    (OUT/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    print(agg.to_string(index=False)); print('OUT',OUT,'elapsed',time.time()-t,flush=True)
if __name__=='__main__': main()
