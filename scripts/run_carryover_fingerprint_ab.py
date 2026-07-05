from __future__ import annotations
import json, math
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.compute as pc
from scipy.stats import mannwhitneyu

MIN_SIZE=20; H=30
ENTRY=0.18; STAY=0.20; LOW_STAY=0.10
FP_CONFIRM=0.10; FP_STAY=0.16; FP_REVIVE=0.20; MAX_WEAK_GAP=1; MAX_DORMANT=3
BASE=Path('/mnt/data/outputs/temporal_theme_research/intraday_1m')
ROOT=Path('/mnt/data')
OUT=Path('/mnt/data/outputs/temporal_theme_research/carryover_fingerprint_ab_v3'); OUT.mkdir(parents=True,exist_ok=True)
DATES=['2026-01-20','2026-01-21','2026-01-22']; COUNTS={'2026-01-20':180,'2026-01-21':185,'2026-01-22':178}
ACTUAL=[('2026-01-20','2026-01-21'),('2026-01-21','2026-01-22')]
NULL=[('2026-01-20','2026-01-22'),('2026-01-22','2026-01-21'),('2026-01-21','2026-01-20')]


def load_state(d,s):
 p=BASE/f'shards_{d}'/f'state_{s:04d}.parquet'
 z=pq.read_table(p).to_pandas(); z=z[z['size']>=MIN_SIZE]
 out=[]
 for _,r in z.iterrows():
  out.append(dict(id=f'{d}_{s}_{int(r.layer_id)}_{int(r.community_id)}',date=d,state=s,time=r.decision_time,
                  layer=int(r.layer_id),size=int(r['size']),members=set(map(int,r.members)),core=set(map(int,r.core_members))))
 return out

def containment(a,b): return len(a&b)/min(len(a),len(b)) if a and b else 0.
def jaccard(a,b): return len(a&b)/len(a|b) if a and b else 0.
def size_sim(a,b): return math.exp(-abs(math.log(max(1,a))-math.log(max(1,b)))/0.7)
def fp_sim(proto,c):
 core=containment(proto['core'],c['core'])
 mem=containment(proto['members'],c['members'])
 jac=jaccard(proto['members'],c['members'])
 return .40*core+.30*mem+.15*jac+.15*size_sim(proto['mean_size'],c['size'])

def update_proto(proto,c):
 p=dict(proto)
 p['member_counts']=Counter(proto['member_counts']); p['core_counts']=Counter(proto['core_counts'])
 p['n']+=1; p['member_counts'].update(c['members']); p['core_counts'].update(c['core'])
 p['members']={x for x,k in p['member_counts'].items() if k/p['n']>=0.35}
 p['core']={x for x,k in p['core_counts'].items() if k/p['n']>=0.35}
 p['mean_size']=((p['mean_size']*(p['n']-1))+c['size'])/p['n']
 return p

def init_proto(prev,openrow):
 mc=Counter(prev['members']); mc.update(openrow['members']); cc=Counter(prev['core']); cc.update(openrow['core'])
 return {'n':2,'member_counts':mc,'core_counts':cc,'members':set(prev['members'])|set(openrow['members']),
         'core':set(prev['core'])|set(openrow['core']),'mean_size':(prev['size']+openrow['size'])/2}

def bridge_candidates(prev,cur):
 inv=defaultdict(list)
 for j,c in enumerate(cur):
  for n in c['members']: inv[n].append(j)
 cand=[]
 for i,p in enumerate(prev):
  cnt=defaultdict(int)
  for n in p['members']:
   for j in inv.get(n,()): cnt[j]+=1
  for j,k in cnt.items():
   s=k/min(len(p['members']),len(cur[j]['members']))
   if s>=ENTRY:
    proto={'members':p['members'],'core':p['core'],'mean_size':p['size']}
    cand.append((s,fp_sim(proto,cur[j]),i,j))
 cand.sort(reverse=True); up=set(); uc=set(); out=[]
 for s,fp,i,j in cand:
  if i in up or j in uc: continue
  up.add(i); uc.add(j); out.append((i,j,s,fp))
 return out

def greedy_step(active,cur,variant):
 inv=defaultdict(list)
 for j,c in enumerate(cur):
  for n in c['members']: inv[n].append(j)
 cand=[]
 for i,a in enumerate(active):
  cnt=defaultdict(int)
  for n in a['last']['members']:
   for j in inv.get(n,()): cnt[j]+=1
  for j,k in cnt.items():
   cont=k/min(len(a['last']['members']),len(cur[j]['members']))
   fp=fp_sim(a['proto'],cur[j])
   ok=False
   if variant in ('A','B'): ok=cont>=STAY
   elif variant in ('C','D'): ok=(cont>=LOW_STAY and fp>=FP_STAY) or cont>=STAY
   if ok: cand.append((.65*cont+.35*fp,cont,fp,i,j))
 cand.sort(reverse=True); ua=set(); uc=set(); matches={}
 for score,cont,fp,i,j in cand:
  if i in ua or j in uc: continue
  ua.add(i); uc.add(j); matches[i]=(j,cont,fp)
 return matches

def follow(prev_roots,open_roots,future,variant):
 active=[]; done=[]; dormant=[]
 for p,o in zip(prev_roots,open_roots):
  active.append({'prev':p,'root':o,'last':o,'proto':init_proto(p,o),'age':1,'active_hits':1,'weak_gap':0,'dormant_gap':0,'revivals':0,'max_size':o['size']})
 for cur in future:
  matches=greedy_step(active,cur,variant)
  nxt=[]
  for i,a in enumerate(active):
   if i in matches:
    j,cont,fp=matches[i]; b=dict(a); b['last']=cur[j]; b['proto']=update_proto(a['proto'],cur[j]); b['age']+=1; b['active_hits']+=1; b['weak_gap']=0; b['max_size']=max(b['max_size'],cur[j]['size']); nxt.append(b)
   elif variant in ('C','D') and a['weak_gap']<MAX_WEAK_GAP:
    b=dict(a); b['age']+=1; b['weak_gap']+=1; nxt.append(b)
   elif variant=='D':
    b=dict(a); b['dormant_gap']=1; dormant.append(b)
   else: done.append(a)
  active=nxt
  if variant=='D' and dormant:
   candidates=[]
   for i,a in enumerate(dormant):
    for j,c in enumerate(cur):
     fp=fp_sim(a['proto'],c)
     if fp>=FP_REVIVE: candidates.append((fp,i,j))
   candidates.sort(reverse=True); ud=set(); uc=set(); revived=[]
   for fp,i,j in candidates:
    if i in ud or j in uc: continue
    ud.add(i); uc.add(j); a=dormant[i]; b=dict(a); b['last']=cur[j]; b['proto']=update_proto(a['proto'],cur[j]); b['age']+=a['dormant_gap']+1; b['active_hits']+=1; b['weak_gap']=0; b['dormant_gap']=0; b['revivals']+=1; b['max_size']=max(b['max_size'],cur[j]['size']); revived.append(b)
   keep=[]
   for i,a in enumerate(dormant):
    if i in ud: continue
    b=dict(a); b['dormant_gap']+=1
    if b['dormant_gap']>MAX_DORMANT: done.append(b)
    else: keep.append(b)
   dormant=keep; active.extend(revived)
 return done+active+dormant

def matched_controls(open_rows, used, roots):
 pool=[(j,r) for j,r in enumerate(open_rows) if j not in used]; controls=[]; taken=set()
 for root in roots:
  opts=[(abs(math.log(max(1,r['size']))-math.log(max(1,root['size']))),j,r) for j,r in pool if j not in taken and r['layer']==root['layer']]
  if not opts: opts=[(abs(math.log(max(1,r['size']))-math.log(max(1,root['size']))),j,r) for j,r in pool if j not in taken]
  if opts:
   _,j,r=min(opts); taken.add(j); controls.append(r)
 return controls

def label_slice(d,t):
 p=ROOT/f'cano_{d}'/f'date={d}'/'labels.parquet'; pf=pq.ParquetFile(p)
 cols=['decision_time','symbol_id','label_5m','label_15m','label_30m','label_60m']; parts=[]; target=pd.Timestamp(t).to_pydatetime()
 for i in range(pf.num_row_groups):
  st=pf.metadata.row_group(i).column(0).statistics
  if st and st.min<=target<=st.max:
   tab=pf.read_row_group(i,columns=cols); tab=tab.filter(pc.equal(tab['decision_time'],target))
   if tab.num_rows: parts.append(tab)
 import pyarrow as pa
 return pa.concat_tables(parts).to_pandas() if parts else pd.DataFrame(columns=cols)

def outcome(paths,kind,pair,variant,labels):
 idx=labels.set_index('symbol_id'); rows=[]
 for a in paths:
  vals=idx.reindex(list(a['root']['members']))
  r={'pair':pair,'kind':kind,'variant':variant,'layer':a['root']['layer'],'open_size':a['root']['size'],'age_states':a['age'],'active_hits':a['active_hits'],'revivals':a.get('revivals',0),
     'persistent_3':a['active_hits']>=3,'persistent_5':a['active_hits']>=5,'persistent_10':a['active_hits']>=10,'persistent_20':a['active_hits']>=20}
  for h in [5,15,30,60]:
   x=vals[f'label_{h}m'].dropna().astype(float); r[f'mean_ret_{h}m']=x.mean() if len(x) else np.nan
  rows.append(r)
 return rows

def run_pair(prevd,curd,prefix):
 close=load_state(prevd,COUNTS[prevd]-1); opens=[load_state(curd,s) for s in range(H)]
 bc=bridge_candidates(close,opens[0]); labels=label_slice(curd,opens[0][0]['time']); pair=f'{prevd}->{curd}'
 allrows=[]; details=[]
 for variant in ['A','B','C','D']:
  chosen=[]
  for i,j,s,fp in bc:
   if variant=='A' and s>=.20: chosen.append((i,j,s,fp))
   elif variant in ('B','C','D') and fp>=FP_CONFIRM: chosen.append((i,j,s,fp))
  prevroots=[close[i] for i,j,s,fp in chosen]; roots=[opens[0][j] for i,j,s,fp in chosen]; used={j for i,j,s,fp in chosen}
  controls=matched_controls(opens[0],used,roots)
  cprev=controls
  paths=follow(prevroots,roots,opens[1:],variant); cpaths=follow(cprev,controls,opens[1:],variant)
  allrows += outcome(paths,f'{prefix}_bridge',pair,variant,labels)
  allrows += outcome(cpaths,f'{prefix}_open_birth_control',pair,variant,labels)
  for i,j,s,fp in chosen: details.append({'pair':pair,'prefix':prefix,'variant':variant,'bridge_containment':s,'fingerprint_score':fp,'prev_layer':close[i]['layer'],'open_layer':opens[0][j]['layer'],'prev_size':close[i]['size'],'open_size':opens[0][j]['size']})
  print(pair,prefix,variant,'bridges',len(chosen),'controls',len(controls),flush=True)
 return allrows,details

def compare(df,a,b,label):
 rows=[]
 metrics=['age_states','active_hits','persistent_3','persistent_5','persistent_10','persistent_20','revivals']+[f'mean_ret_{h}m' for h in [5,15,30,60]]
 for variant in ['A','B','C','D']:
  sub=df[df.variant==variant]
  for m in metrics:
   x=sub[sub.kind==a][m].dropna().astype(float); y=sub[sub.kind==b][m].dropna().astype(float)
   if len(x) and len(y):
    u,p=mannwhitneyu(x,y,alternative='two-sided'); auc=u/(len(x)*len(y))
   else: auc=p=np.nan
   rows.append({'variant':variant,'comparison':label,'metric':m,'a_mean':x.mean() if len(x) else np.nan,'b_mean':y.mean() if len(y) else np.nan,'difference':x.mean()-y.mean() if len(x) and len(y) else np.nan,'auc':auc,'p_value':p,'n_a':len(x),'n_b':len(y)})
 return rows

def main():
 rows=[]; details=[]
 for a,b in ACTUAL:
  r,d=run_pair(a,b,'actual'); rows+=r; details+=d
 for a,b in NULL:
  r,d=run_pair(a,b,'null'); rows+=r; details+=d
 df=pd.DataFrame(rows); dd=pd.DataFrame(details)
 df.to_csv(OUT/'path_outcomes.csv',index=False); dd.to_csv(OUT/'bridge_details.csv',index=False)
 tests=[]
 tests+=compare(df,'actual_bridge','actual_open_birth_control','actual_bridge_vs_birth')
 tests+=compare(df,'actual_bridge','null_bridge','actual_vs_dayorder_bridge')
 tests+=compare(df,'null_bridge','null_open_birth_control','null_bridge_vs_birth')
 tt=pd.DataFrame(tests); tt.to_csv(OUT/'effect_tests.csv',index=False)
 summary=df.groupby(['variant','kind']).agg(n=('active_hits','size'),mean_active_hits=('active_hits','mean'),p3=('persistent_3','mean'),p5=('persistent_5','mean'),p10=('persistent_10','mean'),p20=('persistent_20','mean'),mean_revivals=('revivals','mean'),ret5=('mean_ret_5m','mean'),ret15=('mean_ret_15m','mean'),ret30=('mean_ret_30m','mean'),ret60=('mean_ret_60m','mean')).reset_index()
 summary.to_csv(OUT/'group_summary.csv',index=False)
 manifest={'entry':ENTRY,'stay':STAY,'low_stay':LOW_STAY,'fp_confirm':FP_CONFIRM,'fp_stay':FP_STAY,'fp_revive':FP_REVIVE,'max_weak_gap':MAX_WEAK_GAP,'max_dormant':MAX_DORMANT,'horizon':H}
 (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
 print(summary.to_string(index=False)); print('\nKEY TESTS'); print(tt[(tt.metric.isin(['persistent_3','persistent_5','persistent_10','mean_ret_5m','mean_ret_15m']))].to_string(index=False))
if __name__=='__main__': main()
