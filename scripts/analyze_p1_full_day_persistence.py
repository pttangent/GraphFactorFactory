from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict

BASE=Path('/mnt/data/theme_0106_ex/date=2026-01-06')
OUT=Path('/mnt/data/full_day_p1_layer_persistence')
OUT.mkdir(exist_ok=True)

MIN_SIZE=8
MATCH_J=0.25
TOP_PATHS_PER_LAYER=5

print('loading layer communities metadata...')
_meta = pd.read_parquet(BASE/'layer_communities.parquet', columns=['snapshot_time'])
_all_times = sorted(_meta['snapshot_time'].unique())
_sel_times = _all_times[::5]
print('sampling full day every 5th snapshot:', len(_sel_times), 'of', len(_all_times), _sel_times[0], _sel_times[-1])
lc = pd.read_parquet(
    BASE/'layer_communities.parquet',
    columns=['snapshot_time','layer_id','layer_name','community_id','members','modularity','is_market_mode'],
    filters=[('snapshot_time','in', _sel_times)],
)
print('rows', len(lc))
lc['size'] = lc['members'].map(len).astype(int)
lc = lc[(lc['size']>=MIN_SIZE) & (~lc['is_market_mode'].fillna(False))].copy()
print('rows after min size', len(lc))

sym = pd.read_parquet('/mnt/data/symbols.parquet')
meta = pd.read_parquet('/mnt/data/symbol_metadata.parquet')
sid_to_sym = dict(zip(sym.symbol_id.astype(int), sym.symbol.astype(str)))
meta_by_sym = meta.set_index('symbol')

summary_rows=[]
repr_rows=[]

def interpret_members(member_ids, max_syms=15):
    syms=[sid_to_sym.get(int(x), str(int(x))) for x in member_ids]
    m=meta_by_sym.reindex(syms)
    sectors=m['sector_code'].fillna('UNKNOWN').astype(str)
    inds=m['industry_code'].fillna('UNKNOWN').astype(str)
    qtypes=m['quote_type'].fillna('UNKNOWN').astype(str)
    sec_counts=sectors.value_counts()
    ind_counts=inds.value_counts()
    q_counts=qtypes.value_counts()
    return {
        'symbols': ', '.join(syms[:max_syms]),
        'top_sector': sec_counts.index[0] if len(sec_counts) else 'UNKNOWN',
        'top_sector_share': float(sec_counts.iloc[0]/max(1,len(syms))) if len(sec_counts) else 0,
        'top_industry': ind_counts.index[0] if len(ind_counts) else 'UNKNOWN',
        'top_industry_share': float(ind_counts.iloc[0]/max(1,len(syms))) if len(ind_counts) else 0,
        'quote_type_mix': '; '.join([f'{k}:{v}' for k,v in q_counts.head(3).items()]),
    }

def explain_layer(layer_name):
    lname=str(layer_name)
    if 'return_corr' in lname:
        return '價格共振層：同時漲跌/收益相關的交易籃子，適合板塊同步、擴散與相對強弱研究。'
    if 'volume_expansion' in lname:
        return '放量層：成交量/美元成交額同步擴張，偏事件熱度與資金注意力。'
    if 'trade_intensity' in lname:
        return '交易強度層：成交筆數與平均成交規模同步活躍，偏短線熱門與微結構活躍簇。'
    if 'signed_flow' in lname:
        return '方向性資金流層：買賣壓代理量同步，適合觀察資金推動與同向 order-flow。'
    if 'large_trade_flow' in lname:
        return '大單流層：大額成交/大單占比同步，偏機構交易或大資金行為。'
    if 'odd_lot' in lname:
        return '零股/小單層：odd-lot 活躍，偏散戶/碎單交易結構與高頻碎片化注意力。'
    if 'block_activity' in lname:
        return '大宗/區塊交易層：block 活動同步，常對應大市值或機構調倉。'
    if 'off_exchange' in lname:
        return '場外/暗池層：off-exchange 交易占比同步，偏暗池路由和大市值/ETF 結構。'
    if 'venue_fragmentation' in lname:
        return '交易場所碎片化層：venue fragmentation 同步，反映流動性分散和執行結構。'
    if 'price_impact' in lname:
        return '價格衝擊層：交易對價格衝擊/流動性衝擊相似，偏流動性壓力。'
    if 'absorption' in lname:
        return '吸收層：flow 被價格吸收的相似結構，可能捕捉隱性承接/壓力吸收。'
    if 'flow_return_alignment' in lname:
        return 'flow-return 對齊層：資金流與收益反應一致，偏確認型資金推動信號。'
    if 'report_latency' in lname:
        return '報告延遲層：資料延遲/修正品質相似，更多是資料微結構和數據品質因子。'
    return '一般圖層：該圖層定義下的相似交易結構。'

layer_names = sorted(lc['layer_name'].unique())
for li, lname in enumerate(layer_names,1):
    g = lc[lc['layer_name']==lname].sort_values(['snapshot_time','community_id']).copy()
    times = list(g['snapshot_time'].drop_duplicates())
    next_path=0
    active={}
    path_stats={}
    for ts in times:
        snap = g[g['snapshot_time']==ts]
        curr_sets=[]
        curr_info=[]
        for r in snap.itertuples(index=False):
            s=set(map(int, r.members.tolist() if hasattr(r.members,'tolist') else list(r.members)))
            curr_sets.append(s)
            curr_info.append((int(r.community_id), float(r.modularity), int(r.size)))
        inv=defaultdict(list)
        for pid, s in active.items():
            for m in s:
                inv[m].append(pid)
        matched_prev=set(); new_active={}
        for idx,s in enumerate(curr_sets):
            counts=Counter()
            for m in s:
                for pid in inv.get(m,[]):
                    if pid not in matched_prev:
                        counts[pid]+=1
            best_pid=None; best_j=0.0
            for pid, inter in counts.items():
                union=len(s)+len(active[pid])-inter
                j=inter/union if union else 0.0
                if j>best_j:
                    best_j=j; best_pid=pid
            if best_pid is not None and best_j>=MATCH_J:
                pid=best_pid; matched_prev.add(pid)
            else:
                pid=f'{lname}|p{next_path:06d}'; next_path+=1
                path_stats[pid]={'layer_name':lname,'layer_id':int(g['layer_id'].iloc[0]),'start':ts,'end':ts,'frames':0,'sizes':[],'modularities':[],'jaccards':[],'members_last':None,'members_best':s,'best_size':0}
            st=path_stats[pid]
            st['end']=ts; st['frames']+=1; st['sizes'].append(len(s)); st['modularities'].append(curr_info[idx][1])
            if best_pid is not None: st['jaccards'].append(best_j)
            if len(s)>st.get('best_size',0): st['members_best']=s; st['best_size']=len(s)
            st['members_last']=s
            new_active[pid]=s
        active=new_active

    paths=[]
    for pid,st in path_stats.items():
        frames=st['frames']; sizes=st['sizes']
        paths.append({
            'path_id':pid,'layer_id':st['layer_id'],'layer_name':lname,'start':st['start'],'end':st['end'],
            'frames':frames,'duration_checkpoints':frames,'avg_size':float(np.mean(sizes)),'median_size':float(np.median(sizes)),'max_size':int(np.max(sizes)),
            'avg_modularity':float(np.mean(st['modularities'])) if st['modularities'] else None,
            'avg_match_jaccard':float(np.mean(st['jaccards'])) if st['jaccards'] else None,
            'last_members':st['members_last'],'best_members':st['members_best']
        })
    pdf=pd.DataFrame(paths)
    if pdf.empty: continue
    stable=pdf[pdf['frames']>=5]
    first_snapshot_communities=int(g[g['snapshot_time']==g['snapshot_time'].min()].shape[0])
    communities=int(len(g))
    path_count=int(len(pdf))
    continuation_rate=(communities-path_count)/max(1, communities-first_snapshot_communities)
    summary_rows.append({
        'layer_name':lname,'layer_id':int(g['layer_id'].iloc[0]),'snapshots':len(times),'communities':communities,
        'paths':path_count,'continuation_rate':continuation_rate,
        'stable_paths_ge5':len(stable),'stable_ratio':len(stable)/max(1,path_count),
        'p50_duration':float(pdf['frames'].median()),'p90_duration':float(pdf['frames'].quantile(.9)),'max_duration':int(pdf['frames'].max()),
        'p50_size':float(g['size'].median()),'p90_size':float(g['size'].quantile(.9)),'max_size':int(g['size'].max()),
        'avg_modularity':float(g['modularity'].mean())
    })
    top=pdf.sort_values(['frames','avg_match_jaccard','avg_size'], ascending=[False,False,False]).head(TOP_PATHS_PER_LAYER)
    for rank,r in enumerate(top.itertuples(index=False),1):
        info=interpret_members(r.best_members)
        repr_rows.append({
            'layer_name':lname,'layer_id':r.layer_id,'rank':rank,'path_id':r.path_id,
            'start':r.start,'end':r.end,'frames':r.frames,
            'avg_size':r.avg_size,'max_size':r.max_size,'avg_match_jaccard':r.avg_match_jaccard,
            **info,
            'financial_meaning':explain_layer(lname)
        })
    print(li, '/', len(layer_names), lname, 'paths', len(pdf), 'stable', len(stable), 'maxdur', int(pdf['frames'].max()))

summary=pd.DataFrame(summary_rows).sort_values('continuation_rate', ascending=False)
repr_df=pd.DataFrame(repr_rows).sort_values(['layer_id','layer_name','rank'])
summary.to_csv(OUT/'full_day_layer_stability_ranked.csv', index=False)
repr_df.to_csv(OUT/'full_day_representative_stable_themes.csv', index=False)
print('wrote', OUT)
