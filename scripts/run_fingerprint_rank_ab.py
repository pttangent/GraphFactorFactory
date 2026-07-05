from __future__ import annotations
import math, random
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.compute as pc
from scipy.stats import mannwhitneyu

from run_episode_fingerprint_ab import SRC, OUT, DATES, build_episodes, select_windows, permute_members, wj, cont, struct_sim

MIN_SIZE=20
WINDOW=30


def top_eps(episodes, n=30):
    episodes=[episode for episode in episodes if episode['instances']>=3]
    by_layer=defaultdict(list)
    for episode in episodes:
        by_layer[episode['layer']].append(episode)
    result=[]
    for values in by_layer.values():
        values.sort(key=lambda episode:(-(episode['instances']*math.log1p(episode['mean_size'])),-episode['mean_size'],episode['eid']))
        result.extend(values[:n])
    return result


def idf_from(episodes):
    total=len(episodes)
    frequency=Counter()
    for episode in episodes:
        frequency.update(set(episode['member_freq'])|set(episode['core']))
    return {node:math.log((total+1)/(count+1))+1 for node,count in frequency.items()}


def weighted_jaccard_idf(left,right,idf):
    keys=set(left)|set(right)
    if not keys:
        return 0.0
    numerator=sum(idf.get(key,1)*min(left.get(key,0),right.get(key,0)) for key in keys)
    denominator=sum(idf.get(key,1)*max(left.get(key,0),right.get(key,0)) for key in keys)
    return numerator/denominator if denominator else 0.0


def weighted_containment_idf(left,right,idf):
    if not left or not right:
        return 0.0
    shared=left&right
    return sum(idf.get(key,1) for key in shared)/min(sum(idf.get(key,1) for key in left),sum(idf.get(key,1) for key in right))


def score(left,right,method,idf):
    raw=wj(left['member_freq'],right['member_freq'])
    core=cont(left['core'],right['core'])
    rare=weighted_jaccard_idf(left['member_freq'],right['member_freq'],idf)
    rare_core=weighted_containment_idf(left['core'],right['core'],idf)
    structure=struct_sim(left,right)
    if method=='raw_member':
        return .65*raw+.35*core
    if method=='idf_member':
        return .65*rare+.35*rare_core
    if method=='structure':
        return structure
    if method=='idf_hybrid':
        size=math.exp(-abs(math.log(max(1,left['mean_size']))-math.log(max(1,right['mean_size']))))
        return .60*(.65*rare+.35*rare_core)+.25*structure+.15*size
    raise KeyError(method)


def best_scores(previous,current,method,idf):
    by_layer=defaultdict(list)
    for episode in current:
        by_layer[episode['layer']].append(episode)
    return np.asarray([max((score(episode,candidate,method,idf) for candidate in by_layer.get(episode['layer'],[])),default=0.0) for episode in previous])


def auc_effect(actual,null):
    if not len(actual) or not len(null):
        return np.nan
    statistic=mannwhitneyu(actual,null,alternative='two-sided').statistic
    return float(statistic/(len(actual)*len(null)))


def main():
    table=pq.read_table(SRC,columns=['date','day_state_index','decision_time','layer_id','community_id','size','members','core_members'])
    table=table.filter(pc.greater_equal(table['size'],MIN_SIZE))
    frame=table.to_pandas()
    windows={date:select_windows(frame,date) for date in DATES}
    rows=[]
    distributions=[]
    actual_pairs=[('2026-01-20','2026-01-21'),('2026-01-21','2026-01-22')]
    day_pairs=[('2026-01-20','2026-01-22'),('2026-01-22','2026-01-21'),('2026-01-21','2026-01-20')]
    for gap in [0,2,5]:
        episodes={}
        for date in DATES:
            for side in ['open','mid','close']:
                episodes[(date,side)]=top_eps(build_episodes(windows[date][side],date,side,gap))
        idf=idf_from([episode for values in episodes.values() for episode in values])
        for method in ['raw_member','idf_member','structure','idf_hybrid']:
            groups=defaultdict(list)
            for left,right in actual_pairs:
                groups['actual'].extend(best_scores(episodes[(left,'close')],episodes[(right,'open')],method,idf).tolist())
            for left,right in day_pairs:
                groups['day_order'].extend(best_scores(episodes[(left,'close')],episodes[(right,'open')],method,idf).tolist())
            for left,right in actual_pairs:
                groups['time_window'].extend(best_scores(episodes[(left,'mid')],episodes[(right,'open')],method,idf).tolist())
            for index,(left,right) in enumerate(actual_pairs):
                shuffled=permute_members(episodes[(right,'open')],7000+gap*10+index)
                groups['member_perm'].extend(best_scores(episodes[(left,'close')],shuffled,method,idf).tolist())
            actual=np.asarray(groups['actual'])
            day_null=np.asarray(groups['day_order'])
            threshold=float(np.quantile(day_null,.95))
            for kind,values in groups.items():
                array=np.asarray(values)
                distributions.extend({'gap':gap,'method':method,'kind':kind,'best_score':value} for value in array)
                rows.append({'gap':gap,'method':method,'kind':kind,'n':len(array),'mean':float(array.mean()),'median':float(np.median(array)),'p90':float(np.quantile(array,.9)),'p95':float(np.quantile(array,.95)),'daynull_p95_threshold':threshold,'detection_rate':float((array>=threshold).mean()),'auc_actual_vs_kind':auc_effect(actual,array) if kind!='actual' else .5})
    pd.DataFrame(rows).to_csv(OUT/'rank_ab_summary.csv',index=False)
    pd.DataFrame(distributions).to_parquet(OUT/'rank_ab_distributions.parquet',index=False)


if __name__=='__main__':
    main()
