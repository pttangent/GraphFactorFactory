from __future__ import annotations
import argparse,json,os
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import pandas as pd
from graphfactorfactory.themes.pipeline import ThemeDiscoveryConfig,ThemeDiscoveryPipeline
from graphfactorfactory.themes.production_replay import infer_frame_minutes,prepare_edges
from graphfactorfactory.themes.production_worker import detect_snapshot

def run_day(pipeline,day,workers):
    date=day.name.split('=',1)[1]
    state=Path(pipeline.store.root)/'_run_state'/pipeline.config.run_id
    state.mkdir(parents=True,exist_ok=True)
    marker=state/f'date={date}.json'
    try:
        if json.loads(marker.read_text()).get('status')=='complete':return None
    except (OSError,ValueError):pass
    tmp=marker.with_suffix(f'.json.{os.getpid()}.tmp')
    tmp.write_text(json.dumps({'status':'running','date':date}));tmp.replace(marker)
    edges=pd.read_parquet(day/'edges.parquet')
    nodes=pd.read_parquet(day/'node_features.parquet')
    times=sorted(edges.decision_time.unique())
    cadence=infer_frame_minutes(times,5)
    prepared=prepare_edges(edges,times,pipeline.config)
    node_map={t:nodes[nodes.decision_time==t] for t in times}
    universe_count=len(pd.read_parquet(pipeline.graph_root/'dimensions'/'symbols.parquet'))
    tasks=[(t,prepared[t],pipeline.detector,pipeline.consensus,pipeline.layer_name,universe_count,pipeline.config.run_id) for t in times]
    detected={}
    with ProcessPoolExecutor(max_workers=max(1,workers)) as pool:
        for f in as_completed([pool.submit(detect_snapshot,x) for x in tasks]):
            t,p,c,themes=f.result();detected[t]=(p,c,themes)
    previous=[];records={};community_count=theme_count=0
    for t in times:
        p,c,themes=detected[t]
        themes,life=pipeline.lifecycle.assign(themes,previous,records,timestamp=t,frame_minutes=cadence)
        semantics=pipeline.semantic.label(themes)
        themes=pipeline.quality.score(themes,semantics,life,node_map[t])
        pipeline.store.accumulate_snapshot(snapshot_time=t,temporal_edges=prepared[t],layer_communities=p,subcommunities=c,themes=themes,lifecycle=life,semantics=semantics)
        community_count+=len(p)+len(c);theme_count+=len(themes)
        previous=themes;records={r.theme_instance_id:r for r in life if r.status=='active'}
    target=pipeline.store.write_day(date)
    tmp=marker.with_suffix(f'.json.{os.getpid()}.tmp')
    tmp.write_text(json.dumps({'status':'complete','date':date,'snapshot_count':len(times),'frame_minutes':cadence,'edge_rows':len(edges),'community_count':community_count,'theme_count':theme_count,'output':str(target) if target else None},indent=2));tmp.replace(marker)
    return target

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--graph-root',required=True);p.add_argument('--theme-root',required=True)
    p.add_argument('--date-start');p.add_argument('--date-end');p.add_argument('--workers',type=int,default=16)
    p.add_argument('--run-id',default='phase01_production_v1')
    a=p.parse_args()
    pipe=ThemeDiscoveryPipeline(a.graph_root,a.theme_root,ThemeDiscoveryConfig(run_id=a.run_id,frame_minutes=5))
    done=0
    for day in sorted((Path(a.graph_root)/'canonical').glob('date=*')):
        d=day.name.split('=',1)[1]
        if a.date_start and d<a.date_start or a.date_end and d>a.date_end:continue
        done+=int(run_day(pipe,day,a.workers) is not None)
    pipe.store.build_read_models();print(f'completed_days={done}')
if __name__=='__main__':main()
