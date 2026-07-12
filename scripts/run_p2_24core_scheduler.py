#!/usr/bin/env python3
"""Resource-saturating local scheduler for the full P2 alpha lab.

Full alpha lab stages:
1. P2A direct P0 graph alpha: node features, edge spillover, graph state, P0 eval.
2. P2B P1 theme alpha: theme_returns.
3. P2C P1 relation alpha: relation_spillover, daily_relation_features, intraday eval.

This scheduler is intentionally local-machine oriented.  For a 24-core / 128GB
workstation, use --profile max and tune --inner-workers based on memory pressure.
"""
from __future__ import annotations
import argparse, json, math, os, subprocess, sys, time
from dataclasses import asdict, dataclass
from pathlib import Path

THREAD_CAPS={"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","NUMEXPR_NUM_THREADS":"1","ARROW_NUM_THREADS":"1","POLARS_MAX_THREADS":"1","PYTHONUNBUFFERED":"1"}
@dataclass(frozen=True)
class StagePlan: stage:str; workers:int; inner_workers:int; estimated_slots:int; reason:str

def csv_arg(name,value): return [name,value] if value else []
def build_plan(cores:int,target_cpu:float,profile:str,inner_workers:int)->dict[str,StagePlan]:
    target=max(1,int(math.ceil(cores*target_cpu))); iw=max(1,inner_workers)
    if profile=='safe':
        p0=min(target,12); nested=max(1,min(4,target//iw)); daily=min(target,16)
    elif profile=='balanced':
        p0=min(target,18); nested=max(1,min(8,math.ceil(target/iw))); daily=min(target,20)
    elif profile=='aggressive':
        p0=min(cores,22); nested=max(1,min(12,math.ceil(cores/iw))); daily=min(cores,24)
    else: # max: saturate local workstation, caller may lower workers if RAM says no.
        p0=cores; nested=max(1,math.ceil(cores/iw)); daily=cores
    return {
        'p0-node-features':StagePlan('p0-node-features',p0,1,p0,'P0 rowgroup-streaming stock node features; mostly CPU + parquet IO'),
        'p0-edge-spillover':StagePlan('p0-edge-spillover',p0,1,p0,'P0 direct stock-to-stock spillover; rowgroup-streaming and label time slicing'),
        'p0-graph-state':StagePlan('p0-graph-state',p0,1,p0,'P0 market graph state aggregates; lightest P0 stage'),
        'build-theme-returns':StagePlan('build-theme-returns',nested,iw,nested*iw,'P1 membership + labels; decision_time aligned join with streaming output'),
        'relation-spillover':StagePlan('relation-spillover',nested,iw,nested*iw,'P1 theme relation spillover; decision_time aligned join with streaming output'),
        'daily-relation-features':StagePlan('daily-relation-features',daily,1,daily,'single-level partition aggregation'),
    }
def run_cmd(cmd,env,dry_run):
    print('\n$ '+' '.join(map(str,cmd)),flush=True)
    if dry_run: return
    p=subprocess.run(cmd,env=env)
    if p.returncode!=0: raise SystemExit(p.returncode)
def common(args):
    out=[]; out+=csv_arg('--dates',args.dates); out+=csv_arg('--layers',args.layers); out+=csv_arg('--scales',args.scales); out+=csv_arg('--levels',args.levels); out+=csv_arg('--horizons',args.horizons)
    if args.max_row_groups is not None: out+=['--max-row-groups',str(args.max_row_groups)]
    if args.skip_existing: out.append('--skip-existing')
    return out
def common_p0(args):
    out=[]; out+=csv_arg('--dates',args.dates); out+=csv_arg('--layers',args.layers); out+=csv_arg('--scales',args.scales); out+=csv_arg('--horizons',args.horizons)
    if args.max_row_groups is not None: out+=['--max-row-groups',str(args.max_row_groups)]
    if args.skip_existing: out.append('--skip-existing')
    return out

def main():
    ap=argparse.ArgumentParser(description='Full P2 alpha lab scheduler: P0 graph alpha + P1 theme/relation alpha')
    ap.add_argument('--p0-root'); ap.add_argument('--p1-root'); ap.add_argument('--labels-root',required=True); ap.add_argument('--p2-root',required=True)
    ap.add_argument('--p2-script',default='scripts/p2_alpha_daily_features.py'); ap.add_argument('--p0-script',default='scripts/p2_p0_graph_alpha.py')
    ap.add_argument('--dates'); ap.add_argument('--layers',default='3,6,8,9,11'); ap.add_argument('--scales',default='15m,30m'); ap.add_argument('--levels',default='B50,B35'); ap.add_argument('--horizons',default='5m,15m,30m,60m,120m')
    ap.add_argument('--past-horizon',default='15m'); ap.add_argument('--underreaction-past-horizon',default='15m'); ap.add_argument('--tiers')
    ap.add_argument('--cores',type=int,default=24); ap.add_argument('--target-cpu',type=float,default=1.0); ap.add_argument('--inner-workers',type=int,default=1)
    ap.add_argument('--profile',choices=['safe','balanced','aggressive','max'],default='max')
    ap.add_argument('--stage',choices=['all','p0','p0-node','p0-edge','p0-graph','p0-eval','theme','relation','daily','eval'],default='all')
    ap.add_argument('--max-row-groups',type=int); ap.add_argument('--skip-existing',action='store_true'); ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args()
    if not (0<args.target_cpu<=1.5): raise SystemExit('--target-cpu must be in (0, 1.5]')
    if args.inner_workers<1: raise SystemExit('--inner-workers must be >=1')
    p2_root=Path(args.p2_root); p2_root.mkdir(parents=True,exist_ok=True); plan=build_plan(args.cores,args.target_cpu,args.profile,args.inner_workers)
    payload={'profile':args.profile,'cores':args.cores,'target_cpu':args.target_cpu,'inner_workers':args.inner_workers,'target_slots':int(math.ceil(args.cores*args.target_cpu)),'stage_plan':{k:asdict(v) for k,v in plan.items()},'filters':{'dates':args.dates,'layers':args.layers,'scales':args.scales,'levels':args.levels,'horizons':args.horizons,'past_horizon':args.past_horizon,'underreaction_past_horizon':args.underreaction_past_horizon},'created_at_epoch':time.time()}
    (p2_root/'p2_24core_schedule_plan.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding='utf-8'); print(json.dumps(payload,indent=2,ensure_ascii=False),flush=True)
    env=os.environ.copy(); env.update(THREAD_CAPS); py=sys.executable; p2=args.p2_script; p0=args.p0_script; cfilt=common(args); p0f=common_p0(args)
    if args.stage in ('all','p0','p0-node'):
        if not args.p0_root: raise SystemExit('--p0-root required for P0 stages')
        s=plan['p0-node-features']; run_cmd([py,p0,'node-features','--p0-root',args.p0_root,'--labels-root',args.labels_root,'--out-root',str(p2_root/'p0_node_features'),'--workers',str(s.workers)]+p0f,env,args.dry_run)
    if args.stage in ('all','p0','p0-edge'):
        if not args.p0_root: raise SystemExit('--p0-root required for P0 stages')
        s=plan['p0-edge-spillover']; run_cmd([py,p0,'edge-spillover','--p0-root',args.p0_root,'--labels-root',args.labels_root,'--out-root',str(p2_root/'p0_edge_spillover'),'--past-horizon',args.past_horizon,'--workers',str(s.workers)]+p0f,env,args.dry_run)
    if args.stage in ('all','p0','p0-graph'):
        if not args.p0_root: raise SystemExit('--p0-root required for P0 stages')
        s=plan['p0-graph-state']; run_cmd([py,p0,'graph-state','--p0-root',args.p0_root,'--out-root',str(p2_root/'p0_graph_state'),'--workers',str(s.workers)]+p0f,env,args.dry_run)
    if args.stage in ('all','p0','p0-eval'):
        run_cmd([py,p0,'eval-p0','--p0-alpha-root',str(p2_root),'--out-dir',str(p2_root/'p0_alpha_eval')],env,args.dry_run)
    if args.stage in ('all','theme'):
        if not args.p1_root: raise SystemExit('--p1-root required for P1 theme stages')
        s=plan['build-theme-returns']; run_cmd([py,p2,'build-theme-returns','--p1-root',args.p1_root,'--labels-root',args.labels_root,'--out-root',str(p2_root/'theme_returns'),'--workers',str(s.workers),'--inner-workers',str(s.inner_workers)]+cfilt,env,args.dry_run)
    if args.stage in ('all','relation'):
        if not args.p1_root: raise SystemExit('--p1-root required for P1 relation stages')
        s=plan['relation-spillover']; cmd=[py,p2,'relation-spillover','--p1-root',args.p1_root,'--theme-returns-root',str(p2_root/'theme_returns'),'--out-root',str(p2_root/'relation_spillover'),'--past-horizon',args.past_horizon,'--workers',str(s.workers),'--inner-workers',str(s.inner_workers)]+cfilt; cmd+=csv_arg('--tiers',args.tiers); run_cmd(cmd,env,args.dry_run)
    if args.stage in ('all','daily'):
        s=plan['daily-relation-features']; run_cmd([py,p2,'daily-relation-features','--signals-root',str(p2_root/'relation_spillover'),'--out-root',str(p2_root/'daily_relation_features'),'--workers',str(s.workers),'--underreaction-past-horizon',args.underreaction_past_horizon]+cfilt,env,args.dry_run)
    if args.stage in ('all','eval'):
        run_cmd([py,p2,'evaluate-daily','--features-root',str(p2_root/'daily_relation_features'),'--out-dir',str(p2_root/'daily_relation_eval')],env,args.dry_run)
if __name__=='__main__': main()
