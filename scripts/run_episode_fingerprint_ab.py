from __future__ import annotations
import json, math, random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SRC=Path('/mnt/data/outputs/temporal_theme_research/intraday_1m/theme_instances_2026-01-20_22.parquet')
OUT=Path('/mnt/data/outputs/temporal_theme_research/episode_fingerprint_ab')
OUT.mkdir(parents=True,exist_ok=True)
DATES=['2026-01-20','2026-01-21','2026-01-22']
MIN_SIZE=20
WINDOW=30

@dataclass
class Episode:
    eid:str
    date:str
    side:str
    layer:int
    start_state:int
    end_state:int
    last_state:int
    instances:int=0
    member_counts:Counter=field(default_factory=Counter)
    core_counts:Counter=field(default_factory=Counter)
    sizes:list=field(default_factory=list)
    last_members:set=field(default_factory=set)
    last_core:set=field(default_factory=set)
    gaps:int=0
    def add(self,state,members,core,size):
        if state>self.last_state+1:
            self.gaps += state-self.last_state-1
        self.end_state=state
        self.last_state=state
        self.instances+=1
        self.member_counts.update(members)
        self.core_counts.update(core)
        self.sizes.append(size)
        self.last_members=set(members)
        self.last_core=set(core)
    def proto(self):
        n=max(1,self.instances)
        mf={k:v/n for k,v in self.member_counts.items()}
        cf={k:v/n for k,v in self.core_counts.items()}
        core={k for k,v in cf.items() if v>=0.5}
        persistent={k for k,v in mf.items() if v>=0.5}
        s=np.asarray(self.sizes,dtype=float)
        return dict(eid=self.eid,date=self.date,side=self.side,layer=self.layer,start=self.start_state,end=self.end_state,
                    duration=self.end_state-self.start_state+1,instances=self.instances,gaps=self.gaps,
                    member_freq=mf,core_freq=cf,core=core,persistent=persistent,
                    mean_size=float(s.mean()),std_size=float(s.std()),max_size=int(s.max()),min_size=int(s.min()))

def wj(a,b):
    keys=set(a)|set(b)
    if not keys:
        return 0.0
    num=sum(min(a.get(k,0),b.get(k,0)) for k in keys)
    den=sum(max(a.get(k,0),b.get(k,0)) for k in keys)
    return num/den if den else 0.0

def jac(a,b):
    return len(a&b)/len(a|b) if a and b else 0.0

def cont(a,b):
    return len(a&b)/min(len(a),len(b)) if a and b else 0.0

def struct_sim(a,b):
    size=math.exp(-abs(math.log(max(1,a['mean_size']))-math.log(max(1,b['mean_size'])))/0.8)
    dur=math.exp(-abs(math.log(max(1,a['duration']))-math.log(max(1,b['duration'])))/0.8)
    inst=math.exp(-abs(math.log(max(1,a['instances']))-math.log(max(1,b['instances'])))/0.8)
    gap=math.exp(-abs(a['gaps']-b['gaps'])/3)
    return .45*size+.25*dur+.2*inst+.1*gap

def scores(a,b):
    member=wj(a['member_freq'],b['member_freq'])
    core=cont(a['core'],b['core'])
    persistent=jac(a['persistent'],b['persistent'])
    structure=struct_sim(a,b)
    member_fp=.45*member+.35*core+.20*persistent
    hybrid=.50*member_fp+.30*structure+.20*math.exp(-abs(math.log(max(1,a['mean_size']))-math.log(max(1,b['mean_size']))))
    return {'member':member_fp,'structure':structure,'hybrid':hybrid,'raw_member':member,'core':core,'persistent':persistent}

def select_windows(df,date):
    g=df[df.date==date]
    states=sorted(g.day_state_index.unique())
    mid0=max(0,(len(states)-WINDOW)//2)
    defs={'open':set(states[:WINDOW]),'mid':set(states[mid0:mid0+WINDOW]),'close':set(states[-WINDOW:])}
    return {name:g[g.day_state_index.isin(ix)].copy() for name,ix in defs.items()}

def build_episodes(win,date,side,gap_allow):
    episodes=[]
    active=defaultdict(list)
    nextid=0
    for state in sorted(win.day_state_index.unique()):
        gs=win[win.day_state_index==state]
        for layer,cur in gs.groupby('layer_id',sort=False):
            layer=int(layer)
            valid=[e for e in active[layer] if state-e.last_state<=gap_allow+1]
            inv=defaultdict(list)
            for i,e in enumerate(valid):
                for node in e.last_members:
                    inv[node].append(i)
            candidates=[]
            rows=[]
            for _,r in cur.iterrows():
                members=set(map(int,r.members))
                core=set(map(int,r.core_members))
                size=int(r['size'])
                if size<MIN_SIZE:
                    continue
                ci=len(rows)
                rows.append((members,core,size))
                counts=Counter()
                for node in members:
                    for ei in inv.get(node,()):
                        counts[ei]+=1
                for ei,inter in counts.items():
                    episode=valid[ei]
                    j=inter/(len(members)+len(episode.last_members)-inter)
                    c=inter/min(len(members),len(episode.last_members))
                    cc=cont(core,episode.last_core)
                    score=.45*j+.35*c+.20*cc
                    if (j>=.35 or c>=.50) and score>=.35:
                        candidates.append((score,ci,ei))
            candidates.sort(reverse=True)
            used_cur=set(); used_episode=set(); assignment={}
            for _,ci,ei in candidates:
                if ci in used_cur or ei in used_episode:
                    continue
                used_cur.add(ci); used_episode.add(ei); assignment[ci]=valid[ei]
            for ci,(members,core,size) in enumerate(rows):
                episode=assignment.get(ci)
                if episode is None:
                    episode=Episode(f'{date}_{side}_g{gap_allow}_{nextid}',date,side,layer,int(state),int(state),int(state))
                    nextid+=1
                    episodes.append(episode)
                    active[layer].append(episode)
                episode.add(int(state),members,core,size)
    return [episode.proto() for episode in episodes]
