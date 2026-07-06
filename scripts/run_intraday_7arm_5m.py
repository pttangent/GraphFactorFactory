from __future__ import annotations
import os, glob, time
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Set
import pyarrow.parquet as pq
import pandas as pd
import numpy as np

ROOT = '/mnt/data/phase1_input'
OUT = '/mnt/data/intraday_7arm_5m'
os.makedirs(OUT, exist_ok=True)
MIN_SIZE = 20
SAMPLE_EVERY = 5
ARMS = {
    'A': dict(entry_cont=.20, entry_fp=0, stay=.20, assist_cont=9, assist_fp=9, weak=0, dormant=0, rev_fp=9, breadth=0, post=0),
    'B': dict(entry_cont=.18, entry_fp=.10, stay=.20, assist_cont=9, assist_fp=9, weak=0, dormant=0, rev_fp=9, breadth=0, post=0),
    'C': dict(entry_cont=.18, entry_fp=.10, stay=.20, assist_cont=.10, assist_fp=.16, weak=1, dormant=0, rev_fp=9, breadth=0, post=0),
    'D9': dict(entry_cont=.18, entry_fp=.10, stay=.20, assist_cont=.10, assist_fp=.16, weak=1, dormant=3, rev_fp=.20, breadth=0, post=0),
    'D11': dict(entry_cont=.18, entry_fp=.10, stay=.20, assist_cont=.10, assist_fp=.16, weak=1, dormant=3, rev_fp=.20, breadth=.10, post=0),
    'D13': dict(entry_cont=.18, entry_fp=.10, stay=.20, assist_cont=.10, assist_fp=.16, weak=1, dormant=3, rev_fp=.22, breadth=.10, post=0),
    'D15': dict(entry_cont=.18, entry_fp=.10, stay=.20, assist_cont=.10, assist_fp=.16, weak=1, dormant=3, rev_fp=.30, breadth=.10, post=3),
}


@dataclass
class Path:
    pid: int
    anchor: Set[int]
    last: Set[int]
    first_i: int
    last_i: int
    hits: int = 1
    gap: int = 0
    confirmed: bool = False
    revivals: int = 0
    confirmed_revivals: int = 0
    pending_post: int = 0
    last_size: int = 0
    dead: bool = False


def sim(a: Set[int], b: Set[int]):
    inter = len(a & b)
    if inter == 0:
        return 0.0, 0.0
    cont = inter / min(len(a), len(b))
    jac = inter / (len(a) + len(b) - inter)
    return cont, jac


def run_arm(states: List[List[Set[int]]], cfg: dict):
    nextpid = 0
    paths: Dict[int, Path] = {}
    finished = []
    if not states:
        return []
    for c in states[0]:
        paths[nextpid] = Path(nextpid, set(c), set(c), 0, 0, last_size=len(c))
        nextpid += 1
    for i, comms in enumerate(states[1:], start=1):
        inv = defaultdict(list)
        for ci, c in enumerate(comms):
            for m in c:
                inv[m].append(ci)
        proposals = []
        for pid, p in paths.items():
            if p.dead:
                continue
            cand = set()
            for m in p.last:
                cand.update(inv.get(m, ()))
            for ci in cand:
                c = comms[ci]
                cont, jac = sim(p.last, c)
                fp_j = sim(p.anchor, c)[1]
                ok = False
                typ = 'stay'
                if not p.confirmed:
                    ok = cont >= cfg['entry_cont'] and fp_j >= cfg['entry_fp']
                    typ = 'entry'
                elif p.gap == 0:
                    ok = (cont >= cfg['stay']) or (cont >= cfg['assist_cont'] and fp_j >= cfg['assist_fp'])
                    typ = 'stay'
                else:
                    breadth = len(c) >= p.last_size * (1 + cfg['breadth'])
                    ok = p.gap <= cfg['dormant'] and fp_j >= cfg['rev_fp'] and breadth
                    typ = 'revive'
                if ok:
                    score = cont + .5 * jac + .5 * fp_j
                    proposals.append((score, pid, ci, typ))
        proposals.sort(reverse=True)
        usedp, usedc = set(), set()
        for _, pid, ci, typ in proposals:
            if pid in usedp or ci in usedc:
                continue
            p = paths[pid]
            c = comms[ci]
            was_gap = p.gap
            p.last = set(c)
            p.last_i = i
            p.hits += 1
            p.last_size = len(c)
            p.gap = 0
            if not p.confirmed:
                p.confirmed = True
            if typ == 'revive' and was_gap > 0:
                p.revivals += 1
                if cfg['post'] == 0:
                    p.confirmed_revivals += 1
                else:
                    p.pending_post = cfg['post']
            elif p.pending_post > 0:
                p.pending_post -= 1
                if p.pending_post == 0:
                    p.confirmed_revivals += 1
            usedp.add(pid)
            usedc.add(ci)
        for pid, p in list(paths.items()):
            if p.dead or pid in usedp:
                continue
            p.gap += 1
            allowed = cfg['weak'] if cfg['dormant'] == 0 else cfg['dormant']
            if p.gap > allowed:
                p.dead = True
                finished.append(p)
        for ci, c in enumerate(comms):
            if ci not in usedc:
                paths[nextpid] = Path(nextpid, set(c), set(c), i, i, last_size=len(c))
                nextpid += 1
    finished.extend([p for p in paths.values() if not p.dead])
    return finished


def metrics(paths):
    q = [p for p in paths if p.confirmed]
    n = len(q)
    if n == 0:
        return dict(paths=0, mean_hits=np.nan, median_life=np.nan, s5=np.nan, s15=np.nan, s30=np.nan, s60=np.nan, revival_rate=np.nan, confirmed_revival_rate=np.nan)
    life = np.array([(p.last_i - p.first_i) * 5 + 5 for p in q], float)
    return dict(
        paths=n,
        mean_hits=float(np.mean([p.hits for p in q])),
        median_life=float(np.median(life)),
        s5=float(np.mean(life >= 5)),
        s15=float(np.mean(life >= 15)),
        s30=float(np.mean(life >= 30)),
        s60=float(np.mean(life >= 60)),
        revival_rate=float(np.mean([p.revivals > 0 for p in q])),
        confirmed_revival_rate=float(np.mean([p.confirmed_revivals > 0 for p in q])),
    )


def load_day(path):
    tab = pq.read_table(path, columns=['snapshot_time', 'layer_name', 'members'])
    pdf = tab.to_pandas()
    pdf['size'] = pdf['members'].map(len)
    pdf = pdf[pdf['size'] >= MIN_SIZE]
    times = np.sort(pdf['snapshot_time'].unique())
    keep = set(times[::SAMPLE_EVERY])
    return pdf[pdf['snapshot_time'].isin(keep)]


rows = []
files = sorted(glob.glob(ROOT + '/date=*/layer_communities.parquet'))
print('FILES', len(files), flush=True)
for di, path in enumerate(files, 1):
    date = path.split('date=')[1].split('/')[0]
    t0 = time.time()
    pdf = load_day(path)
    for layer, ldf in pdf.groupby('layer_name', sort=False):
        times = sorted(ldf['snapshot_time'].unique())
        states = []
        for ts in times:
            comms = [set(x) for x in ldf.loc[ldf['snapshot_time'] == ts, 'members'].tolist()]
            states.append(comms)
        for arm, cfg in ARMS.items():
            m = metrics(run_arm(states, cfg))
            rows.append(dict(date=date, layer=layer, arm=arm, states=len(states), **m))
    pd.DataFrame(rows).to_csv(OUT + '/daily_layer_arm.csv', index=False)
    print(f'{di}/{len(files)} {date} rows={len(pdf)} sec={time.time()-t0:.1f}', flush=True)

daily = pd.DataFrame(rows)
agg = []
for (layer, arm), g in daily.groupby(['layer', 'arm']):
    w = g['paths'].fillna(0).to_numpy(float)
    W = w.sum()
    rec = {'layer': layer, 'arm': arm, 'days': g['date'].nunique(), 'confirmed_paths': int(W), 'days_with_paths': int((w > 0).sum())}
    for col in ['mean_hits', 'median_life', 's5', 's15', 's30', 's60', 'revival_rate', 'confirmed_revival_rate']:
        vals = g[col].to_numpy(float)
        mask = np.isfinite(vals) & (w > 0)
        rec[col] = float(np.average(vals[mask], weights=w[mask])) if mask.any() else np.nan
    agg.append(rec)
summary = pd.DataFrame(agg)
summary.to_csv(OUT + '/layer_arm_summary.csv', index=False)
for metric in ['s5', 's15', 's30', 's60', 'revival_rate', 'confirmed_revival_rate', 'median_life']:
    summary.pivot(index='layer', columns='arm', values=metric).to_csv(OUT + f'/{metric}_matrix.csv')
s = summary.copy()
s['score'] = (
    0.35 * s['s15'] + 0.35 * s['s30'] + 0.15 * s['s60']
    + 0.10 * s['confirmed_revival_rate'].fillna(0)
    - 0.10 * ((s['revival_rate'] - s['confirmed_revival_rate']).clip(lower=0).fillna(0))
    + 0.05 * np.minimum(s['confirmed_paths'] / 1000, 1)
)
best = s.sort_values(['layer', 'score'], ascending=[True, False]).groupby('layer').head(1)
best.to_csv(OUT + '/recommended_arm_by_layer.csv', index=False)
print('DONE', OUT, flush=True)
