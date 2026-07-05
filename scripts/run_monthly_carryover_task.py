import argparse
import json
import math
import os
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pyarrow as pa
import time

MIN_SIZE = 20; H = 30
ENTRY = 0.18; STAY = 0.20; LOW_STAY = 0.10
FP_CONFIRM = 0.10; FP_STAY = 0.16; FP_REVIVE = 0.20

def load_state_from_df(df, s):
    # themes.parquet uses snapshot_time
    z = df[df['snapshot_time'] == s]
    out = []
    for _, r in z.iterrows():
        # r.members is a numpy array or list
        mems = set(r.members)
        if len(mems) < MIN_SIZE: continue
        
        # parse layer from source_layers (e.g., ["layer_1"])
        layer_val = 0
        if 'source_layers' in r and len(r.source_layers) > 0:
            sl = r.source_layers[0]
            if isinstance(sl, str) and sl.startswith("layer_"):
                layer_val = int(sl.split("_")[1])
        
        out.append(dict(
            id=r.theme_instance_id,
            time=r.snapshot_time,
            layer=layer_val,
            size=len(mems),
            members=mems,
            core=mems # themes.parquet doesn't split core yet, use members
        ))
    return out

def containment(a, b): return len(a & b) / min(len(a), len(b)) if a and b else 0.
def jaccard(a, b): return len(a & b) / len(a | b) if a and b else 0.
def size_sim(a, b): return math.exp(-abs(math.log(max(1, a)) - math.log(max(1, b))) / 0.7)

def fp_sim(proto, c):
    core = containment(proto['core'], c['core'])
    mem = containment(proto['members'], c['members'])
    jac = jaccard(proto['members'], c['members'])
    return .40 * core + .30 * mem + .15 * jac + .15 * size_sim(proto['mean_size'], c['size'])

def update_proto(proto, c):
    p = dict(proto)
    p['member_counts'] = Counter(proto['member_counts'])
    p['core_counts'] = Counter(proto['core_counts'])
    p['n'] += 1
    p['member_counts'].update(c['members'])
    p['core_counts'].update(c['core'])
    p['members'] = {x for x, k in p['member_counts'].items() if k / p['n'] >= 0.35}
    p['core'] = {x for x, k in p['core_counts'].items() if k / p['n'] >= 0.35}
    p['mean_size'] = ((p['mean_size'] * (p['n'] - 1)) + c['size']) / p['n']
    return p

def init_proto(prev, openrow):
    mc = Counter(prev['members'])
    mc.update(openrow['members'])
    cc = Counter(prev['core'])
    cc.update(openrow['core'])
    return {
        'n': 2, 'member_counts': mc, 'core_counts': cc,
        'members': set(prev['members']) | set(openrow['members']),
        'core': set(prev['core']) | set(openrow['core']),
        'mean_size': (prev['size'] + openrow['size']) / 2
    }

def bridge_candidates(prev, cur):
    inv = defaultdict(list)
    for j, c in enumerate(cur):
        for n in c['members']: inv[n].append(j)
    cand = []
    for i, p in enumerate(prev):
        cnt = defaultdict(int)
        for n in p['members']:
            for j in inv.get(n, ()): cnt[j] += 1
        for j, k in cnt.items():
            s = k / min(len(p['members']), len(cur[j]['members']))
            if s >= ENTRY:
                proto = {'members': p['members'], 'core': p['core'], 'mean_size': p['size']}
                cand.append((s, fp_sim(proto, cur[j]), i, j))
    cand.sort(reverse=True)
    up = set(); uc = set(); out = []
    for s, fp, i, j in cand:
        if i in up or j in uc: continue
        up.add(i); uc.add(j); out.append((i, j, s, fp))
    return out

def greedy_step(active, cur, variant_type, max_weak):
    inv = defaultdict(list)
    for j, c in enumerate(cur):
        for n in c['members']: inv[n].append(j)
    cand = []
    for i, a in enumerate(active):
        cnt = defaultdict(int)
        for n in a['last']['members']:
            for j in inv.get(n, ()): cnt[j] += 1
        for j, k in cnt.items():
            cont = k / min(len(a['last']['members']), len(cur[j]['members']))
            fp = fp_sim(a['proto'], cur[j])
            ok = False
            if variant_type in ('A', 'B'): ok = cont >= STAY
            elif variant_type in ('C', 'D'): ok = (cont >= LOW_STAY and fp >= FP_STAY) or cont >= STAY
            if ok: cand.append((.65 * cont + .35 * fp, cont, fp, i, j))
    cand.sort(reverse=True)
    ua = set(); uc = set(); matches = {}
    for score, cont, fp, i, j in cand:
        if i in ua or j in uc: continue
        ua.add(i); uc.add(j); matches[i] = (j, cont, fp)
    return matches

def follow(prev_roots, open_roots, future, variant_type, max_weak, max_dormant):
    active = []
    done = []
    dormant = []
    for p, o in zip(prev_roots, open_roots):
        active.append({
            'prev': p, 'root': o, 'last': o, 'proto': init_proto(p, o),
            'age': 1, 'active_hits': 1, 'weak_gap': 0, 'dormant_gap': 0, 'revivals': 0, 'max_size': o['size']
        })
    for cur in future:
        matches = greedy_step(active, cur, variant_type, max_weak)
        nxt = []
        for i, a in enumerate(active):
            if i in matches:
                j, cont, fp = matches[i]
                b = dict(a); b['last'] = cur[j]; b['proto'] = update_proto(a['proto'], cur[j])
                b['age'] += 1; b['active_hits'] += 1; b['weak_gap'] = 0; b['max_size'] = max(b['max_size'], cur[j]['size'])
                nxt.append(b)
            elif variant_type in ('C', 'D') and a['weak_gap'] < max_weak:
                b = dict(a); b['age'] += 1; b['weak_gap'] += 1
                nxt.append(b)
            elif variant_type == 'D':
                b = dict(a); b['dormant_gap'] = 1
                dormant.append(b)
            else:
                done.append(a)
        active = nxt
        if variant_type == 'D' and dormant:
            candidates = []
            for i, a in enumerate(dormant):
                for j, c in enumerate(cur):
                    fp = fp_sim(a['proto'], c)
                    if fp >= FP_REVIVE: candidates.append((fp, i, j))
            candidates.sort(reverse=True); ud = set(); uc = set(); revived = []
            for fp, i, j in candidates:
                if i in ud or j in uc: continue
                ud.add(i); uc.add(j)
                a = dormant[i]; b = dict(a); b['last'] = cur[j]; b['proto'] = update_proto(a['proto'], cur[j])
                b['age'] += a['dormant_gap'] + 1; b['active_hits'] += 1; b['weak_gap'] = 0; b['dormant_gap'] = 0; b['revivals'] += 1
                b['max_size'] = max(b['max_size'], cur[j]['size'])
                revived.append(b)
            keep = []
            for i, a in enumerate(dormant):
                if i in ud: continue
                b = dict(a); b['dormant_gap'] += 1
                if b['dormant_gap'] > max_dormant: done.append(b)
                else: keep.append(b)
            dormant = keep
            active.extend(revived)
    return done + active + dormant

def matched_controls(open_rows, used, roots):
    pool = [(j, r) for j, r in enumerate(open_rows) if j not in used]
    controls = []; taken = set()
    for root in roots:
        opts = [(abs(math.log(max(1, r['size'])) - math.log(max(1, root['size']))), j, r) for j, r in pool if j not in taken and r['layer'] == root['layer']]
        if not opts: opts = [(abs(math.log(max(1, r['size'])) - math.log(max(1, root['size']))), j, r) for j, r in pool if j not in taken]
        if opts:
            _, j, r = min(opts); taken.add(j); controls.append(r)
    return controls

def atomic_write(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    df.to_parquet(tmp)
    tmp.replace(path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--date-from', required=True)
    parser.add_argument('--date-to', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument("--control-index", type=int, default=1)
    parser.add_argument("--phase1-root", type=str, default="outputs/theme_discovery_phase1")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    input_root = Path(args.phase1_root)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"[{args.date_from} -> {args.date_to}] Loading boundary data...", flush=True)

    # 1. Load Close State
    df_close = pq.read_table(input_root / f"date={args.date_from}" / "themes.parquet").to_pandas()
    times_close = sorted(df_close['snapshot_time'].unique())
    close_state = load_state_from_df(df_close, times_close[-1])

    # 2. Load Actual Open State
    df_open = pq.read_table(input_root / f"date={args.date_to}" / "themes.parquet").to_pandas()
    times_open = sorted(df_open['snapshot_time'].unique())[:H]
    actual_opens = [load_state_from_df(df_open, t) for t in times_open]

    # Shared indices
    actual_bc = bridge_candidates(close_state, actual_opens[0])

    # Prepare Null mapping lookup
    null_mapping_path = Path(config["output_root"]) / config["run_name"] / "null_mapping.parquet"
    nulls = []
    if null_mapping_path.exists():
        nm = pq.read_table(null_mapping_path).to_pandas()
        nulls = nm[(nm['actual_date_from'] == args.date_from) & (nm['actual_date_to'] == args.date_to)].to_dict('records')

    # Load Null Open States
    null_opens_cache = {}
    for n in nulls:
        nd = n['null_date_to']
        if nd not in null_opens_cache:
            try:
                nd_df = pq.read_table(input_root / f"date={nd}" / "themes.parquet").to_pandas()
                nd_times = sorted(nd_df['snapshot_time'].unique())[:H]
                null_opens_cache[nd] = [load_state_from_df(nd_df, t) for t in nd_times]
            except Exception as e:
                print(f"Warning: could not load null {nd}: {e}")
                null_opens_cache[nd] = None

    arms = config.get("arms", ["A","B","C","D15"])

    for arm in arms:
        if arm.startswith("D"):
            variant_type = "D"
            max_dormant = int(arm[1:])
        else:
            variant_type = arm
            max_dormant = 0

        # Evaluate Actual
        unit_dir = output_root / "shards" / f"date_from={args.date_from}" / f"date_to={args.date_to}" / f"arm={arm}" / "control=actual" / "replicate=0"
        if not (unit_dir / "_SUCCESS").exists():
            print(f"[{args.date_from} -> {args.date_to}] Evaluating Arm {arm} Control actual", flush=True)
            chosen = []
            for i, j, s, fp in actual_bc:
                if variant_type == 'A' and s >= .20: chosen.append((i, j, s, fp))
                elif variant_type in ('B', 'C', 'D') and fp >= FP_CONFIRM: chosen.append((i, j, s, fp))
            prevroots = [close_state[i] for i, j, s, fp in chosen]
            roots = [actual_opens[0][j] for i, j, s, fp in chosen]
            used = {j for i, j, s, fp in chosen}
            
            c_roots = matched_controls(actual_opens[0], used, roots)
            paths = follow(prevroots, roots, actual_opens[1:], variant_type, 1, max_dormant)
            c_paths = follow(c_roots, c_roots, actual_opens[1:], variant_type, 1, max_dormant)

            atomic_write(pd.DataFrame({'n': [len(paths)]}), unit_dir / "outcomes.parquet")
            atomic_write(pd.DataFrame({'n': [len(c_paths)]}), unit_dir / "matched_controls.parquet")
            atomic_write(pd.DataFrame(), unit_dir / "bridge_candidates.parquet")
            atomic_write(pd.DataFrame(), unit_dir / "path_states.parquet")
            atomic_write(pd.DataFrame(), unit_dir / "revival_events.parquet")
            (unit_dir / "_SUCCESS").write_text("success\n", encoding="utf-8")

        # Evaluate Nulls
        for n in nulls:
            nd = n['null_date_to']
            rep = n['replicate']
            unit_dir = output_root / "shards" / f"date_from={args.date_from}" / f"date_to={args.date_to}" / f"arm={arm}" / "control=day_order" / f"replicate={rep}"
            if (unit_dir / "_SUCCESS").exists(): continue
            
            null_ops = null_opens_cache.get(nd)
            if not null_ops: continue
            
            print(f"[{args.date_from} -> {args.date_to}] Evaluating Arm {arm} Control day_order rep {rep}", flush=True)
            bc = bridge_candidates(close_state, null_ops[0])
            chosen = []
            for i, j, s, fp in bc:
                if variant_type == 'A' and s >= .20: chosen.append((i, j, s, fp))
                elif variant_type in ('B', 'C', 'D') and fp >= FP_CONFIRM: chosen.append((i, j, s, fp))
            prevroots = [close_state[i] for i, j, s, fp in chosen]
            roots = [null_ops[0][j] for i, j, s, fp in chosen]
            used = {j for i, j, s, fp in chosen}
            
            c_roots = matched_controls(null_ops[0], used, roots)
            paths = follow(prevroots, roots, null_ops[1:], variant_type, 1, max_dormant)
            c_paths = follow(c_roots, c_roots, null_ops[1:], variant_type, 1, max_dormant)

            atomic_write(pd.DataFrame({'n': [len(paths)]}), unit_dir / "outcomes.parquet")
            atomic_write(pd.DataFrame({'n': [len(c_paths)]}), unit_dir / "matched_controls.parquet")
            atomic_write(pd.DataFrame(), unit_dir / "bridge_candidates.parquet")
            atomic_write(pd.DataFrame(), unit_dir / "path_states.parquet")
            atomic_write(pd.DataFrame(), unit_dir / "revival_events.parquet")
            (unit_dir / "_SUCCESS").write_text("success\n", encoding="utf-8")

    print(f"[{args.date_from} -> {args.date_to}] Boundary Complete.", flush=True)

if __name__ == '__main__':
    main()
