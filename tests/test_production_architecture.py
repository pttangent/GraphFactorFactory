from graphfactorfactory.application.lsh import strict_degree_cap
from graphfactorfactory.themes.production_replay import infer_frame_minutes

def test_strict_degree_cap():
    edges=[(0,1,.99,1,1),(0,2,.98,1,1),(1,2,.97,1,1),(2,3,.96,1,1)]
    kept=strict_degree_cap(edges,1)
    degree={}
    for left,right,*_ in kept:
        degree[left]=degree.get(left,0)+1
        degree[right]=degree.get(right,0)+1
    assert max(degree.values(),default=0)<=1

def test_actual_cadence():
    times=['2026-06-16T13:30:00Z','2026-06-16T13:35:00Z','2026-06-16T13:40:00Z']
    assert infer_frame_minutes(times,15)==5
