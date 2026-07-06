import pandas as pd
from graphfactorfactory.domain.layers import LAYER_SCALES, LAYER_BY_NAME
from graphfactorfactory.application.math_utils import trajectory
from graphfactorfactory.application.lsh import strict_degree_cap


def test_all_layer_scales_update_every_minute():
    assert len(LAYER_SCALES) == 35
    assert all(item.decision_step_minutes == 1 for item in LAYER_SCALES)


def test_strict_degree_cap_both_endpoints():
    edges=[(0,1,.99,1,1),(0,2,.98,1,1),(1,2,.97,1,1),(2,3,.96,1,1)]
    kept=strict_degree_cap(edges,1)
    degree={}
    for left,right,*_ in kept:
        degree[left]=degree.get(left,0)+1; degree[right]=degree.get(right,0)+1
    assert max(degree.values(),default=0) <= 1


def test_rolling_5m_return_trajectory_uses_rolling_returns():
    times=pd.date_range('2026-06-16 14:00', periods=30, freq='min', tz='UTC')
    rows=[]
    for i,t in enumerate(times):
        rows += [
            {'timestamp':t,'symbol':'A','log_ret_1m':float(i+1)},
            {'timestamp':t,'symbol':'B','log_ret_1m':float((i+1)*2)},
        ]
    frame=pd.DataFrame(rows)
    layer=LAYER_BY_NAME['return_corr_cross_sectional_rolling_5m']
    vectors,points,used=trajectory(frame,layer,['A','B'],20)
    assert vectors is not None
    assert points == 26
    assert used == ('log_ret_1m',)
