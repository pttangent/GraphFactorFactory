from __future__ import annotations
import pandas as pd
from .temporal_edges import TemporalEdgeConfig,TemporalEdgeReplay

def infer_frame_minutes(times,fallback=5):
    s=pd.to_datetime(pd.Series(list(times)),utc=True,errors='coerce').dropna().sort_values()
    if len(s)<2:return int(fallback)
    d=s.diff().dropna().dt.total_seconds().div(60);d=d[d>0]
    return max(1,int(round(float(d.median())))) if len(d) else int(fallback)

def prepare_edges(edges,times,config):
    if not getattr(config,'enable_temporal_edges',False):
        return {t:edges[edges.decision_time==t].copy() for t in times}
    replay=TemporalEdgeReplay(TemporalEdgeConfig(
        config.temporal_enter_threshold,config.temporal_exit_threshold,
        config.temporal_smoothing_alpha,config.temporal_missing_grace_frames))
    out={}
    for t in times:out[t]=replay.replay(edges[edges.decision_time==t],t)
    return out
