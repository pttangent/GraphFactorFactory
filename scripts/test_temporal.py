import pandas as pd
from graphfactorfactory.themes.temporal_edges import TemporalEdgeReplay, TemporalEdgeConfig

def test():
    config = TemporalEdgeConfig(enter_threshold=0.75, exit_threshold=0.65, smoothing_alpha=0.6, missing_grace_frames=1)
    replay = TemporalEdgeReplay(config)
    
    # Snapshot 1: New edges
    edges1 = pd.DataFrame([
        {"decision_time": "T1", "layer_id": 1, "src_id": 10, "dst_id": 20, "weight": 0.8},
        {"decision_time": "T1", "layer_id": 1, "src_id": 10, "dst_id": 30, "weight": 0.7} # Below threshold
    ])
    out1 = replay.replay(edges1, "T1")
    print("Out 1:", out1[["layer_id", "src_id", "dst_id", "weight", "temporal_status"]] if not out1.empty else "Empty")
    
    # Snapshot 2: 10->20 drops weight (should stay active due to exit_threshold), 10->30 missing (should be grace)
    edges2 = pd.DataFrame([
        {"decision_time": "T2", "layer_id": 1, "src_id": 10, "dst_id": 20, "weight": 0.6}
    ])
    out2 = replay.replay(edges2, "T2")
    print("Out 2:", out2[["layer_id", "src_id", "dst_id", "weight", "temporal_status"]] if not out2.empty else "Empty")
    
    # Snapshot 3: 10->20 missing again (grace), 10->30 missing again (should drop)
    edges3 = pd.DataFrame()
    out3 = replay.replay(edges3, "T3")
    print("Out 3:", out3[["layer_id", "src_id", "dst_id", "weight", "temporal_status"]] if not out3.empty else "Empty")

if __name__ == '__main__':
    test()
