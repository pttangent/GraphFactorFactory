from __future__ import annotations

def detect_snapshot(args):
    t,edges,detector,consensus,names,universe_count,run_id=args
    parents=[];children=[]
    for layer_id,group in edges.groupby('layer_id',sort=False):
        layer_id=int(layer_id)
        if layer_id==0:continue
        p,c=detector.detect_hierarchy(group,layer_id=layer_id,
            layer_name=names.get(layer_id,str(layer_id)),snapshot_time=t,
            universe_count=universe_count)
        parents.extend(p);children.extend(c)
    themes=consensus.build(parents+children,snapshot_time=t,
        run_id=run_id,universe_count=universe_count)
    return t,parents,children,themes
