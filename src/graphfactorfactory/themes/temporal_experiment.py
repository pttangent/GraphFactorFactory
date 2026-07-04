from __future__ import annotations

from pathlib import Path
import pandas as pd

from .temporal_edges_v2 import LayerRelativeTemporalReplay, LayerTemporalPolicy
from .temporal_community import TwoSliceLeidenDetector
from .sparse_consensus import SparseConsensusBuilder


DEFAULT_POLICIES = {
    "return_corr": LayerTemporalPolicy(0.75, 0.55, 0.20, 0.60, 2),
    "venue_fragmentation": LayerTemporalPolicy(0.80, 0.60, 0.20, 0.60, 2),
    "odd_lot_activity": LayerTemporalPolicy(0.80, 0.60, 0.20, 0.60, 2),
    "signed_flow": LayerTemporalPolicy(0.80, 0.60, 0.15, 0.65, 1),
    "flow_return_alignment": LayerTemporalPolicy(0.80, 0.60, 0.15, 0.65, 1),
    "trade_intensity": LayerTemporalPolicy(0.80, 0.60, 0.15, 0.65, 1),
    "large_trade_flow": LayerTemporalPolicy(0.85, 0.70, 0.05, 0.80, 0),
    "block_activity": LayerTemporalPolicy(0.85, 0.70, 0.05, 0.80, 0),
    "absorption": LayerTemporalPolicy(0.85, 0.70, 0.05, 0.80, 0),
}


class TemporalExperimentRunner:
    def __init__(self, graph_root, *, omega=0.10, seed=20260704):
        self.graph_root = Path(graph_root)
        layers = pd.read_parquet(self.graph_root / "dimensions" / "layers.parquet")
        self.layer_name = dict(zip(layers.layer_id.astype(int), layers.name.astype(str)))
        self.layer_family = dict(zip(layers.name.astype(str), layers.family.astype(str)))
        policies = {
            layer_id: DEFAULT_POLICIES.get(name, LayerTemporalPolicy())
            for layer_id, name in self.layer_name.items()
        }
        self.replay = LayerRelativeTemporalReplay(policies=policies)
        self.detectors = {
            layer_id: TwoSliceLeidenDetector(
                resolution=1.0,
                omega=omega,
                seed=seed,
                min_members=3,
            )
            for layer_id in self.layer_name
        }
        self.consensus = SparseConsensusBuilder(self.layer_family, seed=seed)

    def run_day(self, trade_date, *, start=None, minutes=60, output_dir=None):
        day = self.graph_root / "canonical" / f"date={trade_date}"
        edges = pd.read_parquet(day / "edges.parquet")
        times = sorted(edges.decision_time.unique())
        if start is not None:
            start = pd.Timestamp(start)
            times = [value for value in times if pd.Timestamp(value) >= start]
        times = times[:minutes]

        symbols = pd.read_parquet(self.graph_root / "dimensions" / "symbols.parquet")
        universe_count = len(symbols)
        theme_rows, community_rows, snapshot_rows = [], [], []

        for snapshot_time in times:
            observed = edges[edges.decision_time == snapshot_time]
            effective = self.replay.replay(observed, snapshot_time)
            communities = []
            for raw_layer_id, layer_edges in effective.groupby("layer_id", sort=False):
                layer_id = int(raw_layer_id)
                detector = self.detectors[layer_id]
                found = detector.detect(
                    layer_edges,
                    layer_id=layer_id,
                    layer_name=self.layer_name.get(layer_id, str(layer_id)),
                    snapshot_time=snapshot_time,
                    universe_count=universe_count,
                )
                communities.extend(found)
                for item in found:
                    community_rows.append({
                        "snapshot_time": snapshot_time,
                        "layer_id": layer_id,
                        "layer_name": item.layer_name,
                        "community_id": item.community_id,
                        "member_count": len(item.members),
                        "members": list(item.members),
                        "modularity": item.modularity,
                        "is_market_mode": item.is_market_mode,
                    })

            themes = self.consensus.build(
                communities,
                observed,
                snapshot_time=snapshot_time,
                run_id="temporal_v2",
                universe_count=universe_count,
            )
            for item in themes:
                theme_rows.append({
                    "snapshot_time": snapshot_time,
                    "theme_instance_id": item.theme_instance_id,
                    "member_count": len(item.members),
                    "members": list(item.members),
                    "source_layers": list(item.source_layers),
                    "source_families": list(item.source_families),
                    "consensus_score": item.consensus_score,
                })
            snapshot_rows.append({
                "snapshot_time": snapshot_time,
                "observed_edge_count": len(observed),
                "effective_edge_count": len(effective),
                "prior_only_edge_count": int((effective.temporal_status == "prior_only").sum()) if not effective.empty else 0,
                "community_count": len(communities),
                "theme_count": len(themes),
            })

        outputs = {
            "snapshots": pd.DataFrame(snapshot_rows),
            "communities": pd.DataFrame(community_rows),
            "themes": pd.DataFrame(theme_rows),
        }
        if output_dir is not None:
            target = Path(output_dir)
            target.mkdir(parents=True, exist_ok=True)
            for name, frame in outputs.items():
                frame.to_parquet(target / f"{name}.parquet", index=False)
        return outputs
