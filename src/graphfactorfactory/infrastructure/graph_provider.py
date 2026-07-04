from __future__ import annotations

from pathlib import Path

import pandas as pd


class GraphBatchProvider:
    def __init__(self, graph_store_root: str | Path):
        self.root = Path(graph_store_root).expanduser().resolve()

    def load_snapshot(self, trade_date: str, decision_time, layer_ids: list[int] | None = None) -> dict[str, pd.DataFrame]:
        day = self.root / "canonical" / f"date={trade_date}"
        timestamp = pd.Timestamp(decision_time)
        edges = pd.read_parquet(day / "edges.parquet", filters=[("decision_time", "=", timestamp)])
        nodes = pd.read_parquet(day / "node_features.parquet", filters=[("decision_time", "=", timestamp)])
        if layer_ids is not None:
            wanted = set(layer_ids)
            edges = edges[edges["layer_id"].isin(wanted)]
            nodes = nodes[nodes["layer_id"].isin(wanted | {0})]
        return {"edges": edges, "node_features": nodes}
