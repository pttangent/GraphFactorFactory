from __future__ import annotations

from pathlib import Path

import pandas as pd

from graphfactorfactory.application.causality import audit_source_events
from graphfactorfactory.application.graph import MultilayerGraphBuilder
from graphfactorfactory.application.labels import build_forward_labels
from graphfactorfactory.application.pit import build_point_in_time_panel, decision_grid, filter_regular_session
from graphfactorfactory.domain.config import BuildConfig
from graphfactorfactory.domain.layers import LAYERS
from graphfactorfactory.domain.records import BuildResult
from graphfactorfactory.ports.node_source import NodeFactorSource
from graphfactorfactory.infrastructure.store import CanonicalGraphStore


class GraphFactorPipeline:
    def __init__(self, source: NodeFactorSource, store: CanonicalGraphStore, config: BuildConfig):
        self.source = source
        self.store = store
        self.config = config

    def build_date(self, trade_date: str, universe: list[str] | None = None) -> BuildResult:
        events = filter_regular_session(self.source.load_date(trade_date), self.config)
        if events.empty:
            raise ValueError(f"No regular-session rows for {trade_date}")
        audit_source_events(events)
        if universe is None:
            universe = sorted(events["symbol"].astype(str).unique())
        else:
            universe = sorted(set(map(str, universe)).intersection(events["symbol"].astype(str).unique()))
        symbols = pd.DataFrame({"symbol_id": pd.Series(range(len(universe)), dtype="int32"), "symbol": universe})
        layers = pd.DataFrame([
            {"layer_id": 0, "name": "multiplex", "family": "multiplex", "directed": False, "lag_bars": 0, "columns": ""},
            *[{"layer_id": layer.layer_id, "name": layer.name, "family": layer.family, "directed": layer.directed, "lag_bars": layer.lag_bars, "columns": ",".join(layer.columns)} for layer in LAYERS],
        ])
        self.store.initialize_dimensions(symbols, layers)
        panel = build_point_in_time_panel(events, self.config)
        symbol_lookup = dict(zip(symbols["symbol"], symbols["symbol_id"]))
        label_rows = 0
        with self.store.open_day(trade_date) as writer:
            if self.config.store_labels:
                labels = build_forward_labels(panel, self.config.horizons_minutes)
                labels["symbol_id"] = labels["symbol"].map(symbol_lookup).astype("int32")
                writer.write_labels(labels.drop(columns="symbol"))
                label_rows = len(labels)
            builder = MultilayerGraphBuilder(self.config, symbols)
            graph_decisions = decision_grid(events, self.config)
            graph_decisions = graph_decisions[:: max(1, self.config.graph_step_minutes // 5)]
            for decision_time in graph_decisions:
                products = builder.build_snapshot(events, decision_time)
                writer.write_edges(products.edges)
                writer.write_node_features(products.node_features)
                writer.write_snapshots(products.snapshots)
        catalog = self.store.finalize_catalog()
        manifest = self.store.write_manifest(trade_date=trade_date, source_fingerprint=self.source.fingerprint(), config=self.config, universe_count=len(universe), node_feature_columns=self.source.numeric_feature_columns())
        counts = self.store.count_date_rows(trade_date)
        return BuildResult(root=self.store.root, manifest_path=manifest, catalog_path=catalog, edge_rows=counts["edges"], node_feature_rows=counts["node_features"], snapshot_rows=counts["snapshots"], label_rows=label_rows)
