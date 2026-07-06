from __future__ import annotations

import pandas as pd

from graphfactorfactory.application.adjusted_labels import build_split_adjusted_labels
from graphfactorfactory.application.causality import audit_source_events
from graphfactorfactory.application.graph import MultilayerGraphBuilder
from graphfactorfactory.application.labels import build_forward_labels
from graphfactorfactory.application.pit import build_point_in_time_panel, decision_grid, filter_regular_session
from graphfactorfactory.domain.config import BuildConfig
from graphfactorfactory.domain.layers import LAYERS, MAX_LOOKBACK_MINUTES
from graphfactorfactory.domain.records import BuildResult
from graphfactorfactory.infrastructure.corporate_actions import SplitAdjustmentSource
from graphfactorfactory.infrastructure.store import CanonicalGraphStore
from graphfactorfactory.ports.node_source import NodeFactorSource


def _partition_decisions(decisions, chunk_size: int) -> list[list]:
    size = max(1, int(chunk_size))
    values = list(decisions)
    return [values[index : index + size] for index in range(0, len(values), size)]


def _process_chunk(args):
    chunk_decisions, chunk_data, config, symbols, layers, include_multiplex = args
    from graphfactorfactory.application.graph import MultilayerGraphBuilder

    builder = MultilayerGraphBuilder(config, symbols, layers=tuple(layers), include_multiplex=include_multiplex)
    results = []
    for t in chunk_decisions:
        decision_time = pd.Timestamp(t)
        decision_time = decision_time.tz_localize("UTC") if decision_time.tzinfo is None else decision_time.tz_convert("UTC")
        window_start = decision_time - pd.Timedelta(minutes=MAX_LOOKBACK_MINUTES)
        window = chunk_data[
            (chunk_data["available_time"] <= decision_time)
            & (chunk_data["timestamp"] <= decision_time)
            & (chunk_data["timestamp"] > window_start)
        ]
        results.append((builder.build_snapshot(window, decision_time), t))
    return results


class GraphFactorPipeline:
    def __init__(self, source: NodeFactorSource, store: CanonicalGraphStore, config: BuildConfig):
        self.source = source
        self.store = store
        self.config = config

    def build_date(self, trade_date: str, universe: list[str] | None = None, executor=None) -> BuildResult:
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
            {"layer_id": 0, "name": "multiplex", "family": "multiplex", "directed": False, "lag_bars": 0, "columns": "", "lookbacks_minutes": "30"},
            *[
                {
                    "layer_id": layer.layer_id,
                    "name": layer.name,
                    "family": layer.family,
                    "directed": layer.directed,
                    "lag_bars": layer.lag_bars,
                    "columns": ",".join(layer.columns),
                    "lookbacks_minutes": ",".join(map(str, layer.lookbacks_minutes)),
                }
                for layer in LAYERS
            ],
        ])
        self.store.initialize_dimensions(symbols, layers)
        panel = build_point_in_time_panel(events, self.config)
        symbol_lookup = dict(zip(symbols["symbol"], symbols["symbol_id"]))
        split_source = SplitAdjustmentSource(self.config.split_csv_path) if self.config.split_csv_path else None
        label_rows = 0
        with self.store.open_day(trade_date) as writer:
            if self.config.store_labels:
                labels = build_split_adjusted_labels(panel, self.config.horizons_minutes, split_source) if split_source else build_forward_labels(panel, self.config.horizons_minutes)
                labels["symbol_id"] = labels["symbol"].map(symbol_lookup).astype("int32")
                writer.write_labels(labels.drop(columns="symbol"))
                label_rows = len(labels)
            graph_decisions = decision_grid(events, self.config)

            symbols_list = symbols.sort_values("symbol_id")["symbol"].astype(str).tolist()
            data = events[events["symbol"].isin(symbols_list)].copy()
            data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
            data["available_time"] = pd.to_datetime(data["available_time"], utc=True)

            from concurrent.futures import ProcessPoolExecutor, as_completed
            max_threads = getattr(self, "max_threads", 26)
            task_chunk_size = getattr(self, "task_chunk_size", 3)
            chunks = _partition_decisions(graph_decisions, task_chunk_size)
            chunk_tasks = []
            for chunk in chunks:
                if len(chunk) == 0:
                    continue
                min_t = pd.Timestamp(chunk[0])
                min_t = min_t.tz_localize("UTC") if min_t.tzinfo is None else min_t.tz_convert("UTC")
                max_t = pd.Timestamp(chunk[-1])
                max_t = max_t.tz_localize("UTC") if max_t.tzinfo is None else max_t.tz_convert("UTC")
                chunk_window_start = min_t - pd.Timedelta(minutes=MAX_LOOKBACK_MINUTES)
                chunk_data = data[(data["timestamp"] > chunk_window_start) & (data["available_time"] <= max_t)].copy()
                chunk_tasks.append((list(chunk), chunk_data, self.config, symbols, LAYERS, True))

            def consume(pool):
                import logging
                logger = logging.getLogger(__name__)
                total_chunks = len(chunk_tasks)
                logger.info(f"Phase 0: Submitted {total_chunks} chunk tasks for processing.")
                futures = {
                    pool.submit(_process_chunk, task): index
                    for index, task in enumerate(chunk_tasks)
                }
                buffered = {}
                next_index = 0
                completed = 0
                for future in as_completed(futures):
                    buffered[futures[future]] = future.result()
                    completed += 1
                    if completed % max(1, total_chunks // 10) == 0 or completed == total_chunks:
                        logger.info(f"Phase 0 Progress: {completed}/{total_chunks} chunks completed ({(completed/total_chunks)*100:.1f}%)")
                    while next_index in buffered:
                        for products, _ in buffered.pop(next_index):
                            writer.write_edges(products.edges)
                            writer.write_node_features(products.node_features)
                            writer.write_snapshots(products.snapshots)
                        next_index += 1

            if executor:
                consume(executor)
            else:
                with ProcessPoolExecutor(max_workers=max_threads) as pool:
                    consume(pool)

        catalog = self.store.finalize_catalog()
        manifest = self.store.write_manifest(
            trade_date=trade_date,
            source_fingerprint=self.source.fingerprint(),
            config=self.config,
            universe_count=len(universe),
            node_feature_columns=self.source.numeric_feature_columns(),
            split_source_metadata=split_source.metadata if split_source else None,
        )
        counts = self.store.count_date_rows(trade_date)
        return BuildResult(root=self.store.root, manifest_path=manifest, catalog_path=catalog, edge_rows=counts["edges"], node_feature_rows=counts["node_features"], snapshot_rows=counts["snapshots"], label_rows=label_rows)
