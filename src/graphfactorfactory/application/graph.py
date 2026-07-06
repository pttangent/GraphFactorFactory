from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import pandas as pd

from graphfactorfactory.application.correlation import reciprocal_correlation_graph
from graphfactorfactory.application.lsh import reciprocal_lsh_graph
from graphfactorfactory.application.math_utils import neighbor, trajectory, zscore
from graphfactorfactory.domain.config import BuildConfig
from graphfactorfactory.domain.layers import LAYERS, LayerDefinition


@dataclass
class SnapshotProducts:
    edges: pd.DataFrame
    node_features: pd.DataFrame
    snapshots: pd.DataFrame


class MultilayerGraphBuilder:
    def __init__(self, config: BuildConfig, symbols: pd.DataFrame, *, layers: tuple[LayerDefinition, ...] | None = None, include_multiplex: bool = True):
        self.config = config
        ordered = symbols.sort_values("symbol_id")
        self.symbols = ordered["symbol"].astype(str).tolist()
        self.symbol_ids = ordered["symbol_id"].astype("int32").to_numpy()
        self.layers = tuple(layers or LAYERS)
        self.include_multiplex = bool(include_multiplex)

    def build_snapshot(self, window: pd.DataFrame, decision_time) -> SnapshotProducts:
        started = time.perf_counter()
        decision_time = pd.Timestamp(decision_time)
        decision_time = decision_time.tz_localize("UTC") if decision_time.tzinfo is None else decision_time.tz_convert("UTC")
        window_start = decision_time - pd.Timedelta(minutes=self.config.graph_window_minutes)
        current = window.sort_values(["available_time", "timestamp"]).groupby("symbol").tail(1).set_index("symbol").reindex(self.symbols)
        reversal = -zscore(current["ret_5m"].to_numpy(dtype=np.float32))
        flow_column = "signed_dollar_flow" if "signed_dollar_flow" in current.columns else "signed_dollar_flow_proxy"
        signed_flow = zscore(current[flow_column].to_numpy(dtype=np.float32))
        edge_records, node_records, snapshot_records, adjacencies = [], [], [], []
        for layer in self.layers:
            vectors, window_points, used_columns = trajectory(window, layer, self.symbols, self.config.minimum_window_points, return_corr_benchmarks=self.config.return_corr_benchmarks, return_corr_min_benchmark_points=self.config.return_corr_min_benchmark_points, return_corr_ridge=self.config.return_corr_ridge)
            if vectors is None:
                continue
            if layer.transform.startswith("return_corr_"):
                adjacency, kept_edges, lsh_bits = reciprocal_correlation_graph(vectors, self.config)
            else:
                adjacency, kept_edges, lsh_bits = reciprocal_lsh_graph(vectors, self.config)
            adjacencies.append(adjacency)
            degree = np.diff(adjacency.indptr).astype(np.int16)
            strength = np.asarray(adjacency.sum(axis=1)).ravel().astype(np.float32)
            core = zscore(strength)
            neighbor_reversal = neighbor(adjacency, reversal).astype(np.float32)
            neighbor_flow = neighbor(adjacency, signed_flow).astype(np.float32)
            for left, right, weight, left_rank, right_rank in kept_edges:
                edge_records.append({"decision_time": decision_time, "window_start": window_start, "window_end": decision_time, "layer_id": np.int16(layer.layer_id), "src_id": np.int32(self.symbol_ids[left]), "dst_id": np.int32(self.symbol_ids[right]), "weight": np.float32(weight), "src_rank": np.int16(left_rank), "dst_rank": np.int16(right_rank), "directed": layer.directed, "lag_bars": np.int16(layer.lag_bars), "window_points": np.int16(window_points), "vector_dimension": np.int16(vectors.shape[1])})
            for index, symbol_id in enumerate(self.symbol_ids):
                node_records.append({"decision_time": decision_time, "layer_id": np.int16(layer.layer_id), "symbol_id": np.int32(symbol_id), "degree": degree[index], "strength": strength[index], "core_z": np.float32(core[index]), "neighbor_reversal": neighbor_reversal[index], "neighbor_signed_flow": neighbor_flow[index], "layer_participation": np.float32(degree[index] > 0)})
            snapshot_records.append({"decision_time": decision_time, "window_start": window_start, "window_end": decision_time, "layer_id": np.int16(layer.layer_id), "universe_count": np.int32(len(self.symbols)), "active_nodes": np.int32((degree > 0).sum()), "edge_count": np.int32(len(kept_edges)), "mean_degree": np.float32(degree.mean()), "mean_strength": np.float32(strength.mean()), "window_points": np.int16(window_points), "vector_dimension": np.int16(vectors.shape[1]), "lsh_bits": np.int16(lsh_bits), "used_columns": ",".join(used_columns), "transform": layer.transform})
        if self.include_multiplex and adjacencies:
            multiplex = sum(adjacencies) / np.float32(len(adjacencies))
            degree = np.diff(multiplex.indptr).astype(np.int16)
            strength = np.asarray(multiplex.sum(axis=1)).ravel().astype(np.float32)
            participation = np.mean(np.vstack([np.diff(item.indptr) > 0 for item in adjacencies]), axis=0).astype(np.float32)
            core = zscore(strength)
            multiplex_reversal = neighbor(multiplex, reversal)
            multiplex_flow = neighbor(multiplex, signed_flow)
            for index, symbol_id in enumerate(self.symbol_ids):
                node_records.append({"decision_time": decision_time, "layer_id": np.int16(0), "symbol_id": np.int32(symbol_id), "degree": degree[index], "strength": strength[index], "core_z": np.float32(core[index]), "neighbor_reversal": np.float32(multiplex_reversal[index]), "neighbor_signed_flow": np.float32(multiplex_flow[index]), "layer_participation": participation[index]})
        elapsed_ms = np.int32((time.perf_counter() - started) * 1000)
        for record in snapshot_records:
            record["elapsed_ms_total_snapshot"] = elapsed_ms
        return SnapshotProducts(pd.DataFrame(edge_records), pd.DataFrame(node_records), pd.DataFrame(snapshot_records))
