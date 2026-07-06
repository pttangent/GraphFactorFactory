from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import pandas as pd

from graphfactorfactory.application.correlation import reciprocal_correlation_graph
from graphfactorfactory.application.lsh import reciprocal_lsh_graph
from graphfactorfactory.application.math_utils import neighbor, trajectory, zscore
from graphfactorfactory.domain.config import BuildConfig
from graphfactorfactory.domain.layers import LAYERS, LayerDefinition, layer_scale_definitions


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
        self.layer_scales = layer_scale_definitions(self.layers)
        self.include_multiplex = bool(include_multiplex)

    def _scale_is_due(self, decision_time: pd.Timestamp, step_minutes: int) -> bool:
        if step_minutes <= 1:
            return True
        local = decision_time.tz_convert(self.config.market_timezone)
        market_open = pd.Timestamp(f"{local.date()} {self.config.market_open}", tz=self.config.market_timezone)
        elapsed = int((local - market_open).total_seconds() // 60)
        return elapsed >= 0 and elapsed % step_minutes == 0

    @staticmethod
    def _weight_diagnostics(kept_edges: list[tuple]) -> dict[str, np.float32]:
        if not kept_edges:
            return {
                "weight_p50": np.float32(np.nan),
                "weight_p75": np.float32(np.nan),
                "weight_p90": np.float32(np.nan),
                "weight_p95": np.float32(np.nan),
                "weight_p99": np.float32(np.nan),
                "top_1pct_mean_weight": np.float32(np.nan),
                "tail_mass_95": np.float32(np.nan),
            }
        weights = np.asarray([edge[2] for edge in kept_edges], dtype=np.float32)
        quantiles = np.quantile(weights, [0.50, 0.75, 0.90, 0.95, 0.99])
        p95 = float(quantiles[3])
        top_count = max(1, int(np.ceil(len(weights) * 0.01)))
        top_weights = np.partition(weights, len(weights) - top_count)[-top_count:]
        total_weight = float(weights.sum())
        tail_weight = float(weights[weights >= p95].sum())
        return {
            "weight_p50": np.float32(quantiles[0]),
            "weight_p75": np.float32(quantiles[1]),
            "weight_p90": np.float32(quantiles[2]),
            "weight_p95": np.float32(quantiles[3]),
            "weight_p99": np.float32(quantiles[4]),
            "top_1pct_mean_weight": np.float32(top_weights.mean()),
            "tail_mass_95": np.float32(tail_weight / total_weight if total_weight > 0 else np.nan),
        }

    def build_snapshot(self, window: pd.DataFrame, decision_time) -> SnapshotProducts:
        started = time.perf_counter()
        decision_time = pd.Timestamp(decision_time)
        decision_time = decision_time.tz_localize("UTC") if decision_time.tzinfo is None else decision_time.tz_convert("UTC")
        current = window.sort_values(["available_time", "timestamp"]).groupby("symbol").tail(1).set_index("symbol").reindex(self.symbols)
        return_column = "ret_1m" if "ret_1m" in current.columns else "log_ret_1m"
        reversal = -zscore(current[return_column].to_numpy(dtype=np.float32))
        flow_column = "signed_dollar_flow" if "signed_dollar_flow" in current.columns else "signed_dollar_flow_proxy"
        signed_flow = zscore(current[flow_column].to_numpy(dtype=np.float32))
        edge_records, node_records, snapshot_records = [], [], []
        structural_adjacencies = []

        for scale in self.layer_scales:
            if not self._scale_is_due(decision_time, scale.decision_step_minutes):
                continue
            layer = scale.layer
            parameters = self.config.graph_parameters_for(
                layer_name=layer.name,
                family=layer.family,
                lookback_minutes=scale.lookback_minutes,
            )
            graph_config = self.config.with_graph_parameters(parameters)
            window_start = decision_time - pd.Timedelta(minutes=scale.lookback_minutes)
            scale_window = window[
                (window["available_time"] <= decision_time)
                & (window["timestamp"] <= decision_time)
                & (window["timestamp"] > window_start)
            ]
            vectors, window_points, used_columns = trajectory(
                scale_window,
                layer,
                self.symbols,
                scale.minimum_points,
                return_corr_benchmarks=self.config.return_corr_benchmarks,
                return_corr_min_benchmark_points=self.config.return_corr_min_benchmark_points,
                return_corr_ridge=self.config.return_corr_ridge,
            )
            if vectors is None:
                continue
            if layer.transform.startswith("return_corr_"):
                adjacency, kept_edges, lsh_bits = reciprocal_correlation_graph(vectors, graph_config)
            else:
                adjacency, kept_edges, lsh_bits = reciprocal_lsh_graph(vectors, graph_config)
            if scale.scale_role == "structural":
                structural_adjacencies.append(adjacency)
            degree = np.diff(adjacency.indptr).astype(np.int16)
            strength = np.asarray(adjacency.sum(axis=1)).ravel().astype(np.float32)
            core = zscore(strength)
            neighbor_reversal = neighbor(adjacency, reversal).astype(np.float32)
            neighbor_flow = neighbor(adjacency, signed_flow).astype(np.float32)
            common = {
                "decision_time": decision_time,
                "layer_id": np.int16(layer.layer_id),
                "lookback_minutes": np.int16(scale.lookback_minutes),
                "scale_role": scale.scale_role,
                "decision_step_minutes": np.int16(scale.decision_step_minutes),
                "top_k": np.int16(parameters.top_k),
                "degree_cap": np.int16(parameters.degree_cap),
                "minimum_similarity": np.float32(parameters.minimum_similarity),
            }
            for left, right, weight, left_rank, right_rank in kept_edges:
                edge_records.append({**common, "window_start": window_start, "window_end": decision_time, "src_id": np.int32(self.symbol_ids[left]), "dst_id": np.int32(self.symbol_ids[right]), "weight": np.float32(weight), "src_rank": np.int16(left_rank), "dst_rank": np.int16(right_rank), "directed": layer.directed, "lag_bars": np.int16(layer.lag_bars), "window_points": np.int16(window_points), "vector_dimension": np.int16(vectors.shape[1])})
            for index, symbol_id in enumerate(self.symbol_ids):
                node_records.append({**common, "symbol_id": np.int32(symbol_id), "degree": degree[index], "strength": strength[index], "core_z": np.float32(core[index]), "neighbor_reversal": neighbor_reversal[index], "neighbor_signed_flow": neighbor_flow[index], "layer_participation": np.float32(degree[index] > 0)})
            max_edges = max(1.0, len(self.symbols) * parameters.degree_cap / 2.0)
            diagnostics = self._weight_diagnostics(kept_edges)
            snapshot_records.append({
                **common,
                "window_start": window_start,
                "window_end": decision_time,
                "universe_count": np.int32(len(self.symbols)),
                "active_nodes": np.int32((degree > 0).sum()),
                "node_coverage": np.float32((degree > 0).mean()),
                "isolated_node_ratio": np.float32((degree == 0).mean()),
                "edge_count": np.int32(len(kept_edges)),
                "degree_cap_saturation": np.float32(len(kept_edges) / max_edges),
                "mean_degree": np.float32(degree.mean()),
                "mean_strength": np.float32(strength.mean()),
                "window_points": np.int16(window_points),
                "vector_dimension": np.int16(vectors.shape[1]),
                "lsh_bits": np.int16(lsh_bits),
                "used_columns": ",".join(used_columns),
                "transform": layer.transform,
                **diagnostics,
            })

        if self.include_multiplex and structural_adjacencies:
            multiplex = sum(structural_adjacencies) / np.float32(len(structural_adjacencies))
            degree = np.diff(multiplex.indptr).astype(np.int16)
            strength = np.asarray(multiplex.sum(axis=1)).ravel().astype(np.float32)
            participation = np.mean(np.vstack([np.diff(item.indptr) > 0 for item in structural_adjacencies]), axis=0).astype(np.float32)
            core = zscore(strength)
            multiplex_reversal = neighbor(multiplex, reversal)
            multiplex_flow = neighbor(multiplex, signed_flow)
            for index, symbol_id in enumerate(self.symbol_ids):
                node_records.append({"decision_time": decision_time, "layer_id": np.int16(0), "lookback_minutes": np.int16(30), "scale_role": "structural", "decision_step_minutes": np.int16(5), "top_k": np.int16(0), "degree_cap": np.int16(0), "minimum_similarity": np.float32(np.nan), "symbol_id": np.int32(symbol_id), "degree": degree[index], "strength": strength[index], "core_z": np.float32(core[index]), "neighbor_reversal": np.float32(multiplex_reversal[index]), "neighbor_signed_flow": np.float32(multiplex_flow[index]), "layer_participation": participation[index]})

        elapsed_ms = np.int32((time.perf_counter() - started) * 1000)
        for record in snapshot_records:
            record["elapsed_ms_total_snapshot"] = elapsed_ms
        return SnapshotProducts(pd.DataFrame(edge_records), pd.DataFrame(node_records), pd.DataFrame(snapshot_records))
