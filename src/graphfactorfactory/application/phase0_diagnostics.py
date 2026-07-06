from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


GROUP_KEYS = ["decision_time", "layer_id", "lookback_minutes"]


@dataclass(frozen=True)
class StrongEdgeThresholds:
    table: pd.DataFrame

    @classmethod
    def fit(cls, edges: pd.DataFrame, quantiles: tuple[float, ...] = (0.80, 0.90, 0.95)) -> "StrongEdgeThresholds":
        required = {"layer_id", "lookback_minutes", "weight"}
        missing = required.difference(edges.columns)
        if missing:
            raise ValueError(f"Missing edge columns: {sorted(missing)}")
        rows = []
        for (layer_id, lookback), group in edges.groupby(["layer_id", "lookback_minutes"], sort=True):
            weights = group["weight"].dropna().astype(float)
            if weights.empty:
                continue
            row = {"layer_id": int(layer_id), "lookback_minutes": int(lookback)}
            for q in quantiles:
                row[f"q{int(round(q * 100))}"] = float(weights.quantile(q))
            rows.append(row)
        return cls(pd.DataFrame(rows))


def _edge_pairs(frame: pd.DataFrame) -> set[tuple[int, int]]:
    if frame.empty:
        return set()
    left = np.minimum(frame["src_id"].to_numpy(), frame["dst_id"].to_numpy())
    right = np.maximum(frame["src_id"].to_numpy(), frame["dst_id"].to_numpy())
    return set(zip(left.astype(int), right.astype(int)))


def _community_metrics(frame: pd.DataFrame, universe_count: int, resolution: float = 1.0) -> dict[str, float]:
    if frame.empty:
        return {
            "component_count": 0,
            "giant_component_coverage": 0.0,
            "community_count": 0,
            "community_hhi": np.nan,
            "top_community_coverage": 0.0,
            "weighted_modularity": np.nan,
            "cpm_quality": np.nan,
            "within_community_weight_ratio": np.nan,
        }
    nodes = np.unique(np.concatenate([frame["src_id"].to_numpy(), frame["dst_id"].to_numpy()])).astype(int)
    node_to_local = {node: i for i, node in enumerate(nodes.tolist())}
    rows = frame["src_id"].map(node_to_local).to_numpy()
    cols = frame["dst_id"].map(node_to_local).to_numpy()
    weights = frame["weight"].astype(float).to_numpy()
    adjacency = csr_matrix(
        (np.concatenate([weights, weights]), (np.concatenate([rows, cols]), np.concatenate([cols, rows]))),
        shape=(len(nodes), len(nodes)),
    )
    component_count, labels = connected_components(adjacency, directed=False)
    component_sizes = np.bincount(labels)
    result = {
        "component_count": int(component_count),
        "giant_component_coverage": float(component_sizes.max() / universe_count),
        "community_count": int(component_count),
        "community_hhi": float(np.square(component_sizes / max(1, component_sizes.sum())).sum()),
        "top_community_coverage": float(component_sizes.max() / universe_count),
        "weighted_modularity": np.nan,
        "cpm_quality": np.nan,
        "within_community_weight_ratio": np.nan,
    }
    try:
        import igraph as ig
        import leidenalg

        graph = ig.Graph(n=len(nodes), edges=list(zip(rows.tolist(), cols.tolist())), directed=False)
        graph.es["weight"] = weights.tolist()
        partition = leidenalg.find_partition(
            graph,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=resolution,
            seed=20260704,
        )
        membership = np.asarray(partition.membership, dtype=int)
        sizes = np.bincount(membership)
        same = membership[rows] == membership[cols]
        total_weight = float(weights.sum())
        within_weight = float(weights[same].sum())
        cpm = leidenalg.CPMVertexPartition(
            graph,
            initial_membership=partition.membership,
            weights="weight",
            resolution_parameter=resolution,
        )
        result.update({
            "community_count": int(len(sizes)),
            "community_hhi": float(np.square(sizes / max(1, sizes.sum())).sum()),
            "top_community_coverage": float(sizes.max() / universe_count),
            "weighted_modularity": float(graph.modularity(partition.membership, weights="weight")),
            "cpm_quality": float(cpm.quality()),
            "within_community_weight_ratio": float(within_weight / total_weight if total_weight > 0 else np.nan),
        })
    except ImportError:
        pass
    return result


def compute_snapshot_diagnostics(
    edges: pd.DataFrame,
    *,
    universe_count: int,
    thresholds: StrongEdgeThresholds | None = None,
    community_resolution: float = 1.0,
) -> pd.DataFrame:
    required = set(GROUP_KEYS + ["src_id", "dst_id", "weight"])
    missing = required.difference(edges.columns)
    if missing:
        raise ValueError(f"Missing edge columns: {sorted(missing)}")
    threshold_map = {}
    if thresholds is not None and not thresholds.table.empty:
        threshold_map = thresholds.table.set_index(["layer_id", "lookback_minutes"]).to_dict("index")
    rows = []
    for keys, group in edges.groupby(GROUP_KEYS, sort=True):
        decision_time, layer_id, lookback = keys
        weights = group["weight"].astype(float)
        active_nodes = np.unique(np.concatenate([group["src_id"].to_numpy(), group["dst_id"].to_numpy()]))
        row = {
            "decision_time": pd.Timestamp(decision_time),
            "layer_id": int(layer_id),
            "lookback_minutes": int(lookback),
            "edge_count": int(len(group)),
            "active_nodes": int(len(active_nodes)),
            "node_coverage": float(len(active_nodes) / universe_count),
            "isolated_node_ratio": float(1.0 - len(active_nodes) / universe_count),
            "weight_mean": float(weights.mean()),
            "weight_p50": float(weights.quantile(0.50)),
            "weight_p75": float(weights.quantile(0.75)),
            "weight_p90": float(weights.quantile(0.90)),
            "weight_p95": float(weights.quantile(0.95)),
            "weight_p99": float(weights.quantile(0.99)),
            "top_1pct_mean_weight": float(weights.nlargest(max(1, int(np.ceil(len(weights) * 0.01)))).mean()),
        }
        local_p95 = row["weight_p95"]
        total_weight = float(weights.sum())
        row["tail_mass_95"] = float(weights[weights >= local_p95].sum() / total_weight if total_weight > 0 else np.nan)
        threshold_values = threshold_map.get((int(layer_id), int(lookback)), {})
        for q in (80, 90, 95):
            threshold = threshold_values.get(f"q{q}", np.nan)
            if np.isnan(threshold):
                row[f"strong_edge_ratio_q{q}"] = np.nan
                row[f"strong_node_coverage_q{q}"] = np.nan
                continue
            strong = group[weights.to_numpy() >= float(threshold)]
            strong_nodes = np.unique(np.concatenate([strong["src_id"].to_numpy(), strong["dst_id"].to_numpy()])) if not strong.empty else np.asarray([])
            row[f"strong_edge_ratio_q{q}"] = float(len(strong) / len(group))
            row[f"strong_node_coverage_q{q}"] = float(len(strong_nodes) / universe_count)
        row.update(_community_metrics(group, universe_count, community_resolution))
        rows.append(row)
    result = pd.DataFrame(rows)
    if not result.empty:
        result["trade_date"] = pd.to_datetime(result["decision_time"], utc=True).dt.date.astype(str)
    return result


def compute_temporal_diagnostics(edges: pd.DataFrame, thresholds: StrongEdgeThresholds | None = None) -> pd.DataFrame:
    threshold_map = {}
    if thresholds is not None and not thresholds.table.empty:
        threshold_map = thresholds.table.set_index(["layer_id", "lookback_minutes"]).to_dict("index")
    rows = []
    for (layer_id, lookback), scale in edges.groupby(["layer_id", "lookback_minutes"], sort=True):
        previous_all: set[tuple[int, int]] | None = None
        previous_strong: set[tuple[int, int]] | None = None
        threshold = threshold_map.get((int(layer_id), int(lookback)), {}).get("q90", np.nan)
        for decision_time, frame in scale.groupby("decision_time", sort=True):
            current_all = _edge_pairs(frame)
            current_strong = _edge_pairs(frame[frame["weight"] >= threshold]) if not np.isnan(threshold) else set()
            row = {
                "decision_time": pd.Timestamp(decision_time),
                "layer_id": int(layer_id),
                "lookback_minutes": int(lookback),
            }
            for prefix, current, previous in (("all", current_all, previous_all), ("strong", current_strong, previous_strong)):
                if previous is None:
                    row[f"{prefix}_birth_rate"] = np.nan
                    row[f"{prefix}_death_rate"] = np.nan
                    row[f"{prefix}_persistence"] = np.nan
                    row[f"{prefix}_jaccard"] = np.nan
                else:
                    intersection = current & previous
                    union = current | previous
                    row[f"{prefix}_birth_rate"] = float(len(current - previous) / max(1, len(current)))
                    row[f"{prefix}_death_rate"] = float(len(previous - current) / max(1, len(previous)))
                    row[f"{prefix}_persistence"] = float(len(intersection) / max(1, len(previous)))
                    row[f"{prefix}_jaccard"] = float(len(intersection) / max(1, len(union)))
            rows.append(row)
            previous_all, previous_strong = current_all, current_strong
    result = pd.DataFrame(rows)
    if not result.empty:
        result["trade_date"] = pd.to_datetime(result["decision_time"], utc=True).dt.date.astype(str)
    return result


def compute_resonance_diagnostics(edges: pd.DataFrame, layers: pd.DataFrame) -> pd.DataFrame:
    family_map = layers.set_index("layer_id")["family"].astype(str).to_dict()
    rows = []
    for decision_time, frame in edges.groupby("decision_time", sort=True):
        node_layers: dict[int, set[int]] = {}
        node_families: dict[int, set[str]] = {}
        edge_layers: dict[tuple[int, int], set[int]] = {}
        edge_families: dict[tuple[int, int], set[str]] = {}
        for record in frame.itertuples(index=False):
            layer_id = int(record.layer_id)
            if layer_id == 0:
                continue
            family = family_map.get(layer_id, "unknown")
            pair = tuple(sorted((int(record.src_id), int(record.dst_id))))
            edge_layers.setdefault(pair, set()).add(layer_id)
            edge_families.setdefault(pair, set()).add(family)
            for node in pair:
                node_layers.setdefault(node, set()).add(layer_id)
                node_families.setdefault(node, set()).add(family)
        layer_counts = np.asarray([len(value) for value in edge_layers.values()], dtype=int)
        family_counts = np.asarray([len(value) for value in edge_families.values()], dtype=int)
        node_layer_counts = np.asarray([len(value) for value in node_layers.values()], dtype=int)
        node_family_counts = np.asarray([len(value) for value in node_families.values()], dtype=int)
        rows.append({
            "decision_time": pd.Timestamp(decision_time),
            "resonant_edges_ge2_layers": int((layer_counts >= 2).sum()),
            "resonant_edges_ge3_layers": int((layer_counts >= 3).sum()),
            "resonant_edges_ge2_families": int((family_counts >= 2).sum()),
            "resonant_edges_ge3_families": int((family_counts >= 3).sum()),
            "core_nodes_ge5_layers": int((node_layer_counts >= 5).sum()),
            "core_nodes_ge3_families": int((node_family_counts >= 3).sum()),
            "mean_edge_layer_support": float(layer_counts.mean()) if len(layer_counts) else 0.0,
            "mean_node_layer_participation": float(node_layer_counts.mean()) if len(node_layer_counts) else 0.0,
        })
    result = pd.DataFrame(rows)
    if not result.empty:
        result["trade_date"] = pd.to_datetime(result["decision_time"], utc=True).dt.date.astype(str)
    return result


def aggregate_daily_market_diagnostics(
    snapshot: pd.DataFrame,
    temporal: pd.DataFrame,
    resonance: pd.DataFrame,
) -> pd.DataFrame:
    snapshot_metrics = [
        "edge_count", "node_coverage", "strong_edge_ratio_q90", "strong_node_coverage_q90",
        "weight_p90", "weight_p95", "weight_p99", "top_1pct_mean_weight", "tail_mass_95",
        "giant_component_coverage", "community_hhi", "top_community_coverage",
        "weighted_modularity", "cpm_quality", "within_community_weight_ratio",
    ]
    temporal_metrics = ["all_birth_rate", "all_death_rate", "all_persistence", "strong_birth_rate", "strong_death_rate", "strong_persistence"]
    resonance_metrics = [column for column in resonance.columns if column not in {"decision_time", "trade_date"}]
    daily_snapshot = snapshot.groupby("trade_date")[snapshot_metrics].mean(numeric_only=True)
    daily_temporal = temporal.groupby("trade_date")[temporal_metrics].mean(numeric_only=True)
    daily_resonance = resonance.groupby("trade_date")[resonance_metrics].mean(numeric_only=True)
    daily = daily_snapshot.join(daily_temporal, how="outer").join(daily_resonance, how="outer").reset_index()
    return daily


def load_canonical_edges(root: str | Path, dates: Iterable[str] | None = None) -> pd.DataFrame:
    root = Path(root).expanduser().resolve()
    selected = set(dates or [])
    frames = []
    for day in sorted((root / "canonical").glob("date=*")):
        trade_date = day.name.split("=", 1)[1]
        if selected and trade_date not in selected:
            continue
        path = day / "edges.parquet"
        if path.exists():
            frame = pd.read_parquet(path)
            frame["trade_date"] = trade_date
            frames.append(frame)
    if not frames:
        raise FileNotFoundError("No canonical edges found")
    return pd.concat(frames, ignore_index=True)
