from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import sparse


@dataclass(frozen=True)
class BuildConfig:
    graph_window_minutes: int = 60
    top_k: int = 8
    degree_cap: int = 6


LAYERS = {
    "return_corr": ("ret_5m",),
    "volume_expansion": ("volume_z_30m", "dollar_volume_z_30m"),
    "trade_intensity": ("trade_count_z_30m", "avg_trade_size"),
    "signed_flow": ("volume_ofi_proxy", "count_ofi_proxy", "signed_dollar_flow"),
    "large_trade_flow": ("large_trade_ofi_proxy", "large_trade_dollar_share", "large_trade_count"),
    "odd_lot_activity": ("odd_lot_trade_share", "odd_lot_volume_share"),
    "block_activity": ("block_trade_share", "block_volume_share"),
    "off_exchange": ("off_exchange_dollar_share", "off_exchange_trade_share"),
    "venue_fragmentation": ("venue_fragmentation_proxy",),
    "price_impact": ("price_impact_proxy", "liquidity_impact_proxy"),
    "absorption": ("absorption_proxy", "flow_absorption_proxy"),
    "flow_return_alignment": ("flow_return_alignment",),
    "report_latency": ("avg_report_lag_ns", "max_report_lag_ns", "correction_excluded_share"),
}


def build_point_in_time_panel(events: pd.DataFrame, frequency: str = "5min", stale_tolerance: str = "15min") -> pd.DataFrame:
    frame = events.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["available_time"] = pd.to_datetime(frame["available_time"], utc=True)
    decisions = pd.date_range(frame.available_time.min().ceil(frequency), frame.available_time.max().floor(frequency), freq=frequency)
    symbols = pd.Index(sorted(frame.symbol.astype(str).unique()), name="symbol")
    grid = pd.MultiIndex.from_product([decisions, symbols], names=["decision_time", "symbol"]).to_frame(index=False)
    source = frame.rename(columns={"timestamp": "source_timestamp", "available_time": "source_available_time"}).sort_values(["source_available_time", "symbol"])
    panel = pd.merge_asof(grid.sort_values(["decision_time", "symbol"]), source, left_on="decision_time", right_on="source_available_time", by="symbol", direction="backward", tolerance=pd.Timedelta(stale_tolerance))
    panel = panel[panel.close.notna()].copy()
    audit_pit_frame(panel)
    return panel.sort_values(["decision_time", "symbol"]).reset_index(drop=True)


def build_forward_labels(panel: pd.DataFrame, horizons_minutes=(5, 15, 30, 60, 120), bar_minutes: int = 5) -> pd.DataFrame:
    base = panel[["decision_time", "symbol", "close"]].copy()
    base["decision_time"] = pd.to_datetime(base.decision_time, utc=True)
    px = base.rename(columns={"decision_time": "price_time", "close": "price"})
    out = base[["decision_time", "symbol"]].copy()
    entry_time = out.decision_time + pd.Timedelta(minutes=bar_minutes)
    entry = out.assign(price_time=entry_time).merge(px, on=["symbol", "price_time"], how="left", validate="many_to_one").price
    for horizon in horizons_minutes:
        exit_time = entry_time + pd.Timedelta(minutes=horizon)
        exit_px = out.assign(price_time=exit_time).merge(px, on=["symbol", "price_time"], how="left", validate="many_to_one").price
        out[f"label_{horizon}m"] = exit_px.to_numpy() / entry.to_numpy() - 1.0
    out["label_entry_time"] = entry_time
    return out


def audit_pit_frame(frame: pd.DataFrame) -> dict:
    decision = pd.to_datetime(frame.decision_time, utc=True)
    available = pd.to_datetime(frame.source_available_time, utc=True)
    timestamp = pd.to_datetime(frame.source_timestamp, utc=True)
    violations = int((available > decision).sum() + (timestamp > decision).sum())
    duplicates = int(frame.duplicated(["decision_time", "symbol"]).sum())
    if violations or duplicates:
        raise AssertionError({"availability_violations": violations, "duplicates": duplicates})
    return {"rows": len(frame), "availability_violations": 0, "duplicates": 0}


def _z(values):
    values = np.asarray(values, float)
    std = np.nanstd(values)
    return np.nan_to_num((values - np.nanmean(values)) / (std + 1e-12)) if std > 1e-12 else np.zeros_like(values)


def _trajectory(window, columns, universe):
    blocks = []
    for column in columns:
        if column not in window.columns:
            continue
        pivot = window.pivot_table(index="timestamp", columns="symbol", values=column, aggfunc="last").reindex(columns=universe)
        if len(pivot) < 8:
            continue
        values = pivot.to_numpy(float).T
        medians = np.nanmedian(values, axis=0)
        values = np.where(np.isfinite(values), values, np.where(np.isfinite(medians), medians, 0.0))
        values -= values.mean(axis=1, keepdims=True)
        scales = values.std(axis=1, keepdims=True)
        blocks.append(np.divide(values, scales, out=np.zeros_like(values), where=scales > 1e-12))
    return np.concatenate(blocks, axis=1) if blocks else None


def _lsh_graph(values, top_k, degree_cap):
    values = np.asarray(values, np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    values = np.divide(values, norms, out=np.zeros_like(values), where=norms > 1e-12)
    n, dim = values.shape
    bits = 8 if n >= 1000 else 6
    projections = np.random.default_rng(20260704 + dim).standard_normal((dim, bits), dtype=np.float32)
    signatures = ((values @ projections) > 0).astype(np.uint16)
    codes = np.sum(signatures * (1 << np.arange(bits, dtype=np.uint16)), axis=1)
    buckets = {}
    for index, code in enumerate(codes.tolist()):
        buckets.setdefault(code, []).append(index)
    directed = {}
    for index, code in enumerate(codes.tolist()):
        candidates = list(buckets[code])
        if len(candidates) < top_k + 1:
            for bit in range(bits):
                candidates.extend(buckets.get(code ^ (1 << bit), []))
        candidates = np.array(sorted(set(candidates)), dtype=int)
        candidates = candidates[candidates != index]
        if not len(candidates):
            continue
        similarity = values[candidates] @ values[index]
        chosen = np.argsort(similarity)[-min(top_k, len(similarity)):]
        for other, weight in zip(candidates[chosen], similarity[chosen]):
            if weight > 0.1:
                directed[(index, int(other))] = float(weight)
    pairs = [(i, j, (weight + directed[(j, i)]) / 2) for (i, j), weight in directed.items() if i < j and (j, i) in directed]
    per_node = {}
    for i, j, weight in pairs:
        per_node.setdefault(i, []).append((weight, i, j))
        per_node.setdefault(j, []).append((weight, i, j))
    keep = set()
    for candidates in per_node.values():
        for _, i, j in sorted(candidates, reverse=True)[:degree_cap]:
            keep.add((i, j))
    rows, columns, weights = [], [], []
    for i, j, weight in pairs:
        if (i, j) in keep:
            rows += [i, j]
            columns += [j, i]
            weights += [weight, weight]
    return sparse.csr_matrix((weights, (rows, columns)), shape=(n, n))


class MultilayerGraphBuilder:
    def __init__(self, config: BuildConfig, universe):
        self.config = config
        self.universe = list(map(str, universe))

    def build_snapshot(self, frame: pd.DataFrame, decision_time):
        decision_time = pd.Timestamp(decision_time)
        decision_time = decision_time.tz_localize("UTC") if decision_time.tzinfo is None else decision_time.tz_convert("UTC")
        data = frame.copy()
        data["timestamp"] = pd.to_datetime(data.timestamp, utc=True)
        data["available_time"] = pd.to_datetime(data.available_time, utc=True)
        window = data[(data.available_time <= decision_time) & (data.timestamp <= decision_time) & (data.timestamp > decision_time - pd.Timedelta(minutes=self.config.graph_window_minutes))]
        current = window.sort_values("available_time").groupby("symbol").tail(1).set_index("symbol").reindex(self.universe)
        reversal = -_z(current.ret_5m.to_numpy(float))
        signed_flow = _z(current.signed_dollar_flow.to_numpy(float))
        edge_rows, node_rows = [], []
        for layer, columns in LAYERS.items():
            trajectories = _trajectory(window, columns, self.universe)
            if trajectories is None:
                continue
            adjacency = _lsh_graph(trajectories, self.config.top_k, self.config.degree_cap)
            degree = np.diff(adjacency.indptr)
            strength = np.asarray(adjacency.sum(1)).ravel()
            denominator = np.where(strength > 0, strength, 1.0)
            neighbor_reversal = np.asarray(adjacency @ reversal).ravel() / denominator
            neighbor_flow = np.asarray(adjacency @ signed_flow).ravel() / denominator
            upper = sparse.triu(adjacency, k=1).tocoo()
            edge_rows.extend({"decision_time": decision_time, "layer": layer, "src": self.universe[i], "dst": self.universe[j], "weight": float(weight)} for i, j, weight in zip(upper.row, upper.col, upper.data))
            node_rows.extend({"decision_time": decision_time, "symbol": symbol, "layer": layer, "degree": int(degree[index]), "strength": float(strength[index]), "core_z": float(_z(strength)[index]), "neighbor_reversal": float(neighbor_reversal[index]), "neighbor_signed_flow": float(neighbor_flow[index])} for index, symbol in enumerate(self.universe))
        return pd.DataFrame(edge_rows), pd.DataFrame(node_rows)
