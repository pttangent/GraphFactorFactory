from __future__ import annotations

from pathlib import Path

import pandas as pd

from graphfactorfactory.application.pit import build_point_in_time_panel, filter_regular_session
from graphfactorfactory.domain.config import BuildConfig
from graphfactorfactory.infrastructure.nodefactorfactory import ParquetNodeFactorSource

try:
    from qlib.data.dataset.loader import DataLoader
except ImportError:
    class DataLoader:
        pass


class CanonicalQlibDataLoader(DataLoader):
    """On-demand Qlib view over external node factors and canonical graph data."""

    def __init__(self, node_factor_path: str, graph_store_root: str, config_path: str, include_graph: bool = True):
        self.source = ParquetNodeFactorSource(node_factor_path)
        self.root = Path(graph_store_root).expanduser().resolve()
        self.config = BuildConfig.from_yaml(config_path)
        self.include_graph = include_graph

    def _load_dates(self, start_time, end_time) -> list[str]:
        available = self.source.available_dates()
        if start_time is None and end_time is None:
            return available
        start = pd.Timestamp(start_time).date() if start_time is not None else None
        end = pd.Timestamp(end_time).date() if end_time is not None else None
        return [value for value in available if (start is None or pd.Timestamp(value).date() >= start) and (end is None or pd.Timestamp(value).date() <= end)]

    @staticmethod
    def _label_columns(labels: pd.DataFrame) -> list[str]:
        return [column for column in labels if column.startswith("label_") and not column.startswith("label_exit_time") and not column.startswith("label_exit_price") and column not in {"label_entry_time", "label_entry_price"}]

    def load(self, instruments=None, start_time=None, end_time=None) -> pd.DataFrame:
        frames = []
        numeric_features = self.source.numeric_feature_columns()
        symbols = pd.read_parquet(self.root / "dimensions" / "symbols.parquet")
        symbol_to_id = dict(zip(symbols["symbol"], symbols["symbol_id"]))
        layers = pd.read_parquet(self.root / "dimensions" / "layers.parquet")[["layer_id", "name"]]
        for trade_date in self._load_dates(start_time, end_time):
            events = filter_regular_session(self.source.load_date(trade_date), self.config)
            panel = build_point_in_time_panel(events, self.config)
            columns = [column for column in numeric_features if column in panel.columns]
            base = panel[["decision_time", "symbol", *columns]].copy()
            base["symbol_id"] = base["symbol"].map(symbol_to_id)
            labels_path = self.root / "canonical" / f"date={trade_date}" / "labels.parquet"
            if labels_path.exists():
                labels = pd.read_parquet(labels_path)
                label_columns = self._label_columns(labels)
                base = base.merge(labels[["decision_time", "symbol_id", *label_columns]], on=["decision_time", "symbol_id"], how="left", validate="one_to_one")
            else:
                label_columns = []
            graph_columns: list[str] = []
            if self.include_graph:
                node_path = self.root / "canonical" / f"date={trade_date}" / "node_features.parquet"
                if node_path.exists():
                    node = pd.read_parquet(node_path).merge(layers, on="layer_id", how="left", validate="many_to_one")
                    metrics = ["degree", "strength", "core_z", "neighbor_reversal", "neighbor_signed_flow", "layer_participation"]
                    wide = node.pivot_table(index=["decision_time", "symbol_id"], columns="name", values=metrics, aggfunc="last")
                    wide.columns = [f"graph__{layer}__{metric}" for metric, layer in wide.columns]
                    wide = wide.reset_index()
                    graph_columns = [column for column in wide if column not in {"decision_time", "symbol_id"}]
                    base = base.merge(wide, on=["decision_time", "symbol_id"], how="left", validate="one_to_one")
            base = base.drop(columns="symbol_id").rename(columns={"decision_time": "datetime", "symbol": "instrument"})
            feature_columns = columns + graph_columns
            selected = base[["datetime", "instrument", *feature_columns, *label_columns]].set_index(["datetime", "instrument"]).sort_index()
            selected.columns = pd.MultiIndex.from_tuples([("feature", column) if column in feature_columns else ("label", column) for column in selected.columns])
            frames.append(selected)
        if not frames:
            return pd.DataFrame()
        result = pd.concat(frames).sort_index()
        if instruments is not None and not isinstance(instruments, str):
            wanted = set(map(str, instruments))
            result = result[result.index.get_level_values("instrument").astype(str).isin(wanted)]
        if start_time is not None:
            result = result[result.index.get_level_values("datetime") >= pd.Timestamp(start_time, tz="UTC")]
        if end_time is not None:
            result = result[result.index.get_level_values("datetime") <= pd.Timestamp(end_time, tz="UTC")]
        return result


def materialize_qlib_cache(loader: CanonicalQlibDataLoader, output_path: str | Path, start_time=None, end_time=None) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    loader.load(start_time=start_time, end_time=end_time).to_parquet(output, compression="zstd")
    return output
