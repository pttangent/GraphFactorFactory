from pathlib import Path
import pandas as pd
from qlib.data.dataset.loader import DataLoader

IDENTITY_COLUMNS = {"decision_time", "symbol", "source_timestamp", "source_available_time", "trade_date", "timestamp", "available_time", "frequency", "factor_set_version", "factor_semantics_version", "trade_semantics_version"}


def discover_numeric_features(frame):
    return [column for column in frame.columns if column not in IDENTITY_COLUMNS and not column.startswith("label_") and pd.api.types.is_numeric_dtype(frame[column])]


def export_qlib_table(node_panel, labels, graph_wide, output_path):
    node_features = discover_numeric_features(node_panel)
    label_columns = [column for column in labels if column.startswith("label_") and column != "label_entry_time"]
    base = node_panel[["decision_time", "symbol", *node_features]].merge(labels[["decision_time", "symbol", *label_columns]], on=["decision_time", "symbol"], how="left", validate="one_to_one")
    graph_wide = graph_wide.copy()
    graph_wide["decision_time"] = pd.to_datetime(graph_wide.decision_time, utc=True)
    enriched = base.merge(graph_wide, on=["decision_time", "symbol"], how="left", validate="one_to_one")
    graph_features = [column for column in graph_wide if column not in {"decision_time", "symbol"}]
    features = node_features + graph_features
    data = enriched[["decision_time", "symbol", *features, *label_columns]].rename(columns={"decision_time": "datetime", "symbol": "instrument"})
    if data.duplicated(["datetime", "instrument"]).any():
        raise ValueError("Qlib export requires a unique key")
    data = data.set_index(["datetime", "instrument"]).sort_index()
    data.columns = pd.MultiIndex.from_tuples([("feature", column) if column in features else ("label", column) for column in data.columns])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(output_path, compression="zstd")
    return output_path


class GraphFactorDataLoader(DataLoader):
    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()

    def load(self, instruments=None, start_time=None, end_time=None):
        frame = pd.read_parquet(self.path)
        if not isinstance(frame.index, pd.MultiIndex):
            frame = frame.set_index(["datetime", "instrument"])
        frame.index = frame.index.set_names(["datetime", "instrument"])
        if instruments is not None and not isinstance(instruments, str):
            wanted = set(map(str, instruments))
            frame = frame[frame.index.get_level_values("instrument").astype(str).isin(wanted)]
        if start_time is not None:
            frame = frame[frame.index.get_level_values("datetime") >= pd.Timestamp(start_time)]
        if end_time is not None:
            frame = frame[frame.index.get_level_values("datetime") <= pd.Timestamp(end_time)]
        return frame.sort_index()
