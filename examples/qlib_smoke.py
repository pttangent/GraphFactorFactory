from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

from graphfactorfactory.infrastructure.qlib import CanonicalQlibDataLoader


def run(node_factors: str, graph_store: str, config: str, start: str, end: str):
    loader = CanonicalQlibDataLoader(node_factors, graph_store, config)
    handler = DataHandlerLP(
        instruments=None,
        start_time=start,
        end_time=end,
        data_loader=loader,
        infer_processors=[{"class": "Fillna", "kwargs": {"fields_group": "feature"}}, {"class": "CSZScoreNorm", "kwargs": {"fields_group": "feature"}}],
        learn_processors=[{"class": "DropnaLabel"}],
    )
    dataset = DatasetH(handler=handler, segments={"test": (start, end)})
    return dataset.prepare("test", col_set=["feature", "label"], data_key="learn")
