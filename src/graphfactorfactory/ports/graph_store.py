from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class GraphStore(ABC):
    @abstractmethod
    def initialize_dimensions(self, symbols: pd.DataFrame, layers: pd.DataFrame) -> None: ...

    @abstractmethod
    def open_day(self, trade_date: str): ...

    @abstractmethod
    def finalize_catalog(self) -> Path: ...
