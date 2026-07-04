from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import pandas as pd

from graphfactorfactory.domain.records import SourceFingerprint


class NodeFactorSource(ABC):
    @abstractmethod
    def fingerprint(self) -> SourceFingerprint: ...

    @abstractmethod
    def available_dates(self) -> list[str]: ...

    @abstractmethod
    def load_date(self, trade_date: str, columns: Sequence[str] | None = None) -> pd.DataFrame: ...

    @abstractmethod
    def numeric_feature_columns(self) -> list[str]: ...
