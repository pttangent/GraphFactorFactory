from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import yaml


@dataclass(frozen=True)
class BuildConfig:
    frequency: str = "5min"
    market_timezone: str = "America/New_York"
    market_open: str = "09:30"
    market_close: str = "16:00"
    graph_window_minutes: int = 60
    graph_step_minutes: int = 15
    top_k: int = 8
    degree_cap: int = 6
    minimum_similarity: float = 0.10
    minimum_window_points: int = 8
    horizons_minutes: tuple[int, ...] = (5, 15, 30, 60, 120)
    store_labels: bool = True
    store_qlib_cache: bool = False
    split_csv_path: str | None = None
    parquet_compression: str = "zstd"
    parquet_compression_level: int = 6

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BuildConfig":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        if "horizons_minutes" in raw:
            raw["horizons_minutes"] = tuple(int(value) for value in raw["horizons_minutes"])
        return cls(**raw)

    def to_dict(self) -> dict:
        result = asdict(self)
        result["horizons_minutes"] = list(self.horizons_minutes)
        return result

    @property
    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()
