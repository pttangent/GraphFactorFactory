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
    return_corr_benchmarks: tuple[str, ...] = ("SPY", "QQQ", "IWM")
    return_corr_min_benchmark_points: int = 8
    return_corr_ridge: float = 1e-6

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BuildConfig":
        config_path = Path(path).expanduser().resolve()
        raw = yaml.safe_load(config_path.read_text()) or {}
        if "horizons_minutes" in raw:
            raw["horizons_minutes"] = tuple(int(value) for value in raw["horizons_minutes"])
        if "return_corr_benchmarks" in raw:
            raw["return_corr_benchmarks"] = tuple(str(value).upper() for value in raw["return_corr_benchmarks"])
        split_path = raw.get("split_csv_path")
        if split_path:
            candidate = Path(split_path).expanduser()
            if not candidate.is_absolute():
                repo_relative = (config_path.parent.parent / candidate).resolve()
                config_relative = (config_path.parent / candidate).resolve()
                candidate = repo_relative if repo_relative.exists() else config_relative
            raw["split_csv_path"] = str(candidate)
        return cls(**raw)

    def to_dict(self) -> dict:
        result = asdict(self)
        result["horizons_minutes"] = list(self.horizons_minutes)
        result["return_corr_benchmarks"] = list(self.return_corr_benchmarks)
        return result

    @property
    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()
