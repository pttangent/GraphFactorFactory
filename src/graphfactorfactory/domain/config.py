from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class GraphParameters:
    top_k: int
    degree_cap: int
    minimum_similarity: float

    def validate(self) -> "GraphParameters":
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")
        if self.degree_cap < 1:
            raise ValueError("degree_cap must be >= 1")
        if self.degree_cap > self.top_k:
            raise ValueError("degree_cap cannot exceed top_k")
        if not -1.0 <= self.minimum_similarity <= 1.0:
            raise ValueError("minimum_similarity must be in [-1, 1]")
        return self


@dataclass(frozen=True)
class BuildConfig:
    frequency: str = "1min"
    market_timezone: str = "America/New_York"
    market_open: str = "09:30"
    market_close: str = "16:00"
    graph_window_minutes: int = 30
    graph_step_minutes: int = 1
    top_k: int = 8
    degree_cap: int = 6
    minimum_similarity: float = 0.10
    minimum_window_points: int = 3
    horizons_minutes: tuple[int, ...] = (5, 15, 30, 60, 120)
    store_labels: bool = True
    store_qlib_cache: bool = False
    split_csv_path: str | None = None
    parquet_compression: str = "zstd"
    parquet_compression_level: int = 6
    return_corr_benchmarks: tuple[str, ...] = ("SPY", "QQQ", "IWM")
    return_corr_min_benchmark_points: int = 8
    return_corr_ridge: float = 1e-6
    graph_parameter_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BuildConfig":
        config_path = Path(path).expanduser().resolve()
        raw = yaml.safe_load(config_path.read_text()) or {}
        if "horizons_minutes" in raw:
            raw["horizons_minutes"] = tuple(int(value) for value in raw["horizons_minutes"])
        if "return_corr_benchmarks" in raw:
            raw["return_corr_benchmarks"] = tuple(str(value).upper() for value in raw["return_corr_benchmarks"])
        overrides = raw.get("graph_parameter_overrides") or {}
        if not isinstance(overrides, dict):
            raise ValueError("graph_parameter_overrides must be a mapping")
        raw["graph_parameter_overrides"] = {
            str(key): dict(value or {}) for key, value in overrides.items()
        }
        split_path = raw.get("split_csv_path")
        if split_path:
            candidate = Path(split_path).expanduser()
            if not candidate.is_absolute():
                repo_relative = (config_path.parent.parent / candidate).resolve()
                config_relative = (config_path.parent / candidate).resolve()
                candidate = repo_relative if repo_relative.exists() else config_relative
            raw["split_csv_path"] = str(candidate)
        config = cls(**raw)
        config.base_graph_parameters.validate()
        return config

    @property
    def base_graph_parameters(self) -> GraphParameters:
        return GraphParameters(
            top_k=int(self.top_k),
            degree_cap=int(self.degree_cap),
            minimum_similarity=float(self.minimum_similarity),
        )

    def graph_parameters_for(self, *, layer_name: str, family: str, lookback_minutes: int) -> GraphParameters:
        """Resolve graph parameters from broad to specific.

        Supported override keys, in precedence order:
        ``family:<family>``, ``layer:<layer_name>``, and
        ``scale:<layer_name>:<lookback_minutes>``.
        """
        values = asdict(self.base_graph_parameters)
        keys = (
            f"family:{family}",
            f"layer:{layer_name}",
            f"scale:{layer_name}:{int(lookback_minutes)}",
        )
        for key in keys:
            override = self.graph_parameter_overrides.get(key, {})
            unknown = set(override).difference(values)
            if unknown:
                raise ValueError(f"Unknown graph parameter(s) for {key}: {sorted(unknown)}")
            values.update(override)
        return GraphParameters(
            top_k=int(values["top_k"]),
            degree_cap=int(values["degree_cap"]),
            minimum_similarity=float(values["minimum_similarity"]),
        ).validate()

    def with_graph_parameters(self, parameters: GraphParameters) -> "BuildConfig":
        """Return a lightweight config view accepted by graph constructors."""
        parameters.validate()
        return replace(
            self,
            top_k=parameters.top_k,
            degree_cap=parameters.degree_cap,
            minimum_similarity=parameters.minimum_similarity,
        )

    def to_dict(self) -> dict:
        result = asdict(self)
        result["horizons_minutes"] = list(self.horizons_minutes)
        result["return_corr_benchmarks"] = list(self.return_corr_benchmarks)
        return result

    @property
    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()
