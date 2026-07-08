from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LayerCommunity:
    snapshot_time: Any
    layer_id: int
    layer_name: str
    community_id: int
    members: tuple[int, ...]
    modularity: float
    is_market_mode: bool = False
    parent_community_id: int | None = None
    leiden_seconds: float = 0.0


@dataclass(frozen=True)
class ThemeCandidate:
    theme_instance_id: str
    theme_path_id: str
    snapshot_time: Any
    members: tuple[int, ...]
    source_layers: tuple[str, ...]
    source_families: tuple[str, ...]
    consensus_score: float
    structure_score: float
    member_ratio: float
    is_market_mode: bool = False
    flow_support_score: float = 0.0
    stability_score: float = 0.0
    semantic_coherence_score: float = 0.0
    theme_quality_score: float = 0.0
    quality_breakdown: dict[str, float] = field(default_factory=dict)
    consensus_seconds: float = 0.0


@dataclass(frozen=True)
class LifecycleRecord:
    theme_path_id: str
    theme_instance_id: str
    timestamp: Any
    event_type: str
    status: str
    age_frames: int
    duration_minutes: int
    match_score: float
    member_retention: float
    previous_theme_instance_id: str | None = None
    parent_path_ids: tuple[str, ...] = ()
    child_path_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class SemanticLabel:
    theme_instance_id: str
    label_short: str
    label_long: str
    sector_summary: str
    industry_summary: str
    tags: tuple[str, ...]
    top_companies: tuple[str, ...]
    top_symbols: tuple[str, ...]
    sector_distribution: str
    industry_distribution: str
    market_cap_bucket: str
    semantic_coherence_score: float
    explanation: str
    semantic_method: str
    dictionary_version: str
