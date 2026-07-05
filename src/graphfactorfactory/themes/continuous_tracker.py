from __future__ import annotations

from dataclasses import dataclass, field
from math import exp
from typing import Iterable, Sequence


def _members(theme) -> set[int]:
    values = getattr(theme, "members", getattr(theme, "core_members", ()))
    return {int(value) for value in values}


def _jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _containment(left: set[int], right: set[int]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


def _set_similarity(left: Iterable[str], right: Iterable[str]) -> float:
    return _jaccard(set(left), set(right))


@dataclass(frozen=True)
class ContinuousTrackingConfig:
    method: str = "hybrid_strict"
    jaccard_threshold: float = 0.35
    containment_threshold: float = 0.50
    hybrid_threshold: float = 0.50
    strict_threshold: float = 0.56
    member_evidence_gate: float = 0.20
    overnight_threshold_addition: float = 0.04
    one_to_one: bool = True


@dataclass
class ThemePath:
    lifecycle_id: str
    last_theme: object
    age_states: int = 1
    age_timestamps: int = 1
    first_timestamp: object | None = None
    last_timestamp: object | None = None
    transitions: list[dict] = field(default_factory=list)


class ContinuousThemeTracker:
    """Track themes across adjacent effective graph states without resetting at day boundaries.

    Repeated minute timestamps may increment ``age_timestamps`` but never ``age_states``.
    Overnight transitions preserve lifecycle ids while applying a stricter threshold.
    """

    def __init__(self, config: ContinuousTrackingConfig | None = None):
        self.config = config or ContinuousTrackingConfig()
        self.paths: dict[str, ThemePath] = {}
        self._next_id = 1
        self._active_ids: list[str] = []
        self._last_state_hash: str | None = None

    def _new_path(self, theme, timestamp) -> str:
        lifecycle_id = f"L{self._next_id:08d}"
        self._next_id += 1
        self.paths[lifecycle_id] = ThemePath(
            lifecycle_id=lifecycle_id,
            last_theme=theme,
            first_timestamp=timestamp,
            last_timestamp=timestamp,
        )
        return lifecycle_id

    def _score(self, current, previous) -> tuple[float, float]:
        current_members = _members(current)
        previous_members = _members(previous)
        jaccard = _jaccard(current_members, previous_members)
        containment = _containment(current_members, previous_members)
        method = self.config.method
        if method == "stocknet_j035":
            return jaccard, self.config.jaccard_threshold
        if method == "overlap_c050":
            return containment, self.config.containment_threshold

        family = _set_similarity(
            getattr(current, "families", getattr(current, "family_frequency", {}).keys()),
            getattr(previous, "families", getattr(previous, "family_frequency", {}).keys()),
        )
        layer = _set_similarity(
            getattr(current, "layers", getattr(current, "layer_frequency", {}).keys()),
            getattr(previous, "layers", getattr(previous, "layer_frequency", {}).keys()),
        )
        current_quality = float(getattr(current, "structure_score", getattr(current, "cohesion", 0.5)))
        previous_quality = float(getattr(previous, "structure_score", getattr(previous, "cohesion", 0.5)))
        quality = exp(-abs(current_quality - previous_quality))
        score = 0.30 * containment + 0.25 * jaccard + 0.18 * family + 0.17 * layer + 0.10 * quality
        threshold = self.config.strict_threshold if method == "hybrid_strict" else self.config.hybrid_threshold
        if containment + jaccard < self.config.member_evidence_gate:
            return score, 2.0
        return score, threshold

    def step(
        self,
        themes: Sequence[object],
        *,
        timestamp,
        graph_state_hash: str,
        transition_type: str = "intraday",
    ) -> list[str]:
        if graph_state_hash == self._last_state_hash:
            for lifecycle_id in self._active_ids:
                self.paths[lifecycle_id].age_timestamps += 1
                self.paths[lifecycle_id].last_timestamp = timestamp
            return list(self._active_ids)

        previous_ids = list(self._active_ids)
        candidates: list[tuple[float, int, int, float]] = []
        for current_index, current in enumerate(themes):
            for previous_index, lifecycle_id in enumerate(previous_ids):
                previous = self.paths[lifecycle_id].last_theme
                score, threshold = self._score(current, previous)
                if transition_type == "overnight":
                    threshold += self.config.overnight_threshold_addition
                if score >= threshold:
                    candidates.append((score, current_index, previous_index, threshold))
        candidates.sort(key=lambda value: (-value[0], value[1], value[2]))

        used_current: set[int] = set()
        used_previous: set[int] = set()
        assigned: list[str | None] = [None] * len(themes)
        for score, current_index, previous_index, threshold in candidates:
            if current_index in used_current:
                continue
            if self.config.one_to_one and previous_index in used_previous:
                continue
            lifecycle_id = previous_ids[previous_index]
            path = self.paths[lifecycle_id]
            path.last_theme = themes[current_index]
            path.age_states += 1
            path.age_timestamps += 1
            path.last_timestamp = timestamp
            path.transitions.append(
                {
                    "timestamp": timestamp,
                    "transition_type": transition_type,
                    "score": score,
                    "threshold": threshold,
                }
            )
            assigned[current_index] = lifecycle_id
            used_current.add(current_index)
            used_previous.add(previous_index)

        for index, theme in enumerate(themes):
            if assigned[index] is None:
                assigned[index] = self._new_path(theme, timestamp)

        self._active_ids = [str(value) for value in assigned]
        self._last_state_hash = graph_state_hash
        return list(self._active_ids)
