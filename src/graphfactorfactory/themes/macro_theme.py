from __future__ import annotations

from dataclasses import dataclass


def _containment(left, right):
    left, right = set(left), set(right)
    return len(left & right) / min(len(left), len(right)) if left and right else 0.0


def _weighted_jaccard(left, right):
    keys = set(left) | set(right)
    if not keys:
        return 0.0
    numerator = sum(min(left.get(key, 0.0), right.get(key, 0.0)) for key in keys)
    denominator = sum(max(left.get(key, 0.0), right.get(key, 0.0)) for key in keys)
    return numerator / denominator if denominator else 0.0


@dataclass(frozen=True)
class MacroThemePrototype:
    macro_theme_id: str
    core_members: tuple[int, ...]
    member_frequency: dict[int, float]
    family_frequency: dict[str, float]
    layer_frequency: dict[str, float]
    persistence: float
    mean_consensus_score: float
    effective_state_support: int


@dataclass(frozen=True)
class MacroMatchConfig:
    threshold: float = 0.50
    member_evidence_gate: float = 0.10
    core_weight: float = 0.15
    member_frequency_weight: float = 0.15
    family_weight: float = 0.25
    layer_weight: float = 0.20
    persistence_weight: float = 0.10
    consensus_weight: float = 0.15


class MacroThemeMatcher:
    def __init__(self, config: MacroMatchConfig | None = None):
        self.config = config or MacroMatchConfig()

    def score(self, current: MacroThemePrototype, previous: MacroThemePrototype):
        core = _containment(current.core_members, previous.core_members)
        members = _weighted_jaccard(current.member_frequency, previous.member_frequency)
        families = _weighted_jaccard(current.family_frequency, previous.family_frequency)
        layers = _weighted_jaccard(current.layer_frequency, previous.layer_frequency)
        persistence = 1.0 - abs(current.persistence - previous.persistence)
        consensus = 1.0 - abs(current.mean_consensus_score - previous.mean_consensus_score)
        score = (
            self.config.core_weight * core
            + self.config.member_frequency_weight * members
            + self.config.family_weight * families
            + self.config.layer_weight * layers
            + self.config.persistence_weight * persistence
            + self.config.consensus_weight * consensus
        )
        return score, {
            "core_overlap": core,
            "member_frequency_similarity": members,
            "family_similarity": families,
            "layer_similarity": layers,
            "persistence_similarity": persistence,
            "consensus_similarity": consensus,
        }

    def match(self, current, previous):
        pairs = []
        for current_index, current_item in enumerate(current):
            for previous_index, previous_item in enumerate(previous):
                score, detail = self.score(current_item, previous_item)
                member_evidence = detail["core_overlap"] + detail["member_frequency_similarity"]
                if member_evidence < self.config.member_evidence_gate:
                    continue
                if score >= self.config.threshold:
                    pairs.append((score, current_index, previous_index, detail))
        pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
        used_current, used_previous, result = set(), set(), []
        for pair in pairs:
            if pair[1] in used_current or pair[2] in used_previous:
                continue
            used_current.add(pair[1])
            used_previous.add(pair[2])
            result.append(pair)
        return result
