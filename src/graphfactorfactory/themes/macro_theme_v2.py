from __future__ import annotations

from dataclasses import dataclass
from math import exp, log


def _weighted_jaccard(left, right, weights=None):
    keys = set(left) | set(right)
    if not keys:
        return 0.0
    numerator = 0.0
    denominator = 0.0
    for key in keys:
        weight = weights.get(str(key), 1.0) if weights else 1.0
        numerator += weight * min(left.get(key, 0.0), right.get(key, 0.0))
        denominator += weight * max(left.get(key, 0.0), right.get(key, 0.0))
    return numerator / denominator if denominator else 0.0


def _weighted_containment(left, right, weights=None):
    left = {str(value) for value in left}
    right = {str(value) for value in right}
    if not left or not right:
        return 0.0
    shared = left & right
    if weights:
        shared_weight = sum(weights.get(value, 1.0) for value in shared)
        left_weight = sum(weights.get(value, 1.0) for value in left)
        right_weight = sum(weights.get(value, 1.0) for value in right)
        return shared_weight / min(left_weight, right_weight)
    return len(shared) / min(len(left), len(right))


def _numeric_similarity(left, right, scale):
    return exp(-abs(float(left) - float(right)) / scale)


def inverse_document_frequency(documents):
    documents = list(documents)
    total = len(documents)
    counts = {}
    for document in documents:
        for token in document:
            key = str(token)
            counts[key] = counts.get(key, 0) + 1
    return {
        key: log((total + 1) / (count + 1)) + 1.0
        for key, count in counts.items()
    }


@dataclass(frozen=True)
class StructuralMacroConfig:
    threshold: float = 0.50
    member_evidence_gate: float = 0.08
    quality_gamma: float = 0.08
    quality_anchor: float = 0.72
    core_weight: float = 0.16
    member_frequency_weight: float = 0.18
    family_weight: float = 0.10
    layer_weight: float = 0.08
    rare_family_weight: float = 0.10
    rare_layer_weight: float = 0.08
    persistence_weight: float = 0.06
    consensus_weight: float = 0.08
    cohesion_weight: float = 0.05
    core_ratio_weight: float = 0.035
    member_entropy_weight: float = 0.035
    size_weight: float = 0.075


class StructuralMacroMatcher:
    def __init__(self, config=None, node_idf=None, family_idf=None, layer_idf=None):
        self.config = config or StructuralMacroConfig()
        self.node_idf = node_idf or {}
        self.family_idf = family_idf or {}
        self.layer_idf = layer_idf or {}

    def score(self, current, previous):
        core = _weighted_containment(current.core_members, previous.core_members, self.node_idf)
        members = _weighted_jaccard(current.member_frequency, previous.member_frequency, self.node_idf)
        families = _weighted_jaccard(current.family_frequency, previous.family_frequency)
        layers = _weighted_jaccard(current.layer_frequency, previous.layer_frequency)
        rare_families = _weighted_jaccard(current.family_frequency, previous.family_frequency, self.family_idf)
        rare_layers = _weighted_jaccard(current.layer_frequency, previous.layer_frequency, self.layer_idf)
        persistence = 1.0 - abs(current.persistence - previous.persistence)
        consensus = 1.0 - abs(current.mean_consensus_score - previous.mean_consensus_score)
        cohesion = 1.0 - abs(current.cohesion - previous.cohesion)
        core_ratio = _numeric_similarity(current.core_ratio, previous.core_ratio, 0.25)
        entropy = _numeric_similarity(current.member_entropy, previous.member_entropy, 0.25)
        size = _numeric_similarity(current.log_size, previous.log_size, 0.70)
        values = (
            core, members, families, layers, rare_families, rare_layers,
            persistence, consensus, cohesion, core_ratio, entropy, size,
        )
        weights = (
            self.config.core_weight,
            self.config.member_frequency_weight,
            self.config.family_weight,
            self.config.layer_weight,
            self.config.rare_family_weight,
            self.config.rare_layer_weight,
            self.config.persistence_weight,
            self.config.consensus_weight,
            self.config.cohesion_weight,
            self.config.core_ratio_weight,
            self.config.member_entropy_weight,
            self.config.size_weight,
        )
        score = sum(weight * value for weight, value in zip(weights, values))
        quality = 0.5 * min(current.persistence, previous.persistence) + 0.5 * min(current.cohesion, previous.cohesion)
        threshold = self.config.threshold + self.config.quality_gamma * (self.config.quality_anchor - quality)
        return score, threshold, core, members

    def match(self, current, previous):
        pairs = []
        for current_index, current_item in enumerate(current):
            for previous_index, previous_item in enumerate(previous):
                score, threshold, core, members = self.score(current_item, previous_item)
                if core + members < self.config.member_evidence_gate:
                    continue
                if score >= threshold:
                    pairs.append((score, current_index, previous_index))
        pairs.sort(key=lambda value: (-value[0], value[1], value[2]))
        used_current = set()
        used_previous = set()
        selected = []
        for value in pairs:
            if value[1] in used_current or value[2] in used_previous:
                continue
            used_current.add(value[1])
            used_previous.add(value[2])
            selected.append(value)
        return selected
