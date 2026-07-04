from collections import Counter


def stable_core(history, fallback, min_hits=2):
    if not history:
        return set(fallback)
    counts = Counter(node for frame in history for node in frame)
    core = {node for node, count in counts.items() if count >= min_hits}
    return core or set(history[-1])


def path_score(candidate, previous, core):
    current = set(candidate.members)
    shared = len(current & core)
    containment = shared / min(len(current), len(core)) if current and core else 0.0
    jaccard = shared / len(current | core) if current or core else 0.0
    left, right = set(candidate.source_families), set(previous.source_families)
    family = len(left & right) / len(left | right) if left or right else 0.0
    size = min(len(current), len(core)) / max(len(current), len(core)) if current and core else 0.0
    score = 0.55 * containment + 0.15 * jaccard + 0.20 * family + 0.10 * size
    return score, containment
