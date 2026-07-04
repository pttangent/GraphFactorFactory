from .theme_path_score import stable_core, path_score


def match_paths(current, previous, history, threshold=0.45, min_hits=2):
    previous_by_path = {item.theme_path_id: item for item in previous}
    pairs = []
    for candidate in current:
        for path_id, item in previous_by_path.items():
            core = stable_core(history.get(path_id), item.members, min_hits)
            score, retention = path_score(candidate, item, core)
            if score >= threshold:
                pairs.append((score, retention, candidate, item))
    pairs.sort(key=lambda row: (-row[0], -row[1], row[2].theme_instance_id, row[3].theme_path_id))
    selected = {}
    used_paths = set()
    for score, retention, candidate, item in pairs:
        if candidate.theme_instance_id in selected or item.theme_path_id in used_paths:
            continue
        selected[candidate.theme_instance_id] = (item, score, retention)
        used_paths.add(item.theme_path_id)
    return selected, used_paths
