from __future__ import annotations

from dataclasses import replace
from .models import ThemeCandidate, LifecycleRecord


def overlap_small(left, right):
    left, right = set(left), set(right)
    return len(left & right) / min(len(left), len(right)) if left and right else 0.0


class ThemeLifecycleTracker:
    def __init__(self, min_overlap=0.5):
        self.min_overlap = min_overlap

    def assign(self, current: list[ThemeCandidate], previous: list[ThemeCandidate], previous_records: dict[str, LifecycleRecord], *, timestamp, frame_minutes: int):
        ranked = {candidate.theme_instance_id: sorted([(item, overlap_small(candidate.members, item.members)) for item in previous if overlap_small(candidate.members, item.members) >= self.min_overlap], key=lambda value: (-value[1], value[0].theme_instance_id)) for candidate in current}
        reverse = {item.theme_instance_id: [candidate for candidate in current if overlap_small(candidate.members, item.members) >= self.min_overlap] for item in previous}
        used = set(); assigned=[]; records=[]
        for candidate in current:
            matches = ranked[candidate.theme_instance_id]
            best, score = matches[0] if matches else (None, 0.0)
            if best is None:
                updated = candidate
                record = LifecycleRecord(candidate.theme_path_id, candidate.theme_instance_id, timestamp, "birth", "active", 1, frame_minutes, 1.0, 1.0)
            elif best.theme_instance_id not in used:
                used.add(best.theme_instance_id)
                prior = previous_records.get(best.theme_instance_id)
                age = (prior.age_frames if prior else 1) + 1
                event = "merge" if len(matches) > 1 else "continuation"
                updated = replace(candidate, theme_path_id=best.theme_path_id, stability_score=score)
                record = LifecycleRecord(updated.theme_path_id, updated.theme_instance_id, timestamp, event, "active", age, age * frame_minutes, score, score, best.theme_instance_id, tuple(item.theme_path_id for item, _ in matches[1:]), ())
            else:
                event = "split" if len(reverse.get(best.theme_instance_id, [])) > 1 else "birth"
                updated = candidate
                record = LifecycleRecord(updated.theme_path_id, updated.theme_instance_id, timestamp, event, "active", 1, frame_minutes, score, score, best.theme_instance_id, (best.theme_path_id,) if event == "split" else (), (updated.theme_path_id,) if event == "split" else ())
            assigned.append(updated); records.append(record)
        for item in previous:
            if item.theme_instance_id not in used and not reverse.get(item.theme_instance_id):
                prior = previous_records.get(item.theme_instance_id)
                records.append(LifecycleRecord(item.theme_path_id, item.theme_instance_id, timestamp, "death", "inactive", prior.age_frames if prior else 1, prior.duration_minutes if prior else frame_minutes, 0.0, 0.0, item.theme_instance_id))
        return assigned, records
