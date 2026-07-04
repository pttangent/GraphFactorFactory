from dataclasses import replace

from .models import LifecycleRecord
from .path_state import PathState
from .stable_core_match import match_paths


class StableCoreTracker:
    def __init__(self, threshold=0.45, history_frames=3, min_hits=2, grace_frames=1):
        self.threshold = threshold
        self.min_hits = min_hits
        self.grace_frames = grace_frames
        self.state = PathState(history_frames)

    def assign(self, current, previous, previous_records, *, timestamp, frame_minutes):
        selected, used_paths = match_paths(
            current,
            previous,
            self.state.members,
            threshold=self.threshold,
            min_hits=self.min_hits,
        )
        assigned = []
        records = []
        previous_by_path = {item.theme_path_id: item for item in previous}

        for candidate in current:
            match = selected.get(candidate.theme_instance_id)
            if match is None:
                updated = candidate
                record = LifecycleRecord(
                    candidate.theme_path_id,
                    candidate.theme_instance_id,
                    timestamp,
                    "birth",
                    "tentative",
                    1,
                    frame_minutes,
                    1.0,
                    1.0,
                )
            else:
                old, score, retention = match
                prior = previous_records.get(old.theme_instance_id)
                age = (prior.age_frames if prior else 1) + 1
                event = "revival" if self.state.missed.get(old.theme_path_id, 0) else "continuation"
                updated = replace(candidate, theme_path_id=old.theme_path_id, stability_score=score)
                record = LifecycleRecord(
                    updated.theme_path_id,
                    updated.theme_instance_id,
                    timestamp,
                    event,
                    "active",
                    age,
                    age * frame_minutes,
                    score,
                    retention,
                    old.theme_instance_id,
                )
            assigned.append(updated)
            records.append(record)
            self.state.observe(updated.theme_path_id, updated.members)

        for path_id, old in previous_by_path.items():
            if path_id in used_paths:
                continue
            prior = previous_records.get(old.theme_instance_id)
            missed = self.state.miss(path_id)
            if missed <= self.grace_frames:
                records.append(LifecycleRecord(
                    path_id, old.theme_instance_id, timestamp,
                    "dormant", "dormant",
                    prior.age_frames if prior else 1,
                    prior.duration_minutes if prior else frame_minutes,
                    0.0, 0.0, old.theme_instance_id,
                ))
            else:
                records.append(LifecycleRecord(
                    path_id, old.theme_instance_id, timestamp,
                    "death", "inactive",
                    prior.age_frames if prior else 1,
                    prior.duration_minutes if prior else frame_minutes,
                    0.0, 0.0, old.theme_instance_id,
                ))
                self.state.clear(path_id)
        return assigned, records
