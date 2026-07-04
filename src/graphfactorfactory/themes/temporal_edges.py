from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class TemporalEdgeConfig:
    enter_threshold: float = 0.75
    exit_threshold: float = 0.65
    smoothing_alpha: float = 0.6
    missing_grace_frames: int = 1


class TemporalEdgeReplay:
    def __init__(self, config: TemporalEdgeConfig | None = None):
        self.config = config or TemporalEdgeConfig()
        self.state = {}

    def replay(self, edges: pd.DataFrame, snapshot_time):
        seen = set()
        rows = []
        for row in edges.itertuples(index=False):
            key = (int(row.layer_id), int(row.src_id), int(row.dst_id))
            seen.add(key)
            previous = self.state.get(key)
            raw_weight = float(row.weight)
            if previous is None:
                smoothed = raw_weight
            else:
                smoothed = self.config.smoothing_alpha * raw_weight + (1.0 - self.config.smoothing_alpha) * previous["weight"]
            threshold = self.config.exit_threshold if previous and previous["active"] else self.config.enter_threshold
            active = smoothed >= threshold
            self.state[key] = {"weight": smoothed, "active": active, "missing_frames": 0}
            if active:
                payload = row._asdict()
                payload["weight"] = smoothed
                payload["raw_weight"] = raw_weight
                payload["temporal_status"] = "active"
                rows.append(payload)
        for key, previous in list(self.state.items()):
            if key in seen:
                continue
            missing_frames = previous["missing_frames"] + 1
            if previous["active"] and missing_frames <= self.config.missing_grace_frames:
                layer_id, src_id, dst_id = key
                rows.append({"decision_time": snapshot_time, "window_start": snapshot_time, "window_end": snapshot_time, "layer_id": layer_id, "src_id": src_id, "dst_id": dst_id, "weight": previous["weight"], "raw_weight": None, "src_rank": None, "dst_rank": None, "directed": False, "lag_bars": 0, "window_points": 0, "vector_dimension": 0, "temporal_status": "grace"})
                self.state[key] = {**previous, "missing_frames": missing_frames}
            else:
                self.state.pop(key, None)
        return pd.DataFrame(rows)
