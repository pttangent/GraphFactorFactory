from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class LayerTemporalPolicy:
    enter_quantile: float = 0.80
    exit_quantile: float = 0.60
    prior_lambda: float = 0.15
    smoothing_alpha: float = 0.60
    grace_frames: int = 1


class LayerRelativeTemporalReplay:
    """Keep observed evidence separate from decayed temporal prior."""

    def __init__(self, policies=None, default=None):
        self.policies = policies or {}
        self.default = default or LayerTemporalPolicy()
        self.state = {}

    def _policy(self, layer_id):
        return self.policies.get(int(layer_id), self.default)

    def replay(self, edges: pd.DataFrame, snapshot_time):
        if edges.empty and not self.state:
            return edges.copy()

        output = []
        current_keys = set()
        for layer_id, group in edges.groupby("layer_id", sort=False):
            policy = self._policy(layer_id)
            weights = group["weight"].astype(float)
            enter = float(weights.quantile(policy.enter_quantile))
            exit_ = float(weights.quantile(policy.exit_quantile))

            for row in group.to_dict("records"):
                key = (int(layer_id), int(row["src_id"]), int(row["dst_id"]))
                current_keys.add(key)
                old = self.state.get(key)
                raw = float(row["weight"])
                prior = float(old["effective_weight"]) if old else 0.0
                active_before = bool(old and old["active"])
                threshold = exit_ if active_before else enter
                smoothed = policy.smoothing_alpha * raw + (1.0 - policy.smoothing_alpha) * prior
                active = smoothed >= threshold
                effective = raw + policy.prior_lambda * prior
                row.update({
                    "raw_observed_weight": raw,
                    "temporal_prior_weight": prior,
                    "effective_weight": effective,
                    "weight": effective,
                    "temporal_status": "active" if active else "inactive",
                    "edge_age": int(old["edge_age"] + 1) if old else 1,
                    "missing_frames": 0,
                })
                self.state[key] = row.copy()
                self.state[key]["active"] = active
                if active:
                    output.append(row)

        for key, old in list(self.state.items()):
            if key in current_keys:
                continue
            policy = self._policy(key[0])
            missing = int(old.get("missing_frames", 0)) + 1
            if bool(old.get("active")) and missing <= policy.grace_frames:
                prior = float(old["effective_weight"])
                grace = old.copy()
                grace.update({
                    "decision_time": snapshot_time,
                    "raw_observed_weight": 0.0,
                    "temporal_prior_weight": prior,
                    "effective_weight": policy.prior_lambda * prior,
                    "weight": policy.prior_lambda * prior,
                    "temporal_status": "prior_only",
                    "missing_frames": missing,
                    "window_points": 0,
                    "vector_dimension": 0,
                })
                self.state[key] = grace.copy()
                self.state[key]["active"] = True
                output.append(grace)
            else:
                self.state.pop(key, None)

        return pd.DataFrame(output)
