from __future__ import annotations

import hashlib
import pandas as pd


STATE_COLUMNS = ("layer_id", "src_id", "dst_id", "weight")


def graph_state_hash(edges: pd.DataFrame) -> str:
    if edges.empty:
        return "empty"
    frame = edges.loc[:, STATE_COLUMNS].copy()
    frame = frame.sort_values(["layer_id", "src_id", "dst_id"], kind="mergesort")
    hashed = pd.util.hash_pandas_object(frame, index=False).values.tobytes()
    return hashlib.sha1(hashed).hexdigest()


class GraphStateClock:
    """Separate wall-clock snapshots from effective graph-state transitions."""

    def __init__(self):
        self.previous_hash = None
        self.state_index = -1

    def observe(self, edges: pd.DataFrame):
        current_hash = graph_state_hash(edges)
        changed = current_hash != self.previous_hash
        if changed:
            self.state_index += 1
            self.previous_hash = current_hash
        return changed, self.state_index, current_hash
