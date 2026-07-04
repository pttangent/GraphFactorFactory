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
        self.state = pd.DataFrame(columns=["weight", "active", "missing_frames"])
        self.state.index = pd.MultiIndex.from_tuples([], names=["layer_id", "src_id", "dst_id"])

    def replay(self, edges: pd.DataFrame, snapshot_time):
        if edges.empty and self.state.empty:
            return pd.DataFrame()

        # Prepare new edges
        if not edges.empty:
            new_edges = edges.set_index(["layer_id", "src_id", "dst_id"])
        else:
            new_edges = pd.DataFrame(index=pd.MultiIndex.from_tuples([], names=["layer_id", "src_id", "dst_id"]))
            new_edges["weight"] = []

        # Full outer join of new_edges and state
        # state has columns: weight, active, missing_frames
        joined = new_edges.join(self.state, how="outer", rsuffix="_state")
        
        # Vectorized calculation of smoothed weight
        alpha = self.config.smoothing_alpha
        # A: In both -> alpha * new + (1-alpha) * old
        smoothed = alpha * joined["weight"] + (1.0 - alpha) * joined["weight_state"]
        # B: Only in new -> new
        smoothed = smoothed.fillna(joined["weight"])
        # C: Only in old (missing) -> old
        smoothed = smoothed.fillna(joined["weight_state"])
        
        # Calculate threshold
        thresholds = pd.Series(self.config.enter_threshold, index=joined.index)
        thresholds.loc[joined["active"] == True] = self.config.exit_threshold
        
        # New active status
        new_active = smoothed >= thresholds
        
        # Calculate missing frames
        is_missing = joined["weight"].isna()
        prev_missing = joined["missing_frames"].fillna(0)
        new_missing = (prev_missing + 1).where(is_missing, 0)
        
        # Update self.state (keep newly active, OR previous active within grace period)
        keep_mask = (~is_missing) | (joined["active"] == True) & (new_missing <= self.config.missing_grace_frames)
        
        new_state = pd.DataFrame({
            "weight": smoothed,
            "active": new_active,
            "missing_frames": new_missing
        })
        self.state = new_state[keep_mask]
        
        # Prepare output
        output_mask_active = (~is_missing) & new_active
        output_mask_grace = is_missing & keep_mask
        
        out_frames = []
        if output_mask_active.any():
            active_out = joined.loc[output_mask_active].copy()
            active_out["raw_weight"] = active_out["weight"]
            active_out["weight"] = smoothed.loc[output_mask_active]
            active_out["temporal_status"] = "active"
            active_out = active_out.drop(columns=["weight_state", "active", "missing_frames"], errors='ignore')
            out_frames.append(active_out)
            
        if output_mask_grace.any():
            grace_out = joined.loc[output_mask_grace].copy()
            grace_out["weight"] = smoothed.loc[output_mask_grace]
            grace_out["raw_weight"] = None
            grace_out["temporal_status"] = "grace"
            # Fill mandatory columns that were NaN because these edges were missing in new_edges
            grace_out["decision_time"] = snapshot_time
            grace_out["window_start"] = snapshot_time
            grace_out["window_end"] = snapshot_time
            grace_out["src_rank"] = None
            grace_out["dst_rank"] = None
            grace_out["directed"] = False
            grace_out["lag_bars"] = 0
            grace_out["window_points"] = 0
            grace_out["vector_dimension"] = 0
            grace_out = grace_out.drop(columns=["weight_state", "active", "missing_frames"], errors='ignore')
            out_frames.append(grace_out)
            
        if not out_frames:
            return pd.DataFrame()
            
        out = pd.concat(out_frames).reset_index()
        return out

