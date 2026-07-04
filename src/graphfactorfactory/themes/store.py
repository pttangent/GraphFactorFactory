from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import pandas as pd


class ThemeStore:
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._accumulator = []

    def accumulate_snapshot(self, *, snapshot_time, temporal_edges, layer_communities, subcommunities, themes, lifecycle, semantics):
        self._accumulator.append({
            "temporal_edges": temporal_edges if not temporal_edges.empty else pd.DataFrame(),
            "layer_communities": pd.DataFrame([asdict(item) for item in layer_communities]) if layer_communities else pd.DataFrame(),
            "subcommunities": pd.DataFrame([asdict(item) for item in subcommunities]) if subcommunities else pd.DataFrame(),
            "themes": pd.DataFrame([{**asdict(item), "quality_breakdown": json.dumps(item.quality_breakdown, sort_keys=True)} for item in themes]) if themes else pd.DataFrame(),
            "lifecycle": pd.DataFrame([asdict(item) for item in lifecycle]) if lifecycle else pd.DataFrame(),
            "semantics": pd.DataFrame([asdict(item) for item in semantics]) if semantics else pd.DataFrame()
        })

    def write_day(self, trade_date):
        if not self._accumulator:
            return None
        target = self.root / f"date={trade_date}"
        target.mkdir(parents=True, exist_ok=True)
        
        for key in ["temporal_edges", "layer_communities", "subcommunities", "themes", "lifecycle", "semantics"]:
            frames = [item[key] for item in self._accumulator if not item[key].empty]
            if frames:
                pd.concat(frames, ignore_index=True).to_parquet(target / f"{key}.parquet", index=False)
                
        self._accumulator = []
        return target

    def build_read_models(self):
        theme_files = sorted(self.root.glob("date=*/themes.parquet"))
        lifecycle_files = sorted(self.root.glob("date=*/lifecycle.parquet"))
        semantic_files = sorted(self.root.glob("date=*/semantics.parquet"))
        themes = pd.concat([pd.read_parquet(path) for path in theme_files], ignore_index=True) if theme_files else pd.DataFrame()
        lifecycle = pd.concat([pd.read_parquet(path) for path in lifecycle_files], ignore_index=True) if lifecycle_files else pd.DataFrame()
        semantics = pd.concat([pd.read_parquet(path) for path in semantic_files], ignore_index=True) if semantic_files else pd.DataFrame()
        read_root = self.root / "read_models"; read_root.mkdir(exist_ok=True)
        themes.to_parquet(read_root / "theme_events.parquet", index=False)
        lifecycle.to_parquet(read_root / "theme_lifecycle.parquet", index=False)
        semantics.to_parquet(read_root / "theme_semantics.parquet", index=False)
        if not themes.empty:
            themes.sort_values("snapshot_time").groupby("theme_path_id").tail(1).to_parquet(read_root / "active_theme_paths.parquet", index=False)
        return read_root

