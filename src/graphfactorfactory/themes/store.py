from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import pandas as pd


class ThemeStore:
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def write_snapshot(self, *, trade_date, snapshot_time, layer_communities, subcommunities, themes, lifecycle, semantics):
        stamp = pd.Timestamp(snapshot_time).strftime("%H%M%S")
        target = self.root / f"date={trade_date}" / f"snapshot={stamp}"
        target.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([asdict(item) for item in layer_communities]).to_parquet(target / "layer_communities.parquet", index=False)
        pd.DataFrame([asdict(item) for item in subcommunities]).to_parquet(target / "subcommunities.parquet", index=False)
        pd.DataFrame([{**asdict(item), "quality_breakdown": json.dumps(item.quality_breakdown, sort_keys=True)} for item in themes]).to_parquet(target / "themes.parquet", index=False)
        pd.DataFrame([asdict(item) for item in lifecycle]).to_parquet(target / "lifecycle.parquet", index=False)
        pd.DataFrame([asdict(item) for item in semantics]).to_parquet(target / "semantics.parquet", index=False)
        return target

    def build_read_models(self):
        theme_files = sorted(self.root.glob("date=*/snapshot=*/themes.parquet"))
        lifecycle_files = sorted(self.root.glob("date=*/snapshot=*/lifecycle.parquet"))
        semantic_files = sorted(self.root.glob("date=*/snapshot=*/semantics.parquet"))
        themes = pd.concat([pd.read_parquet(path) for path in theme_files], ignore_index=True) if theme_files else pd.DataFrame()
        lifecycle = pd.concat([pd.read_parquet(path) for path in lifecycle_files], ignore_index=True) if lifecycle_files else pd.DataFrame()
        semantics = pd.concat([pd.read_parquet(path) for path in semantic_files], ignore_index=True) if semantic_files else pd.DataFrame()
        read_root = self.root / "read_models"; read_root.mkdir(exist_ok=True)
        themes.to_parquet(read_root / "theme_events.parquet", index=False)
        lifecycle.to_parquet(read_root / "theme_lifecycle.parquet", index=False)
        semantics.to_parquet(read_root / "theme_semantics.parquet", index=False)
        if not themes.empty:
            active = themes.sort_values("snapshot_time").groupby("theme_path_id").tail(1)
            active.to_parquet(read_root / "active_theme_paths.parquet", index=False)
        return read_root
