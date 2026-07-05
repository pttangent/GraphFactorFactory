from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from dataclasses import asdict

import numpy as np
import pandas as pd

from .temporal import ThemeLifecycleTracker
from .models import ThemeCandidate, LifecycleRecord
from .pipeline import ThemeDiscoveryConfig

logger = logging.getLogger(__name__)

class ThemeStorePhase2:
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def write_day_layer(self, trade_date, layer_name, themes, lifecycle):
        target = self.root / f"date={trade_date}" / f"layer={layer_name}"
        target.mkdir(parents=True, exist_ok=True)
        
        if themes:
            pd.DataFrame([{**asdict(item), "quality_breakdown": json.dumps(item.quality_breakdown, sort_keys=True)} for item in themes]).to_parquet(target / "theme_paths.parquet", index=False)
        if lifecycle:
            pd.DataFrame([asdict(item) for item in lifecycle]).to_parquet(target / "lifecycle_states.parquet", index=False)
        
        (target / "_SUCCESS").write_text("success", encoding="utf-8")
        return target

def _process_layer_temporal(args):
    layer_name, layer_themes, config = args
    tracker = ThemeLifecycleTracker(min_overlap=config.min_overlap)
    
    snapshot_times = sorted(layer_themes["snapshot_time"].unique())
    previous_candidates = []
    previous_records = {}
    
    all_updated_candidates = []
    all_lifecycle_records = []
    
    for t in snapshot_times:
        current_df = layer_themes[layer_themes["snapshot_time"] == t]
        
        # Reconstruct ThemeCandidate objects
        current_candidates = []
        for _, row in current_df.iterrows():
            quality_breakdown = json.loads(row["quality_breakdown"]) if isinstance(row["quality_breakdown"], str) else row["quality_breakdown"]
            members = tuple(row["members"]) if isinstance(row["members"], (list, np.ndarray)) else row["members"]
            core_members = tuple(row["core_members"]) if isinstance(row["core_members"], (list, np.ndarray)) else row["core_members"]
            
            cand = ThemeCandidate(
                theme_instance_id=row["theme_instance_id"],
                theme_path_id=row["theme_path_id"],
                layer_id=row["layer_id"],
                layer_name=row["layer_name"],
                snapshot_time=row["snapshot_time"],
                community_id_local=row["community_id_local"],
                members=members,
                core_members=core_members,
                size=row["size"],
                community_quality=row["community_quality"],
                stability_score=row.get("stability_score", 0.0),
                source_graph_id=row["source_graph_id"],
                quality_breakdown=quality_breakdown,
                quality_score=row.get("quality_score", 0.0),
                run_id=row["run_id"]
            )
            current_candidates.append(cand)
            
        assigned, records = tracker.assign(
            current_candidates,
            previous_candidates,
            previous_records,
            timestamp=t,
            frame_minutes=config.frame_minutes
        )
        
        all_updated_candidates.extend(assigned)
        all_lifecycle_records.extend(records)
        
        previous_candidates = assigned
        previous_records = {r.theme_instance_id: r for r in records if r.status == "active"}
        
    return layer_name, all_updated_candidates, all_lifecycle_records


class ThemeTemporalPhase2Pipeline:
    def __init__(self, phase1_root, phase2_root, config: ThemeDiscoveryConfig):
        self.phase1_root = Path(phase1_root).expanduser().resolve()
        self.store = ThemeStorePhase2(phase2_root)
        self.config = config

    def run(self, date_start=None, date_end=None, max_layer_workers=6):
        outputs = []

        for day in sorted(self.phase1_root.glob("date=*")):
            trade_date = day.name.split("=", 1)[1]
            if date_start and trade_date < date_start:
                continue
            if date_end and trade_date > date_end:
                continue
                
            t0 = time.time()
            themes_path = day / "themes.parquet"
            if not themes_path.exists():
                logger.warning(f"Phase 1 themes missing for {trade_date}, skipping Phase 2.")
                continue
                
            themes_df = pd.read_parquet(themes_path)
            if themes_df.empty:
                logger.info(f"Phase 1 themes empty for {trade_date}, skipping Phase 2.")
                continue
                
            tasks = []
            for layer_name, layer_themes in themes_df.groupby("layer_name"):
                tasks.append((layer_name, layer_themes, self.config))
                
            import concurrent.futures
            with ProcessPoolExecutor(max_workers=max_layer_workers) as executor:
                futures = [executor.submit(_process_layer_temporal, task) for task in tasks]
                for future in concurrent.futures.as_completed(futures):
                    layer_name, all_updated_candidates, all_lifecycle_records = future.result()
                    
                    target = self.store.write_day_layer(trade_date, layer_name, all_updated_candidates, all_lifecycle_records)
                    outputs.append(target)
            
            logger.info(f"[{trade_date}] Phase 2 (Intraday Lifecycle) finished in {time.time() - t0:.1f}s")

        return outputs
