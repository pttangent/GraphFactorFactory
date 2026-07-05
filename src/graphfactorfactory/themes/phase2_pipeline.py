from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .models import LifecycleRecord, ThemeCandidate
from .pipeline import ThemeDiscoveryConfig
from .temporal import ThemeLifecycleTracker

logger = logging.getLogger(__name__)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temp, index=False)
    os.replace(temp, path)


def _tuple_value(value: Any) -> tuple:
    if value is None:
        return ()
    if isinstance(value, float) and pd.isna(value):
        return ()
    if isinstance(value, np.ndarray):
        return tuple(value.tolist())
    if isinstance(value, list):
        return tuple(value)
    if isinstance(value, tuple):
        return value
    return (value,)


def _candidate_from_layer_row(row: pd.Series) -> ThemeCandidate:
    timestamp = pd.Timestamp(row["snapshot_time"])
    layer_id = int(row["layer_id"])
    layer_name = str(row["layer_name"])
    community_id = int(row["community_id"])
    members = tuple(sorted(int(item) for item in row["members"]))
    instance_id = f"layer:{layer_id}:{timestamp.isoformat()}:{community_id}"
    return ThemeCandidate(
        theme_instance_id=instance_id,
        theme_path_id=instance_id,
        snapshot_time=timestamp,
        members=members,
        source_layers=(layer_name,),
        source_families=(layer_name,),
        consensus_score=float(row.get("modularity", 0.0) or 0.0),
        structure_score=float(row.get("modularity", 0.0) or 0.0),
        member_ratio=0.0,
        is_market_mode=bool(row.get("is_market_mode", False)),
    )


def _candidate_to_row(candidate: ThemeCandidate) -> dict[str, Any]:
    row = asdict(candidate)
    row["members"] = list(candidate.members)
    row["source_layers"] = list(candidate.source_layers)
    row["source_families"] = list(candidate.source_families)
    row["quality_breakdown"] = json.dumps(candidate.quality_breakdown, sort_keys=True)
    return row


def _candidate_from_state_row(row: pd.Series) -> ThemeCandidate:
    breakdown = row.get("quality_breakdown", "{}")
    if isinstance(breakdown, str):
        breakdown = json.loads(breakdown)
    return ThemeCandidate(
        theme_instance_id=str(row["theme_instance_id"]),
        theme_path_id=str(row["theme_path_id"]),
        snapshot_time=pd.Timestamp(row["snapshot_time"]),
        members=tuple(int(item) for item in _tuple_value(row["members"])),
        source_layers=tuple(str(item) for item in _tuple_value(row["source_layers"])),
        source_families=tuple(str(item) for item in _tuple_value(row["source_families"])),
        consensus_score=float(row.get("consensus_score", 0.0)),
        structure_score=float(row.get("structure_score", 0.0)),
        member_ratio=float(row.get("member_ratio", 0.0)),
        is_market_mode=bool(row.get("is_market_mode", False)),
        flow_support_score=float(row.get("flow_support_score", 0.0)),
        stability_score=float(row.get("stability_score", 0.0)),
        semantic_coherence_score=float(row.get("semantic_coherence_score", 0.0)),
        theme_quality_score=float(row.get("theme_quality_score", 0.0)),
        quality_breakdown=breakdown,
    )


def _record_from_state_row(row: pd.Series) -> LifecycleRecord:
    previous = row.get("previous_theme_instance_id")
    return LifecycleRecord(
        theme_path_id=str(row["theme_path_id"]),
        theme_instance_id=str(row["theme_instance_id"]),
        timestamp=pd.Timestamp(row["timestamp"]),
        event_type=str(row["event_type"]),
        status=str(row["status"]),
        age_frames=int(row["age_frames"]),
        duration_minutes=int(row["duration_minutes"]),
        match_score=float(row["match_score"]),
        member_retention=float(row["member_retention"]),
        previous_theme_instance_id=(None if previous is None or (isinstance(previous, float) and pd.isna(previous)) else str(previous)),
        parent_path_ids=tuple(str(item) for item in _tuple_value(row.get("parent_path_ids"))),
        child_path_ids=tuple(str(item) for item in _tuple_value(row.get("child_path_ids"))),
    )


class ThemeStorePhase2:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_root = self.root / "_state"
        self.state_root.mkdir(parents=True, exist_ok=True)

    def day_dir(self, trade_date: str) -> Path:
        return self.root / f"date={trade_date}"

    def state_dir(self, trade_date: str, layer_name: str) -> Path:
        return self.state_root / f"date={trade_date}" / f"layer={layer_name}"

    def write_day_layer(
        self,
        trade_date: str,
        layer_name: str,
        themes: list[ThemeCandidate],
        lifecycle: list[LifecycleRecord],
        active_candidates: list[ThemeCandidate],
        active_records: dict[str, LifecycleRecord],
    ) -> Path:
        target = self.day_dir(trade_date) / f"layer={layer_name}"
        target.mkdir(parents=True, exist_ok=True)
        _atomic_parquet(pd.DataFrame([_candidate_to_row(item) for item in themes]), target / "theme_paths.parquet")
        _atomic_parquet(pd.DataFrame([asdict(item) for item in lifecycle]), target / "lifecycle_states.parquet")

        state = self.state_dir(trade_date, layer_name)
        state.mkdir(parents=True, exist_ok=True)
        _atomic_parquet(pd.DataFrame([_candidate_to_row(item) for item in active_candidates]), state / "active_candidates.parquet")
        _atomic_parquet(pd.DataFrame([asdict(item) for item in active_records.values()]), state / "active_records.parquet")
        (target / "_SUCCESS").write_text("success\n", encoding="utf-8")
        (state / "_SUCCESS").write_text("success\n", encoding="utf-8")
        return target

    def mark_day_success(self, trade_date: str) -> None:
        self.day_dir(trade_date).mkdir(parents=True, exist_ok=True)
        (self.day_dir(trade_date) / "_SUCCESS").write_text("success\n", encoding="utf-8")
        state_day = self.state_root / f"date={trade_date}"
        state_day.mkdir(parents=True, exist_ok=True)
        (state_day / "_SUCCESS").write_text("success\n", encoding="utf-8")

    def load_state(self, trade_date: str) -> dict[str, tuple[list[ThemeCandidate], dict[str, LifecycleRecord]]]:
        result: dict[str, tuple[list[ThemeCandidate], dict[str, LifecycleRecord]]] = {}
        day = self.state_root / f"date={trade_date}"
        if not (day / "_SUCCESS").exists():
            return result
        for layer in sorted(day.glob("layer=*")):
            if not (layer / "_SUCCESS").exists():
                continue
            layer_name = layer.name.split("=", 1)[1]
            candidates_path = layer / "active_candidates.parquet"
            records_path = layer / "active_records.parquet"
            candidates: list[ThemeCandidate] = []
            records: dict[str, LifecycleRecord] = {}
            if candidates_path.exists():
                frame = pd.read_parquet(candidates_path)
                candidates = [_candidate_from_state_row(row) for _, row in frame.iterrows()]
            if records_path.exists():
                frame = pd.read_parquet(records_path)
                records = {
                    record.theme_instance_id: record
                    for record in (_record_from_state_row(row) for _, row in frame.iterrows())
                }
            result[layer_name] = (candidates, records)
        return result

    def invalidate_from(self, trade_date: str) -> None:
        for root in (self.root, self.state_root):
            for day in root.glob("date=*"):
                current = day.name.split("=", 1)[1]
                if current >= trade_date:
                    shutil.rmtree(day, ignore_errors=True)


class ThemeTemporalPhase2Pipeline:
    """Stateful and restartable temporal linking over immutable Phase 1 output."""

    def __init__(self, phase1_root: str | Path, phase2_root: str | Path, config: ThemeDiscoveryConfig):
        self.phase1_root = Path(phase1_root).expanduser().resolve()
        self.store = ThemeStorePhase2(phase2_root)
        self.config = config

    def _phase1_dates(self) -> list[str]:
        return sorted(path.name.split("=", 1)[1] for path in self.phase1_root.glob("date=*") if path.is_dir())

    def run(
        self,
        date_start: str | None = None,
        date_end: str | None = None,
        *,
        rebuild_from: str | None = None,
        resume: bool = True,
    ) -> list[Path]:
        dates = [date for date in self._phase1_dates() if (not date_start or date >= date_start) and (not date_end or date <= date_end)]
        if not dates:
            return []

        effective_start = rebuild_from or dates[0]
        if rebuild_from:
            self.store.invalidate_from(rebuild_from)

        all_phase1_dates = self._phase1_dates()
        previous_dates = [date for date in all_phase1_dates if date < effective_start]
        state: dict[str, tuple[list[ThemeCandidate], dict[str, LifecycleRecord]]] = {}
        if previous_dates:
            state = self.store.load_state(previous_dates[-1])

        outputs: list[Path] = []
        tracker = ThemeLifecycleTracker(min_overlap=self.config.min_overlap)

        for trade_date in dates:
            if trade_date < effective_start:
                continue
            day_success = self.store.day_dir(trade_date) / "_SUCCESS"
            state_success = self.store.state_root / f"date={trade_date}" / "_SUCCESS"
            if resume and day_success.exists() and state_success.exists():
                state = self.store.load_state(trade_date)
                logger.info("[%s] Phase 2 already complete; loaded serialized carry state.", trade_date)
                continue

            source = self.phase1_root / f"date={trade_date}" / "layer_communities.parquet"
            if not source.exists():
                raise FileNotFoundError(f"Missing Phase 1 layer communities: {source}")
            frame = pd.read_parquet(source)
            if frame.empty:
                self.store.mark_day_success(trade_date)
                state = {}
                continue

            if "snapshot_time" not in frame.columns:
                raise ValueError(f"{source} is missing snapshot_time")
            frame = frame.sort_values(["layer_name", "snapshot_time", "community_id"], kind="mergesort")
            next_state: dict[str, tuple[list[ThemeCandidate], dict[str, LifecycleRecord]]] = {}

            for layer_name, layer_frame in frame.groupby("layer_name", sort=True):
                previous_candidates, previous_records = state.get(str(layer_name), ([], {}))
                all_candidates: list[ThemeCandidate] = []
                all_records: list[LifecycleRecord] = []

                for timestamp, snapshot in layer_frame.groupby("snapshot_time", sort=True):
                    current = [_candidate_from_layer_row(row) for _, row in snapshot.iterrows()]
                    assigned, records = tracker.assign(
                        current,
                        previous_candidates,
                        previous_records,
                        timestamp=pd.Timestamp(timestamp),
                        frame_minutes=self.config.frame_minutes,
                    )
                    all_candidates.extend(assigned)
                    all_records.extend(records)
                    previous_candidates = assigned
                    previous_records = {record.theme_instance_id: record for record in records if record.status == "active"}

                outputs.append(
                    self.store.write_day_layer(
                        trade_date,
                        str(layer_name),
                        all_candidates,
                        all_records,
                        previous_candidates,
                        previous_records,
                    )
                )
                next_state[str(layer_name)] = (previous_candidates, previous_records)

            self.store.mark_day_success(trade_date)
            state = next_state
            logger.info("[%s] Phase 2 completed with cross-day carry state.", trade_date)

        return outputs
