import argparse
import json
import logging
import multiprocessing
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from graphfactorfactory.application.pipeline import GraphFactorPipeline
from graphfactorfactory.domain.config import BuildConfig
from graphfactorfactory.domain.layers import LAYER_SCALES
from graphfactorfactory.infrastructure.nodefactorfactory.parquet_source import ParquetNodeFactorSource
from graphfactorfactory.infrastructure.store import CanonicalGraphStore
from graphfactorfactory.themes.pipeline import ThemeDiscoveryConfig, ThemeDiscoveryPipeline
from scripts.run_phase01_production import run_day

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("1H_1M_QA")


def _et_mask(series, start="10:00", end="10:59"):
    local = pd.to_datetime(series, utc=True).dt.tz_convert("America/New_York")
    hhmm = local.dt.strftime("%H:%M")
    return (hhmm >= start) & (hhmm <= end)


def _key_set(frame, columns):
    if frame.empty:
        return set()
    return set(map(tuple, frame[columns].drop_duplicates().to_numpy()))


def run_qa(month_pack_root, graph_root, theme_root, workers, config_path, overwrite=False):
    date = "2026-06-16"
    graph_root = Path(graph_root).resolve()
    theme_root = Path(theme_root).resolve()
    if overwrite:
        shutil.rmtree(graph_root, ignore_errors=True)
        shutil.rmtree(theme_root, ignore_errors=True)
    if (graph_root / "canonical" / f"date={date}").exists() or (theme_root / f"date={date}").exists():
        raise FileExistsError("QA output already exists; use new roots or --overwrite")

    config = BuildConfig.from_yaml(config_path)
    config = replace(config, frequency="1min", market_open="09:30", market_close="10:59", graph_step_minutes=1)
    if config.graph_step_minutes != 1:
        raise AssertionError("graph_step_minutes must equal 1")
    bad_steps = [
        (item.layer.name, item.lookback_minutes, item.decision_step_minutes)
        for item in LAYER_SCALES
        if item.decision_step_minutes != 1
    ]
    if bad_steps:
        raise AssertionError(f"all layer-scales must update every minute: {bad_steps}")

    glob_pattern = str(Path(month_pack_root) / "month=*" / "node_factors_1m" / "date=*" / "*.parquet")
    source = ParquetNodeFactorSource(glob_pattern)
    source_rows = source.load_date(date)
    candidate_symbols = int(source_rows["symbol"].astype(str).nunique())

    store = CanonicalGraphStore(graph_root, config)
    pipe = GraphFactorPipeline(source, store, config)
    pipe.max_threads = max(1, workers)
    pipe.task_chunk_size = 1
    logger.info("Executing Phase 0 with 09:30-09:59 retained as lookback warmup")
    pipe.build_date(date)

    theme_config = ThemeDiscoveryConfig(run_id="run_1h_1m_all35", frame_minutes=1)
    theme_pipe = ThemeDiscoveryPipeline(graph_root, theme_root, theme_config)
    run_day(theme_pipe, graph_root / "canonical" / f"date={date}", workers)
    theme_pipe.store.build_read_models()

    day_graph = graph_root / "canonical" / f"date={date}"
    edges = pd.read_parquet(day_graph / "edges.parquet")
    snapshots = pd.read_parquet(day_graph / "snapshots.parquet")
    edges = edges[_et_mask(edges["decision_time"])].copy()
    snapshots = snapshots[_et_mask(snapshots["decision_time"])].copy()

    day_theme = theme_root / f"date={date}"
    communities = pd.read_parquet(day_theme / "layer_communities.parquet")
    communities = communities[_et_mask(communities["snapshot_time"])].copy()
    communities["lookback_minutes"] = pd.to_numeric(
        communities["layer_name"].astype(str).str.extract(r"@(\d+)m$")[0],
        errors="coerce",
    ).fillna(0).astype(int)

    temporal_edges = pd.read_parquet(day_theme / "temporal_edges.parquet")
    temporal_edges = temporal_edges[_et_mask(temporal_edges["decision_time"])].copy()

    expected_frames = 60
    expected_scales = len(LAYER_SCALES)
    expected_rows = expected_frames * expected_scales
    snapshot_keys = ["decision_time", "layer_id", "lookback_minutes"]
    community_keys = ["snapshot_time", "layer_id", "lookback_minutes"]

    frame_count = int(snapshots["decision_time"].nunique())
    actual_snapshot_keys = _key_set(snapshots, snapshot_keys)
    phase0_nonempty_keys = _key_set(edges, snapshot_keys)
    phase1_processed_keys = _key_set(temporal_edges, snapshot_keys)
    phase1_community_keys = _key_set(communities, community_keys)

    degree_violations = []
    for key, group in edges.groupby(snapshot_keys, sort=False):
        degree = pd.concat([group["src_id"], group["dst_id"]]).value_counts()
        cap = int(group["degree_cap"].iloc[0])
        if (degree > cap).any():
            degree_violations.append({
                "key": tuple(map(str, key)),
                "cap": cap,
                "max_degree": int(degree.max()),
                "violations": int((degree > cap).sum()),
            })

    duplicate_edges = int(edges.duplicated(subset=snapshot_keys + ["src_id", "dst_id"]).sum())
    duplicate_memberships = 0
    for _, group in communities.groupby(community_keys, sort=False):
        seen = set()
        for members in group["members"]:
            overlap = seen.intersection(members)
            duplicate_memberships += len(overlap)
            seen.update(members)

    self_loops = int((edges["src_id"] == edges["dst_id"]).sum())
    nonfinite_weights = int((~np.isfinite(edges["weight"])).sum())
    phase1_missing_nonempty = phase0_nonempty_keys - phase1_processed_keys
    phase1_unexpected = phase1_processed_keys - phase0_nonempty_keys

    report = {
        "date": date,
        "window_et": "10:00-10:59",
        "config_path": str(Path(config_path).resolve()),
        "parameter_set_id": config.parameter_set_id,
        "config_hash": config.config_hash,
        "candidate_symbols_day": candidate_symbols,
        "decision_frames_expected": expected_frames,
        "decision_frames_actual": frame_count,
        "layer_scales_expected_per_frame": expected_scales,
        "snapshot_layer_scale_rows_expected": expected_rows,
        "snapshot_layer_scale_rows_actual": len(actual_snapshot_keys),
        "phase1_layer_scale_rows_with_communities": len(phase1_community_keys),
        "phase1_nonempty_graphs_expected": len(phase0_nonempty_keys),
        "phase1_nonempty_graphs_processed": len(phase1_processed_keys),
        "phase1_missing_nonempty_graphs": len(phase1_missing_nonempty),
        "phase1_unexpected_graphs": len(phase1_unexpected),
        "edges": int(len(edges)),
        "communities": int(len(communities)),
        "degree_cap_violation_groups": len(degree_violations),
        "degree_cap_violation_details": degree_violations[:20],
        "duplicate_edge_keys": duplicate_edges,
        "self_loops": self_loops,
        "nonfinite_weights": nonfinite_weights,
        "empty_community_members": int(communities["members"].apply(len).eq(0).sum()) if not communities.empty else 0,
        "communities_below_min_members": int(communities["members"].apply(len).lt(3).sum()) if not communities.empty else 0,
        "duplicate_same_scale_memberships": duplicate_memberships,
        "largest_community": int(communities["members"].apply(len).max()) if not communities.empty else 0,
        "all_layer_scales_step_one": all(item.decision_step_minutes == 1 for item in LAYER_SCALES),
    }
    report["pass"] = (
        frame_count == expected_frames
        and len(actual_snapshot_keys) == expected_rows
        and not phase1_missing_nonempty
        and not phase1_unexpected
        and len(degree_violations) == 0
        and duplicate_edges == 0
        and self_loops == 0
        and nonfinite_weights == 0
        and duplicate_memberships == 0
        and report["empty_community_members"] == 0
        and report["communities_below_min_members"] == 0
    )

    theme_root.mkdir(parents=True, exist_ok=True)
    (theme_root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    snapshots.to_csv(theme_root / "qa_snapshots.csv", index=False)
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise SystemExit(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--month-pack-root", required=True)
    parser.add_argument("--graph-root", required=True)
    parser.add_argument("--theme-root", required=True)
    parser.add_argument("--config", default="configs/phase0_ab_selected_v1.yaml")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_qa(args.month_pack_root, args.graph_root, args.theme_root, args.workers, args.config, args.overwrite)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
