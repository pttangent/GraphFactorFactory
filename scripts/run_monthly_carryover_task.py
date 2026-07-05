from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


def atomic_write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temp, index=False)
    os.replace(temp, path)


def as_members(value: Any) -> set[int]:
    return {int(item) for item in value}


def load_day(root: Path, trade_date: str, min_size: int) -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    path = root / f"date={trade_date}" / "layer_communities.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pq.read_table(path).to_pandas()
    required = {"snapshot_time", "layer_id", "layer_name", "community_id", "members"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["snapshot_time"] = pd.to_datetime(frame["snapshot_time"])
    frame["size"] = frame["members"].map(len)
    frame = frame[frame["size"] >= min_size]
    frame = frame.sort_values(["snapshot_time", "layer_id", "community_id"], kind="mergesort")
    return frame, sorted(frame["snapshot_time"].unique())


def state_at(frame: pd.DataFrame, timestamp: pd.Timestamp) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in frame[frame["snapshot_time"] == timestamp].iterrows():
        members = as_members(row["members"])
        rows.append(
            {
                "id": f"{int(row['layer_id'])}:{pd.Timestamp(timestamp).isoformat()}:{int(row['community_id'])}",
                "time": pd.Timestamp(timestamp),
                "layer": int(row["layer_id"]),
                "layer_name": str(row["layer_name"]),
                "community_id": int(row["community_id"]),
                "size": len(members),
                "members": members,
                "core": members,
            }
        )
    return rows


def containment(left: set[int], right: set[int]) -> float:
    return len(left & right) / min(len(left), len(right)) if left and right else 0.0


def jaccard(left: set[int], right: set[int]) -> float:
    return len(left & right) / len(left | right) if left and right else 0.0


def size_similarity(left: float, right: float) -> float:
    return math.exp(-abs(math.log(max(1.0, left)) - math.log(max(1.0, right))) / 0.7)


def fingerprint(proto: dict[str, Any], current: dict[str, Any]) -> float:
    if int(proto["layer"]) != int(current["layer"]):
        return 0.0
    return (
        0.40 * containment(proto["core"], current["core"])
        + 0.30 * containment(proto["members"], current["members"])
        + 0.15 * jaccard(proto["members"], current["members"])
        + 0.15 * size_similarity(proto["mean_size"], current["size"])
    )


def init_proto(previous: dict[str, Any], opened: dict[str, Any]) -> dict[str, Any]:
    member_counts = Counter(previous["members"])
    member_counts.update(opened["members"])
    core_counts = Counter(previous["core"])
    core_counts.update(opened["core"])
    return {
        "layer": previous["layer"],
        "n": 2,
        "member_counts": member_counts,
        "core_counts": core_counts,
        "members": set(previous["members"]) | set(opened["members"]),
        "core": set(previous["core"]) | set(opened["core"]),
        "mean_size": (previous["size"] + opened["size"]) / 2.0,
    }


def update_proto(proto: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    result = dict(proto)
    result["member_counts"] = Counter(proto["member_counts"])
    result["core_counts"] = Counter(proto["core_counts"])
    result["n"] += 1
    result["member_counts"].update(current["members"])
    result["core_counts"].update(current["core"])
    result["members"] = {item for item, count in result["member_counts"].items() if count / result["n"] >= 0.35}
    result["core"] = {item for item, count in result["core_counts"].items() if count / result["n"] >= 0.35}
    result["mean_size"] = ((proto["mean_size"] * proto["n"]) + current["size"]) / result["n"]
    return result


def bridge_candidates(previous: list[dict[str, Any]], current: list[dict[str, Any]], entry: float) -> list[dict[str, Any]]:
    inverted: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, community in enumerate(current):
        for member in community["members"]:
            inverted[(community["layer"], member)].append(index)
    candidates: list[dict[str, Any]] = []
    for previous_index, prior in enumerate(previous):
        counts: dict[int, int] = defaultdict(int)
        for member in prior["members"]:
            for current_index in inverted.get((prior["layer"], member), ()):
                counts[current_index] += 1
        for current_index, intersection in counts.items():
            opened = current[current_index]
            score = intersection / min(len(prior["members"]), len(opened["members"]))
            if score >= entry:
                proto = {"layer": prior["layer"], "members": prior["members"], "core": prior["core"], "mean_size": prior["size"]}
                candidates.append(
                    {
                        "previous_index": previous_index,
                        "current_index": current_index,
                        "containment": score,
                        "fingerprint": fingerprint(proto, opened),
                        "layer": prior["layer"],
                    }
                )
    candidates.sort(key=lambda item: (-item["containment"], -item["fingerprint"], item["previous_index"], item["current_index"]))
    used_previous: set[int] = set()
    used_current: set[int] = set()
    selected: list[dict[str, Any]] = []
    for candidate in candidates:
        if candidate["previous_index"] in used_previous or candidate["current_index"] in used_current:
            continue
        used_previous.add(candidate["previous_index"])
        used_current.add(candidate["current_index"])
        selected.append(candidate)
    return selected


def match_active(active: list[dict[str, Any]], current: list[dict[str, Any]], arm: dict[str, Any]) -> dict[int, tuple[int, float, float]]:
    candidates: list[tuple[float, int, int, float, float]] = []
    for active_index, path in enumerate(active):
        for current_index, community in enumerate(current):
            if path["last"]["layer"] != community["layer"]:
                continue
            overlap = containment(path["last"]["members"], community["members"])
            fp_score = fingerprint(path["proto"], community)
            accepted = overlap >= float(arm.get("stay_containment", 0.20))
            assisted_overlap = arm.get("assisted_stay_containment")
            assisted_fp = arm.get("assisted_stay_fingerprint")
            if assisted_overlap is not None and assisted_fp is not None:
                accepted = accepted or (overlap >= float(assisted_overlap) and fp_score >= float(assisted_fp))
            if accepted:
                candidates.append((0.65 * overlap + 0.35 * fp_score, active_index, current_index, overlap, fp_score))
    candidates.sort(reverse=True)
    used_active: set[int] = set()
    used_current: set[int] = set()
    result: dict[int, tuple[int, float, float]] = {}
    for _, active_index, current_index, overlap, fp_score in candidates:
        if active_index in used_active or current_index in used_current:
            continue
        used_active.add(active_index)
        used_current.add(current_index)
        result[active_index] = (current_index, overlap, fp_score)
    return result


def follow_paths(
    previous_roots: list[dict[str, Any]],
    open_roots: list[dict[str, Any]],
    future_states: list[list[dict[str, Any]]],
    arm_name: str,
    arm: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    active: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []
    dormant: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    revival_rows: list[dict[str, Any]] = []
    max_weak = int(arm.get("max_weak_gap", 0))
    revival_enabled = bool(arm.get("revival", False))
    max_dormant = int(arm.get("max_dormant_states", 0))
    max_revivals = int(arm.get("max_revivals", 0))
    pre_active = int(arm.get("pre_revival_active_states", 0))
    revival_fp = float(arm.get("revival_fingerprint", 1.0))
    breadth = arm.get("breadth_expansion")
    post_confirm = int(arm.get("post_revival_confirmation_states", 0))

    for path_index, (prior, opened) in enumerate(zip(previous_roots, open_roots)):
        path = {
            "path_id": f"{arm_name}:{prior['id']}->{opened['id']}",
            "last": opened,
            "proto": init_proto(prior, opened),
            "age": 1,
            "active_hits": 1,
            "weak_gap": 0,
            "dormant_gap": 0,
            "revivals": 0,
            "post_revival_hits": 0,
        }
        active.append(path)
        state_rows.append({"path_id": path["path_id"], "time": opened["time"], "status": "active", "layer": opened["layer"], "size": opened["size"]})

    for state_index, current in enumerate(future_states, start=1):
        matches = match_active(active, current, arm)
        next_active: list[dict[str, Any]] = []
        used_current: set[int] = set()
        for active_index, path in enumerate(active):
            if active_index in matches:
                current_index, overlap, fp_score = matches[active_index]
                used_current.add(current_index)
                community = current[current_index]
                updated = dict(path)
                updated["last"] = community
                updated["proto"] = update_proto(path["proto"], community)
                updated["age"] += 1
                updated["active_hits"] += 1
                updated["weak_gap"] = 0
                if updated["revivals"]:
                    updated["post_revival_hits"] += 1
                next_active.append(updated)
                state_rows.append({"path_id": updated["path_id"], "time": community["time"], "status": "active", "layer": community["layer"], "size": community["size"], "containment": overlap, "fingerprint": fp_score})
            elif path["weak_gap"] < max_weak:
                updated = dict(path)
                updated["age"] += 1
                updated["weak_gap"] += 1
                next_active.append(updated)
                state_rows.append({"path_id": updated["path_id"], "time": current[0]["time"] if current else None, "status": "weak", "layer": updated["last"]["layer"], "size": updated["last"]["size"]})
            elif revival_enabled and path["active_hits"] >= pre_active and path["revivals"] < max_revivals:
                updated = dict(path)
                updated["dormant_gap"] = 1
                dormant.append(updated)
            else:
                completed.append(path)
        active = next_active

        if revival_enabled and dormant:
            revival_candidates: list[tuple[float, int, int]] = []
            for dormant_index, path in enumerate(dormant):
                for current_index, community in enumerate(current):
                    if current_index in used_current or path["last"]["layer"] != community["layer"]:
                        continue
                    fp_score = fingerprint(path["proto"], community)
                    breadth_ok = breadth is None or community["size"] >= path["proto"]["mean_size"] * (1.0 + float(breadth))
                    if fp_score >= revival_fp and breadth_ok:
                        revival_candidates.append((fp_score, dormant_index, current_index))
            revival_candidates.sort(reverse=True)
            used_dormant: set[int] = set()
            for fp_score, dormant_index, current_index in revival_candidates:
                if dormant_index in used_dormant or current_index in used_current:
                    continue
                used_dormant.add(dormant_index)
                used_current.add(current_index)
                path = dormant[dormant_index]
                community = current[current_index]
                updated = dict(path)
                updated["last"] = community
                updated["proto"] = update_proto(path["proto"], community)
                updated["age"] += updated["dormant_gap"] + 1
                updated["active_hits"] += 1
                updated["revivals"] += 1
                updated["post_revival_hits"] = 1
                updated["dormant_gap"] = 0
                active.append(updated)
                revival_rows.append({
                    "path_id": updated["path_id"],
                    "time": community["time"],
                    "layer": community["layer"],
                    "fingerprint": fp_score,
                    "size": community["size"],
                    "prototype_size": path["proto"]["mean_size"],
                    "breadth_expansion": community["size"] / max(1.0, path["proto"]["mean_size"]) - 1.0,
                    "dormant_gap": path["dormant_gap"],
                })
            retained: list[dict[str, Any]] = []
            for dormant_index, path in enumerate(dormant):
                if dormant_index in used_dormant:
                    continue
                updated = dict(path)
                updated["dormant_gap"] += 1
                if updated["dormant_gap"] > max_dormant:
                    completed.append(updated)
                else:
                    retained.append(updated)
            dormant = retained

    final_paths = completed + active + dormant
    outcome_rows = []
    for path in final_paths:
        confirmed_revival = bool(path["revivals"] and path["post_revival_hits"] >= max(1, post_confirm))
        outcome_rows.append({
            "path_id": path["path_id"],
            "layer": path["last"]["layer"],
            "age_states": path["age"],
            "active_hits": path["active_hits"],
            "revivals": path["revivals"],
            "post_revival_hits": path["post_revival_hits"],
            "confirmed_revival": confirmed_revival,
            "persistent_3": path["active_hits"] >= 3,
            "persistent_5": path["active_hits"] >= 5,
            "persistent_10": path["active_hits"] >= 10,
            "persistent_20": path["active_hits"] >= 20,
        })
    return pd.DataFrame(state_rows), pd.DataFrame(revival_rows), pd.DataFrame(outcome_rows)


def matched_controls(open_rows: list[dict[str, Any]], used: set[int], roots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pool = [(index, row) for index, row in enumerate(open_rows) if index not in used]
    selected: list[dict[str, Any]] = []
    taken: set[int] = set()
    for root in roots:
        options = [
            (abs(math.log(max(1, row["size"])) - math.log(max(1, root["size"]))), index, row)
            for index, row in pool
            if index not in taken and row["layer"] == root["layer"]
        ]
        if options:
            _, index, row = min(options)
            taken.add(index)
            selected.append(row)
    return selected


def evaluate_unit(
    close_state: list[dict[str, Any]],
    open_states: list[list[dict[str, Any]]],
    arm_name: str,
    arm: dict[str, Any],
    unit_dir: Path,
    control_name: str,
    replicate: int,
) -> None:
    entry = float(arm.get("entry_containment", 0.18))
    candidates = bridge_candidates(close_state, open_states[0], entry)
    fingerprint_confirm = arm.get("fingerprint_confirm")
    chosen = [candidate for candidate in candidates if fingerprint_confirm is None or candidate["fingerprint"] >= float(fingerprint_confirm)]
    previous_roots = [close_state[item["previous_index"]] for item in chosen]
    open_roots = [open_states[0][item["current_index"]] for item in chosen]
    used = {item["current_index"] for item in chosen}
    controls = matched_controls(open_states[0], used, open_roots)

    path_states, revival_events, outcomes = follow_paths(previous_roots, open_roots, open_states[1:], arm_name, arm)
    control_states, control_revivals, control_outcomes = follow_paths(controls, controls, open_states[1:], arm_name, arm)
    for frame, kind in ((outcomes, "bridge"), (control_outcomes, "open_birth_control")):
        if not frame.empty:
            frame["kind"] = kind
            frame["control"] = control_name
            frame["replicate"] = replicate
    combined_outcomes = pd.concat([outcomes, control_outcomes], ignore_index=True) if not outcomes.empty or not control_outcomes.empty else pd.DataFrame()

    bridge_frame = pd.DataFrame(chosen)
    if not bridge_frame.empty:
        bridge_frame["arm"] = arm_name
        bridge_frame["control"] = control_name
        bridge_frame["replicate"] = replicate
    atomic_write(bridge_frame, unit_dir / "bridge_candidates.parquet")
    atomic_write(pd.concat([path_states, control_states], ignore_index=True), unit_dir / "path_states.parquet")
    atomic_write(pd.concat([revival_events, control_revivals], ignore_index=True), unit_dir / "revival_events.parquet")
    atomic_write(pd.DataFrame([{"matched_controls": len(controls)}]), unit_dir / "matched_controls.parquet")
    atomic_write(combined_outcomes, unit_dir / "outcomes.parquet")
    manifest = {
        "arm": arm_name,
        "control": control_name,
        "replicate": replicate,
        "bridge_count": len(chosen),
        "control_count": len(controls),
    }
    (unit_dir / "task_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (unit_dir / "_SUCCESS").write_text("success\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--date-from", required=True)
    parser.add_argument("--date-to", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--phase1-root", default="outputs/theme_discovery_phase1")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    phase1_root = Path(args.phase1_root)
    output_root = Path(args.output_dir)
    min_size = int(config.get("min_community_size", 20))
    horizon = int(config.get("open_horizon_states", 30))

    close_frame, close_times = load_day(phase1_root, args.date_from, min_size)
    open_frame, open_times = load_day(phase1_root, args.date_to, min_size)
    close_state = state_at(close_frame, close_times[-1])
    actual_states = [state_at(open_frame, timestamp) for timestamp in open_times[:horizon]]
    if not actual_states:
        raise RuntimeError(f"No open states for {args.date_to}")

    arms = config.get("arms", {})
    if not isinstance(arms, dict) or not arms:
        raise ValueError("config.arms must be a non-empty mapping")

    null_mapping_path = output_root / "null_mapping.parquet"
    null_rows = pd.read_parquet(null_mapping_path) if null_mapping_path.exists() else pd.DataFrame()
    boundary_nulls = null_rows[
        (null_rows["actual_date_from"] == args.date_from)
        & (null_rows["actual_date_to"] == args.date_to)
    ] if not null_rows.empty else pd.DataFrame()
    null_cache: dict[str, list[list[dict[str, Any]]]] = {}

    expected_units: list[Path] = []
    for arm_name, raw_arm in arms.items():
        arm = dict(raw_arm)
        if "inherit" in arm:
            parent = dict(arms[arm["inherit"]])
            parent.update({key: value for key, value in arm.items() if key != "inherit"})
            arm = parent

        actual_dir = output_root / "shards" / f"date_from={args.date_from}" / f"date_to={args.date_to}" / f"arm={arm_name}" / "control=actual" / "replicate=0"
        expected_units.append(actual_dir)
        if not (actual_dir / "_SUCCESS").exists():
            evaluate_unit(close_state, actual_states, arm_name, arm, actual_dir, "actual", 0)

        for _, null_row in boundary_nulls.iterrows():
            null_date = str(null_row["null_date_to"])
            replicate = int(null_row["replicate"])
            if null_date not in null_cache:
                null_frame, null_times = load_day(phase1_root, null_date, min_size)
                null_cache[null_date] = [state_at(null_frame, timestamp) for timestamp in null_times[:horizon]]
            unit_dir = output_root / "shards" / f"date_from={args.date_from}" / f"date_to={args.date_to}" / f"arm={arm_name}" / "control=day_order" / f"replicate={replicate}"
            expected_units.append(unit_dir)
            if not (unit_dir / "_SUCCESS").exists():
                evaluate_unit(close_state, null_cache[null_date], arm_name, arm, unit_dir, "day_order", replicate)

    missing = [str(path) for path in expected_units if not (path / "_SUCCESS").exists()]
    if missing:
        raise RuntimeError(f"Boundary incomplete; missing units: {missing[:10]}")
    print(f"[{args.date_from} -> {args.date_to}] completed {len(expected_units)} units", flush=True)


if __name__ == "__main__":
    main()
