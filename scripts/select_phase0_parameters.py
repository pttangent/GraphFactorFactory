from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


GOOD_METRICS = (
    "strong_edge_ratio_q90",
    "strong_node_coverage_q90",
    "weight_p95",
    "weight_p99",
    "weighted_modularity",
    "within_community_weight_ratio",
    "strong_persistence",
)
BAD_METRICS = ("degree_cap_saturation", "empty_rate")
REGIME_METRICS = (
    "strong_edge_ratio_q90",
    "strong_node_coverage_q90",
    "weight_p95",
    "weight_p99",
    "tail_mass_95",
    "node_coverage",
    "community_hhi",
    "weighted_modularity",
    "within_community_weight_ratio",
    "strong_birth_rate",
    "strong_death_rate",
    "strong_persistence",
)


def parse_candidate(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("candidate must be NAME=/path/to/diagnostics")
    name, raw_path = value.split("=", 1)
    return name, Path(raw_path).expanduser().resolve()


def zscore_within(frame: pd.DataFrame, column: str) -> pd.Series:
    def transform(values: pd.Series) -> pd.Series:
        std = values.std(ddof=0)
        return (values - values.mean()) / (std if std and np.isfinite(std) else 1.0)

    return frame.groupby(["layer_id", "lookback_minutes"])[column].transform(transform)


def load_candidate(name: str, root: Path, parameters: dict) -> pd.DataFrame:
    snapshot = pd.read_parquet(root / "snapshot_market_diagnostics.parquet")
    temporal = pd.read_parquet(root / "temporal_edge_diagnostics.parquet")
    keys = ["trade_date", "decision_time", "layer_id", "lookback_minutes"]
    merged = snapshot.merge(temporal, on=keys, how="left")
    merged["candidate"] = name
    merged["empty_rate"] = (merged["edge_count"] == 0).astype(float)
    merged["top_k"] = int(parameters[name]["top_k"])
    merged["degree_cap"] = int(parameters[name]["degree_cap"])
    merged["minimum_similarity"] = float(parameters[name]["minimum_similarity"])
    if "degree_cap_saturation" not in merged:
        max_edges = merged.get("universe_count", pd.Series(1.0, index=merged.index)) * merged["degree_cap"] / 2.0
        merged["degree_cap_saturation"] = merged["edge_count"] / max_edges.clip(lower=1.0)
    return merged


def score_candidates(frame: pd.DataFrame, active_dates: set[str], inactive_dates: set[str]) -> pd.DataFrame:
    group_keys = ["candidate", "layer_id", "lookback_minutes"]
    available_good = [column for column in GOOD_METRICS if column in frame]
    available_bad = [column for column in BAD_METRICS if column in frame]
    aggregate_columns = available_good + available_bad
    scores = frame.groupby(group_keys)[aggregate_columns].mean(numeric_only=True).reset_index()

    regime_rows = []
    for keys, group in frame.groupby(group_keys, sort=True):
        row = dict(zip(group_keys, keys))
        effect_sizes = []
        for metric in REGIME_METRICS:
            if metric not in group:
                continue
            active = group[group["trade_date"].astype(str).isin(active_dates)][metric]
            inactive = group[group["trade_date"].astype(str).isin(inactive_dates)][metric]
            pooled_std = group[metric].std()
            if active.empty or inactive.empty or not pooled_std or not np.isfinite(pooled_std):
                continue
            effect_sizes.append(abs(float(active.mean() - inactive.mean()) / float(pooled_std)))
        row["regime_separation"] = float(np.mean(effect_sizes)) if effect_sizes else 0.0
        regime_rows.append(row)
    scores = scores.merge(pd.DataFrame(regime_rows), on=group_keys, how="left")

    reward_columns = available_good + ["regime_separation"]
    penalty_columns = available_bad
    for column in reward_columns + penalty_columns:
        scores[f"{column}_z"] = zscore_within(scores, column)
    scores["reward_score"] = scores[[f"{column}_z" for column in reward_columns]].mean(axis=1)
    scores["penalty_score"] = scores[[f"{column}_z" for column in penalty_columns]].mean(axis=1) if penalty_columns else 0.0
    scores["multiobjective_score"] = scores["reward_score"] - 0.75 * scores["penalty_score"]
    return scores


def build_registry(winners: pd.DataFrame, parameters: dict, parameter_set_id: str) -> dict:
    overrides = {}
    for row in winners.itertuples(index=False):
        selected = parameters[row.candidate]
        key = f"scale:{row.layer_name}:{int(row.lookback_minutes)}"
        overrides[key] = {
            "top_k": int(selected["top_k"]),
            "degree_cap": int(selected["degree_cap"]),
            "minimum_similarity": float(selected["minimum_similarity"]),
        }
    return {
        "parameter_set_id": parameter_set_id,
        "graph_parameter_overrides": overrides,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Select Phase 0 parameters by layer and scale")
    parser.add_argument("--candidate", action="append", required=True, type=parse_candidate)
    parser.add_argument("--parameters-json", required=True)
    parser.add_argument("--layers", required=True, help="dimensions/layers.parquet")
    parser.add_argument("--active-dates", nargs="+", required=True)
    parser.add_argument("--inactive-dates", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--parameter-set-id", default="phase0_selected_v1")
    args = parser.parse_args()

    parameters = json.loads(Path(args.parameters_json).read_text())
    frames = [load_candidate(name, root, parameters) for name, root in args.candidate]
    combined = pd.concat(frames, ignore_index=True)
    scores = score_candidates(combined, set(args.active_dates), set(args.inactive_dates))

    layers = pd.read_parquet(args.layers)[["layer_id", "name", "family"]].rename(columns={"name": "layer_name"})
    scores = scores.merge(layers, on="layer_id", how="left")
    winners = (
        scores.sort_values("multiobjective_score", ascending=False)
        .groupby(["layer_id", "lookback_minutes"], as_index=False)
        .first()
    )

    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output / "parameter_candidate_scores.csv", index=False)
    winners.to_csv(output / "selected_parameters_by_layer_scale.csv", index=False)
    registry = build_registry(winners, parameters, args.parameter_set_id)
    (output / "selected_parameter_registry.yaml").write_text(yaml.safe_dump(registry, sort_keys=False), encoding="utf-8")
    print(yaml.safe_dump(registry, sort_keys=False))


if __name__ == "__main__":
    main()
