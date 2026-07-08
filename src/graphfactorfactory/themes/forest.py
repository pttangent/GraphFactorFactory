from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ThemeForestConfig:
    """Post-process broad themes into a graph-structural theme forest.

    Semantic metadata is used only for labeling and diagnostics. It is never used
    to decide splits.
    """

    min_root_members_to_split: int = 30
    min_leaf_members: int = 5
    max_leaf_members: int = 160
    max_children_per_root: int = 12
    duplicate_jaccard_threshold: float = 0.72
    max_child_root_ratio: float = 0.85


@dataclass(frozen=True)
class ThemeForestResult:
    nodes: pd.DataFrame
    edges: pd.DataFrame
    members: pd.DataFrame


class ThemeForestBuilder:
    def __init__(self, *, config: ThemeForestConfig | None = None, metadata: pd.DataFrame | None = None):
        self.config = config or ThemeForestConfig()
        self.metadata = self._normalize_metadata(metadata)

    @staticmethod
    def _normalize_metadata(metadata: pd.DataFrame | None) -> pd.DataFrame:
        if metadata is None or metadata.empty:
            return pd.DataFrame()
        frame = metadata.copy()
        frame = frame.rename(
            columns={
                "company_name": "company",
                "sector_code": "sector",
                "industry_code": "industry",
            }
        )
        if "symbol_id" in frame.columns:
            frame = frame.set_index("symbol_id", drop=False)
        return frame

    @staticmethod
    def _market_cap_bucket(value) -> str:
        try:
            if pd.isna(value):
                return "Unknown"
            cap = float(value)
        except Exception:
            return "Unknown"
        if cap >= 200e9:
            return "Mega-Cap"
        if cap >= 10e9:
            return "Large-Cap"
        if cap >= 2e9:
            return "Mid-Cap"
        if cap >= 300e6:
            return "Small-Cap"
        return "Micro-Cap"

    @staticmethod
    def _jaccard(left: set[int], right: set[int]) -> float:
        union = len(left | right)
        return len(left & right) / union if union else 0.0

    def _member_summary(self, members: Iterable[int]) -> dict:
        values = list(map(int, members))
        if not values or self.metadata.empty:
            return {
                "metadata_coverage": 0.0,
                "top_symbols": "",
                "top_companies": "",
                "top_sector": "Unknown",
                "top_sector_share": 0.0,
                "top_industry": "Unknown",
                "top_industry_share": 0.0,
                "market_cap_bucket": "Unknown",
            }
        rows = self.metadata.reindex(values)
        coverage = float(rows.get("symbol", pd.Series(index=rows.index)).notna().mean())
        symbols = rows.get("symbol", pd.Series(dtype=str)).dropna().astype(str).head(15).tolist()
        companies = rows.get("company", pd.Series(dtype=str)).dropna().astype(str).head(8).tolist()
        sectors = rows.get("sector", pd.Series(dtype=str)).dropna().astype(str)
        industries = rows.get("industry", pd.Series(dtype=str)).dropna().astype(str)
        sector_counts = Counter(sectors)
        industry_counts = Counter(industries)
        top_sector, top_sector_count = sector_counts.most_common(1)[0] if sector_counts else ("Unknown", 0)
        top_industry, top_industry_count = industry_counts.most_common(1)[0] if industry_counts else ("Unknown", 0)
        caps = rows.get("market_cap", pd.Series(dtype=float)).dropna()
        return {
            "metadata_coverage": coverage,
            "top_symbols": ",".join(symbols),
            "top_companies": " | ".join(companies),
            "top_sector": top_sector,
            "top_sector_share": float(top_sector_count / max(1, len(sectors))),
            "top_industry": top_industry,
            "top_industry_share": float(top_industry_count / max(1, len(industries))),
            "market_cap_bucket": self._market_cap_bucket(caps.median()) if len(caps) else "Unknown",
        }

    def build_snapshot_forest(
        self,
        *,
        snapshot_time,
        themes: pd.DataFrame,
        layer_communities: pd.DataFrame | None = None,
        subcommunities: pd.DataFrame | None = None,
    ) -> ThemeForestResult:
        unit_frames = []
        for frame in (subcommunities, layer_communities):
            if frame is not None and not frame.empty:
                needed = frame[frame["snapshot_time"] == snapshot_time]
                if not needed.empty:
                    unit_frames.append(needed[["layer_name", "members"]])
        units = pd.concat(unit_frames, ignore_index=True) if unit_frames else pd.DataFrame(columns=["layer_name", "members"])
        structural_units = []
        for row in units.itertuples(index=False):
            members = set(map(int, row.members))
            if self.config.min_leaf_members <= len(members) <= self.config.max_leaf_members:
                structural_units.append((str(row.layer_name), members))

        nodes: list[dict] = []
        edges: list[dict] = []
        memberships: list[dict] = []
        for theme in themes.itertuples(index=False):
            root_members = set(map(int, theme.members))
            root_id = f"{theme.theme_instance_id}:root"
            root_summary = self._member_summary(sorted(root_members))
            root = {
                "snapshot_time": snapshot_time,
                "forest_node_id": root_id,
                "parent_forest_node_id": None,
                "theme_instance_id": theme.theme_instance_id,
                "level": 0,
                "node_type": "root_market" if bool(theme.is_market_mode) else "root",
                "member_count": len(root_members),
                "split_status": "pending",
                "split_method": "all_layer_structural_subcommunity_greedy_v1",
                "semantic_used_for_split": False,
                "source_layers": ",".join(map(str, getattr(theme, "source_layers", []))),
                "consensus_score": float(getattr(theme, "consensus_score", np.nan)),
                "theme_quality_score": float(getattr(theme, "theme_quality_score", np.nan)),
                "child_count": 0,
                "evidence_units": 0,
                "evidence_layers": 0,
                **root_summary,
            }
            nodes.append(root)
            if len(root_members) < self.config.min_root_members_to_split:
                nodes[-1]["split_status"] = "too_small"
                memberships.extend(self._membership_rows(snapshot_time, root_id, root_members, "root_leaf"))
                continue

            candidates: dict[tuple[int, ...], set[int]] = {}
            support_layers: dict[tuple[int, ...], set[str]] = defaultdict(set)
            for layer_name, unit_members in structural_units:
                child_members = root_members & unit_members
                size = len(child_members)
                if size < self.config.min_leaf_members:
                    continue
                if size > min(self.config.max_leaf_members, int(self.config.max_child_root_ratio * len(root_members))):
                    continue
                key = tuple(sorted(child_members))
                candidates[key] = child_members
                support_layers[key].add(layer_name)

            nodes[-1]["evidence_units"] = len(candidates)
            nodes[-1]["evidence_layers"] = len(set().union(*support_layers.values())) if support_layers else 0
            if len(candidates) < 2:
                nodes[-1]["split_status"] = "no_structural_children"
                memberships.extend(self._membership_rows(snapshot_time, root_id, root_members, "root_leaf"))
                continue

            selected: list[set[int]] = []
            selected_keys: list[tuple[int, ...]] = []
            ranked = sorted(candidates.items(), key=lambda kv: (-len(support_layers[kv[0]]), -len(kv[1]), kv[0]))
            for key, child_members in ranked:
                if len(selected) >= self.config.max_children_per_root:
                    break
                if any(self._jaccard(child_members, existing) > self.config.duplicate_jaccard_threshold for existing in selected):
                    continue
                selected.append(child_members)
                selected_keys.append(key)

            if len(selected) < 2:
                nodes[-1]["split_status"] = "no_diverse_children"
                memberships.extend(self._membership_rows(snapshot_time, root_id, root_members, "root_leaf"))
                continue

            nodes[-1]["split_status"] = "split"
            nodes[-1]["child_count"] = len(selected)
            for child_index, child_members in enumerate(selected, start=1):
                child_id = f"{theme.theme_instance_id}:leaf:{child_index:02d}"
                key = selected_keys[child_index - 1]
                layers = sorted(support_layers.get(key, set()))
                nodes.append(
                    {
                        "snapshot_time": snapshot_time,
                        "forest_node_id": child_id,
                        "parent_forest_node_id": root_id,
                        "theme_instance_id": theme.theme_instance_id,
                        "level": 1,
                        "node_type": "leaf_market" if bool(theme.is_market_mode) else "leaf",
                        "member_count": len(child_members),
                        "split_status": "leaf",
                        "split_method": "all_layer_structural_subcommunity_greedy_v1",
                        "semantic_used_for_split": False,
                        "source_layers": ",".join(layers),
                        "consensus_score": float(getattr(theme, "consensus_score", np.nan)),
                        "theme_quality_score": float(getattr(theme, "theme_quality_score", np.nan)),
                        "child_count": 0,
                        "evidence_units": 1,
                        "evidence_layers": len(layers),
                        **self._member_summary(sorted(child_members)),
                    }
                )
                edges.append(
                    {
                        "snapshot_time": snapshot_time,
                        "parent_forest_node_id": root_id,
                        "child_forest_node_id": child_id,
                        "relation_type": "structural_child",
                        "child_member_count": len(child_members),
                        "child_ratio": float(len(child_members) / max(1, len(root_members))),
                        "source_layer_count": len(layers),
                    }
                )
                memberships.extend(self._membership_rows(snapshot_time, child_id, child_members, "leaf"))
        return ThemeForestResult(pd.DataFrame(nodes), pd.DataFrame(edges), pd.DataFrame(memberships))

    @staticmethod
    def _membership_rows(snapshot_time, forest_node_id: str, members: Iterable[int], role: str) -> list[dict]:
        return [
            {"snapshot_time": snapshot_time, "forest_node_id": forest_node_id, "symbol_id": int(symbol_id), "membership_role": role}
            for symbol_id in sorted(map(int, members))
        ]
