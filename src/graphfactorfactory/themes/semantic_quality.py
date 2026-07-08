from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
import pandas as pd

from .models import SemanticLabel, ThemeCandidate, LifecycleRecord


class MetadataSemanticLabeler:
    def __init__(self, metadata: pd.DataFrame | None = None, dictionary_version="metadata-v1"):
        self.metadata = metadata.copy() if metadata is not None else pd.DataFrame()
        self.dictionary_version = dictionary_version
        if not self.metadata.empty and "symbol_id" in self.metadata:
            self.metadata = self.metadata.set_index("symbol_id", drop=False)

    def label(self, themes: list[ThemeCandidate]) -> list[SemanticLabel]:
        output=[]
        for theme in themes:
            rows = self.metadata.reindex(theme.members).dropna(how="all") if not self.metadata.empty else pd.DataFrame()
            symbols = rows.get("symbol", pd.Series(index=rows.index, dtype=str)).dropna().astype(str).tolist()[:5]
            sectors = Counter(rows.get("sector", pd.Series(dtype=str)).dropna().astype(str))
            industries = Counter(rows.get("industry", pd.Series(dtype=str)).dropna().astype(str))
            companies = rows.get("company", pd.Series(index=rows.index, dtype=str)).dropna().astype(str).tolist()[:5]
            
            total_valid = len(rows.get("sector", pd.Series()).dropna()) or 1
            sector_dist = {k: float(v/total_valid) for k, v in sectors.most_common(5)}
            total_valid_ind = len(rows.get("industry", pd.Series()).dropna()) or 1
            industry_dist = {k: float(v/total_valid_ind) for k, v in industries.most_common(5)}
            
            sector = sectors.most_common(1)[0][0] if sectors else "Mixed"
            industry = industries.most_common(1)[0][0] if industries else "Mixed"
            
            market_cap_bucket = "Unknown"
            if "market_cap" in rows and not rows["market_cap"].dropna().empty:
                med_cap = rows["market_cap"].median()
                if med_cap >= 200e9: market_cap_bucket = "Mega-Cap"
                elif med_cap >= 10e9: market_cap_bucket = "Large-Cap"
                elif med_cap >= 2e9: market_cap_bucket = "Mid-Cap"
                elif med_cap >= 300e6: market_cap_bucket = "Small-Cap"
                else: market_cap_bucket = "Micro-Cap"
            
            tags = tuple(sorted(set(theme.source_families) | ({sector} if sector != "Mixed" else set()) | ({industry} if industry != "Mixed" else set())))
            coherence = (sectors.most_common(1)[0][1] / len(rows)) if sectors and len(rows) else min(1.0, 0.4 + 0.1 * len(theme.source_families))
            fallback = [str(value) for value in theme.members[:3]]
            title_members = symbols[:3] or fallback
            
            if theme.is_market_mode:
                short = f"Market Mode: {sector}/{industry} ({market_cap_bucket})"
            else:
                short = f"{sector}/{industry} ({market_cap_bucket}): {', '.join(title_members)}"
            
            output.append(SemanticLabel(
                theme_instance_id=theme.theme_instance_id,
                label_short=short,
                label_long=short,
                sector_summary=sector,
                industry_summary=industry,
                tags=tags,
                top_companies=tuple(companies),
                top_symbols=tuple(symbols),
                sector_distribution=json.dumps(sector_dist),
                industry_distribution=json.dumps(industry_dist),
                market_cap_bucket=market_cap_bucket,
                semantic_coherence_score=float(coherence),
                explanation="Deterministic metadata-first label from members and supporting layer families.",
                semantic_method="metadata_dictionary",
                dictionary_version=self.dictionary_version
            ))
        return output


class ThemeQualityScorer:
    def score(self, themes: list[ThemeCandidate], semantics: list[SemanticLabel], lifecycle: list[LifecycleRecord], node_features: pd.DataFrame | None = None) -> list[ThemeCandidate]:
        semantic_map={item.theme_instance_id:item for item in semantics}; life_map={item.theme_instance_id:item for item in lifecycle if item.status=="active"}
        result=[]
        for theme in themes:
            semantic = semantic_map.get(theme.theme_instance_id); life=life_map.get(theme.theme_instance_id)
            semantic_score = semantic.semantic_coherence_score if semantic else 0.0
            stability = life.member_retention if life else theme.stability_score
            flow_support=0.0
            if node_features is not None and not node_features.empty:
                rows=node_features[node_features.symbol_id.isin(theme.members)]
                if "neighbor_signed_flow" in rows:
                    flow_support=float(min(1.0, rows.neighbor_signed_flow.abs().mean()))
            quality = 0.35*theme.structure_score + 0.25*theme.consensus_score + 0.20*stability + 0.10*semantic_score + 0.10*flow_support
            result.append(replace(theme, flow_support_score=flow_support, stability_score=stability, semantic_coherence_score=semantic_score, theme_quality_score=float(quality), quality_breakdown={"structure":theme.structure_score,"consensus":theme.consensus_score,"stability":stability,"semantic":semantic_score,"flow":flow_support}))
        return result
