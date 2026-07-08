import pandas as pd
from pathlib import Path
import json
import logging
import sys
from dataclasses import dataclass, asdict

from graphfactorfactory.themes.semantic_quality import MetadataSemanticLabeler, ThemeQualityScorer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)

@dataclass
class MockTheme:
    theme_instance_id: str
    members: tuple[int, ...]
    source_families: tuple[str, ...]
    is_market_mode: bool
    structure_score: float
    consensus_score: float
    stability_score: float
    flow_support_score: float

def main():
    graph_root = Path("D:/DEV/US-Stock/GraphFactorFactory/data/graph_store_6m")
    theme_root = Path("D:/DEV/US-Stock/GraphFactorFactory/data/theme_store_6m")
    metadata_path = "D:/DEV/US-Stock/GraphFactorFactory/data/metadata/symbol_metadata.parquet"

    symbols = pd.read_parquet(graph_root / "dimensions" / "symbols.parquet")
    meta = pd.read_parquet(metadata_path)
    meta = meta.rename(columns={
        "company_name": "company",
        "sector_code": "sector",
        "industry_code": "industry",
    })
    metadata = pd.merge(symbols, meta, on='symbol', how='left')

    labeler = MetadataSemanticLabeler(metadata)

    dates = sorted([d for d in theme_root.glob("date=*") if d.is_dir()])
    
    for day in dates:
        themes_path = day / "themes.parquet"
        semantics_path = day / "semantics.parquet"
        
        if not themes_path.exists():
            continue
            
        logging.info(f"Patching {day.name}...")
        
        themes_df = pd.read_parquet(themes_path)
        if themes_df.empty: continue
            
        themes = []
        for _, row in themes_df.iterrows():
            themes.append(MockTheme(
                theme_instance_id=row['theme_instance_id'],
                members=tuple(row['members']),
                source_families=tuple(row['source_families']),
                is_market_mode=row['is_market_mode'],
                structure_score=row['structure_score'],
                consensus_score=row['consensus_score'],
                stability_score=row['stability_score'],
                flow_support_score=row.get('flow_support_score', 0.0)
            ))
            
        new_semantics = labeler.label(themes)
        
        semantic_map = {item.theme_instance_id: item for item in new_semantics}
        
        for i, row in themes_df.iterrows():
            tid = row['theme_instance_id']
            sem = semantic_map[tid]
            themes_df.at[i, 'semantic_coherence_score'] = sem.semantic_coherence_score
            
            qual = 0.35 * row['structure_score'] + 0.25 * row['consensus_score'] + 0.20 * row['stability_score'] + 0.10 * sem.semantic_coherence_score + 0.10 * row.get('flow_support_score', 0.0)
            themes_df.at[i, 'theme_quality_score'] = float(qual)
            
            try:
                bd = json.loads(row['quality_breakdown'])
            except:
                bd = {}
            bd['semantic'] = sem.semantic_coherence_score
            themes_df.at[i, 'quality_breakdown'] = json.dumps(bd, sort_keys=True)
            
        themes_df.to_parquet(themes_path, index=False)
        semantics_df = pd.DataFrame([asdict(item) for item in new_semantics])
        semantics_df.to_parquet(semantics_path, index=False)
        
        logging.info(f"Successfully patched {len(themes)} themes for {day.name}")

if __name__ == "__main__":
    main()
