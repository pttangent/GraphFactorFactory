import pandas as pd
from pathlib import Path

OUT_DIR = Path("artifacts/p2_alpha_lab")
REPORT_PATH = OUT_DIR / "P2_ALPHA_REPORT.md"

def load_ic(name):
    p = OUT_DIR / f"{name}_ic.csv"
    if p.exists():
        return pd.read_csv(p)
    return None

df_spillover = load_ic("relation_spillover")
df_core_peri = load_ic("core_peripheral")
df_theme_birth = load_ic("theme_birth")

with open(REPORT_PATH, 'w') as f:
    f.write("# P2 Alpha Lab - Initial Sanity Report\n\n")
    f.write("This report evaluates three structural graph alpha concepts using 20 days of P1 output.\n\n")
    
    f.write("## 1. Alpha 1: Relation Spillover\n")
    f.write("> **Hypothesis**: If Theme A and Theme B have a fuzzy relation, the past return of A combined with the relation strength predicts the future return of B.\n\n")
    if df_spillover is not None:
        f.write(df_spillover.to_markdown(index=False))
        f.write("\n\n")
    else:
        f.write("No data.\n\n")
        
    f.write("## 2. Alpha 4: Core-to-Peripheral Diffusion\n")
    f.write("> **Hypothesis**: Inside a theme, the past return of the core members (top 20%) predicts the forward return of the peripheral members (bottom 50%).\n\n")
    if df_core_peri is not None:
        f.write(df_core_peri.to_markdown(index=False))
        f.write("\n\n")
    else:
        f.write("No data.\n\n")
        
    f.write("## 3. Alpha 3: New Theme Birth\n")
    f.write("> **Hypothesis**: Themes that appear for the very first time (no fuzzy predecessor) represent newly traded topics and exhibit stronger forward momentum.\n\n")
    if df_theme_birth is not None:
        # Rename columns to match formatting if needed
        f.write(df_theme_birth.to_markdown(index=False))
        f.write("\n\n")
    else:
        f.write("No data.\n\n")

print(f"Report generated at {REPORT_PATH}")
