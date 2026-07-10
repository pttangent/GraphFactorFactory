$ErrorActionPreference = 'Stop'

Write-Host "1/6: Flattening P1 data..."
python scripts\p2_flatten_p1.py

Write-Host "2/6: Building Theme Returns..."
python scripts\p2_build_theme_returns.py

Write-Host "3/6: Running Relation Spillover Alpha (Alpha 1)..."
python scripts\p2_relation_spillover_alpha.py

Write-Host "4/6: Running Core-Peripheral Alpha (Alpha 4)..."
python scripts\p2_core_peripheral_alpha.py

Write-Host "5/6: Running Theme Birth Alpha (Alpha 3)..."
python scripts\p2_theme_birth_alpha.py

Write-Host "6/6: Generating Final Report..."
python scripts\p2_alpha_report.py

Write-Host "P2 Pipeline Completed!"
