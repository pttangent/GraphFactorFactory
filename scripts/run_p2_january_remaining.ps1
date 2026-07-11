$dates = "2026-01-02,2026-01-16,2026-01-20,2026-01-21,2026-01-22,2026-01-23,2026-01-26,2026-01-27,2026-01-28,2026-01-29,2026-01-30"
$layers = "3,8,9"
$scales = "30m"
$levels = "B50,B35"
$workers = 20

Write-Host "Running P2 Step 1: build-theme-returns"
python scripts/p2_alpha_daily_features.py build-theme-returns --p1-root C:\GFF_Cache\p1_b50_b35_sharded --labels-root D:\DEV\US-Stock\GraphFactorFactory\data\graph_store_6m\canonical --out-root C:\GFF_Cache\p2_alpha_lab\theme_returns --dates $dates --layers $layers --scales $scales --levels $levels --workers $workers

Write-Host "Running P2 Step 2: relation-spillover"
python scripts/p2_alpha_daily_features.py relation-spillover --p1-root C:\GFF_Cache\p1_b50_b35_sharded --theme-returns-root C:\GFF_Cache\p2_alpha_lab\theme_returns --out-root C:\GFF_Cache\p2_alpha_lab\relation_spillover --dates $dates --layers $layers --scales $scales --levels $levels --past-horizon 15m --workers $workers

Write-Host "Running P2 Step 3: daily-relation-features"
python scripts/p2_alpha_daily_features.py daily-relation-features --signals-root C:\GFF_Cache\p2_alpha_lab\relation_spillover --out-root C:\GFF_Cache\p2_alpha_lab\daily_relation_features --dates $dates --layers $layers --scales $scales --workers $workers

Write-Host "Running P2 Step 4: evaluate-daily"
python scripts/p2_alpha_daily_features.py evaluate-daily --features-root C:\GFF_Cache\p2_alpha_lab\daily_relation_features --out-dir C:\GFF_Cache\p2_alpha_lab\daily_relation_eval
