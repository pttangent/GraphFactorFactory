$metrics_file = "C:\GFF_Cache\p2_alpha_lab\daily_relation_eval\daily_alpha_metrics.csv"
Write-Host "Waiting for P2 to complete up to 2026-01-30..."
while ($true) {
    if (Test-Path $metrics_file) {
        $done = (Get-Content $metrics_file | Select-String "2026-01-30")
        if ($done) {
            Write-Host "P2 complete. Starting full pack..."
            break
        }
    }
    Start-Sleep -Seconds 20
}

# Run the python script
python scripts\pack_january_full.py
