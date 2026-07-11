$total = 700
$cache_dir = "C:\GFF_Cache\p1_b50_b35_sharded"

Write-Host "Waiting for P1 to reach 100%..."
while ($true) {
    $completed = (Get-ChildItem -Path $cache_dir -Recurse -Filter "theme_memberships.parquet" -ErrorAction SilentlyContinue).Count
    if ($completed -ge $total) {
        Write-Host "P1 is 100% complete. Starting packing process..."
        break
    }
    Start-Sleep -Seconds 10
}

python scripts\pack_january_smokerun.py
