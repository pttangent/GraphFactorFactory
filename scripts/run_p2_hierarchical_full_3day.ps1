param(
    [Parameter(Mandatory = $true)]
    [string]$P1Root,

    [Parameter(Mandatory = $true)]
    [string]$Raw1mRoot,

    [Parameter(Mandatory = $true)]
    [string]$Symbols,

    [Parameter(Mandatory = $true)]
    [string]$Layers,

    [Parameter(Mandatory = $true)]
    [string]$OutputDir,

    [string]$Dates = "2026-01-06,2026-01-07,2026-01-08",
    [string]$Levels = "B50,B35",
    [int]$Workers = 24,
    [double]$MemoryBudgetGb = 88.0,
    [int]$TasksPerChild = 4,
    [int]$MaxInFlight = 24,
    [switch]$ResetCheckpoints
)

$ErrorActionPreference = "Stop"

# Every worker is deliberately single-threaded. Parallelism is owned by the
# 24-process partition scheduler, avoiding 24 x BLAS/Arrow thread explosions.
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
$env:ARROW_NUM_THREADS = "1"
$env:POLARS_MAX_THREADS = "1"
$env:PYTHONUNBUFFERED = "1"
$env:GFF_MAX_TASKS_PER_CHILD = [string]$TasksPerChild

foreach ($Path in @($P1Root, $Raw1mRoot, $Symbols, $Layers)) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Required input does not exist: $Path"
    }
}

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Runner = Join-Path $PSScriptRoot "p2_hierarchical_alpha.py"
if (-not (Test-Path -LiteralPath $Runner)) {
    throw "Runner not found: $Runner"
}

$Arguments = @(
    $Runner,
    "--p1-root", $P1Root,
    "--labels-root", $Raw1mRoot,
    "--symbols", $Symbols,
    "--layers", $Layers,
    "--output-dir", $OutputDir,
    "--dates", $Dates,
    "--levels", $Levels,
    "--workers", [string]$Workers,
    "--memory-budget-gb", [string]$MemoryBudgetGb,
    "--tasks-per-child", [string]$TasksPerChild,
    "--max-in-flight", [string]$MaxInFlight
)
if ($ResetCheckpoints) {
    $Arguments += "--reset-checkpoints"
}

Write-Host "[P2 full] dates=$Dates levels=$Levels workers=$Workers memory=${MemoryBudgetGb}GB"
Write-Host "[P2 full] P1=$P1Root"
Write-Host "[P2 full] raw_1m=$Raw1mRoot"
Write-Host "[P2 full] output=$OutputDir"

Push-Location $RepoRoot
try {
    & python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "p2_hierarchical_alpha.py failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

$Summary = Join-Path $OutputDir "run_summary.json"
$Report = Join-Path $OutputDir "P2_HIERARCHICAL_ALPHA_3DAY_REPORT.md"
if (-not (Test-Path -LiteralPath $Summary)) {
    throw "Run ended without summary: $Summary"
}
if (-not (Test-Path -LiteralPath $Report)) {
    throw "Run ended without report: $Report"
}

Write-Host "[P2 full] complete"
Write-Host "[P2 full] summary: $Summary"
Write-Host "[P2 full] report:  $Report"
