# Windows shim for the Makefile. Same commands, no GNU make required.
param([string]$Target = "test", [int]$Seed = 42)
$cmd = @{
    setup   = "python -m pip install -e `".[dev]`""
    eval    = "python -m rakshak.eval.harness --seed $Seed; python -m rakshak.eval.baf --seed $Seed"
    baf     = "python -m rakshak.eval.baf --seed $Seed"
    figures = "python -m rakshak.eval.harness --seed $Seed --figures-only"
    test    = "python -m pytest"
    lint    = "python -m ruff check src tests"
}[$Target]
if (-not $cmd) { Write-Error "unknown target: $Target"; exit 1 }
Write-Host "> $cmd"
Invoke-Expression $cmd
exit $LASTEXITCODE
