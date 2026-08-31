# Windows shim for the Makefile. Same commands, no GNU make required.
param([string]$Target = "test", [int]$Seed = 42)

# T-0021: mirrors the Makefile's BLAS thread pin. The hand-written Baum-Welch
# fit's summation order depends on thread count, so without this the committed
# ablations.md and sensitivity PNGs are not reproducible on a machine with a
# different core count. Keep in sync with the Makefile.
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"

$cmd = @{
    setup   = "python -m pip install -e `".[dev]`""
    eval    = "python -m rakshak.generator --seed $Seed; python -m rakshak.eval.harness --seed $Seed; python -m rakshak.eval.verdict --seed $Seed; python -m rakshak.eval.ablations --seed $Seed; python -m rakshak.eval.lag_probe --seed $Seed; python -m rakshak.eval.typology --seed $Seed; python -m rakshak.explain.reasons --seed $Seed; python -m rakshak.eval.baf --seed $Seed"
    baf     = "python -m rakshak.eval.baf --seed $Seed"
    blackswan = "python -m rakshak.generator.generate --seed $Seed --shock-day 194 --shock-magnitude 6.0; python -m rakshak.eval.blackswan --seed $Seed"
    profile = "python -m rakshak.data.profile --seed $Seed"
    figures = "python -m rakshak.eval.harness --seed $Seed --figures-only"
    test    = "python -m pytest"
    lint    = "python -m ruff check src tests"
}[$Target]
if (-not $cmd) { Write-Error "unknown target: $Target"; exit 1 }
Write-Host "> $cmd"
Invoke-Expression $cmd
exit $LASTEXITCODE
