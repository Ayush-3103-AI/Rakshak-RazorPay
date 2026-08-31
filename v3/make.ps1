# Windows shim: GNU make is not installed here, but the Makefile is still the source of
# truth. This parses the recipe lines for one target and runs them, rather than keeping a
# second copy of the command list that will drift from the first.
#
#   ./make.ps1 test          ./make.ps1 train -Vars @{RUNG=3}
param([string]$Target = "test", [hashtable]$Vars = @{})

$defaults = @{ SEED = "42"; RUNG = "2"; PY = "uv run python"; PYTEST = "uv run pytest" }
foreach ($k in $Vars.Keys) { $defaults[$k] = [string]$Vars[$k] }

$lines = Get-Content (Join-Path $PSScriptRoot "Makefile")

function Get-Recipe([string]$name) {
    $recipe = @()
    $inTarget = $false
    foreach ($line in $lines) {
        if ($line -match "^([A-Za-z0-9_-]+):(.*)$") {
            if ($inTarget) { break }
            if ($Matches[1] -eq $name) {
                $inTarget = $true
                # Expand a prerequisite list (e.g. `all: lint test`) depth-first.
                $prereqs = ($Matches[2] -replace '##.*$', '').Trim()
                if ($prereqs) { foreach ($p in $prereqs -split '\s+') { $recipe += Get-Recipe $p } }
            }
            continue
        }
        if ($inTarget -and $line -match "^`t(.*)$") { $recipe += $Matches[1] }
    }
    if (-not $inTarget) { throw "unknown target: $name" }
    return $recipe
}

foreach ($cmd in Get-Recipe $Target) {
    if ($cmd -match '^\s*$' -or $cmd -match '^\s*#') { continue }
    $ignoreFailure = $cmd.StartsWith("-")
    if ($ignoreFailure) { $cmd = $cmd.Substring(1) }
    foreach ($k in $defaults.Keys) { $cmd = $cmd -replace [regex]::Escape("`$($k)"), $defaults[$k] }
    Write-Host "> $cmd" -ForegroundColor Cyan
    Invoke-Expression $cmd
    if ($LASTEXITCODE -ne 0 -and -not $ignoreFailure) { exit $LASTEXITCODE }
}
exit 0
