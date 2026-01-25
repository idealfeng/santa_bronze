param(
    [string]$Python = $env:SANTA_PYTHON,
    [string]$InputCsv,
    [string]$DonorCsv,
    [string]$OutCsv,
    [double]$MinGap = 1e-12,
    [int[]]$OnlyGroups = @()
)

$ErrorActionPreference = "Stop"

if (-not $InputCsv) { throw "-InputCsv is required." }
if (-not $DonorCsv) { throw "-DonorCsv is required." }
if (-not $OutCsv) { throw "-OutCsv is required." }

$ProjectDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectDir

if (-not $Python -or -not (Test-Path $Python)) {
    $cands = @(
        "E:\anaconda\envs\torch312\python.exe",
        (Join-Path $env:CONDA_PREFIX "python.exe"),
        (Get-Command python -ErrorAction SilentlyContinue).Source
    ) | Where-Object { $_ -and (Test-Path $_) }
    $Python = $cands | Select-Object -First 1
}
if (-not $Python) { throw "Cannot find a usable Python. Pass -Python or set env var SANTA_PYTHON." }

$env:PYTHONDONTWRITEBYTECODE = "1"

$args = @(
    (Join-Path $ProjectDir "kaggle_safe.py"),
    "--input", $InputCsv,
    "--donor", $DonorCsv,
    "--out", $OutCsv,
    "--min-gap", $MinGap.ToString("G17", [Globalization.CultureInfo]::InvariantCulture)
)
if ($OnlyGroups -and $OnlyGroups.Count -gt 0) {
    $args += "--only-groups"
    $args += ($OnlyGroups | ForEach-Object { [string]$_ })
}

& $Python @args
