param(
    [string]$Python = $env:SANTA_PYTHON,
    [string]$WslDistro = "Ubuntu",
    [string]$ShakePath = "baseline\\shake_public",
    [string]$InputCsv = "baseline_csv\\submission.csv",
    [string]$DonorCsv = "",
    [string]$OutCsv = "baseline_csv\\submission_shake.csv",
    [string]$Workdir = "shake_work",
    [int]$TimeoutSec = 3600,
    [int]$OmpThreads = 32,
    [int]$Decimals = 16
)

$ErrorActionPreference = "Stop"

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

if (-not $Workdir -or $Workdir.Trim().Length -eq 0) { $Workdir = "shake_work" }
if ($Workdir -eq "shake_work") {
    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    $Workdir = ("shake_work\\run_" + $ts)
}

Write-Host ("Shake: " + $ShakePath)
Write-Host ("InputCsv: " + $InputCsv)
if ($DonorCsv -and $DonorCsv.Trim().Length -gt 0) { Write-Host ("DonorCsv: " + $DonorCsv) }
Write-Host ("OutCsv: " + $OutCsv)
Write-Host ("Workdir: " + $Workdir)
Write-Host ("OMP: " + $OmpThreads + " TimeoutSec: " + $TimeoutSec)

if (-not $DonorCsv -or $DonorCsv.Trim().Length -eq 0) { $DonorCsv = $InputCsv }

& $Python (Join-Path $ProjectDir "shake_runner.py") `
    --input $InputCsv `
    --donor $DonorCsv `
    --shake $ShakePath `
    --workdir $Workdir `
    --wsl-distro $WslDistro `
    --timeout $TimeoutSec `
    --omp-threads $OmpThreads `
    --fix-direction `
    --suffix-min-propagate `
    --suffix-min-mode prefix `
    --decimals $Decimals `
    --out $OutCsv
