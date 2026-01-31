param(
    [string]$Python = $env:SANTA_PYTHON,
    [string]$SparrowExe = "sparrow_sm\\target\\release\\sparrow.exe",
    [string]$BaseCsv = "baseline_csv\\submission.csv",
    [string]$OutPrefix = "baseline_csv\\sparrow",
    [int]$NMin = 2,
    [int]$NMax = 60,
    [int]$TimePerN = 60,
    [double]$StripMargin = 1.003,
    [int]$StripMin = 2000,
    [int]$Decimals = 16,
    [double]$MinImprove = 1e-12,
    [int[]]$Seeds = @(1, 42, 77, 202)
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

$sparrow = $SparrowExe
if (-not (Test-Path $sparrow)) { $sparrow = Join-Path $ProjectDir $sparrow }
if (-not (Test-Path $sparrow)) { throw "Missing sparrow exe: $SparrowExe" }

$base = $BaseCsv
if (-not (Test-Path $base)) { $base = Join-Path $ProjectDir $base }
if (-not (Test-Path $base)) { throw "Missing base CSV: $BaseCsv" }

New-Item -ItemType Directory -Force -Path (Join-Path $ProjectDir "logs") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $ProjectDir "sparrow_work") | Out-Null

$procs = @()
foreach ($seed in $Seeds) {
    $seedStr = [string]$seed
    $workdir = Join-Path $ProjectDir ("sparrow_work\\s" + $seedStr)
    New-Item -ItemType Directory -Force -Path $workdir | Out-Null

    $outCsv = Join-Path $ProjectDir ($OutPrefix + "_s" + $seedStr + ".csv")
    $logOut = Join-Path $ProjectDir ("logs\\sparrow_s" + $seedStr + ".out.log")
    $logErr = Join-Path $ProjectDir ("logs\\sparrow_s" + $seedStr + ".err.log")

    $args = @(
        "-u",
        (Join-Path $ProjectDir "run_sparrow_small.py"),
        "--sparrow-exe", $sparrow,
        "--base", $base,
        "--out", $outCsv,
        "--workdir", $workdir,
        "--nmin", $NMin.ToString(),
        "--nmax", $NMax.ToString(),
        "--time", $TimePerN.ToString(),
        "--seed", $seedStr,
        "--strip-margin", $StripMargin.ToString("G17", [Globalization.CultureInfo]::InvariantCulture),
        "--strip-min", $StripMin.ToString(),
        "--decimals", $Decimals.ToString(),
        "--min-improve", $MinImprove.ToString("G17", [Globalization.CultureInfo]::InvariantCulture)
    )

    $p = Start-Process -FilePath $Python -ArgumentList $args -WorkingDirectory $ProjectDir `
        -RedirectStandardOutput $logOut -RedirectStandardError $logErr -PassThru
    Write-Host ("Started sparrow seed=" + $seedStr + " pid=" + $p.Id + " out=" + $outCsv)
    $procs += $p
}

Write-Host ("Waiting for " + $procs.Count + " sparrow job(s)...")
foreach ($p in $procs) {
    try { Wait-Process -Id $p.Id } catch {}
}
Write-Host "Done."

