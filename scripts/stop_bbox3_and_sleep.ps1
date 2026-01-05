$ErrorActionPreference = "Stop"

param(
    [string]$Python = $env:SANTA_PYTHON,
    [int]$SleepHours = 3,
    [int]$KillTimeoutSec = 30
)

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
if (-not $Python) {
    throw "Cannot find a usable Python. Pass -Python or set env var SANTA_PYTHON."
}

New-Item -ItemType Directory -Force -Path (Join-Path $ProjectDir "best") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $ProjectDir "best\\history") | Out-Null

$train = Join-Path $ProjectDir "train.py"

function Get-ScoreFromCsv {
    param([string]$Path)
    try {
        $outText = & $Python $train --score-file $Path 2>&1
        $m = [regex]::Match(($outText | Out-String), "Score:\s*([0-9]+(?:\.[0-9]+)?)")
        if ($m.Success) { return [double]$m.Groups[1].Value }
    } catch {}
    return $null
}

# 1) Save current best among known outputs.
$cands = @(
    "submission_bbox3_w1.csv",
    "submission_bbox3_w2.csv",
    "submission_bbox3_w3.csv",
    "submission_bbox3_w4.csv",
    "submission_best.csv",
    "best\\submission_best.csv"
) | ForEach-Object { Join-Path $ProjectDir $_ } | Where-Object { Test-Path $_ }

$scored = @()
foreach ($p in $cands) {
    $s = Get-ScoreFromCsv -Path $p
    if ($s -ne $null) {
        $scored += [pscustomobject]@{ Path = $p; Score = $s }
    }
}

if ($scored.Count -gt 0) {
    $best = $scored | Sort-Object Score | Select-Object -First 1
    $dstBest = Join-Path $ProjectDir "best\\submission_best.csv"
    Copy-Item -Force $best.Path $dstBest
    Copy-Item -Force $best.Path (Join-Path $ProjectDir "submission_best.csv")
    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    Copy-Item -Force $best.Path (Join-Path $ProjectDir ("best\\history\\submission_best_" + $ts + ".csv"))
    Set-Content -Path (Join-Path $ProjectDir "best\\best_score.txt") -Value ($best.Score.ToString("G17"))
    Write-Host ("Saved best: " + $best.Score.ToString("F12") + "  " + $best.Path)
} else {
    Write-Host "No scorable outputs found to save."
}

# 2) Kill bbox3_runner processes (by command line match).
$procs = @()
try {
    $procs = Get-CimInstance Win32_Process |
        Where-Object {
            $_.Name -match "python" -and
            $_.CommandLine -and
            ($_.CommandLine -like "*bbox3_runner.py*") -and
            ($_.CommandLine -like "*$ProjectDir*")
        }
} catch {}

if ($procs.Count -gt 0) {
    Write-Host ("Stopping bbox3_runner processes: " + ($procs | ForEach-Object { $_.ProcessId } | Sort-Object | ForEach-Object { $_.ToString() } -join ", "))
    foreach ($p in $procs) {
        try { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue } catch {}
    }
    Start-Sleep -Seconds $KillTimeoutSec
}

# 3) Sleep for N hours
$secs = [Math]::Max(60, $SleepHours * 3600)
Write-Host ("Sleeping for " + $SleepHours + " hour(s)...")
rundll32.exe powrprof.dll,SetSuspendState 0,1,0

