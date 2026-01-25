param(
    [string]$Python = $env:SANTA_PYTHON,
    [int]$KillTimeoutSec = 5,
    [switch]$SkipSave = $false
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
if (-not $Python) {
    throw "Cannot find a usable Python. Pass -Python or set env var SANTA_PYTHON."
}

$env:PYTHONDONTWRITEBYTECODE = "1"

New-Item -ItemType Directory -Force -Path (Join-Path $ProjectDir "best") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $ProjectDir "best\\history") | Out-Null

$train = Join-Path $ProjectDir "train.py"
$pidFile = Join-Path $ProjectDir "bbox3_work\\latest_pids.tsv"

function Get-ScoreFromCsv {
    param([string]$Path)
    try {
        $outText = & $Python $train --score-file $Path 2>&1
        $m = [regex]::Match(($outText | Out-String), "Score:\\s*([0-9]+(?:\\.[0-9]+)?)")
        if ($m.Success) { return [double]::Parse($m.Groups[1].Value, [Globalization.CultureInfo]::InvariantCulture) }
    } catch {}

    # Fallback: use 'python' on PATH (helps when $Python resolves to an env missing deps).
    try {
        $outText = & python $train --score-file $Path 2>&1
        $m = [regex]::Match(($outText | Out-String), "Score:\\s*([0-9]+(?:\\.[0-9]+)?)")
        if ($m.Success) { return [double]::Parse($m.Groups[1].Value, [Globalization.CultureInfo]::InvariantCulture) }
    } catch {}
    return $null
}

if (-not $SkipSave) {
    # 1) Save best among known outputs (prefer latest launcher outputs).
    $cands = @()

    if (Test-Path $pidFile) {
        try {
            $lines = Get-Content -Path $pidFile -ErrorAction SilentlyContinue | Where-Object { $_ -and $_.Trim() -ne "" }
            foreach ($ln in $lines) {
                $parts = $ln -split "`t"
                if ($parts.Count -ge 4) {
                    $outRel = [string]$parts[3]
                    $outRel = $outRel.Trim()
                    if ($outRel.Length -gt 0) {
                        $cand = Join-Path $ProjectDir $outRel
                        if (Test-Path $cand) { $cands += $cand }
                    }
                }
            }
        } catch {}
    }

    try {
        $cands += (Get-ChildItem (Join-Path $ProjectDir "baseline_csv") -File -Filter "submission.csv" -ErrorAction SilentlyContinue | ForEach-Object { $_.FullName })
    } catch {}

    $cands += @(
        (Join-Path $ProjectDir "submission_best.csv"),
        (Join-Path $ProjectDir "best\\submission_best.csv")
    ) | Where-Object { Test-Path $_ }

    $cands = $cands | Where-Object { $_ -and (Test-Path $_) } | Sort-Object -Unique

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
        Copy-Item -Force $best.Path (Join-Path $ProjectDir "baseline_csv\\submission_bbox3_best.csv")
        $ts = Get-Date -Format "yyyyMMdd_HHmmss"
        Copy-Item -Force $best.Path (Join-Path $ProjectDir ("best\\history\\submission_best_" + $ts + ".csv"))
        Set-Content -Path (Join-Path $ProjectDir "best\\best_score.txt") -Value ($best.Score.ToString("G17", [Globalization.CultureInfo]::InvariantCulture))
        Write-Host ("Saved best: " + $best.Score.ToString("F12") + "  " + $best.Path)
    } else {
        Write-Host "No scorable outputs found to save. Tip: run scripts\\promote_best.ps1 to pick the best from baseline_csv."
    }
} else {
    Write-Host "Save skipped (-SkipSave)."
}

# 2) Kill bbox3_runner processes (by command line match).
$killed = @()

# Prefer PID list from the launcher script (doesn't require Win32_Process privileges).
if (Test-Path $pidFile) {
    $lines = Get-Content -Path $pidFile -ErrorAction SilentlyContinue | Where-Object { $_ -and $_.Trim() -ne "" }
    foreach ($ln in $lines) {
        $parts = $ln -split "`t"
        if ($parts.Count -ge 2) {
            $targetPid = 0
            if ([int]::TryParse($parts[1], [ref]$targetPid)) {
                try {
                    Stop-Process -Id $targetPid -Force -ErrorAction SilentlyContinue
                    $killed += $targetPid
                } catch {}
            }
        }
    }
}

if ($killed.Count -gt 0) {
    Write-Host ("Stopped PIDs from " + $pidFile + ": " + (($killed | Sort-Object -Unique) -join ", "))
    Start-Sleep -Seconds $KillTimeoutSec
    return
}

# Fallback: try command-line match (may require permissions on some machines).
$procs = @()
try {
    $procs = Get-CimInstance Win32_Process |
        Where-Object {
            $_.Name -match "python" -and
            $_.CommandLine -and
            ($_.CommandLine -like "*bbox3_runner.py*")
        }
} catch {}

if ($procs.Count -gt 0) {
    Write-Host ("Stopping bbox3_runner processes: " + ($procs | ForEach-Object { $_.ProcessId } | Sort-Object | ForEach-Object { $_.ToString() } -join ", "))
    foreach ($p in $procs) {
        try { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue } catch {}
    }
    Start-Sleep -Seconds $KillTimeoutSec
} else {
    Write-Host "No bbox3_runner.py processes found (if they were started without the launcher script, reboot also stops them)."
}
