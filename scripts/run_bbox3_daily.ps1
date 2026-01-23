param(
    [string]$Python = $env:SANTA_PYTHON,
    [string]$WslDistro = "Ubuntu",
    [string]$InputCsv = "submission_delprop.csv",
    [string]$DonorCsv = "submission_delprop.csv",
    [int]$OmpThreads = 4,
    [int]$RoundsFast = 600,
    [int]$TimeoutFast = 1200,
    [int]$RoundsDeep = 300,
    [int]$TimeoutDeep = 2400,
    [int]$StatusIntervalSec = 30,
    [string]$StashRoot = "stash",
    [int]$StashEvery = 0,
    [int]$StashKeep = 200,
    [double]$StashMinImprove = 1e-10,
    [int]$Decimals = 16,
    [ValidateSet("Normal", "BelowNormal", "Idle", "AboveNormal", "High", "RealTime")]
    [string]$Priority = "BelowNormal",
    [string]$SleepAfter = "false"
)

$ErrorActionPreference = "Stop"

$sleepAfterNorm = $SleepAfter
if ([string]::IsNullOrWhiteSpace($sleepAfterNorm)) { $sleepAfterNorm = "true" }
$sleepAfterNorm = $sleepAfterNorm.Trim().ToLowerInvariant()
$doSleepAfter = -not ($sleepAfterNorm -in @("0", "false", "no", "off"))

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

New-Item -ItemType Directory -Force -Path (Join-Path $ProjectDir "logs") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $ProjectDir "best") | Out-Null

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$runDir = Join-Path $ProjectDir ("logs\\run_" + $ts)
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

function Start-BBox3Worker {
    param(
        [string]$Name,
        [string]$Workdir,
        [string]$OutCsv,
        [int[]]$NValues,
        [int[]]$RValues,
        [int]$Rounds,
        [int]$Timeout
    )

    $bbox3Args = @(
        (Join-Path $ProjectDir "bbox3_runner.py"),
        "--input", $InputCsv,
        "--donor", $DonorCsv,
        "--wsl-distro", $WslDistro,
        "--workdir", $Workdir,
        "--out", $OutCsv,
        "--rounds", $Rounds.ToString(),
        "--timeout", $Timeout.ToString(),
        "--omp-threads", $OmpThreads.ToString(),
        "--decimals", $Decimals.ToString(),
        "--n"
    ) + ($NValues | ForEach-Object { $_.ToString() }) + @("--r") + ($RValues | ForEach-Object { $_.ToString() }) + @(
        "--fix-direction",
        "--delete-propagate"
    )

    $stashNorm = $StashRoot
    if (-not [string]::IsNullOrWhiteSpace($stashNorm)) {
        $stashDir = Join-Path $ProjectDir (Join-Path $stashNorm $Name)
        $bbox3Args += @(
            "--stash-dir", $stashDir,
            "--stash-every", $StashEvery.ToString(),
            "--stash-keep", $StashKeep.ToString(),
            "--stash-min-improve", $StashMinImprove.ToString("G17")
        )
    }

    $stdout = Join-Path $runDir ($Name + ".out.log")
    $stderr = Join-Path $runDir ($Name + ".err.log")

    return Start-Process `
        -FilePath $Python `
        -ArgumentList $bbox3Args `
        -WorkingDirectory $ProjectDir `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -PassThru
}

$workers = @(
    @{ Name = "w1"; Workdir = "bbox3_w1"; Out = "submission_bbox3_w1.csv"; N = @(900, 1100, 1300, 1500, 1700); R = @(10, 20, 30); Rounds = $RoundsFast; Timeout = $TimeoutFast },
    @{ Name = "w2"; Workdir = "bbox3_w2"; Out = "submission_bbox3_w2.csv"; N = @(900, 1100, 1300, 1500, 1700); R = @(40, 60, 80); Rounds = $RoundsFast; Timeout = $TimeoutFast },
    @{ Name = "w3"; Workdir = "bbox3_w3"; Out = "submission_bbox3_w3.csv"; N = @(900, 1100, 1300, 1500, 1700); R = @(90, 120, 150); Rounds = $RoundsFast; Timeout = $TimeoutFast },
    @{ Name = "w4"; Workdir = "bbox3_w4"; Out = "submission_bbox3_w4.csv"; N = @(1800, 2000, 2200, 2500); R = @(30, 60, 90); Rounds = $RoundsDeep; Timeout = $TimeoutDeep }
)

Write-Host ("RunDir: " + $runDir)
Write-Host ("Python: " + $Python)

$procs = @()
$start = Get-Date
try {
    foreach ($w in $workers) {
        $p = Start-BBox3Worker `
            -Name $w.Name `
            -Workdir $w.Workdir `
            -OutCsv $w.Out `
            -NValues $w.N `
            -RValues $w.R `
            -Rounds $w.Rounds `
            -Timeout $w.Timeout
        $procs += $p
        try { $p.PriorityClass = $Priority } catch {}
        Write-Host ("Started " + $w.Name + " pid=" + $p.Id)
    }

    function Get-ScoreFromCsv {
        param([string]$Path)
        try {
            $outText = & $Python $train --score-file $Path 2>&1
            $m = [regex]::Match(($outText | Out-String), "Score:\s*([0-9]+(?:\.[0-9]+)?)")
            if ($m.Success) { return [double]$m.Groups[1].Value }
        } catch {}
        return $null
    }

    $train = Join-Path $ProjectDir "train.py"
    $fileState = @{}
    foreach ($w in $workers) {
        $path = Join-Path $ProjectDir $w.Out
        $fileState[$w.Name] = @{
            Path = $path
            LastWrite = $null
            Score = $null
        }
    }

    $bestEver = $null
    while ($true) {
        $running = @()
        foreach ($p in $procs) {
            $pp = Get-Process -Id $p.Id -ErrorAction SilentlyContinue
            if ($pp) { $running += $pp }
        }

        $keys = @($fileState.Keys)
        foreach ($k in $keys) {
            $st = $fileState[$k]
            $path = $st.Path
            if (Test-Path $path) {
                $lw = (Get-Item $path).LastWriteTimeUtc
                if ($st.LastWrite -eq $null -or $lw -gt $st.LastWrite) {
                    $st.LastWrite = $lw
                    $s = Get-ScoreFromCsv -Path $path
                    if ($s -ne $null) {
                        $st.Score = $s
                        if ($bestEver -eq $null -or $s -lt $bestEver) { $bestEver = $s }
                    }
                }
            }
            $fileState[$k] = $st
        }

        $elapsed = (New-TimeSpan -Start $start -End (Get-Date)).TotalMinutes
        $statusParts = @()
        foreach ($w in $workers) {
            $st = $fileState[$w.Name]
            $sc = if ($st.Score -ne $null) { "{0:F12}" -f $st.Score } else { "n/a" }
            $statusParts += ($w.Name + "=" + $sc)
        }
        $bestStr = if ($bestEver -ne $null) { "{0:F12}" -f $bestEver } else { "n/a" }
        Write-Host ("[{0}] elapsed={1:F1}m running={2} best={3} | {4}" -f (Get-Date -Format "HH:mm:ss"), $elapsed, $running.Count, $bestStr, ($statusParts -join " "))

        if ($running.Count -eq 0) { break }
        Start-Sleep -Seconds ([Math]::Max(5, $StatusIntervalSec))
    }

    $candidates = @()

    foreach ($w in $workers) {
        $p = Join-Path $ProjectDir $w.Out
        if (-not (Test-Path $p)) { continue }
        $outText = & $Python $train --score-file $p 2>&1
        $m = [regex]::Match(($outText | Out-String), "Score:\\s*([0-9]+(?:\\.[0-9]+)?)")
        if ($m.Success) {
            $candidates += [pscustomobject]@{ Path = $p; Score = [double]$m.Groups[1].Value }
        }
    }

    $bestPath = Join-Path $ProjectDir "best\\submission_best.csv"
    if (Test-Path $bestPath) {
        $outText = & $Python $train --score-file $bestPath 2>&1
        $m = [regex]::Match(($outText | Out-String), "Score:\\s*([0-9]+(?:\\.[0-9]+)?)")
        if ($m.Success) {
            $candidates += [pscustomobject]@{ Path = $bestPath; Score = [double]$m.Groups[1].Value }
        }
    }

    if ($candidates.Count -gt 0) {
        $best = $candidates | Sort-Object Score | Select-Object -First 1
        Copy-Item -Force $best.Path $bestPath
        Copy-Item -Force $best.Path (Join-Path $ProjectDir "submission_best.csv")
        Set-Content -Path (Join-Path $ProjectDir "best\\best_score.txt") -Value ($best.Score.ToString("G17"))
        Write-Host ("Best: " + $best.Score.ToString("G17") + "  " + $best.Path)
    } else {
        Write-Host "No scorable outputs found."
    }
}
finally {
    Set-Content -Path (Join-Path $runDir "done.txt") -Value (Get-Date).ToString("O")
    if ($doSleepAfter) {
        Start-Sleep -Seconds 5
        rundll32.exe powrprof.dll,SetSuspendState 0,1,0
    }
}
