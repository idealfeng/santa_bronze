param(
    [string]$Python = $env:SANTA_PYTHON,
    [string]$BaselineDir = "baseline_csv",
    [int]$Top = 20,
    [int]$ValidateTop = 10,
    [switch]$SetSubmissionCsv = $false,
    [switch]$NoScoreNamedCopy = $false,
    [int]$ScoreDigits = 4,
    [switch]$KaggleSafe = $false,
    [string]$KaggleSafeDonorCsv = "",
    [double]$KaggleMinGap = 1e-12,
    [int[]]$KaggleOnlyGroups = @()
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

function New-UniquePath {
    param([string]$Path)
    if (-not (Test-Path $Path)) { return $Path }
    $dir = Split-Path -Parent $Path
    $base = Split-Path -LeafBase $Path
    $ext = [System.IO.Path]::GetExtension($Path)
    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    $cand = Join-Path $dir ($base + "_" + $ts + $ext)
    return $cand
}

function Get-BestValidatedCsv {
    param(
        [string]$Dir,
        [int]$TopN,
        [int]$ValidateN
    )

    $outText = & $Python (Join-Path $ProjectDir "train.py") --score-dir $Dir --top $TopN --validate-top $ValidateN 2>&1
    $text = ($outText | Out-String)

    $best = $null
    $re = [regex]'full=([0-9]+(?:\.[0-9]+)?)\s+\(fast=[^\)]*\)\s+(.+\.csv)\s*$'
    foreach ($ln in ($text -split "`r?`n")) {
        $m = $re.Match($ln)
        if (-not $m.Success) { continue }
        $score = [double]::Parse($m.Groups[1].Value, [Globalization.CultureInfo]::InvariantCulture)
        $path = $m.Groups[2].Value.Trim()
        if (-not $best -or $score -lt $best.Score) {
            $best = [pscustomobject]@{ Score = $score; Path = $path }
        }
    }
    return $best
}

New-Item -ItemType Directory -Force -Path (Join-Path $ProjectDir "best") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $ProjectDir "best\\history") | Out-Null

$best = Get-BestValidatedCsv -Dir $BaselineDir -TopN $Top -ValidateN $ValidateTop
if (-not $best) {
    throw "No validated (overlap-free) CSV found under '$BaselineDir'. Try increasing -Top / -ValidateTop."
}

$src = $best.Path
if (-not (Test-Path $src)) {
    $src2 = Join-Path $ProjectDir $src
    if (Test-Path $src2) { $src = $src2 } else { throw "Best path not found on disk: $($best.Path)" }
}

$promoteSrc = $src
if ($KaggleSafe) {
    if (-not $KaggleOnlyGroups -or $KaggleOnlyGroups.Count -eq 0) {
        throw "KaggleSafe requires -KaggleOnlyGroups (e.g. -KaggleOnlyGroups 176)."
    }

    $donor = $KaggleSafeDonorCsv
    if (-not $donor -or $donor.Trim().Length -eq 0) {
        $donorCands = @(
            "baseline_csv\\latest_shake.csv",
            "baseline_csv\\70.4397_ensemble_shake.csv",
            "baseline_csv\\ensemble_latest.csv",
            "best\\submission_best.csv"
        ) | ForEach-Object { Join-Path $ProjectDir $_ } | Where-Object { Test-Path $_ }
        $donor = $donorCands | Select-Object -First 1
    } else {
        if (-not (Test-Path $donor)) { $donor = Join-Path $ProjectDir $donor }
    }
    if (-not $donor -or -not (Test-Path $donor)) {
        throw "KaggleSafe enabled but no donor CSV found. Pass -KaggleSafeDonorCsv or generate baseline_csv\\latest_shake.csv first."
    }

    $safeOut = Join-Path $ProjectDir "baseline_csv\\submission_kaggle_safe.csv"
    $pyArgs = @(
        (Join-Path $ProjectDir "kaggle_safe.py"),
        "--input", $src,
        "--donor", $donor,
        "--out", $safeOut,
        "--min-gap", $KaggleMinGap.ToString("G17", [Globalization.CultureInfo]::InvariantCulture),
        "--only-groups"
    ) + ($KaggleOnlyGroups | ForEach-Object { [string]$_ })

    & $Python @pyArgs

    if (-not (Test-Path $safeOut)) { throw "KaggleSafe failed to write: $safeOut" }
    $promoteSrc = $safeOut
}

$dstBest = Join-Path $ProjectDir "best\\submission_best.csv"
Copy-Item -Force $promoteSrc $dstBest
Copy-Item -Force $promoteSrc (Join-Path $ProjectDir "submission_best.csv")
Copy-Item -Force $promoteSrc (Join-Path $ProjectDir "baseline_csv\\submission_bbox3_best.csv")

if ($SetSubmissionCsv) {
    Copy-Item -Force $promoteSrc (Join-Path $ProjectDir "baseline_csv\\submission.csv")
}

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
Copy-Item -Force $promoteSrc (Join-Path $ProjectDir ("best\\history\\submission_best_" + $ts + ".csv"))
Set-Content -Path (Join-Path $ProjectDir "best\\best_score.txt") -Value ($best.Score.ToString("G17", [Globalization.CultureInfo]::InvariantCulture))

if (-not $NoScoreNamedCopy) {
    $digits = [Math]::Max(1, [Math]::Min(8, [int]$ScoreDigits))
    $scoreStr = $best.Score.ToString(("F" + $digits), [Globalization.CultureInfo]::InvariantCulture)
    $leaf = Split-Path -Leaf $promoteSrc
    if ($leaf -notmatch '^[0-9]+(?:\.[0-9]+)?_') {
        $dst = Join-Path $ProjectDir ("baseline_csv\\" + $scoreStr + "_" + $leaf)
        $dst = New-UniquePath -Path $dst
        Copy-Item -Force $promoteSrc $dst
        Write-Host ("Score-tagged copy: " + $dst)
    }
}

Write-Host ("Promoted best: " + $best.Score.ToString("F12", [Globalization.CultureInfo]::InvariantCulture) + "  " + $promoteSrc)
