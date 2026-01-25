param(
    [string]$Python = $env:SANTA_PYTHON,
    [string]$WslDistro = "Ubuntu",
    [string]$BaselineDir = "baseline_csv",
    [int]$Top = 20,
    [int]$ValidateTop = 10,
    [string]$ShakePath = "baseline\\shake_public",
    [string]$Bbox3Path = "why_not\\bbox3",
    [string]$RunName = "",
    [int]$TotalThreads = 32,
    [int]$ShakeTimeoutSec = 3600,
    [int]$BboxRounds = 5000,
    [int]$BboxTimeoutSec = 1200,
    [int]$Decimals = 16,
    [switch]$SkipShake = $false,
    [switch]$SkipBbox3 = $false,
    [switch]$NoTagOutputs = $false,
    [int]$ScoreDigits = 4
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
    return (Join-Path $dir ($base + "_" + $ts + $ext))
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

function Score-Csv {
    param([string]$Path)
    $outText = & $Python (Join-Path $ProjectDir "train.py") --score-file $Path 2>&1
    $text = ($outText | Out-String)
    $m = [regex]::Match($text, "Score:\\s*([0-9]+(?:\\.[0-9]+)?)")
    if (-not $m.Success) { return $null }
    return [double]::Parse($m.Groups[1].Value, [Globalization.CultureInfo]::InvariantCulture)
}

function Tag-CsvCopy {
    param(
        [string]$Path,
        [double]$Score,
        [string]$Suffix
    )
    if ($NoTagOutputs) { return $null }
    $digits = [Math]::Max(1, [Math]::Min(8, [int]$ScoreDigits))
    $scoreStr = $Score.ToString(("F" + $digits), [Globalization.CultureInfo]::InvariantCulture)
    $leaf = Split-Path -Leaf $Path
    $leafBase = Split-Path -LeafBase $Path
    $ext = [System.IO.Path]::GetExtension($Path)
    $dstLeaf = if ($Suffix -and $Suffix.Trim().Length -gt 0) { ($leafBase + "_" + $Suffix + $ext) } else { $leaf }
    $dst = Join-Path $ProjectDir ("baseline_csv\\" + $scoreStr + "_" + $dstLeaf)
    $dst = New-UniquePath -Path $dst
    Copy-Item -Force $Path $dst
    return $dst
}

New-Item -ItemType Directory -Force -Path (Join-Path $ProjectDir "baseline_csv") | Out-Null

$best = Get-BestValidatedCsv -Dir $BaselineDir -TopN $Top -ValidateN $ValidateTop
if (-not $best) {
    throw "No validated (overlap-free) CSV found under '$BaselineDir'. Try increasing -Top / -ValidateTop."
}

$startCsv = $best.Path
if (-not (Test-Path $startCsv)) {
    $start2 = Join-Path $ProjectDir $startCsv
    if (Test-Path $start2) { $startCsv = $start2 } else { throw "Best path not found on disk: $($best.Path)" }
}

Copy-Item -Force $startCsv (Join-Path $ProjectDir "baseline_csv\\latest_start.csv")
Write-Host ("StartCsv: " + $startCsv + "  score=" + $best.Score.ToString("F12", [Globalization.CultureInfo]::InvariantCulture))

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$runNameNorm = ""
if ($RunName -ne $null) { $runNameNorm = [string]$RunName }
$runNameNorm = $runNameNorm.Trim()
if ($runNameNorm.Length -eq 0) { $runNameNorm = ("best_" + $ts) }

$currentCsv = $startCsv

if (-not $SkipShake) {
    $shakeOut = Join-Path $ProjectDir "baseline_csv\\latest_shake.csv"
    & (Join-Path $ProjectDir "scripts\\run_shake_public.ps1") `
        -Python $Python `
        -WslDistro $WslDistro `
        -ShakePath $ShakePath `
        -InputCsv $currentCsv `
        -DonorCsv $currentCsv `
        -OutCsv $shakeOut `
        -Workdir ("shake_work\\run_" + $runNameNorm) `
        -TimeoutSec $ShakeTimeoutSec `
        -OmpThreads $TotalThreads `
        -Decimals $Decimals

    $shakeScore = Score-Csv -Path $shakeOut
    if ($shakeScore -ne $null) {
        Write-Host ("Shake score: " + $shakeScore.ToString("F12", [Globalization.CultureInfo]::InvariantCulture))
        $tagged = Tag-CsvCopy -Path $shakeOut -Score $shakeScore -Suffix ""
        if ($tagged) { Write-Host ("Tagged shake: " + $tagged) }
    }
    $currentCsv = $shakeOut
} else {
    Write-Host "Shake skipped."
}

if (-not $SkipBbox3) {
    & (Join-Path $ProjectDir "scripts\\run_bbox3_baseline_csv.ps1") `
        -Python $Python `
        -WslDistro $WslDistro `
        -Bbox3Path $Bbox3Path `
        -InputCsv $currentCsv `
        -DonorCsv $currentCsv `
        -RunName $runNameNorm `
        -TotalThreads $TotalThreads `
        -Rounds $BboxRounds `
        -TimeoutSec $BboxTimeoutSec `
        -Decimals $Decimals
} else {
    Write-Host "BBox3 skipped."
}

Write-Host ("Started. RunName=" + $runNameNorm)
