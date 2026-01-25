param(
    [string]$Python = $env:SANTA_PYTHON,
    [string]$WslDistro = "Ubuntu",
    [string]$Bbox3Path = "data\\improvesanta\\archive\\bbox3",
    [string]$ShakePath = "baseline\\shake_public",
    [string]$InputsDir = "baseline_csv",
    [int]$EnsembleLimit = 200,
    [int]$Decimals = 16,
    [switch]$RunShake = $false,
    [int]$ShakeTimeoutSec = 3600,
    [switch]$RunSa140 = $false,
    [int]$SaTopK = 60,
    [int]$SaSteps = 200000,
    [int]$SaRestarts = 3,
    [double]$SaT0 = 0.2,
    [double]$SaAlpha = 0.9997,
    [double]$SaMoveXY = 0.10,
    [double]$SaMoveDeg = 15,
    [int]$TotalThreads = 32,
    [int]$BboxRounds = 5000,
    [int]$BboxTimeoutSec = 600
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

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$tag = "push_" + $ts

$ens = Join-Path $ProjectDir ("baseline_csv\\{0}_ensemble.csv" -f $tag)
$suf = Join-Path $ProjectDir ("baseline_csv\\{0}_ensemble_suffixmin.csv" -f $tag)
$sa  = Join-Path $ProjectDir ("baseline_csv\\{0}_sa140.csv" -f $tag)
$shk = Join-Path $ProjectDir ("baseline_csv\\{0}_shake.csv" -f $tag)

Write-Host ("[1/4] Ensemble from: " + $InputsDir)
& $Python (Join-Path $ProjectDir "train.py") `
    --ensemble-dir $InputsDir `
    --ensemble-limit $EnsembleLimit `
    --ensemble-validate-overlap `
    --ensemble-validate-topk 5 `
    --fix-direction `
    --suffix-min-propagate `
    --suffix-min-mode prefix `
    --decimals $Decimals `
    --out $ens `
    --score

Write-Host ("[2/4] Suffix-min again (cheap pass)")
& $Python (Join-Path $ProjectDir "train.py") `
    --clean-in $ens `
    --suffix-min-propagate `
    --suffix-min-mode prefix `
    --decimals $Decimals `
    --out $suf `
    --score

$startCsv = $suf
if ($RunSa140) {
    Write-Host ("[3/4] SA optimize N<=140 (topK=" + $SaTopK + ")")
    & $Python (Join-Path $ProjectDir "train.py") `
        --opt-small-in $suf `
        --opt-small-nmax 140 `
        --opt-small-top $SaTopK `
        --opt-small-steps $SaSteps `
        --opt-small-restarts $SaRestarts `
        --opt-small-t0 $SaT0 `
        --opt-small-alpha $SaAlpha `
        --opt-small-move-xy $SaMoveXY `
        --opt-small-move-deg $SaMoveDeg `
        --fix-direction `
        --suffix-min-propagate `
        --suffix-min-mode prefix `
        --decimals $Decimals `
        --out $sa `
        --score
    $startCsv = $sa
} else {
    Write-Host "[3/4] SA skipped"
}

if ($RunShake) {
    Write-Host ("[3.5/4] Run shake_public from: " + $startCsv)
    & (Join-Path $ProjectDir "scripts\\run_shake_public.ps1") `
        -Python $Python `
        -WslDistro $WslDistro `
        -ShakePath $ShakePath `
        -InputCsv $startCsv `
        -OutCsv $shk `
        -Workdir ("shake_work\\" + $tag) `
        -TimeoutSec $ShakeTimeoutSec `
        -OmpThreads $TotalThreads `
        -Decimals $Decimals
    $startCsv = $shk
} else {
    Write-Host "[3.5/4] Shake skipped"
}

Write-Host ("[4/4] Start bbox3 workers from: " + $startCsv)
& (Join-Path $ProjectDir "scripts\\run_bbox3_baseline_csv.ps1") `
    -Python $Python `
    -WslDistro $WslDistro `
    -Bbox3Path $Bbox3Path `
    -InputCsv $startCsv `
    -DonorCsv $startCsv `
    -RunName $tag `
    -TotalThreads $TotalThreads `
    -Rounds $BboxRounds `
    -TimeoutSec $BboxTimeoutSec

Write-Host ("Started. RunName=" + $tag)
