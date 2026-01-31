param(
    [string]$Python = $env:SANTA_PYTHON,
    [string]$WslDistro = "Ubuntu",
    [string]$Bbox3Path = "data\\improvesanta\\archive\\bbox3",
    [string]$InputCsv = "baseline_csv\\submission.csv",
    [string]$DonorCsv = "baseline_csv\\submission.csv",
    [string]$RunName = "",
    [int]$TotalThreads = 32,
    [Alias("Workers")]
    [ValidateSet(1, 2, 4)]
    [int]$WorkerCount = 4,
    [int]$Rounds = 400,
    [int]$TimeoutSec = 600,
    [int]$Decimals = 16,
    [switch]$DeletePropagate = $false,
    [switch]$InsertPropagate = $false,
    [int]$InsertPropagateNmax = 60,
    [int]$InsertPropagateTopK = 5,
    [int]$StashEvery = 0,
    [int]$StashKeep = 200,
    [double]$StashMinImprove = 1e-10,
    [string]$StashRoot = "",
    [ValidateSet("Normal", "BelowNormal", "Idle", "AboveNormal", "High", "RealTime")]
    [string]$Priority = "BelowNormal"
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

New-Item -ItemType Directory -Force -Path (Join-Path $ProjectDir "logs") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $ProjectDir "bbox3_work") | Out-Null

$runNameNorm = ""
if ($RunName -ne $null) { $runNameNorm = [string]$RunName }
$runNameNorm = $runNameNorm.Trim()
if ($runNameNorm.Length -gt 0) {
    $runNameNorm = ($runNameNorm -replace "[^A-Za-z0-9_-]", "_")
}
$runNamePart = if ($runNameNorm.Length -gt 0) { "_" + $runNameNorm } else { "" }

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$runDir = Join-Path $ProjectDir ("logs\\bbox3_" + $ts + $runNamePart)
try {
    New-Item -ItemType Directory -Force -Path $runDir | Out-Null
} catch {
    # Some environments deny creating under ./logs; fall back to bbox3_work.
    $runDir = Join-Path $ProjectDir ("bbox3_work\\logs\\bbox3_" + $ts + $runNamePart)
    New-Item -ItemType Directory -Force -Path $runDir | Out-Null
}
$latestPidFile = Join-Path $ProjectDir "bbox3_work\\latest_pids.tsv"
Set-Content -Path $latestPidFile -Value "" -Encoding UTF8

$pidTag = if ($runNameNorm.Length -gt 0) { $runNameNorm } else { "latest" }
$stablePidFile = Join-Path $ProjectDir ("bbox3_work\\pids_" + $pidTag + ".tsv")
Set-Content -Path $stablePidFile -Value "" -Encoding UTF8

function Start-BBox3Worker {
    param(
        [string]$Name,
        [string]$Workdir,
        [string]$OutCsv,
        [int[]]$NValues,
        [int[]]$RValues,
        [int]$OmpThreads,
        [string]$StashDir
    )

    $args = @(
        "-u",
        (Join-Path $ProjectDir "bbox3_runner.py"),
        "--input", $InputCsv,
        "--donor", $DonorCsv,
        "--bbox3", $Bbox3Path,
        "--wsl-distro", $WslDistro,
        "--workdir", $Workdir,
        "--out", $OutCsv,
        "--rounds", $Rounds.ToString(),
        "--timeout", $TimeoutSec.ToString(),
        "--omp-threads", $OmpThreads.ToString(),
        "--fix-direction",
        "--decimals", $Decimals.ToString(),
        "--n"
    ) + ($NValues | ForEach-Object { $_.ToString() }) + @("--r") + ($RValues | ForEach-Object { $_.ToString() })

    if ($DeletePropagate) {
        $args += "--delete-propagate"
    }
    if ($InsertPropagate) {
        $args += @(
            "--insert-propagate",
            "--insert-propagate-nmax", $InsertPropagateNmax.ToString(),
            "--insert-propagate-topk", $InsertPropagateTopK.ToString()
        )
    }

    if ($StashDir -and $StashDir.Trim().Length -gt 0) {
        $args += @(
            "--stash-dir", $StashDir,
            "--stash-every", ([Math]::Max(0, [int]$StashEvery)).ToString(),
            "--stash-keep", ([Math]::Max(1, [int]$StashKeep)).ToString(),
            "--stash-min-improve", ([double]$StashMinImprove).ToString("G17", [Globalization.CultureInfo]::InvariantCulture)
        )
    }

    $stdout = Join-Path $runDir ($Name + ".out.log")
    $stderr = Join-Path $runDir ($Name + ".err.log")

    return Start-Process `
        -FilePath $Python `
        -ArgumentList $args `
        -WorkingDirectory $ProjectDir `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -PassThru
}

$ompPerWorker = [Math]::Max(1, [Math]::Floor($TotalThreads / [Math]::Max(1, [int]$WorkerCount)))

$workRoot = if ($runNameNorm.Length -gt 0) { ("bbox3_work\\" + $runNameNorm) } else { "bbox3_work" }
$stashRoot = ""
if ($StashRoot -and $StashRoot.Trim().Length -gt 0) {
    $stashRoot = [string]$StashRoot
    $stashRoot = $stashRoot.Trim()
} elseif ($StashEvery -gt 0) {
    $stashRoot = ($workRoot + "\\stash")
}

$nVals = @(1200, 1500, 1800, 2000)
$r1 = @(11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67)
$r2 = @(71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139)
$r3 = @(149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227)
$r4 = @(229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311)

$workerSpecs = @()
if ($WorkerCount -eq 1) {
    $workerSpecs = @(
        @{ Name = "w1"; Workdir = ($workRoot + "\\w1"); Out = ("baseline_csv\\submission_bbox3" + $runNamePart + "_w1.csv"); N = $nVals; R = ($r1 + $r2 + $r3 + $r4) }
    )
} elseif ($WorkerCount -eq 2) {
    $workerSpecs = @(
        @{ Name = "w1"; Workdir = ($workRoot + "\\w1"); Out = ("baseline_csv\\submission_bbox3" + $runNamePart + "_w1.csv"); N = $nVals; R = ($r1 + $r2) },
        @{ Name = "w2"; Workdir = ($workRoot + "\\w2"); Out = ("baseline_csv\\submission_bbox3" + $runNamePart + "_w2.csv"); N = $nVals; R = ($r3 + $r4) }
    )
} else {
    $workerSpecs = @(
        @{ Name = "w1"; Workdir = ($workRoot + "\\w1"); Out = ("baseline_csv\\submission_bbox3" + $runNamePart + "_w1.csv"); N = $nVals; R = $r1 },
        @{ Name = "w2"; Workdir = ($workRoot + "\\w2"); Out = ("baseline_csv\\submission_bbox3" + $runNamePart + "_w2.csv"); N = $nVals; R = $r2 },
        @{ Name = "w3"; Workdir = ($workRoot + "\\w3"); Out = ("baseline_csv\\submission_bbox3" + $runNamePart + "_w3.csv"); N = $nVals; R = $r3 },
        @{ Name = "w4"; Workdir = ($workRoot + "\\w4"); Out = ("baseline_csv\\submission_bbox3" + $runNamePart + "_w4.csv"); N = $nVals; R = $r4 }
    )
}

Write-Host ("RunDir: " + $runDir)
Write-Host ("Python: " + $Python)
if ($runNameNorm.Length -gt 0) { Write-Host ("RunName: " + $runNameNorm) }
Write-Host ("Workers: " + $WorkerCount + " | OMP per worker: " + $ompPerWorker + " (TotalThreads=" + $TotalThreads + ")")
Write-Host ("PID files: " + $latestPidFile + " ; " + $stablePidFile)

foreach ($w in $workerSpecs) {
    $stashDir = ""
    if ($stashRoot -and $stashRoot.Trim().Length -gt 0) {
        $stashDir = ($stashRoot + "\\" + $w.Name)
        New-Item -ItemType Directory -Force -Path (Join-Path $ProjectDir $stashDir) | Out-Null
    }

    $p = Start-BBox3Worker -Name $w.Name -Workdir $w.Workdir -OutCsv $w.Out -NValues $w.N -RValues $w.R -OmpThreads $ompPerWorker -StashDir $stashDir
    try { $p.PriorityClass = $Priority } catch {}
    Write-Host ("Started " + $w.Name + " pid=" + $p.Id + " out=" + $w.Out)

    $pidLine = ($w.Name + "`t" + $p.Id + "`t" + $w.Workdir + "`t" + $w.Out + "`t" + $runNameNorm)
    Add-Content -Path (Join-Path $runDir "pids.tsv") -Value $pidLine
    Add-Content -Path $latestPidFile -Value $pidLine
    Add-Content -Path $stablePidFile -Value $pidLine
}

Write-Host "Tip: monitor logs with e.g. Get-Content -Wait <logfile>"
