# Continuous ratchet sweep — static and slow poisoning, num_malicious 2..10
# Usage:  .\run_continuous_sweep.ps1
#         .\run_continuous_sweep.ps1 -Preset lite -Malicious 2,3,4   (quick test)

param(
    [ValidateSet("lite","medium","paper")]
    [string]$Preset = "paper",

    [int[]]$Malicious = @(2,3,4,5,6,7,8,9,10),

    [string]$Backdoor = "semantic"
)

$ErrorActionPreference = "Stop"

$staticDir = "results\static_continuous"
$slowDir   = "results\slow_continuous"

foreach ($dir in @($staticDir, $slowDir)) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}

$malArgs = $Malicious -join " "

function Run-Sweep {
    param([string]$Label, [string]$Schedule, [string]$OutDir)

    Write-Host ""
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host "  $Label" -ForegroundColor Cyan
    Write-Host "  mal: $malArgs  |  preset: $Preset  |  defense: safesplit_continuous_ratchet" -ForegroundColor Cyan
    Write-Host "================================================================" -ForegroundColor Cyan

    python sweep_malicious.py `
        --preset          $Preset `
        --defense         safesplit_continuous_ratchet `
        --backdoor        $Backdoor `
        --attack-schedule $Schedule `
        --malicious       $Malicious `
        --out-dir         $OutDir

    if ($LASTEXITCODE -ne 0) {
        Write-Host "$Label failed (exit $LASTEXITCODE)." -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

Run-Sweep "STATIC  |  safesplit_continuous_ratchet" "static" $staticDir
Run-Sweep "SLOW    |  safesplit_continuous_ratchet" "slow"   $slowDir

Write-Host ""
Write-Host "================================================================" -ForegroundColor Green
Write-Host "  Continuous ratchet sweeps complete."                            -ForegroundColor Green
Write-Host "  results\static_continuous\  static + safesplit_continuous_ratchet" -ForegroundColor Green
Write-Host "  results\slow_continuous\    slow   + safesplit_continuous_ratchet" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
