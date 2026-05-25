# Full malicious client sweep — static and slow poisoning, with and without ratchet
# Usage:  .\run_full_sweep.ps1
#         .\run_full_sweep.ps1 -Preset lite -Malicious 1,2,3,4   (quick test)

param(
    [ValidateSet("lite","medium","paper")]
    [string]$Preset = "paper",

    [int[]]$Malicious = @(1,2,3,4,5,6,7,8,9,10),

    [string]$Backdoor = "semantic"
)

$ErrorActionPreference = "Stop"

$staticDir        = "results\static"
$slowDir          = "results\slow"
$staticRatchetDir = "results\static_ratchet"
$slowRatchetDir   = "results\slow_ratchet"

foreach ($dir in @($staticDir, $slowDir, $staticRatchetDir, $slowRatchetDir)) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}

$malArgs = $Malicious -join " "

function Run-Sweep {
    param([string]$Label, [string]$Defense, [string]$Schedule, [string]$OutDir)

    Write-Host ""
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host "  $Label" -ForegroundColor Cyan
    Write-Host "  mal: $malArgs  |  preset: $Preset  |  defense: $Defense" -ForegroundColor Cyan
    Write-Host "================================================================" -ForegroundColor Cyan

    python sweep_malicious.py `
        --preset          $Preset `
        --defense         $Defense `
        --backdoor        $Backdoor `
        --attack-schedule $Schedule `
        --malicious       $Malicious `
        --out-dir         $OutDir

    if ($LASTEXITCODE -ne 0) {
        Write-Host "$Label failed (exit $LASTEXITCODE)." -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

Run-Sweep "STATIC  |  safesplit_trust"         "safesplit_trust"         "static" $staticDir
Run-Sweep "STATIC  |  safesplit_trust_ratchet"  "safesplit_trust_ratchet" "static" $staticRatchetDir
Run-Sweep "SLOW    |  safesplit_trust"          "safesplit_trust"         "slow"   $slowDir
Run-Sweep "SLOW    |  safesplit_trust_ratchet"  "safesplit_trust_ratchet" "slow"   $slowRatchetDir

# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "================================================================" -ForegroundColor Green
Write-Host "  All sweeps complete."                                           -ForegroundColor Green
Write-Host "  results\static\          static  + safesplit_trust"            -ForegroundColor Green
Write-Host "  results\static_ratchet\  static  + safesplit_trust_ratchet"    -ForegroundColor Green
Write-Host "  results\slow\            slow    + safesplit_trust"             -ForegroundColor Green
Write-Host "  results\slow_ratchet\    slow    + safesplit_trust_ratchet"     -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
