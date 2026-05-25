# Activation clustering sweep — slow poisoning only, num_malicious 2..10
# Usage:  .\run_activation_sweep.ps1
#         .\run_activation_sweep.ps1 -NumRounds 20
#         .\run_activation_sweep.ps1 -Preset lite -Malicious 2,3 -NumRounds 2   (smoke test)

param(
    [ValidateSet("lite","medium","paper")]
    [string]$Preset = "paper",

    [int[]]$Malicious = @(3,4,5,6,7,8),

    [string]$Backdoor = "semantic",

    [int]$NumRounds = 0    # 0 = use preset default
)

$ErrorActionPreference = "Stop"

$outDir = if ($NumRounds -gt 0) {
    "results\activation_clustering_slow_${NumRounds}rounds"
} else {
    "results\activation_clustering_slow"
}

New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$malArgs = $Malicious -join " "

$numRoundsArgs = if ($NumRounds -gt 0) { @("--num-rounds", $NumRounds) } else { @() }

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  SLOW    |  safesplit_activation_clustering" -ForegroundColor Cyan
Write-Host "  mal: $malArgs  |  preset: $Preset  |  rounds: $(if ($NumRounds -gt 0) { $NumRounds } else { 'preset default' })" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan

python sweep_malicious.py `
    --preset          $Preset `
    --defense         safesplit_activation_clustering `
    --backdoor        $Backdoor `
    --attack-schedule slow `
    --malicious       $Malicious `
    --out-dir         $outDir `
    @numRoundsArgs

if ($LASTEXITCODE -ne 0) {
    Write-Host "Sweep failed (exit $LASTEXITCODE)." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "================================================================" -ForegroundColor Green
Write-Host "  Activation clustering sweep complete." -ForegroundColor Green
Write-Host "  Output: $outDir\" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
