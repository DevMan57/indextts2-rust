param(
    [switch]$SkipBuild
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

if (-not $SkipBuild -and -not (Test-Path '.\target\release\indextts2.exe')) {
    $env:CUDA_COMPUTE_CAP = '90'
    cargo build --release --features cuda
}

python .\scripts\ui_launcher.py
