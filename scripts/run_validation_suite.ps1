param(
    [string]$OutDir = 'debug/validation_suite_auto',
    [string]$Speaker = 'checkpoints/speaker_16k.wav'
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot\..

$env:CUDA_COMPUTE_CAP = '90'
$exe = '.\target\release\indextts2.exe'
if (-not (Test-Path $exe)) {
    cargo build --release --features cuda
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$text = 'This is an automated validation run for voice cloning and emotion control.'
$common = @('infer','--speaker',$Speaker,'--text',$text,'--top-k','0','--top-p','1.0','--temperature','0.8','--repetition-penalty','1.05','--flow-steps','25','--flow-cfg-rate','0.7','--de-rumble','--de-rumble-cutoff-hz','180')

$runs = @(
    @{name='voice_clone'; extra=@()},
    @{name='emotion_audio'; extra=@('--emotion-audio','speaker.wav','--emotion-alpha','1.0')},
    @{name='emotion_audio_blend'; extra=@('--emotion-audio','speaker.wav','--emotion-alpha','0.35')},
    @{name='emotion_vector'; extra=@('--emotion-vector','0.60,0.00,0.00,0.00,0.00,0.00,0.10,0.20','--emotion-alpha','0.9')},
    @{name='emotion_text'; extra=@('--use-emo-text','--emo-text','I am very happy and excited today.','--emotion-alpha','0.9')}
)

foreach ($run in $runs) {
    $wav = Join-Path $OutDir ($run.name + '.wav')
    $log = Join-Path $OutDir ($run.name + '.log')
    $args = @($common + @('--output', $wav) + $run.extra)
    Write-Host "RUNNING: $($run.name)"
    & $exe @args 2>&1 | Out-File -FilePath $log -Encoding utf8
    if ($LASTEXITCODE -ne 0) {
        throw "Failed run: $($run.name)"
    }
}

Write-Host "Validation suite complete in $OutDir"
