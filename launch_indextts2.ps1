param(
    [Parameter(Mandatory = $true)]
    [string]$Text,

    [string]$Speaker = 'checkpoints/speaker_16k.wav',
    [string]$Output = 'debug/quick_run.wav',

    [ValidateSet('VoiceClone', 'EmotionAudio', 'EmotionAudioBlend', 'EmotionVector', 'EmotionText')]
    [string]$Mode = 'VoiceClone',

    [string]$EmotionAudio = 'speaker.wav',
    [double]$EmotionAlpha = 1.0,
    [string]$EmotionVector = '0.60,0.00,0.00,0.00,0.00,0.00,0.10,0.20',
    [string]$EmotionText = 'I feel happy and excited today.',

    [double]$Temperature = 0.8,
    [int]$TopK = 0,
    [double]$TopP = 1.0,
    [double]$RepetitionPenalty = 1.05,
    [int]$FlowSteps = 25,
    [double]$FlowCfgRate = 0.7,

    [switch]$Cpu,
    [switch]$NoDerumble,
    [double]$DerumbleCutoffHz = 180.0,
    [switch]$SkipBuild
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$exe = '.\target\release\indextts2.exe'
if (-not $SkipBuild -and -not (Test-Path $exe)) {
    $env:CUDA_COMPUTE_CAP = '90'
    cargo build --release --features cuda
}

if (-not (Test-Path $Speaker)) {
    throw "Speaker file not found: $Speaker"
}

$outDir = Split-Path -Parent $Output
if ($outDir -and -not (Test-Path $outDir)) {
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
}

$args = @('infer')
if ($Cpu) {
    $args = @('--cpu', 'infer')
}

$args += @(
    '--text', $Text,
    '--speaker', $Speaker,
    '--output', $Output,
    '--temperature', "$Temperature",
    '--top-k', "$TopK",
    '--top-p', "$TopP",
    '--repetition-penalty', "$RepetitionPenalty",
    '--flow-steps', "$FlowSteps",
    '--flow-cfg-rate', "$FlowCfgRate"
)

if (-not $NoDerumble) {
    $args += @('--de-rumble', '--de-rumble-cutoff-hz', "$DerumbleCutoffHz")
}

switch ($Mode) {
    'EmotionAudio' {
        $args += @('--emotion-audio', $EmotionAudio, '--emotion-alpha', "$EmotionAlpha")
    }
    'EmotionAudioBlend' {
        $args += @('--emotion-audio', $EmotionAudio, '--emotion-alpha', "$EmotionAlpha")
    }
    'EmotionVector' {
        $args += @('--emotion-vector', $EmotionVector, '--emotion-alpha', "$EmotionAlpha")
    }
    'EmotionText' {
        $args += @('--use-emo-text', '--emo-text', $EmotionText, '--emotion-alpha', "$EmotionAlpha")
    }
    default {
    }
}

Write-Host "Running: $exe $($args -join ' ')"
& $exe @args
if ($LASTEXITCODE -ne 0) {
    throw "Inference failed with exit code $LASTEXITCODE"
}

Write-Host "Done: $Output"
