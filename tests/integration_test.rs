//! Integration tests for IndexTTS2
//!
//! Tests the full pipeline from text to audio.

use anyhow::Result;
use candle_core::Device;
use std::path::Path;

// Import the library
use indextts2::config::ModelConfig;
use indextts2::text::{TextNormalizer, TextTokenizer, segment_text};
use indextts2::audio::{MelSpectrogram, Resampler};
use indextts2::models::semantic::SemanticEncoder;
use indextts2::models::speaker::CAMPPlus;
use indextts2::models::gpt::UnifiedVoice;
use indextts2::models::s2mel::{LengthRegulator, DiffusionTransformer, FlowMatching};
use indextts2::models::vocoder::BigVGAN;
use indextts2::inference::{InferenceConfig, StreamingSynthesizer};

/// Test text normalization
#[test]
fn test_text_normalization() {
    let normalizer = TextNormalizer::new();

    // Numbers
    assert!(normalizer.normalize("I have 42 apples").contains("forty"));

    // Abbreviations
    let result = normalizer.normalize("Dr. Smith works at St. Mary's");
    assert!(!result.contains("Dr."));
    assert!(!result.contains("St."));
}

/// Test text segmentation
#[test]
fn test_text_segmentation() {
    let text = "Hello world. This is a test. How are you doing today?";
    let segments = segment_text(text, 50);

    assert!(!segments.is_empty());
    for seg in &segments {
        assert!(seg.len() <= 100); // Some margin for words
    }
}

/// Test mel spectrogram computation
#[test]
fn test_mel_spectrogram() {
    let mel = MelSpectrogram::new(
        1024, // n_fft
        256,  // hop_length
        1024, // win_length
        80,   // n_mels
        22050, // sample_rate
        0.0,  // fmin
    ).unwrap();

    // Generate sine wave
    let samples: Vec<f32> = (0..22050)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 22050.0).sin())
        .collect();

    let spec = mel.compute(&samples).unwrap();
    assert!(!spec.is_empty());
}

/// Test semantic encoder initialization
#[test]
fn test_semantic_encoder_init() {
    let device = Device::Cpu;
    let encoder = SemanticEncoder::new(&device).unwrap();
    assert!(!encoder.is_initialized());
}

/// Test speaker encoder initialization
#[test]
fn test_speaker_encoder_init() {
    let device = Device::Cpu;
    let encoder = CAMPPlus::new(&device).unwrap();
    // Should create without error
}

/// Test GPT model initialization
#[test]
fn test_gpt_init() {
    let device = Device::Cpu;
    let gpt = UnifiedVoice::new(&device).unwrap();
    // Should create without error
}

/// Test length regulator initialization
#[test]
fn test_length_regulator_init() {
    let device = Device::Cpu;
    let regulator = LengthRegulator::new(&device).unwrap();
    assert_eq!(regulator.output_channels(), 512);
}

/// Test DiT initialization
#[test]
fn test_dit_init() {
    let device = Device::Cpu;
    let dit = DiffusionTransformer::new(&device).unwrap();
    assert_eq!(dit.hidden_dim(), 512);
    assert_eq!(dit.output_channels(), 80);
}

/// Test flow matching initialization
#[test]
fn test_flow_matching_init() {
    let device = Device::Cpu;
    let fm = FlowMatching::new(&device);
    assert_eq!(fm.num_steps(), 25);
    assert!((fm.cfg_rate() - 0.7).abs() < 0.001);
}

/// Test BigVGAN vocoder initialization
#[test]
fn test_vocoder_init() {
    let device = Device::Cpu;
    let vocoder = BigVGAN::new(&device).unwrap();
    assert_eq!(vocoder.sample_rate(), 22050);
    assert_eq!(vocoder.upsample_factor(), 256);
}

/// Test inference config defaults
#[test]
fn test_inference_config_defaults() {
    let config = InferenceConfig::default();
    assert_eq!(config.temperature, 0.8);
    assert_eq!(config.top_k, 50);
    assert_eq!(config.flow_steps, 25);
    assert!(!config.use_gpu);
}

/// Test streaming synthesizer
#[test]
fn test_streaming_synthesizer() {
    let device = Device::Cpu;
    let mut synth = StreamingSynthesizer::new(&device).unwrap();

    let chunks = synth.generate_all("Hello world. This is a test.").unwrap();
    assert!(!chunks.is_empty());

    // Check that last chunk is marked final
    let last = chunks.last().unwrap();
    assert!(last.is_final);
}

/// Test resampler
#[test]
fn test_resampler() {
    // Generate 1 second of 48kHz audio
    let samples: Vec<f32> = (0..48000)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48000.0).sin())
        .collect();

    // Resample to 22050 Hz
    let resampled = Resampler::resample(&samples, 48000, 22050).unwrap();

    // Should be approximately half the length
    let expected_len = (48000 * 22050 / 48000) as usize;
    assert!((resampled.len() as i32 - expected_len as i32).abs() < 100);
}

/// Test full model chain (without weights)
#[test]
fn test_model_chain_placeholder() {
    use candle_core::{Tensor, DType};

    let device = Device::Cpu;

    // 1. Text processing
    let normalizer = TextNormalizer::new();
    let normalized = normalizer.normalize("Hello world");
    assert!(!normalized.is_empty());

    // 2. Mel spectrogram
    let mel_extractor = MelSpectrogram::new(1024, 256, 1024, 80, 22050, 0.0).unwrap();
    let samples: Vec<f32> = vec![0.0; 22050];
    let mel = mel_extractor.compute(&samples).unwrap();

    // 3. Length regulator (placeholder)
    let regulator = LengthRegulator::new(&device).unwrap();
    let codes = Tensor::randn(0.0f32, 1.0, (1, 50, 1024), &device).unwrap();
    let (features, durations) = regulator.forward(&codes, Some(&[100])).unwrap();
    assert_eq!(features.dims3().unwrap().0, 1); // Batch

    // 4. DiT (placeholder)
    let dit = DiffusionTransformer::new(&device).unwrap();
    let x = Tensor::randn(0.0f32, 1.0, (1, 100, 80), &device).unwrap();
    let t = Tensor::new(&[0.5f32], &device).unwrap();
    let content = Tensor::randn(0.0f32, 1.0, (1, 100, 512), &device).unwrap();
    let output = dit.forward(&x, &t, &content, None).unwrap();
    assert_eq!(output.dims3().unwrap(), (1, 100, 80));

    // 5. Flow matching
    let fm = FlowMatching::new(&device);
    let noise = fm.sample_noise(&[1, 100, 80]).unwrap();
    let mel_spec = fm.sample(&dit, &noise, &content, None).unwrap();
    assert_eq!(mel_spec.dims3().unwrap(), (1, 100, 80));

    // 6. Vocoder (placeholder)
    let vocoder = BigVGAN::new(&device).unwrap();
    let mel_input = Tensor::randn(0.0f32, 1.0, (1, 80, 100), &device).unwrap();
    let audio = vocoder.forward(&mel_input).unwrap();
    let (batch, channels, samples) = audio.dims3().unwrap();
    assert_eq!(batch, 1);
    assert_eq!(channels, 1);
    assert_eq!(samples, 100 * 256); // 256x upsampling
}

/// Test model config parsing (if config exists)
#[test]
fn test_config_parse_example() {
    let yaml = r#"
dataset:
    bpe_model: bpe.model
    sample_rate: 24000
    squeeze: false
    mel:
        sample_rate: 24000
        n_fft: 1024
        hop_length: 256
        win_length: 1024
        n_mels: 100
        mel_fmin: 0
        normalize: false

gpt:
    model_dim: 1280
    max_mel_tokens: 1815
    max_text_tokens: 600
    heads: 20
    use_mel_codes_as_input: true
    mel_length_compression: 1024
    layers: 24
    number_text_tokens: 12000
    number_mel_codes: 8194
    start_mel_token: 8192
    stop_mel_token: 8193
    start_text_token: 0
    stop_text_token: 1
    train_solo_embeddings: false
    condition_type: "conformer_perceiver"
    condition_module:
        output_size: 512
        linear_units: 2048
        attention_heads: 8
        num_blocks: 6
        input_layer: "conv2d2"
        perceiver_mult: 2
    emo_condition_module:
        output_size: 512
        linear_units: 1024
        attention_heads: 4
        num_blocks: 4
        input_layer: "conv2d2"
        perceiver_mult: 2

semantic_codec:
    codebook_size: 8192
    hidden_size: 1024
    codebook_dim: 8
    vocos_dim: 384
    vocos_intermediate_dim: 2048
    vocos_num_layers: 12

s2mel:
    preprocess_params:
        sr: 22050
        spect_params:
            n_fft: 1024
            win_length: 1024
            hop_length: 256
            n_mels: 80
            fmin: 0
            fmax: "None"
    dit_type: "DiT"
    reg_loss_type: "l1"
    style_encoder:
        dim: 192
    length_regulator:
        channels: 512
        is_discrete: false
        in_channels: 1024
        content_codebook_size: 2048
        sampling_ratios: [1, 1, 1, 1]
        vector_quantize: false
        n_codebooks: 1
        quantizer_dropout: 0.0
        f0_condition: false
        n_f0_bins: 512
    DiT:
        hidden_dim: 512
        num_heads: 8
        depth: 13
        class_dropout_prob: 0.1
        block_size: 8192
        in_channels: 80
        style_condition: true
        final_layer_type: 'wavenet'
        target: 'mel'
        content_dim: 512
        content_codebook_size: 1024
        content_type: 'discrete'
        f0_condition: false
        n_f0_bins: 512
        content_codebooks: 1
        is_causal: false
        long_skip_connection: true
        zero_prompt_speech_token: false
        time_as_token: false
        style_as_token: false
        uvit_skip_connection: true
        add_resblock_in_transformer: false
    wavenet:
        hidden_dim: 512
        num_layers: 8
        kernel_size: 5
        dilation_rate: 1
        p_dropout: 0.2
        style_condition: true

gpt_checkpoint: gpt.pth
w2v_stat: wav2vec2bert_stats.pt
s2mel_checkpoint: s2mel.pth
emo_matrix: feat2.pt
spk_matrix: feat1.pt
emo_num: [3, 17, 2, 8, 4, 5, 10, 24]
qwen_emo_path: qwen0.6bemo4-merge/
vocoder:
    type: "bigvgan"
    name: "nvidia/bigvgan_v2_22khz_80band_256x"
version: 2.0
"#;

    let config: ModelConfig = serde_yaml::from_str(yaml).unwrap();
    assert_eq!(config.gpt.model_dim, 1280);
    assert_eq!(config.gpt.layers, 24);
    assert_eq!(config.gpt.stop_mel_token, 8193);
    assert_eq!(config.s2mel.dit.hidden_dim, 512);
    assert_eq!(config.s2mel.dit.depth, 13);
}
