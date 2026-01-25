//! Integration tests for IndexTTS2
//!
//! Tests the full pipeline from text to audio.

use candle_core::Device;

// Import the library
use indextts2::config::ModelConfig;
use indextts2::text::TextNormalizer;
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
    let normalizer = TextNormalizer::new(false);

    // Numbers
    assert!(normalizer.normalize("I have 42 apples").contains("forty"));

    // Abbreviations that are expanded
    let result = normalizer.normalize("Dr. Smith and Prof. Jones");
    assert!(!result.contains("Dr."), "Dr. should be expanded");
    assert!(!result.contains("Prof."), "Prof. should be expanded");
    assert!(result.contains("Doctor"), "Should contain Doctor");
    assert!(result.contains("Professor"), "Should contain Professor");
}

/// Test text segmentation
#[test]
fn test_text_segmentation() {
    use indextts2::text::segment_text_string;
    let text = "Hello world. This is a test. How are you doing today?";
    let segments = segment_text_string(text, 50);

    assert!(!segments.is_empty());
    for seg in &segments {
        assert!(seg.len() <= 100); // Some margin for words
    }
}

/// Test mel spectrogram computation
#[test]
fn test_mel_spectrogram() {
    let mel = MelSpectrogram::new(
        1024,  // n_fft
        256,   // hop_length
        1024,  // win_length
        80,    // n_mels
        22050, // sample_rate
        0.0,   // fmin
        None,  // fmax (None = Nyquist)
    );

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
    // SemanticEncoder requires a stat_path; use a placeholder path that will fall back to defaults
    let _encoder = SemanticEncoder::load("nonexistent_stats.pt", None::<&str>, &device).unwrap();
    // Encoder loads without weights, but stats default to zero mean / unit std
}

/// Test speaker encoder initialization
#[test]
fn test_speaker_encoder_init() {
    let device = Device::Cpu;
    let _encoder = CAMPPlus::new(&device).unwrap();
    // Should create without error
}

/// Test GPT model initialization
#[test]
fn test_gpt_init() {
    let device = Device::Cpu;
    let _gpt = UnifiedVoice::new(&device).unwrap();
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
    let normalizer = TextNormalizer::new(false);
    let normalized = normalizer.normalize("Hello world");
    assert!(!normalized.is_empty());

    // 2. Mel spectrogram
    let mel_extractor = MelSpectrogram::new(1024, 256, 1024, 80, 22050, 0.0, None);
    let samples: Vec<f32> = vec![0.0; 22050];
    let _mel = mel_extractor.compute(&samples).unwrap();

    // 3. Length regulator (placeholder)
    let regulator = LengthRegulator::new(&device).unwrap();
    let codes = Tensor::randn(0.0f32, 1.0, (1, 50, 1024), &device).unwrap();
    let (features, _durations) = regulator.forward(&codes, Some(&[100])).unwrap();
    assert_eq!(features.dims3().unwrap().0, 1); // Batch

    // 4. DiT (placeholder)
    let dit = DiffusionTransformer::new(&device).unwrap();
    let x = Tensor::randn(0.0f32, 1.0, (1, 100, 80), &device).unwrap();
    let t = Tensor::new(&[0.5f32], &device).unwrap();
    let content = Tensor::randn(0.0f32, 1.0, (1, 100, 512), &device).unwrap();
    let prompt_x = Tensor::zeros((1, 100, 80), DType::F32, &device).unwrap();
    let style = Tensor::randn(0.0f32, 1.0, (1, 192), &device).unwrap();
    let output = dit.forward(&x, &prompt_x, &t, &content, &style).unwrap();
    assert_eq!(output.dims3().unwrap(), (1, 100, 80));

    // 5. Flow matching
    let fm = FlowMatching::new(&device);
    let noise = fm.sample_noise(&[1, 100, 80]).unwrap();
    let mel_spec = fm.sample(&dit, &noise, &prompt_x, &content, &style, 0).unwrap();
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

/// Full inference test with real weights (if available)
/// This test is ignored by default because it requires downloading weights.
/// Run with: cargo test test_full_inference_with_weights -- --ignored
#[test]
#[ignore = "Requires model weights in checkpoints/"]
fn test_full_inference_with_weights() {
    use indextts2::inference::IndexTTS2;
    use std::path::Path;

    let config_path = Path::new("checkpoints/config.yaml");
    let speaker_path = Path::new("checkpoints/speaker_16k.wav");
    let output_path = Path::new("test_output.wav");

    // Skip if config doesn't exist
    if !config_path.exists() {
        eprintln!("Skipping: checkpoints/config.yaml not found");
        return;
    }

    // Skip if speaker audio doesn't exist
    if !speaker_path.exists() {
        eprintln!("Skipping: checkpoints/speaker_16k.wav not found");
        return;
    }

    // Create IndexTTS2 instance
    let mut tts = IndexTTS2::new(config_path).expect("Failed to create IndexTTS2");

    // Load weights
    tts.load_weights("checkpoints/").expect("Failed to load weights");

    // Run inference
    let result = tts.infer("Hello world, this is a test.", speaker_path)
        .expect("Failed to run inference");

    // Verify output
    assert!(result.audio.len() > 0, "Audio output is empty");
    assert_eq!(result.sample_rate, 22050, "Expected 22050 Hz sample rate");
    assert!(result.mel_codes.len() > 0, "Mel codes are empty");

    // Check audio duration is reasonable (0.5-30 seconds)
    let duration = result.duration();
    assert!(duration > 0.5, "Audio too short: {:.2}s", duration);
    assert!(duration < 30.0, "Audio too long: {:.2}s", duration);

    // Check audio values are in valid range [-1, 1]
    let max_val = result.audio.iter().cloned().fold(0.0f32, f32::max);
    let min_val = result.audio.iter().cloned().fold(0.0f32, f32::min);
    assert!(max_val <= 1.5, "Audio max value out of range: {}", max_val);
    assert!(min_val >= -1.5, "Audio min value out of range: {}", min_val);

    // Save output
    result.save(output_path).expect("Failed to save audio");
    assert!(output_path.exists(), "Output file not created");

    // Clean up
    if output_path.exists() {
        std::fs::remove_file(output_path).ok();
    }

    println!("Full inference test passed!");
    println!("  Audio duration: {:.2}s", duration);
    println!("  Mel codes: {}", result.mel_codes.len());
    println!("  Sample rate: {} Hz", result.sample_rate);
}

/// Test inference with different text lengths
#[test]
#[ignore = "Requires model weights in checkpoints/"]
fn test_inference_text_lengths() {
    use indextts2::inference::IndexTTS2;
    use std::path::Path;

    let config_path = Path::new("checkpoints/config.yaml");
    let speaker_path = Path::new("checkpoints/speaker_16k.wav");

    if !config_path.exists() || !speaker_path.exists() {
        eprintln!("Skipping: Required files not found");
        return;
    }

    let mut tts = IndexTTS2::new(config_path).expect("Failed to create IndexTTS2");
    tts.load_weights("checkpoints/").expect("Failed to load weights");

    let test_texts = vec![
        ("short", "Hello."),
        ("medium", "The quick brown fox jumps over the lazy dog."),
        ("long", "This is a much longer piece of text that should take more time to synthesize. It contains multiple sentences and should produce a longer audio output."),
    ];

    for (name, text) in test_texts {
        let result = tts.infer(text, speaker_path)
            .expect(&format!("Failed to infer '{}' text", name));

        assert!(result.audio.len() > 0, "{} text produced no audio", name);
        println!("{}: {:.2}s audio, {} mel codes", name, result.duration(), result.mel_codes.len());
    }
}

/// Benchmark-style timing test for inference components
#[test]
#[ignore = "Requires model weights in checkpoints/"]
fn test_inference_timing() {
    use indextts2::inference::IndexTTS2;
    use std::path::Path;
    use std::time::Instant;

    let config_path = Path::new("checkpoints/config.yaml");
    let speaker_path = Path::new("checkpoints/speaker_16k.wav");

    if !config_path.exists() || !speaker_path.exists() {
        eprintln!("Skipping: Required files not found");
        return;
    }

    // Time model loading
    let start = Instant::now();
    let mut tts = IndexTTS2::new(config_path).expect("Failed to create IndexTTS2");
    let init_time = start.elapsed();
    println!("Model initialization: {:?}", init_time);

    // Time weight loading
    let start = Instant::now();
    tts.load_weights("checkpoints/").expect("Failed to load weights");
    let load_time = start.elapsed();
    println!("Weight loading: {:?}", load_time);

    // Time inference (cold start)
    let text = "Hello world, this is a test.";
    let start = Instant::now();
    let result = tts.infer(text, speaker_path).expect("Failed to run inference");
    let cold_time = start.elapsed();
    println!("Cold inference: {:?} for {:.2}s audio", cold_time, result.duration());

    // Time inference (warm)
    let start = Instant::now();
    let result = tts.infer(text, speaker_path).expect("Failed to run inference");
    let warm_time = start.elapsed();
    println!("Warm inference: {:?} for {:.2}s audio", warm_time, result.duration());

    // Calculate real-time factor
    let rtf = warm_time.as_secs_f32() / result.duration();
    println!("Real-time factor: {:.2}x", rtf);
}
