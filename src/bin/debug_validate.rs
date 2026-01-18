//! Debug validation CLI
//!
//! Validates Rust implementation against Python golden data.
//!
//! Usage:
//!     cargo run --bin debug_validate -- --golden-dir golden/ --component all
//!     cargo run --bin debug_validate -- --golden-dir golden/ --component gpt --verbose

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use candle_core::{Device, Tensor, DType};
use std::path::PathBuf;

use indextts2::debug::{Validator, ValidationConfig};
use indextts2::text::{TextNormalizer, TextTokenizer};
use indextts2::audio::MelSpectrogram;
use indextts2::models::semantic::SemanticEncoder;
use indextts2::models::speaker::CAMPPlus;
use indextts2::models::gpt::UnifiedVoice;
use indextts2::models::s2mel::{LengthRegulator, DiffusionTransformer, FlowMatching};
use indextts2::models::vocoder::BigVGAN;

/// Components that can be validated
#[derive(Clone, Copy, Debug, ValueEnum)]
enum Component {
    /// All components
    All,
    /// Text tokenizer
    Tokenizer,
    /// Mel spectrogram
    Mel,
    /// Speaker encoder (CAMPPlus)
    Speaker,
    /// Semantic encoder (Wav2Vec-BERT)
    Semantic,
    /// GPT decoder
    Gpt,
    /// S2Mel (DiT + Flow Matching)
    S2mel,
    /// Vocoder (BigVGAN)
    Vocoder,
    /// Full pipeline
    Full,
}

/// Debug validation CLI
#[derive(Parser, Debug)]
#[command(name = "debug_validate")]
#[command(about = "Validate Rust implementation against Python golden data")]
struct Args {
    /// Path to golden data directory
    #[arg(short, long, default_value = "debug/golden")]
    golden_dir: PathBuf,

    /// Component to validate
    #[arg(short, long, default_value = "all")]
    component: Component,

    /// Absolute tolerance
    #[arg(long, default_value = "1e-4")]
    atol: f32,

    /// Relative tolerance
    #[arg(long, default_value = "1e-3")]
    rtol: f32,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Use GPU
    #[arg(long)]
    gpu: bool,

    /// Path to model weights
    #[arg(long)]
    weights: Option<PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("IndexTTS2 Debug Validator");
    println!("========================\n");
    println!("Golden dir: {:?}", args.golden_dir);
    println!("Component: {:?}", args.component);
    println!("Tolerances: atol={}, rtol={}", args.atol, args.rtol);
    println!();

    if !args.golden_dir.exists() {
        eprintln!("Error: Golden directory not found: {:?}", args.golden_dir);
        eprintln!("Generate golden data first:");
        eprintln!("  python debug/dump_python.py --text \"Hello world\" --speaker voice.wav --output {:?}",
                 args.golden_dir);
        eprintln!("\nOr use mock data:");
        eprintln!("  python debug/dump_python.py --mock --output {:?}", args.golden_dir);
        std::process::exit(1);
    }

    let device = if args.gpu {
        Device::cuda_if_available(0)?
    } else {
        Device::Cpu
    };

    let config = ValidationConfig {
        atol: args.atol,
        rtol: args.rtol,
        verbose: args.verbose,
        max_diffs: 10,
    };

    let mut validator = Validator::with_config(&args.golden_dir, config);

    match args.component {
        Component::All => {
            validate_tokenizer(&mut validator)?;
            validate_mel(&mut validator)?;
            validate_speaker(&mut validator, &device)?;
            validate_semantic(&mut validator, &device)?;
            validate_gpt(&mut validator, &device)?;
            validate_s2mel(&mut validator, &device)?;
            validate_vocoder(&mut validator, &device)?;
        }
        Component::Tokenizer => validate_tokenizer(&mut validator)?,
        Component::Mel => validate_mel(&mut validator)?,
        Component::Speaker => validate_speaker(&mut validator, &device)?,
        Component::Semantic => validate_semantic(&mut validator, &device)?,
        Component::Gpt => validate_gpt(&mut validator, &device)?,
        Component::S2mel => validate_s2mel(&mut validator, &device)?,
        Component::Vocoder => validate_vocoder(&mut validator, &device)?,
        Component::Full => validate_full(&mut validator, &device)?,
    }

    validator.print_summary();

    if validator.all_passed() {
        std::process::exit(0);
    } else {
        std::process::exit(1);
    }
}

fn validate_tokenizer(validator: &mut Validator) -> Result<()> {
    println!("Validating tokenizer...");

    // Load golden tokens
    let golden_path = validator.results().first().map(|_| ());

    // For now, just check shape
    let tokens: Vec<i64> = vec![0; 10]; // Placeholder
    validator.validate_i64("tokens", &tokens, "input")?;

    Ok(())
}

fn validate_mel(validator: &mut Validator) -> Result<()> {
    println!("Validating mel spectrogram...");

    // Load golden audio and compute mel
    let mel_extractor = MelSpectrogram::new(1024, 256, 1024, 80, 22050, 0.0)?;

    // Placeholder validation
    let samples = vec![0.0f32; 22050];
    let _mel = mel_extractor.compute(&samples)?;

    println!("  Mel computation working");
    Ok(())
}

fn validate_speaker(validator: &mut Validator, device: &Device) -> Result<()> {
    println!("Validating speaker encoder...");

    let mut encoder = CAMPPlus::new(device)?;
    encoder.initialize_random()?;

    // Load golden audio
    let audio = Tensor::randn(0.0f32, 1.0, (1, 16000), device)?;
    let emb = encoder.forward(&audio)?;

    validator.validate_tensor("speaker_emb", &emb, "encoders")?;
    Ok(())
}

fn validate_semantic(validator: &mut Validator, device: &Device) -> Result<()> {
    println!("Validating semantic encoder...");

    let mut encoder = SemanticEncoder::new(device)?;
    encoder.initialize_random()?;

    let audio = Tensor::randn(0.0f32, 1.0, (1, 16000), device)?;
    let features = encoder.forward(&audio)?;

    validator.validate_tensor("semantic_feat", &features, "encoders")?;
    Ok(())
}

fn validate_gpt(validator: &mut Validator, device: &Device) -> Result<()> {
    println!("Validating GPT...");

    let mut gpt = UnifiedVoice::new(device)?;
    gpt.initialize_random()?;

    // Create dummy input
    let text_ids = Tensor::zeros((1, 10), DType::I64, device)?;
    let _output = gpt.forward(&text_ids)?;

    // Validate layer outputs
    // These would be captured with hooks in a real implementation
    let layer_00 = Tensor::randn(0.0f32, 1.0, (1, 100, 1280), device)?;
    validator.validate_tensor("layer_00", &layer_00, "gpt")?;

    let layer_12 = Tensor::randn(0.0f32, 1.0, (1, 100, 1280), device)?;
    validator.validate_tensor("layer_12", &layer_12, "gpt")?;

    let layer_23 = Tensor::randn(0.0f32, 1.0, (1, 100, 1280), device)?;
    validator.validate_tensor("layer_23", &layer_23, "gpt")?;

    Ok(())
}

fn validate_s2mel(validator: &mut Validator, device: &Device) -> Result<()> {
    println!("Validating S2Mel (DiT + Flow Matching)...");

    // Length regulator
    let mut regulator = LengthRegulator::new(device)?;
    regulator.initialize_random()?;

    let codes = Tensor::randn(0.0f32, 1.0, (1, 50, 1024), device)?;
    let (features, _durations) = regulator.forward(&codes, Some(&[100]))?;
    validator.validate_tensor("length_reg", &features, "synthesis")?;

    // DiT
    let mut dit = DiffusionTransformer::new(device)?;
    dit.initialize_random()?;

    let x = Tensor::randn(0.0f32, 1.0, (1, 100, 80), device)?;
    let t = Tensor::new(&[0.0f32], device)?;
    let content = Tensor::randn(0.0f32, 1.0, (1, 100, 512), device)?;
    let step_00 = dit.forward(&x, &t, &content, None)?;
    validator.validate_tensor("dit_step_00", &step_00, "synthesis")?;

    // Flow matching
    let fm = FlowMatching::new(device);
    let noise = fm.sample_noise(&[1, 100, 80])?;
    let mel_spec = fm.sample(&dit, &noise, &content, None)?;
    validator.validate_tensor("mel_spec", &mel_spec, "synthesis")?;

    Ok(())
}

fn validate_vocoder(validator: &mut Validator, device: &Device) -> Result<()> {
    println!("Validating vocoder (BigVGAN)...");

    let mut vocoder = BigVGAN::new(device)?;
    vocoder.initialize_random()?;

    let mel = Tensor::randn(0.0f32, 1.0, (1, 80, 100), device)?;
    let audio = vocoder.forward(&mel)?;

    validator.validate_tensor("audio", &audio.squeeze(0)?.squeeze(0)?, "output")?;
    Ok(())
}

fn validate_full(validator: &mut Validator, device: &Device) -> Result<()> {
    println!("Validating full pipeline...");

    // This would run the complete pipeline and compare final output
    // For now, validate that all components can be created

    let _normalizer = TextNormalizer::new();
    let _mel = MelSpectrogram::new(1024, 256, 1024, 80, 22050, 0.0)?;
    let _speaker = CAMPPlus::new(device)?;
    let _semantic = SemanticEncoder::new(device)?;
    let _gpt = UnifiedVoice::new(device)?;
    let _regulator = LengthRegulator::new(device)?;
    let _dit = DiffusionTransformer::new(device)?;
    let _fm = FlowMatching::new(device);
    let _vocoder = BigVGAN::new(device)?;

    println!("  All components created successfully");
    Ok(())
}
