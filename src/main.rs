//! IndexTTS2 CLI - Command-line interface for text-to-speech synthesis

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};
use std::path::PathBuf;
use std::time::Instant;
use tracing::{info, warn, Level};
use tracing_subscriber::FmtSubscriber;

use indextts2::{IndexTTS2, ModelConfig, VERSION};
use indextts2::inference::{InferenceConfig, StreamingSynthesizer};

/// IndexTTS2 - High-performance zero-shot text-to-speech in Rust
#[derive(Parser, Debug)]
#[command(name = "indextts2")]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Enable verbose logging
    #[arg(short, long, global = true)]
    verbose: bool,

    /// Use CPU instead of GPU
    #[arg(long, global = true)]
    cpu: bool,

    /// Use FP16 precision (faster, slightly lower quality)
    #[arg(long, global = true)]
    fp16: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Synthesize speech from text
    Infer {
        /// Text to synthesize
        #[arg(short, long)]
        text: String,

        /// Path to speaker reference audio
        #[arg(short, long)]
        speaker: PathBuf,

        /// Output audio file path
        #[arg(short, long, default_value = "output.wav")]
        output: PathBuf,

        /// Path to emotion reference audio (optional)
        #[arg(long)]
        emotion_audio: Option<PathBuf>,

        /// Emotion blending alpha (0.0 - 1.0)
        #[arg(long, default_value = "1.0")]
        emotion_alpha: f32,

        /// Emotion vector as comma-separated values
        /// Order: happy,angry,sad,afraid,disgusted,melancholic,surprised,calm
        #[arg(long)]
        emotion_vector: Option<String>,

        /// Path to model config file
        #[arg(short, long, default_value = "checkpoints/config.yaml")]
        config: PathBuf,

        /// Maximum text tokens per segment
        #[arg(long, default_value = "120")]
        max_tokens: usize,

        /// Generation temperature (0.0-1.0)
        #[arg(long, default_value = "0.8")]
        temperature: f32,

        /// Top-k sampling (0 = disabled)
        #[arg(long, default_value = "50")]
        top_k: usize,

        /// Top-p (nucleus) sampling (1.0 = disabled)
        #[arg(long, default_value = "0.95")]
        top_p: f32,

        /// Enable streaming output (play as generated)
        #[arg(long)]
        stream: bool,
    },

    /// Start streaming TTS server
    Serve {
        /// Port to listen on
        #[arg(short, long, default_value = "8080")]
        port: u16,

        /// Path to model config file
        #[arg(short, long, default_value = "checkpoints/config.yaml")]
        config: PathBuf,
    },

    /// Download model weights from HuggingFace
    Download {
        /// Model version to download (1.5 or 2)
        #[arg(short, long, default_value = "2")]
        version: String,

        /// Output directory for checkpoints
        #[arg(short, long, default_value = "checkpoints")]
        output: PathBuf,
    },

    /// Show model information
    Info {
        /// Path to model config file
        #[arg(short, long, default_value = "checkpoints/config.yaml")]
        config: PathBuf,
    },
}

fn setup_logging(verbose: bool) {
    let level = if verbose { Level::DEBUG } else { Level::INFO };
    let subscriber = FmtSubscriber::builder()
        .with_max_level(level)
        .with_target(false)
        .with_thread_ids(false)
        .compact()
        .finish();
    tracing::subscriber::set_global_default(subscriber).expect("Failed to set tracing subscriber");
}

fn create_progress_bar(msg: &str) -> ProgressBar {
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .unwrap(),
    );
    pb.set_message(msg.to_string());
    pb
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    setup_logging(cli.verbose);

    info!("IndexTTS2 v{}", VERSION);

    match cli.command {
        Commands::Infer {
            text,
            speaker,
            output,
            emotion_audio,
            emotion_alpha,
            emotion_vector,
            config,
            max_tokens,
            temperature,
            top_k,
            top_p,
            stream,
        } => {
            // Validate inputs
            if !speaker.exists() {
                anyhow::bail!("Speaker audio file not found: {:?}", speaker);
            }

            let pb = create_progress_bar("Loading model...");
            let start = Instant::now();

            // Create inference config
            let inference_config = InferenceConfig {
                temperature,
                top_k,
                top_p,
                use_gpu: !cli.cpu,
                ..Default::default()
            };

            // Load model
            let mut tts = IndexTTS2::with_config(&config, inference_config)
                .context("Failed to load model config")?;

            // Load weights if config directory has checkpoints
            if let Some(model_dir) = config.parent() {
                if model_dir.join("gpt.pth").exists() || model_dir.join("s2mel.pth").exists() {
                    pb.set_message("Loading model weights...");
                    tts.load_weights(model_dir)?;
                } else {
                    warn!("No model weights found in {:?}, using random initialization", model_dir);
                }
            }

            pb.finish_with_message(format!("Model loaded in {:.1}s", start.elapsed().as_secs_f32()));

            info!("Text: {} ({} chars)", &text[..text.len().min(50)], text.len());
            info!("Speaker: {:?}", speaker);
            info!("Output: {:?}", output);

            // Perform inference
            let pb = create_progress_bar("Generating speech...");
            let start = Instant::now();

            let result = if emotion_audio.is_some() {
                tts.infer_with_emotion(&text, &speaker, emotion_audio.as_ref())
            } else {
                tts.infer(&text, &speaker)
            }
            .context("Inference failed")?;

            let duration = result.duration();
            pb.finish_with_message(format!(
                "Generated {:.1}s of audio in {:.1}s (RTF: {:.2}x)",
                duration,
                start.elapsed().as_secs_f32(),
                duration / start.elapsed().as_secs_f32()
            ));

            // Save output
            result.save(&output)
                .context("Failed to save audio")?;

            info!("Saved to {:?}", output);
            info!("Generated {} mel codes", result.mel_codes.len());

            Ok(())
        }

        Commands::Serve { port, config } => {
            info!("Starting TTS server on port {}", port);
            // TODO: Implement HTTP/WebSocket server
            eprintln!("🚧 Server mode not yet implemented.");
            Ok(())
        }

        Commands::Download { version, output } => {
            info!("Downloading IndexTTS{} models to {:?}", version, output);
            // TODO: Implement model download from HuggingFace
            eprintln!("🚧 Model download not yet implemented.");
            eprintln!("For now, use: hf download IndexTeam/IndexTTS-2 --local-dir=checkpoints");
            Ok(())
        }

        Commands::Info { config } => {
            info!("Loading config from {:?}", config);
            
            if config.exists() {
                let cfg = ModelConfig::load(&config)
                    .context("Failed to load config")?;
                println!("{:#?}", cfg);
            } else {
                eprintln!("Config file not found: {:?}", config);
                eprintln!("Download models first with: indextts2 download");
            }
            
            Ok(())
        }
    }
}
