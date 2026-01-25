//! Basic inference example for IndexTTS2
//!
//! Demonstrates how to use the library for text-to-speech synthesis.

use anyhow::Result;
use indextts2::inference::IndexTTS2;
use std::path::Path;

fn main() -> Result<()> {
    println!("IndexTTS2 Basic Inference Example");
    println!("==================================");

    // Check for required files
    let checkpoint_dir = Path::new("checkpoints");
    if !checkpoint_dir.exists() {
        eprintln!("Error: checkpoints/ directory not found.");
        eprintln!("Please download the model weights first.");
        return Ok(());
    }

    // Initialize the model
    println!("Loading model...");
    let config_path = checkpoint_dir.join("config.yaml");
    let mut tts = IndexTTS2::new(&config_path)?;

    // Load weights
    println!("Loading weights...");
    tts.load_weights(checkpoint_dir)?;

    // Synthesize speech
    let text = "Hello world! This is a test of the IndexTTS2 text-to-speech system.";
    let speaker_path = Path::new("speaker_16k.wav");

    if !speaker_path.exists() {
        eprintln!("Warning: speaker_16k.wav not found. Using default speaker.");
    }

    println!("Synthesizing: \"{}\"", text);
    let audio = tts.infer(text, speaker_path)?;

    // Save output
    let output_path = "output.wav";
    audio.save(output_path)?;
    println!("Saved to: {}", output_path);

    Ok(())
}
