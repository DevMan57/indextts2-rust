//! Streaming TTS example for IndexTTS2
//!
//! Demonstrates how to use the streaming synthesizer for real-time audio generation.

use anyhow::Result;
use indextts2::inference::StreamingSynthesizer;
use candle_core::Device;

fn main() -> Result<()> {
    println!("IndexTTS2 Streaming TTS Example");
    println!("================================");

    // Use CPU for this example
    let device = Device::Cpu;

    // Create streaming synthesizer
    println!("Initializing streaming synthesizer...");
    let mut synth = StreamingSynthesizer::new(&device)?;

    // Long text that will be processed in chunks
    let text = "This is a demonstration of streaming text-to-speech synthesis. \
                The audio is generated in chunks as the model processes each sentence. \
                This allows for lower latency in real-time applications.";

    println!("Generating audio for: \"{}\"", text);
    println!();

    // Generate all chunks
    let chunks = synth.generate_all(text)?;

    println!("Generated {} audio chunks:", chunks.len());
    for (i, chunk) in chunks.iter().enumerate() {
        let duration_ms = chunk.samples.len() as f32 / 22.05; // 22050 Hz
        println!(
            "  Chunk {}: {} samples ({:.1}ms){}",
            i + 1,
            chunk.samples.len(),
            duration_ms,
            if chunk.is_final { " [FINAL]" } else { "" }
        );
    }

    println!();
    println!("In a real application, each chunk would be sent to an audio player");
    println!("as soon as it's generated, enabling low-latency streaming.");

    Ok(())
}
