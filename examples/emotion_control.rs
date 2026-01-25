//! Emotion control example for IndexTTS2
//!
//! Demonstrates how to use emotion embeddings to control speech style.

use anyhow::Result;

fn main() -> Result<()> {
    println!("IndexTTS2 Emotion Control Example");
    println!("==================================");

    println!();
    println!("Note: Emotion control requires the Qwen emotion model.");
    println!("This feature is not yet fully implemented in the Rust version.");
    println!();
    println!("Supported emotions (when available):");
    println!("  - neutral");
    println!("  - happy");
    println!("  - sad");
    println!("  - angry");
    println!("  - surprised");
    println!("  - fearful");
    println!("  - disgusted");
    println!();
    println!("Usage:");
    println!("  cargo run --release --bin indextts2 -- --cpu infer \\");
    println!("    --text \"Hello world\" \\");
    println!("    --speaker speaker.wav \\");
    println!("    --emotion happy \\");
    println!("    --output output.wav");

    Ok(())
}
