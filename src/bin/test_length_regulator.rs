//! Targeted CPU test for the LengthRegulator forward() pass
//!
//! Creates a test input with known statistics (mean=0, var=2.3) and runs
//! forward() with target_length=172 to capture per-block debug output.
//!
//! Usage: cargo run --release --bin test_length_regulator --no-default-features

use anyhow::Result;
use candle_core::{Device, Tensor, D};
use indextts2::models::s2mel::{LengthRegulator, LengthRegulatorConfig};

fn main() -> Result<()> {
    let device = Device::Cpu;

    // Create length regulator with default config
    let config = LengthRegulatorConfig::default();
    let mut lr = LengthRegulator::with_config(config, &device)?;

    // Load weights from checkpoint
    lr.load_weights("checkpoints/s2mel.safetensors")?;

    // Create test input matching S_infer statistics
    // S_infer has mean~0, var~2.3, shape [1, T, 1024]
    // randn gives N(0,1), multiply by sqrt(2.3) ~ 1.517 to get var=2.3
    let input = (Tensor::randn(0.0f32, 1.0, (1, 100, 1024), &device)? * 1.517)?;
    let in_mean: f32 = input.mean_all()?.to_scalar()?;
    let in_var: f32 = input.var(D::Minus1)?.mean_all()?.to_scalar()?;
    eprintln!("INPUT: shape={:?}, mean={:.6}, var={:.6}", input.shape(), in_mean, in_var);

    // Run forward pass with target_length=172 (100 * 1.72)
    let target_len = 172;
    eprintln!("Running forward with target_length={}...", target_len);
    let (output, _durations) = lr.forward(&input, Some(&[target_len]))?;

    let out_mean: f32 = output.mean_all()?.to_scalar()?;
    let out_var: f32 = output.var(D::Minus1)?.mean_all()?.to_scalar()?;
    eprintln!("OUTPUT: shape={:?}, mean={:.6}, var={:.6}", output.shape(), out_mean, out_var);

    // Summary
    eprintln!("\n--- SUMMARY ---");
    eprintln!("Input  variance: {:.6}", in_var);
    eprintln!("Output variance: {:.6}", out_var);
    eprintln!("Variance ratio:  {:.6}", out_var / in_var);

    Ok(())
}
