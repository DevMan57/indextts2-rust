//! BigVGAN weight loading utilities
//!
//! Handles loading weights from safetensors format, including
//! weight normalization conversion (weight_g, weight_v -> weight).

use anyhow::{Context, Result};
use candle_core::{safetensors, Device, Tensor};
use std::collections::HashMap;
use std::path::Path;

/// Load BigVGAN weights from safetensors file
pub fn load_bigvgan_weights<P: AsRef<Path>>(
    path: P,
    device: &Device,
) -> Result<HashMap<String, Tensor>> {
    let path = path.as_ref();
    tracing::info!("Loading BigVGAN weights from {:?}", path);

    let tensors = safetensors::load(path, device)
        .with_context(|| format!("Failed to load BigVGAN weights from {:?}", path))?;

    tracing::info!("Loaded {} tensors", tensors.len());

    // Convert weight normalization (weight_g, weight_v) to regular weights
    let mut converted = HashMap::new();
    let mut processed = std::collections::HashSet::new();

    for (name, tensor) in tensors.iter() {
        if processed.contains(name) {
            continue;
        }

        if name.ends_with(".weight_v") {
            let base_name = name.strip_suffix(".weight_v").unwrap();
            let g_name = format!("{}.weight_g", base_name);

            if let Some(weight_g) = tensors.get(&g_name) {
                // Apply weight normalization: weight = g * v / ||v||
                let weight = apply_weight_norm(weight_g, tensor)?;
                converted.insert(format!("{}.weight", base_name), weight);
                processed.insert(name.clone());
                processed.insert(g_name);
            } else {
                // No weight_g found, just use weight_v as-is
                converted.insert(format!("{}.weight", base_name), tensor.clone());
                processed.insert(name.clone());
            }
        } else if name.ends_with(".weight_g") {
            // Skip - handled with weight_v
            continue;
        } else {
            // Regular tensor
            converted.insert(name.clone(), tensor.clone());
            processed.insert(name.clone());
        }
    }

    tracing::info!("Converted to {} tensors", converted.len());
    Ok(converted)
}

/// Apply weight normalization: weight = g * (v / ||v||)
fn apply_weight_norm(weight_g: &Tensor, weight_v: &Tensor) -> Result<Tensor> {
    // Compute L2 norm of weight_v along all dims except first
    // weight_v shape: [out_channels, in_channels, kernel_size]
    // weight_g shape: [out_channels, 1, 1]

    let v_flat = weight_v.flatten(1, 2)?; // [out_channels, in_channels * kernel_size]
    let norm = v_flat
        .sqr()?
        .sum_keepdim(1)?
        .sqrt()?
        .unsqueeze(2)?; // [out_channels, 1, 1]

    // Avoid division by zero
    let norm = norm.clamp(1e-12, f64::MAX)?;

    // Normalize v and scale by g
    let v_normalized = weight_v.broadcast_div(&norm)?;
    weight_g.broadcast_mul(&v_normalized).map_err(Into::into)
}

/// Get tensor by name with optional default
pub fn get_tensor(
    weights: &HashMap<String, Tensor>,
    name: &str,
    default: Option<&Tensor>,
) -> Result<Tensor> {
    if let Some(tensor) = weights.get(name) {
        Ok(tensor.clone())
    } else if let Some(def) = default {
        Ok(def.clone())
    } else {
        anyhow::bail!("Tensor not found: {}", name)
    }
}

/// Get alpha tensor for Snake activation
pub fn get_alpha(weights: &HashMap<String, Tensor>, prefix: &str) -> Option<Tensor> {
    weights.get(&format!("{}.alpha", prefix)).cloned()
}

/// Get beta tensor for Snake-Beta activation
pub fn get_beta(weights: &HashMap<String, Tensor>, prefix: &str) -> Option<Tensor> {
    weights.get(&format!("{}.beta", prefix)).cloned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weight_norm() {
        let device = Device::Cpu;

        // Create test weight_v and weight_g
        let weight_v = Tensor::new(&[[[1.0f32, 2.0, 3.0]]], &device).unwrap(); // [1, 1, 3]
        let weight_g = Tensor::new(&[[[2.0f32]]], &device).unwrap(); // [1, 1, 1]

        let weight = apply_weight_norm(&weight_g, &weight_v).unwrap();

        // ||v|| = sqrt(1 + 4 + 9) = sqrt(14) ≈ 3.742
        // weight = 2 * [1, 2, 3] / 3.742 ≈ [0.535, 1.069, 1.604]
        let values: Vec<f32> = weight.flatten_all().unwrap().to_vec1().unwrap();
        assert!((values[0] - 0.535).abs() < 0.01);
        assert!((values[1] - 1.069).abs() < 0.01);
        assert!((values[2] - 1.604).abs() < 0.01);
    }
}
