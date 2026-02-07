//! Semantic codec for vector quantization
//!
//! Quantizes semantic embeddings from Wav2Vec-BERT to discrete codes.
//! Uses a learned codebook with 8192 entries and 8-dimensional codes.
//!
//! Architecture:
//! - Input: Semantic embeddings (batch, seq_len, hidden_size=1024)
//! - Projection: hidden_size -> codebook_dim (1024 -> 8)
//! - VQ: Find nearest codebook entry for each frame
//! - Output: Quantized embeddings + discrete codes

use anyhow::{bail, Result};
use candle_core::{safetensors, Device, Tensor, DType, D};
use candle_nn::{Linear, Module, VarBuilder};
use std::path::Path;

use super::VocosBackbone;

/// Default codebook size (8192 entries)
const DEFAULT_CODEBOOK_SIZE: usize = 8192;
/// Default hidden size from Wav2Vec-BERT
const DEFAULT_HIDDEN_SIZE: usize = 1024;
/// Default codebook dimension
const DEFAULT_CODEBOOK_DIM: usize = 8;

/// Semantic codec with vector quantization
///
/// Quantizes continuous semantic features to discrete codes using
/// a learned codebook. This enables the GPT model to work with
/// discrete mel codes.
pub struct SemanticCodec {
    device: Device,
    /// Codebook size (number of entries)
    codebook_size: usize,
    /// Hidden size (input dimension)
    hidden_size: usize,
    /// Codebook dimension (code vector size)
    codebook_dim: usize,
    /// Projection from hidden_size to codebook_dim
    proj_in: Option<Linear>,
    /// Projection from codebook_dim back to hidden_size
    proj_out: Option<Linear>,
    /// Codebook embeddings (codebook_size, codebook_dim)
    codebook: Option<Tensor>,
    /// VocosBackbone encoder (12 ConvNeXtV2 blocks + linear)
    encoder: Option<VocosBackbone>,
}

impl SemanticCodec {
    /// Create a new semantic codec with default parameters
    pub fn new(device: &Device) -> Result<Self> {
        Self::with_config(DEFAULT_CODEBOOK_SIZE, DEFAULT_HIDDEN_SIZE, DEFAULT_CODEBOOK_DIM, device)
    }

    /// Create a new semantic codec with custom configuration
    pub fn with_config(
        codebook_size: usize,
        hidden_size: usize,
        codebook_dim: usize,
        device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            device: device.clone(),
            codebook_size,
            hidden_size,
            codebook_dim,
            proj_in: None,
            proj_out: None,
            codebook: None,
            encoder: None,
        })
    }

    /// Load codec weights from safetensors file
    pub fn load<P: AsRef<Path>>(path: P, device: &Device) -> Result<Self> {
        let mut codec = Self::new(device)?;
        codec.load_weights(path)?;
        Ok(codec)
    }

    /// Load weights from safetensors file
    pub fn load_weights<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let path = path.as_ref();

        if !path.exists() {
            // Initialize with random codebook for testing
            self.initialize_random_codebook()?;
            return Ok(());
        }

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[path], DType::F32, &self.device)?
        };

        // Load projection layers
        self.proj_in = Some(candle_nn::linear(
            self.hidden_size,
            self.codebook_dim,
            vb.pp("proj_in"),
        )?);

        self.proj_out = Some(candle_nn::linear(
            self.codebook_dim,
            self.hidden_size,
            vb.pp("proj_out"),
        )?);

        // Load codebook
        let codebook = vb.get((self.codebook_size, self.codebook_dim), "codebook")?;
        self.codebook = Some(codebook);

        Ok(())
    }

    /// Load quantizer weights from a MaskGCT semantic codec checkpoint.
    ///
    /// Extracts the first quantizer's codebook and weight-normalized output
    /// projection from the checkpoint, computing the actual linear weight via:
    ///
    ///   actual_weight = weight_g * (weight_v / ||weight_v||_per_row)
    ///
    /// # Arguments
    /// * `path` - Path to the MaskGCT `model.safetensors` file
    ///
    /// # Tensor keys used
    /// - `quantizer.quantizers.0.codebook.weight` [8192, 8]
    /// - `quantizer.quantizers.0.out_project.weight_g` [1024, 1, 1]
    /// - `quantizer.quantizers.0.out_project.weight_v` [1024, 8, 1]
    /// - `quantizer.quantizers.0.out_project.bias` [1024]
    pub fn load_quantizer_weights<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let path = path.as_ref();

        if !path.exists() {
            bail!(
                "MaskGCT semantic codec checkpoint not found: {:?}",
                path
            );
        }

        let tensors = safetensors::load(path, &self.device)?;

        // --- Codebook ---
        let codebook_key = "quantizer.quantizers.0.codebook.weight";
        let codebook = tensors
            .get(codebook_key)
            .ok_or_else(|| anyhow::anyhow!("Key not found: {}", codebook_key))?
            .clone();

        let (cb_size, cb_dim) = codebook.dims2()?;
        if cb_size != self.codebook_size || cb_dim != self.codebook_dim {
            bail!(
                "Codebook shape mismatch: expected [{}, {}], got [{}, {}]",
                self.codebook_size,
                self.codebook_dim,
                cb_size,
                cb_dim
            );
        }
        self.codebook = Some(codebook.to_dtype(DType::F32)?);

        // --- Weight-normalized output projection (Conv1d kernel=1 → Linear) ---
        let wg_key = "quantizer.quantizers.0.out_project.weight_g";
        let wv_key = "quantizer.quantizers.0.out_project.weight_v";
        let bias_key = "quantizer.quantizers.0.out_project.bias";

        let weight_g = tensors
            .get(wg_key)
            .ok_or_else(|| anyhow::anyhow!("Key not found: {}", wg_key))?
            .to_dtype(DType::F32)?;
        let weight_v = tensors
            .get(wv_key)
            .ok_or_else(|| anyhow::anyhow!("Key not found: {}", wv_key))?
            .to_dtype(DType::F32)?;
        let bias = tensors
            .get(bias_key)
            .ok_or_else(|| anyhow::anyhow!("Key not found: {}", bias_key))?
            .to_dtype(DType::F32)?;

        // weight_g: [1024, 1, 1] → squeeze to [1024, 1]
        let weight_g = weight_g.squeeze(D::Minus1)?.squeeze(D::Minus1)?.unsqueeze(D::Minus1)?;
        // weight_g is now [1024, 1]

        // weight_v: [1024, 8, 1] → squeeze to [1024, 8]
        let weight_v = weight_v.squeeze(D::Minus1)?;
        // weight_v is now [1024, 8]

        // Compute per-output-channel L2 norm of weight_v: ||weight_v||_dim1 → [1024, 1]
        let norm_v = weight_v
            .sqr()?
            .sum(D::Minus1)?
            .sqrt()?
            .unsqueeze(D::Minus1)?;
        // norm_v is [1024, 1]

        // Avoid division by zero
        let norm_v = norm_v.clamp(1e-12, f64::MAX)?;

        // actual_weight = weight_g * weight_v / norm_v → [1024, 8]
        let actual_weight = weight_g
            .broadcast_mul(&weight_v)?
            .broadcast_div(&norm_v)?;

        // bias: [1024]
        let bias = bias.flatten_all()?;

        self.proj_out = Some(Linear::new(actual_weight, Some(bias)));

        // --- Weight-normalized input projection (Conv1d kernel=1 → Linear) ---
        let in_wg_key = "quantizer.quantizers.0.in_project.weight_g";
        let in_wv_key = "quantizer.quantizers.0.in_project.weight_v";
        let in_bias_key = "quantizer.quantizers.0.in_project.bias";

        let in_weight_g = tensors
            .get(in_wg_key)
            .ok_or_else(|| anyhow::anyhow!("Key not found: {}", in_wg_key))?
            .to_dtype(DType::F32)?;
        let in_weight_v = tensors
            .get(in_wv_key)
            .ok_or_else(|| anyhow::anyhow!("Key not found: {}", in_wv_key))?
            .to_dtype(DType::F32)?;
        let in_bias = tensors
            .get(in_bias_key)
            .ok_or_else(|| anyhow::anyhow!("Key not found: {}", in_bias_key))?
            .to_dtype(DType::F32)?;

        // in_weight_g: [8, 1, 1] → squeeze to [8, 1]
        let in_weight_g = in_weight_g.squeeze(D::Minus1)?.squeeze(D::Minus1)?.unsqueeze(D::Minus1)?;
        // in_weight_g is now [8, 1]

        // in_weight_v: [8, 1024, 1] → squeeze to [8, 1024]
        let in_weight_v = in_weight_v.squeeze(D::Minus1)?;
        // in_weight_v is now [8, 1024]

        // Compute per-output-channel L2 norm of in_weight_v: ||weight_v||_dim1 → [8, 1]
        let in_norm_v = in_weight_v
            .sqr()?
            .sum(D::Minus1)?
            .sqrt()?
            .unsqueeze(D::Minus1)?;
        let in_norm_v = in_norm_v.clamp(1e-12, f64::MAX)?;

        // actual_weight = weight_g * weight_v / norm_v → [8, 1024]
        let actual_weight_in = in_weight_g
            .broadcast_mul(&in_weight_v)?
            .broadcast_div(&in_norm_v)?;

        // bias: [8]
        let in_bias = in_bias.flatten_all()?;

        self.proj_in = Some(Linear::new(actual_weight_in, Some(in_bias)));

        // --- VocosBackbone encoder ---
        self.encoder = Some(VocosBackbone::load(&tensors, &self.device)?);

        Ok(())
    }

    /// Initialize random codebook for testing/placeholder
    fn initialize_random_codebook(&mut self) -> Result<()> {
        // Random codebook with small values
        let codebook = Tensor::randn(
            0.0f32,
            0.02,
            (self.codebook_size, self.codebook_dim),
            &self.device,
        )?;
        self.codebook = Some(codebook);
        Ok(())
    }

    /// Quantize embeddings to discrete codes
    ///
    /// # Arguments
    /// * `embeddings` - Input embeddings (batch, seq_len, hidden_size)
    ///
    /// # Returns
    /// * Tuple of (quantized embeddings, discrete codes)
    /// * quantized: (batch, seq_len, hidden_size)
    /// * codes: (batch, seq_len) with values in [0, codebook_size)
    pub fn quantize(&self, embeddings: &Tensor) -> Result<(Tensor, Tensor)> {
        let (_batch_size, _seq_len, hidden) = embeddings.dims3()?;

        // Project to codebook dimension if projection layer exists
        let projected = if let Some(ref proj) = self.proj_in {
            proj.forward(embeddings)?
        } else {
            // Direct projection if no learned layer
            if hidden != self.codebook_dim {
                // Simple linear projection placeholder
                // Flatten to 2D, project, reshape back to 3D
                let (batch, seq, _) = embeddings.dims3()?;
                let flat = embeddings.reshape((batch * seq, hidden))?;
                let weight = Tensor::randn(
                    0.0f32,
                    0.02,
                    (hidden, self.codebook_dim),
                    &self.device,
                )?;
                let projected_flat = flat.matmul(&weight)?;
                projected_flat.reshape((batch, seq, self.codebook_dim))?
            } else {
                embeddings.clone()
            }
        };

        // Get codebook or create placeholder
        let codebook = self.codebook.as_ref().map_or_else(
            || {
                Tensor::randn(
                    0.0f32,
                    0.02,
                    (self.codebook_size, self.codebook_dim),
                    &self.device,
                )
            },
            |cb| Ok(cb.clone()),
        )?;

        // Find nearest codebook entries using L2-normalized cosine distance
        // projected: (batch, seq, codebook_dim)
        // codebook: (codebook_size, codebook_dim)
        let codes = self.find_nearest_codes(&projected, &codebook)?;

        // Look up quantized embeddings from codebook
        let quantized_codes = self.lookup_codes(&codes, &codebook)?;

        // Project back to hidden size if projection layer exists
        let quantized = if let Some(ref proj) = self.proj_out {
            proj.forward(&quantized_codes)?
        } else if self.codebook_dim != self.hidden_size {
            // Simple projection back
            // Flatten to 2D, project, reshape back to 3D
            let (batch, seq, _) = quantized_codes.dims3()?;
            let flat = quantized_codes.reshape((batch * seq, self.codebook_dim))?;
            let weight = Tensor::randn(
                0.0f32,
                0.02,
                (self.codebook_dim, self.hidden_size),
                &self.device,
            )?;
            let projected_flat = flat.matmul(&weight)?;
            projected_flat.reshape((batch, seq, self.hidden_size))?
        } else {
            quantized_codes
        };

        Ok((quantized, codes))
    }

    /// Full encode + quantize pipeline for reference audio features.
    ///
    /// Runs the VocosBackbone encoder (if loaded) followed by vector
    /// quantization, producing both quantized embeddings and discrete codes.
    ///
    /// # Arguments
    /// * `features` - W2V-BERT features `(B, T, 1024)` -- raw semantic embeddings
    ///
    /// # Returns
    /// * `(quantized_embeddings, codes)` where:
    ///   - `quantized_embeddings`: `(B, T, 1024)` -- reconstructed through VQ + out_project
    ///   - `codes`: `(B, T)` -- discrete codebook indices
    ///
    /// # Pipeline
    /// `features -> encoder -> in_project -> L2-norm VQ -> out_project`
    pub fn encode_and_quantize(&self, features: &Tensor) -> Result<(Tensor, Tensor)> {
        // Step 1: Run through VocosBackbone encoder (if loaded)
        let encoded = if let Some(ref enc) = self.encoder {
            enc.forward(features)?
        } else {
            features.clone()
        };

        // Step 2: Quantize using in_project + L2-norm VQ + out_project
        self.quantize(&encoded)
    }

    /// L2-normalize a tensor along the last dimension.
    ///
    /// For each vector v, computes v / max(||v||, 1e-12).
    fn l2_normalize(x: &Tensor) -> Result<Tensor> {
        let norm = x
            .sqr()?
            .sum(D::Minus1)?
            .sqrt()?
            .clamp(1e-12, f64::MAX)?
            .unsqueeze(D::Minus1)?;
        x.broadcast_div(&norm).map_err(Into::into)
    }

    /// Find nearest codebook entry for each embedding vector using
    /// L2-normalized (cosine) distance, matching the Python MaskGCT
    /// `use_l2_normlize=True` behavior.
    fn find_nearest_codes(&self, projected: &Tensor, codebook: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, _) = projected.dims3()?;

        // Flatten to (batch * seq, codebook_dim)
        let flat = projected.reshape((batch_size * seq_len, self.codebook_dim))?;

        // L2-normalize both encodings and codebook before distance computation
        let flat_norm = Self::l2_normalize(&flat)?;
        let codebook_norm = Self::l2_normalize(codebook)?;

        // Compute L2 distance on normalized vectors:
        // ||a_norm - b_norm||^2 = ||a_norm||^2 + ||b_norm||^2 - 2*a_norm·b_norm
        // flat_norm: (N, D), codebook_norm: (K, D)

        // ||a_norm||^2: (N, 1)
        let flat_sq = flat_norm.sqr()?.sum(D::Minus1)?.unsqueeze(1)?;

        // ||b_norm||^2: (1, K)
        let codebook_sq = codebook_norm.sqr()?.sum(D::Minus1)?.unsqueeze(0)?;

        // a_norm · b_norm: (N, K)
        let dot = flat_norm.matmul(&codebook_norm.t()?)?;

        // Distance: (N, K)
        let dist = (flat_sq.broadcast_add(&codebook_sq)? - (dot * 2.0)?)?;

        // Find argmin along codebook dimension
        let codes = dist.argmin(D::Minus1)?;

        // Reshape back to (batch, seq)
        codes.reshape((batch_size, seq_len)).map_err(Into::into)
    }

    /// Look up embeddings from codebook using codes
    fn lookup_codes(&self, codes: &Tensor, codebook: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len) = codes.dims2()?;

        // Flatten codes to 1D for indexing
        let flat_codes = codes.flatten_all()?;

        // Index into codebook
        let flat_embeddings = codebook.index_select(&flat_codes, 0)?;

        // Reshape to (batch, seq, codebook_dim)
        flat_embeddings
            .reshape((batch_size, seq_len, self.codebook_dim))
            .map_err(Into::into)
    }

    /// Convert discrete codes back to embeddings
    ///
    /// # Arguments
    /// * `codes` - Discrete codes (batch, seq_len)
    ///
    /// # Returns
    /// * Embeddings (batch, seq_len, hidden_size)
    pub fn vq2emb(&self, codes: &Tensor) -> Result<Tensor> {
        let codebook = self.codebook.as_ref().map_or_else(
            || {
                Tensor::randn(
                    0.0f32,
                    0.02,
                    (self.codebook_size, self.codebook_dim),
                    &self.device,
                )
            },
            |cb| Ok(cb.clone()),
        )?;

        // Look up embeddings
        let quantized_codes = self.lookup_codes(codes, &codebook)?;

        // Project back to hidden size
        if let Some(ref proj) = self.proj_out {
            proj.forward(&quantized_codes).map_err(Into::into)
        } else if self.codebook_dim != self.hidden_size {
            let weight = Tensor::randn(
                0.0f32,
                0.02,
                (self.codebook_dim, self.hidden_size),
                &self.device,
            )?;
            quantized_codes.broadcast_matmul(&weight).map_err(Into::into)
        } else {
            Ok(quantized_codes)
        }
    }

    /// Convert codes to embeddings (alias for vq2emb)
    pub fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        self.vq2emb(codes)
    }

    /// Get the codebook size
    pub fn codebook_size(&self) -> usize {
        self.codebook_size
    }

    /// Get the hidden size
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Get the codebook dimension
    pub fn codebook_dim(&self) -> usize {
        self.codebook_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_codec_new() {
        let device = Device::Cpu;
        let codec = SemanticCodec::new(&device).unwrap();

        assert_eq!(codec.codebook_size(), 8192);
        assert_eq!(codec.hidden_size(), 1024);
        assert_eq!(codec.codebook_dim(), 8);
    }

    #[test]
    fn test_quantize_and_decode() {
        let device = Device::Cpu;
        let mut codec = SemanticCodec::new(&device).unwrap();
        codec.initialize_random_codebook().unwrap();

        // Create dummy input (batch=2, seq=10, hidden=1024)
        let input = Tensor::randn(0.0f32, 1.0, (2, 10, 1024), &device).unwrap();

        // Quantize
        let (quantized, codes) = codec.quantize(&input).unwrap();

        assert_eq!(quantized.dims3().unwrap(), (2, 10, 1024));
        assert_eq!(codes.dims2().unwrap(), (2, 10));

        // Decode codes back
        let decoded = codec.vq2emb(&codes).unwrap();
        assert_eq!(decoded.dims3().unwrap(), (2, 10, 1024));
    }

    #[test]
    fn test_codes_in_range() {
        let device = Device::Cpu;
        let mut codec = SemanticCodec::new(&device).unwrap();
        codec.initialize_random_codebook().unwrap();

        let input = Tensor::randn(0.0f32, 1.0, (1, 50, 1024), &device).unwrap();
        let (_, codes) = codec.quantize(&input).unwrap();

        // Verify codes are in valid range (argmin returns U32)
        let codes_vec: Vec<u32> = codes.flatten_all().unwrap().to_vec1().unwrap();
        for code in codes_vec {
            assert!(code < 8192, "Code {} out of range", code);
        }
    }

    #[test]
    fn test_l2_normalize() {
        let device = Device::Cpu;
        // Create a simple 2D tensor
        let x = Tensor::new(&[[3.0f32, 4.0], [0.0, 0.0], [1.0, 0.0]], &device).unwrap();
        let normed = SemanticCodec::l2_normalize(&x).unwrap();
        let vals: Vec<Vec<f32>> = normed.to_vec2().unwrap();

        // [3, 4] -> [0.6, 0.8]  (norm = 5)
        assert!((vals[0][0] - 0.6).abs() < 1e-5);
        assert!((vals[0][1] - 0.8).abs() < 1e-5);

        // [0, 0] -> clamped to small epsilon norm, values near 0
        assert!(vals[1][0].abs() < 1e-3);
        assert!(vals[1][1].abs() < 1e-3);

        // [1, 0] -> [1, 0]  (already unit norm)
        assert!((vals[2][0] - 1.0).abs() < 1e-5);
        assert!(vals[2][1].abs() < 1e-5);
    }

    #[test]
    fn test_l2_normalized_quantize_selects_cosine_nearest() {
        let device = Device::Cpu;
        let mut codec = SemanticCodec::with_config(3, 2, 2, &device).unwrap();

        // Create a codebook with 3 entries of dim 2
        // Entries point in distinct directions
        let codebook = Tensor::new(
            &[[1.0f32, 0.0], [0.0, 1.0], [1.0, 1.0]],
            &device,
        ).unwrap();
        codec.codebook = Some(codebook);

        // Input vector [100, 1] -- large magnitude pointing mostly along dim 0
        // Under raw L2 distance, this might not pick [1,0] due to magnitude.
        // Under L2-normalized distance, [100,1] normalizes to ~[1, 0.01]
        // which is closest to codebook entry [1, 0] (cosine similarity ~1.0).
        let input = Tensor::new(&[[[100.0f32, 1.0]]], &device).unwrap();
        let (_, codes) = codec.quantize(&input).unwrap();
        let code: Vec<u32> = codes.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(code[0], 0, "Should select [1,0] entry via cosine distance");
    }

    #[test]
    fn test_encode_and_quantize_without_encoder() {
        let device = Device::Cpu;
        let mut codec = SemanticCodec::new(&device).unwrap();
        codec.initialize_random_codebook().unwrap();

        // Without encoder loaded, encode_and_quantize should behave like quantize
        let input = Tensor::randn(0.0f32, 1.0, (1, 5, 1024), &device).unwrap();
        let (quantized, codes) = codec.encode_and_quantize(&input).unwrap();

        assert_eq!(quantized.dims3().unwrap(), (1, 5, 1024));
        assert_eq!(codes.dims2().unwrap(), (1, 5));
    }

    #[test]
    fn test_encoder_field_is_none_by_default() {
        let device = Device::Cpu;
        let codec = SemanticCodec::new(&device).unwrap();
        assert!(codec.encoder.is_none());
    }
}
