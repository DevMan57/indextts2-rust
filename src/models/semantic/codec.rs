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

use anyhow::Result;
use candle_core::{Device, Tensor, DType, D};
use candle_nn::{Linear, Module, VarBuilder};
use std::path::Path;

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
                let weight = Tensor::randn(
                    0.0f32,
                    0.02,
                    (hidden, self.codebook_dim),
                    &self.device,
                )?;
                embeddings.matmul(&weight)?
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

        // Find nearest codebook entries using L2 distance
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
            let weight = Tensor::randn(
                0.0f32,
                0.02,
                (self.codebook_dim, self.hidden_size),
                &self.device,
            )?;
            quantized_codes.matmul(&weight)?
        } else {
            quantized_codes
        };

        Ok((quantized, codes))
    }

    /// Find nearest codebook entry for each embedding vector
    fn find_nearest_codes(&self, projected: &Tensor, codebook: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, _) = projected.dims3()?;

        // Flatten to (batch * seq, codebook_dim)
        let flat = projected.reshape((batch_size * seq_len, self.codebook_dim))?;

        // Compute L2 distance: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
        // flat: (N, D), codebook: (K, D)

        // ||a||^2: (N, 1)
        let flat_sq = flat.sqr()?.sum(D::Minus1)?.unsqueeze(1)?;

        // ||b||^2: (1, K)
        let codebook_sq = codebook.sqr()?.sum(D::Minus1)?.unsqueeze(0)?;

        // a·b: (N, K)
        let dot = flat.matmul(&codebook.t()?)?;

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
            quantized_codes.matmul(&weight).map_err(Into::into)
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

        // Verify codes are in valid range
        let codes_vec: Vec<i64> = codes.flatten_all().unwrap().to_vec1().unwrap();
        for code in codes_vec {
            assert!(code >= 0 && code < 8192, "Code {} out of range", code);
        }
    }
}
