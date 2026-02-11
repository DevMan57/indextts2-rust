//! BigVGAN v2 Vocoder
//!
//! Neural vocoder for converting mel spectrograms to audio waveforms.
//! Based on BigVGAN v2 22kHz 80-band configuration.
//!
//! Architecture:
//! - Input mel spectrogram: (batch, mel_channels, time)
//! - Upsampling blocks with anti-aliased activations
//! - Multi-resolution fusion
//! - Output waveform: (batch, 1, samples)

use super::weights::load_bigvgan_weights;
use anyhow::Result;
use candle_core::{Device, DType, Tensor};
use std::collections::HashMap;
use std::path::Path;

/// BigVGAN configuration
#[derive(Clone)]
pub struct BigVGANConfig {
    /// Number of mel channels (input)
    pub num_mels: usize,
    /// Initial hidden channels
    pub upsample_initial_channel: usize,
    /// Upsampling rates
    pub upsample_rates: Vec<usize>,
    /// Upsampling kernel sizes
    pub upsample_kernel_sizes: Vec<usize>,
    /// ResBlock kernel sizes
    pub resblock_kernel_sizes: Vec<usize>,
    /// ResBlock dilation sizes
    pub resblock_dilation_sizes: Vec<Vec<usize>>,
    /// Sample rate
    pub sample_rate: usize,
}

impl Default for BigVGANConfig {
    fn default() -> Self {
        // BigVGAN v2 22kHz 80-band 256x configuration
        Self {
            num_mels: 80,
            upsample_initial_channel: 1536,
            upsample_rates: vec![4, 4, 2, 2, 2, 2],
            upsample_kernel_sizes: vec![8, 8, 4, 4, 4, 4],
            resblock_kernel_sizes: vec![3, 7, 11],
            resblock_dilation_sizes: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
            sample_rate: 22050,
        }
    }
}

/// Snake-Beta activation: x + (1/beta) * sin^2(x * alpha)
///
/// When alpha_logscale=True (which is the case for BigVGAN v2):
/// - Both alpha and beta are stored in log scale (initialized to zeros)
/// - Forward pass: alpha_exp = exp(alpha), beta_exp = exp(beta)
/// - Formula: x + (1/beta_exp) * sin^2(x * alpha_exp)
fn snake_beta_activation(x: &Tensor, alpha: &Tensor, beta: &Tensor) -> Result<Tensor> {
    // alpha and beta have shape [channels], need to broadcast to [1, channels, 1]
    let alpha = alpha.unsqueeze(0)?.unsqueeze(2)?;
    let beta = beta.unsqueeze(0)?.unsqueeze(2)?;

    // BigVGAN v2 uses alpha_logscale=True, so both alpha AND beta must be exponentiated
    // This is critical for correct activation behavior!
    let alpha_exp = alpha.exp()?;
    let beta_exp = beta.exp()?;
    let beta_safe = beta_exp.clamp(1e-9, 1e9)?;  // Use 1e-9 to match Python epsilon

    let scaled = x.broadcast_mul(&alpha_exp)?;
    let sin_sq = scaled.sin()?.sqr()?;
    let term = sin_sq.broadcast_div(&beta_safe)?;
    (x + term).map_err(Into::into)
}

/// Snake activation with scalar alpha
fn snake_activation_scalar(x: &Tensor, alpha: f32) -> Result<Tensor> {
    let scaled = (x * alpha as f64)?;
    let sin_sq = scaled.sin()?.sqr()?;
    let div = (sin_sq / alpha as f64)?;
    (x + div).map_err(Into::into)
}

/// 1D Convolution with loaded weights
struct Conv1d {
    weight: Tensor,
    bias: Option<Tensor>,
    stride: usize,
    padding: usize,
    dilation: usize,
}

impl Conv1d {
    fn from_weights(
        weights: &HashMap<String, Tensor>,
        prefix: &str,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> Result<Self> {
        let weight = weights
            .get(&format!("{}.weight", prefix))
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Missing weight: {}.weight", prefix))?;
        let bias = weights.get(&format!("{}.bias", prefix)).cloned();

        Ok(Self {
            weight,
            bias,
            stride,
            padding,
            dilation,
        })
    }

    fn new_random(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        device: &Device,
    ) -> Result<Self> {
        let weight =
            Tensor::randn(0.0f32, 0.02, (out_channels, in_channels, kernel_size), device)?;
        let bias = Some(Tensor::zeros((out_channels,), DType::F32, device)?);

        Ok(Self {
            weight,
            bias,
            stride,
            padding,
            dilation,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.conv1d(&self.weight, self.padding, self.stride, self.dilation, 1)?;
        if let Some(ref bias) = self.bias {
            let bias = bias.unsqueeze(0)?.unsqueeze(2)?;
            x.broadcast_add(&bias).map_err(Into::into)
        } else {
            Ok(x)
        }
    }
}

/// Transposed 1D Convolution for upsampling
struct ConvTranspose1d {
    weight: Tensor,
    bias: Option<Tensor>,
    stride: usize,
    padding: usize,
}

impl ConvTranspose1d {
    fn from_weights(
        weights: &HashMap<String, Tensor>,
        prefix: &str,
        stride: usize,
        padding: usize,
    ) -> Result<Self> {
        let weight = weights
            .get(&format!("{}.weight", prefix))
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Missing weight: {}.weight", prefix))?;
        let bias = weights.get(&format!("{}.bias", prefix)).cloned();

        Ok(Self {
            weight,
            bias,
            stride,
            padding,
        })
    }

    fn new_random(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        device: &Device,
    ) -> Result<Self> {
        let weight =
            Tensor::randn(0.0f32, 0.02, (in_channels, out_channels, kernel_size), device)?;
        let bias = Some(Tensor::zeros((out_channels,), DType::F32, device)?);

        Ok(Self {
            weight,
            bias,
            stride,
            padding,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // True transposed convolution (matches PyTorch ConvTranspose1d).
        // Candle signature: conv_transpose1d(kernel, padding, output_padding, stride, dilation, groups)
        let x = x.conv_transpose1d(&self.weight, self.padding, 0, self.stride, 1, 1)?;

        if let Some(ref bias) = self.bias {
            let bias = bias.unsqueeze(0)?.unsqueeze(2)?;
            x.broadcast_add(&bias).map_err(Into::into)
        } else {
            Ok(x)
        }
    }
}

/// Activation module with Snake-Beta
struct Activation {
    alpha: Tensor,
    beta: Tensor,
}

impl Activation {
    fn from_weights(weights: &HashMap<String, Tensor>, prefix: &str) -> Result<Self> {
        let alpha = weights
            .get(&format!("{}.act.alpha", prefix))
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Missing {}.act.alpha", prefix))?;
        let beta = weights
            .get(&format!("{}.act.beta", prefix))
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Missing {}.act.beta", prefix))?;

        Ok(Self { alpha, beta })
    }

    fn new_random(channels: usize, device: &Device) -> Result<Self> {
        let alpha = Tensor::ones((channels,), DType::F32, device)?;
        let beta = Tensor::ones((channels,), DType::F32, device)?;
        Ok(Self { alpha, beta })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        snake_beta_activation(x, &self.alpha, &self.beta)
    }
}

/// Anti-Aliased Multi-Periodicity (AMP) Block - a single resblock
struct AMPBlock {
    convs1: Vec<Conv1d>,
    convs2: Vec<Conv1d>,
    activations: Vec<Activation>,
}

impl AMPBlock {
    fn from_weights(
        weights: &HashMap<String, Tensor>,
        prefix: &str,
        kernel_size: usize,
        dilations: &[usize],
    ) -> Result<Self> {
        let mut convs1 = Vec::new();
        let mut convs2 = Vec::new();
        let mut activations = Vec::new();

        for (i, &dilation) in dilations.iter().enumerate() {
            let padding = (kernel_size * dilation - dilation) / 2;

            // Load convs1
            convs1.push(Conv1d::from_weights(
                weights,
                &format!("{}.convs1.{}", prefix, i),
                1,
                padding,
                dilation,
            )?);

            // Load convs2
            convs2.push(Conv1d::from_weights(
                weights,
                &format!("{}.convs2.{}", prefix, i),
                1,
                kernel_size / 2,
                1,
            )?);

            // Load activations (2 per convolution pair)
            activations.push(Activation::from_weights(
                weights,
                &format!("{}.activations.{}", prefix, i * 2),
            )?);
            activations.push(Activation::from_weights(
                weights,
                &format!("{}.activations.{}", prefix, i * 2 + 1),
            )?);
        }

        Ok(Self {
            convs1,
            convs2,
            activations,
        })
    }

    fn new_random(channels: usize, kernel_size: usize, dilations: &[usize], device: &Device) -> Result<Self> {
        let mut convs1 = Vec::new();
        let mut convs2 = Vec::new();
        let mut activations = Vec::new();

        for &dilation in dilations {
            let padding = (kernel_size * dilation - dilation) / 2;
            convs1.push(Conv1d::new_random(
                channels, channels, kernel_size, 1, padding, dilation, device,
            )?);
            convs2.push(Conv1d::new_random(
                channels,
                channels,
                kernel_size,
                1,
                kernel_size / 2,
                1,
                device,
            )?);
            activations.push(Activation::new_random(channels, device)?);
            activations.push(Activation::new_random(channels, device)?);
        }

        Ok(Self {
            convs1,
            convs2,
            activations,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut out = x.clone();

        for i in 0..self.convs1.len() {
            let xt = self.activations[i * 2].forward(&out)?;
            let xt = self.convs1[i].forward(&xt)?;
            let xt = self.activations[i * 2 + 1].forward(&xt)?;
            let xt = self.convs2[i].forward(&xt)?;
            out = (out + xt)?;
        }

        Ok(out)
    }
}

/// Multi-Resolution Fusion Block
/// Contains 3 AMPBlocks with different kernel sizes (3, 7, 11) that are averaged
struct MRFBlock {
    resblocks: Vec<AMPBlock>,
}

impl MRFBlock {
    fn from_weights(
        weights: &HashMap<String, Tensor>,
        start_index: usize,
        kernel_sizes: &[usize],
        dilations: &[usize],
    ) -> Result<Self> {
        let mut resblocks = Vec::new();

        for (i, &kernel_size) in kernel_sizes.iter().enumerate() {
            let block_idx = start_index + i;
            resblocks.push(AMPBlock::from_weights(
                weights,
                &format!("resblocks.{}", block_idx),
                kernel_size,
                dilations,
            )?);
        }

        Ok(Self { resblocks })
    }

    fn new_random(
        channels: usize,
        kernel_sizes: &[usize],
        dilations: &[usize],
        device: &Device,
    ) -> Result<Self> {
        let mut resblocks = Vec::new();
        for &kernel_size in kernel_sizes {
            resblocks.push(AMPBlock::new_random(channels, kernel_size, dilations, device)?);
        }
        Ok(Self { resblocks })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if self.resblocks.is_empty() {
            return Ok(x.clone());
        }

        let mut out = self.resblocks[0].forward(x)?;
        for block in self.resblocks.iter().skip(1) {
            out = (out + block.forward(x)?)?;
        }

        // Average
        (out / self.resblocks.len() as f64).map_err(Into::into)
    }
}

/// BigVGAN Vocoder
pub struct BigVGAN {
    device: Device,
    config: BigVGANConfig,
    /// Input convolution
    conv_pre: Option<Conv1d>,
    /// Upsampling layers
    ups: Vec<ConvTranspose1d>,
    /// MRF blocks (one per upsample layer, each containing 3 resblocks)
    mrf_blocks: Vec<MRFBlock>,
    /// Post-activation
    activation_post: Option<Activation>,
    /// Output convolution
    conv_post: Option<Conv1d>,
    /// Whether initialized
    weights_loaded: bool,
}

impl BigVGAN {
    /// Create with default config
    pub fn new(device: &Device) -> Result<Self> {
        Self::with_config(BigVGANConfig::default(), device)
    }

    /// Create with custom config
    pub fn with_config(config: BigVGANConfig, device: &Device) -> Result<Self> {
        Ok(Self {
            device: device.clone(),
            config,
            conv_pre: None,
            ups: Vec::new(),
            mrf_blocks: Vec::new(),
            activation_post: None,
            conv_post: None,
            weights_loaded: false,
        })
    }

    /// Load from checkpoint
    pub fn load<P: AsRef<Path>>(path: P, device: &Device) -> Result<Self> {
        let mut vocoder = Self::new(device)?;
        vocoder.load_weights(path)?;
        Ok(vocoder)
    }

    /// Initialize with random weights
    pub fn initialize_random(&mut self) -> Result<()> {
        let h = self.config.upsample_initial_channel;

        // Input convolution
        self.conv_pre = Some(Conv1d::new_random(
            self.config.num_mels,
            h,
            7,
            1,
            3,
            1,
            &self.device,
        )?);

        // Upsampling blocks
        self.ups.clear();
        self.mrf_blocks.clear();

        let mut ch = h;
        for (rate, kernel) in self
            .config
            .upsample_rates
            .iter()
            .zip(self.config.upsample_kernel_sizes.iter())
        {
            let out_ch = ch / 2;
            let padding = (kernel - rate) / 2;

            self.ups.push(ConvTranspose1d::new_random(
                ch,
                out_ch,
                *kernel,
                *rate,
                padding,
                &self.device,
            )?);

            // MRF block with 3 resblocks at different kernel sizes
            self.mrf_blocks.push(MRFBlock::new_random(
                out_ch,
                &self.config.resblock_kernel_sizes,
                &self.config.resblock_dilation_sizes[0],
                &self.device,
            )?);

            ch = out_ch;
        }

        // Post activation
        self.activation_post = Some(Activation::new_random(ch, &self.device)?);

        // Output convolution
        self.conv_post = Some(Conv1d::new_random(ch, 1, 7, 1, 3, 1, &self.device)?);

        self.weights_loaded = true;
        Ok(())
    }

    /// Load weights from file
    pub fn load_weights<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let path = path.as_ref();
        if !path.exists() {
            tracing::warn!("BigVGAN weights not found at {:?}, using random", path);
            return self.initialize_random();
        }

        let weights = load_bigvgan_weights(path, &self.device)?;

        // Load conv_pre
        self.conv_pre = Some(Conv1d::from_weights(&weights, "conv_pre", 1, 3, 1)?);

        // Load upsampling layers and MRF blocks
        self.ups.clear();
        self.mrf_blocks.clear();

        let num_ups = self.config.upsample_rates.len();
        let num_resblocks_per_mrf = self.config.resblock_kernel_sizes.len();

        for i in 0..num_ups {
            let rate = self.config.upsample_rates[i];
            let kernel = self.config.upsample_kernel_sizes[i];
            let padding = (kernel - rate) / 2;

            self.ups.push(ConvTranspose1d::from_weights(
                &weights,
                &format!("ups.{}.0", i),
                rate,
                padding,
            )?);

            // Load MRF block for this layer
            // Each MRF block has 3 resblocks (indices: i*3, i*3+1, i*3+2)
            let start_idx = i * num_resblocks_per_mrf;
            self.mrf_blocks.push(MRFBlock::from_weights(
                &weights,
                start_idx,
                &self.config.resblock_kernel_sizes,
                &self.config.resblock_dilation_sizes[0],
            )?);
        }

        // Load post activation
        self.activation_post = Some(Activation::from_weights(&weights, "activation_post")?);

        // Load conv_post
        self.conv_post = Some(Conv1d::from_weights(&weights, "conv_post", 1, 3, 1)?);

        self.weights_loaded = true;
        tracing::info!("BigVGAN weights loaded successfully");
        Ok(())
    }

    /// Forward pass
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        if !self.weights_loaded {
            let (batch, _channels, time) = mel.dims3()?;
            let total_upsample: usize = self.config.upsample_rates.iter().product();
            let samples = time * total_upsample;
            return Tensor::zeros((batch, 1, samples), DType::F32, &self.device)
                .map_err(Into::into);
        }

        // Initial convolution
        let mut x = if let Some(ref conv) = self.conv_pre {
            conv.forward(mel)?
        } else {
            mel.clone()
        };
        eprintln!("DEBUG: BigVGAN after conv_pre: mean={:.4}, min={:.4}, max={:.4}",
            x.mean_all()?.to_scalar::<f32>()?,
            x.flatten_all()?.min(0)?.to_scalar::<f32>()?,
            x.flatten_all()?.max(0)?.to_scalar::<f32>()?);

        // Upsampling with MRF blocks
        // Upstream BigVGAN applies ConvTranspose directly (no extra fixed Snake here).
        for (i, (up, mrf)) in self.ups.iter().zip(self.mrf_blocks.iter()).enumerate() {
            x = up.forward(&x)?;
            eprintln!("DEBUG: BigVGAN {} after up: mean={:.4}, min={:.4}, max={:.4}",
                i,
                x.mean_all()?.to_scalar::<f32>()?,
                x.flatten_all()?.min(0)?.to_scalar::<f32>()?,
                x.flatten_all()?.max(0)?.to_scalar::<f32>()?);
            x = mrf.forward(&x)?;
            eprintln!("DEBUG: BigVGAN {} after mrf: mean={:.4}, min={:.4}, max={:.4}",
                i,
                x.mean_all()?.to_scalar::<f32>()?,
                x.flatten_all()?.min(0)?.to_scalar::<f32>()?,
                x.flatten_all()?.max(0)?.to_scalar::<f32>()?);
        }

        // Final activation
        if let Some(ref act) = self.activation_post {
            x = act.forward(&x)?;
        }

        // Output convolution
        if let Some(ref conv) = self.conv_post {
            x = conv.forward(&x)?;
        }

        // Debug: check pre-tanh values
        let pre_tanh_mean: f32 = x.mean_all()?.to_scalar()?;
        let pre_tanh_min: f32 = x.flatten_all()?.min(0)?.to_scalar()?;
        let pre_tanh_max: f32 = x.flatten_all()?.max(0)?.to_scalar()?;
        eprintln!("DEBUG: BigVGAN pre-tanh: mean={:.4}, min={:.4}, max={:.4}",
            pre_tanh_mean, pre_tanh_min, pre_tanh_max);

        // Tanh to normalize output
        x.tanh().map_err(Into::into)
    }

    /// Synthesize audio from mel spectrogram
    pub fn synthesize(&self, mel: &Tensor, transpose: bool) -> Result<Vec<f32>> {
        let mel = if transpose {
            mel.transpose(1, 2)?
        } else {
            mel.clone()
        };

        let audio = self.forward(&mel)?;
        let audio = audio.squeeze(0)?.squeeze(0)?;
        audio.to_vec1().map_err(Into::into)
    }

    /// Get sample rate
    pub fn sample_rate(&self) -> usize {
        self.config.sample_rate
    }

    /// Get total upsampling factor
    pub fn upsample_factor(&self) -> usize {
        self.config.upsample_rates.iter().product()
    }

    /// Check if initialized
    pub fn is_initialized(&self) -> bool {
        self.weights_loaded
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bigvgan_config_default() {
        let config = BigVGANConfig::default();
        assert_eq!(config.num_mels, 80);
        assert_eq!(config.sample_rate, 22050);
        assert_eq!(config.upsample_rates.len(), 6);

        let total: usize = config.upsample_rates.iter().product();
        assert_eq!(total, 256);
    }

    #[test]
    fn test_snake_activation_scalar() {
        let device = Device::Cpu;
        let x = Tensor::new(&[0.0f32, 1.0, -1.0], &device).unwrap();
        let y = snake_activation_scalar(&x, 1.0).unwrap();
        let values: Vec<f32> = y.to_vec1().unwrap();
        assert!((values[0] - 0.0).abs() < 0.001);
        assert!((values[1] - 1.708).abs() < 0.01);
    }

    #[test]
    fn test_bigvgan_new() {
        let device = Device::Cpu;
        let vocoder = BigVGAN::new(&device).unwrap();
        assert_eq!(vocoder.sample_rate(), 22050);
        assert_eq!(vocoder.upsample_factor(), 256);
    }

    #[test]
    fn test_bigvgan_placeholder() {
        let device = Device::Cpu;
        let vocoder = BigVGAN::new(&device).unwrap();

        let mel = Tensor::randn(0.0f32, 1.0, (1, 80, 100), &device).unwrap();
        let audio = vocoder.forward(&mel).unwrap();

        let (batch, channels, samples) = audio.dims3().unwrap();
        assert_eq!(batch, 1);
        assert_eq!(channels, 1);
        assert_eq!(samples, 100 * 256);
    }

    #[test]
    fn test_bigvgan_initialized() {
        let device = Device::Cpu;
        let mut vocoder = BigVGAN::new(&device).unwrap();
        vocoder.initialize_random().unwrap();

        assert!(vocoder.is_initialized());

        let mel = Tensor::randn(0.0f32, 1.0, (1, 80, 50), &device).unwrap();
        let audio = vocoder.forward(&mel).unwrap();

        let (batch, channels, samples) = audio.dims3().unwrap();
        assert_eq!(batch, 1);
        assert_eq!(channels, 1);
        assert!(samples > 50 * 100);
    }
}
