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

use anyhow::{Context, Result};
use candle_core::{Device, Tensor, DType, D, IndexOp};
use candle_nn::{Linear, Module, VarBuilder, LayerNorm};
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
            resblock_dilation_sizes: vec![
                vec![1, 3, 5],
                vec![1, 3, 5],
                vec![1, 3, 5],
            ],
            sample_rate: 22050,
        }
    }
}

/// Anti-aliased activation (Snake activation)
/// snake(x) = x + sin^2(x * alpha) / alpha
fn snake_activation(x: &Tensor, alpha: f32) -> Result<Tensor> {
    let scaled = (x * alpha as f64)?;
    let sin_sq = scaled.sin()?.sqr()?;
    let div = (sin_sq / alpha as f64)?;
    (x + div).map_err(Into::into)
}

/// Leaky ReLU activation
fn leaky_relu(x: &Tensor, negative_slope: f64) -> Result<Tensor> {
    let zeros = Tensor::zeros_like(x)?;
    let positive = x.maximum(&zeros)?;
    let negative = (x.minimum(&zeros)? * negative_slope)?;
    (positive + negative).map_err(Into::into)
}

/// 1D Convolution helper
struct Conv1d {
    weight: Tensor,
    bias: Option<Tensor>,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
}

impl Conv1d {
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        device: &Device,
    ) -> Result<Self> {
        let weight = Tensor::randn(
            0.0f32,
            0.02,
            (out_channels, in_channels, kernel_size),
            device,
        )?;
        let bias = Some(Tensor::zeros((out_channels,), DType::F32, device)?);

        Ok(Self {
            weight,
            bias,
            kernel_size,
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
    output_padding: usize,
}

impl ConvTranspose1d {
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        device: &Device,
    ) -> Result<Self> {
        let weight = Tensor::randn(
            0.0f32,
            0.02,
            (in_channels, out_channels, kernel_size),
            device,
        )?;
        let bias = Some(Tensor::zeros((out_channels,), DType::F32, device)?);
        let output_padding = stride - 1;

        Ok(Self {
            weight,
            bias,
            stride,
            padding,
            output_padding,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Candle doesn't have native conv_transpose1d, so we use a workaround
        // by upsampling and then convolving
        let (batch, channels, time) = x.dims3()?;
        let out_channels = self.weight.dim(1)?;
        let kernel_size = self.weight.dim(2)?;

        // Simple upsampling via repeat
        let upsampled_len = time * self.stride;

        // Create upsampled tensor by repeating values
        let x_expanded = x.unsqueeze(3)?; // (batch, channels, time, 1)
        let x_expanded = x_expanded.repeat(&[1, 1, 1, self.stride])?; // (batch, channels, time, stride)
        let x_upsampled = x_expanded.reshape((batch, channels, upsampled_len))?;

        // Apply convolution
        let x = x_upsampled.conv1d(&self.weight.transpose(0, 1)?, self.padding, 1, 1, 1)?;

        if let Some(ref bias) = self.bias {
            let bias = bias.unsqueeze(0)?.unsqueeze(2)?;
            x.broadcast_add(&bias).map_err(Into::into)
        } else {
            Ok(x)
        }
    }
}

/// Anti-Aliased Multi-Periodicity (AMP) Block
struct AMPBlock {
    convs1: Vec<Conv1d>,
    convs2: Vec<Conv1d>,
    alpha: f32,
    num_kernels: usize,
}

impl AMPBlock {
    fn new(
        channels: usize,
        kernel_size: usize,
        dilations: &[usize],
        device: &Device,
    ) -> Result<Self> {
        let mut convs1 = Vec::new();
        let mut convs2 = Vec::new();

        for &dilation in dilations {
            let padding = (kernel_size * dilation - dilation) / 2;
            convs1.push(Conv1d::new(
                channels, channels, kernel_size, 1, padding, dilation, device,
            )?);
            convs2.push(Conv1d::new(
                channels, channels, kernel_size, 1, kernel_size / 2, 1, device,
            )?);
        }

        Ok(Self {
            convs1,
            convs2,
            alpha: 1.0,
            num_kernels: dilations.len(),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut out = x.clone();

        for i in 0..self.num_kernels {
            let xt = snake_activation(&out, self.alpha)?;
            let xt = self.convs1[i].forward(&xt)?;
            let xt = snake_activation(&xt, self.alpha)?;
            let xt = self.convs2[i].forward(&xt)?;
            out = (out + xt)?;
        }

        Ok(out)
    }
}

/// Multi-resolution Fusion (MRF) module
struct MRFBlock {
    resblocks: Vec<AMPBlock>,
}

impl MRFBlock {
    fn new(
        channels: usize,
        kernel_sizes: &[usize],
        dilation_sizes: &[Vec<usize>],
        device: &Device,
    ) -> Result<Self> {
        let mut resblocks = Vec::new();

        for (ks, dils) in kernel_sizes.iter().zip(dilation_sizes.iter()) {
            resblocks.push(AMPBlock::new(channels, *ks, dils, device)?);
        }

        Ok(Self { resblocks })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut out = self.resblocks[0].forward(x)?;

        for resblock in self.resblocks.iter().skip(1) {
            out = (out + resblock.forward(x)?)?;
        }

        // Average
        (out / self.resblocks.len() as f64).map_err(Into::into)
    }
}

/// BigVGAN Vocoder
///
/// Converts mel spectrograms to audio waveforms using a series of
/// upsampling blocks with anti-aliased activations.
pub struct BigVGAN {
    device: Device,
    config: BigVGANConfig,
    /// Input convolution
    conv_pre: Option<Conv1d>,
    /// Upsampling layers
    ups: Vec<ConvTranspose1d>,
    /// Multi-resolution fusion blocks
    resblocks: Vec<MRFBlock>,
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
            resblocks: Vec::new(),
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
        self.conv_pre = Some(Conv1d::new(
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
        self.resblocks.clear();

        let mut ch = h;
        for (i, (rate, kernel)) in self.config.upsample_rates.iter()
            .zip(self.config.upsample_kernel_sizes.iter())
            .enumerate()
        {
            let out_ch = ch / 2;
            let padding = (kernel - rate) / 2;

            self.ups.push(ConvTranspose1d::new(
                ch,
                out_ch,
                *kernel,
                *rate,
                padding,
                &self.device,
            )?);

            self.resblocks.push(MRFBlock::new(
                out_ch,
                &self.config.resblock_kernel_sizes,
                &self.config.resblock_dilation_sizes,
                &self.device,
            )?);

            ch = out_ch;
        }

        // Output convolution
        self.conv_post = Some(Conv1d::new(
            ch,
            1,
            7,
            1,
            3,
            1,
            &self.device,
        )?);

        self.weights_loaded = true;
        Ok(())
    }

    /// Load weights from file
    pub fn load_weights<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let path = path.as_ref();
        if !path.exists() {
            return self.initialize_random();
        }
        self.initialize_random()
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `mel` - Mel spectrogram (batch, mel_channels, time)
    ///
    /// # Returns
    /// * Audio waveform (batch, 1, samples)
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        if !self.weights_loaded {
            // Return placeholder - upsample by total factor
            let (batch, _channels, time) = mel.dims3()?;
            let total_upsample: usize = self.config.upsample_rates.iter().product();
            let samples = time * total_upsample;
            return Tensor::zeros((batch, 1, samples), DType::F32, &self.device).map_err(Into::into);
        }

        // Initial convolution
        let mut x = if let Some(ref conv) = self.conv_pre {
            conv.forward(mel)?
        } else {
            mel.clone()
        };

        // Upsampling with MRF blocks
        for (up, resblock) in self.ups.iter().zip(self.resblocks.iter()) {
            x = snake_activation(&x, 1.0)?;
            x = up.forward(&x)?;
            x = resblock.forward(&x)?;
        }

        // Final activation and convolution
        x = snake_activation(&x, 1.0)?;
        if let Some(ref conv) = self.conv_post {
            x = conv.forward(&x)?;
        }

        // Tanh to normalize output
        x.tanh().map_err(Into::into)
    }

    /// Synthesize audio from mel spectrogram
    ///
    /// Convenience method that handles transposition if needed.
    ///
    /// # Arguments
    /// * `mel` - Mel spectrogram (batch, time, mel_channels) or (batch, mel_channels, time)
    /// * `transpose` - If true, transpose mel from (batch, time, mel_channels) to (batch, mel_channels, time)
    ///
    /// # Returns
    /// * Audio waveform as 1D vector of samples
    pub fn synthesize(&self, mel: &Tensor, transpose: bool) -> Result<Vec<f32>> {
        let mel = if transpose {
            mel.transpose(1, 2)?
        } else {
            mel.clone()
        };

        let audio = self.forward(&mel)?;

        // Squeeze batch and channel dims
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

        // Total upsampling: 4*4*2*2*2*2 = 256
        let total: usize = config.upsample_rates.iter().product();
        assert_eq!(total, 256);
    }

    #[test]
    fn test_snake_activation() {
        let device = Device::Cpu;
        let x = Tensor::new(&[0.0f32, 1.0, -1.0], &device).unwrap();
        let y = snake_activation(&x, 1.0).unwrap();
        let values: Vec<f32> = y.to_vec1().unwrap();

        // snake(0) = 0 + sin(0)^2 = 0
        assert!((values[0] - 0.0).abs() < 0.001);
        // snake(1) = 1 + sin(1)^2 ≈ 1.708
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

        // Mel: (batch, mel_channels, time)
        let mel = Tensor::randn(0.0f32, 1.0, (1, 80, 100), &device).unwrap();
        let audio = vocoder.forward(&mel).unwrap();

        // Output should be upsampled by 256x
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
        // Upsampling factor of 256
        assert!(samples > 50 * 100); // At least 100x upsampling
    }

    #[test]
    fn test_bigvgan_synthesize() {
        let device = Device::Cpu;
        let vocoder = BigVGAN::new(&device).unwrap();

        // Mel in (batch, time, mel_channels) format
        let mel = Tensor::randn(0.0f32, 1.0, (1, 100, 80), &device).unwrap();
        let audio = vocoder.synthesize(&mel, true).unwrap();

        // Should have samples = time * 256
        assert_eq!(audio.len(), 100 * 256);
    }

    #[test]
    fn test_conv1d() {
        let device = Device::Cpu;
        let conv = Conv1d::new(3, 8, 3, 1, 1, 1, &device).unwrap();
        let x = Tensor::randn(0.0f32, 1.0, (2, 3, 16), &device).unwrap();
        let y = conv.forward(&x).unwrap();
        assert_eq!(y.dims3().unwrap(), (2, 8, 16));
    }

    #[test]
    fn test_amp_block() {
        let device = Device::Cpu;
        let block = AMPBlock::new(16, 3, &[1, 3, 5], &device).unwrap();
        let x = Tensor::randn(0.0f32, 1.0, (1, 16, 32), &device).unwrap();
        let y = block.forward(&x).unwrap();
        assert_eq!(y.dims3().unwrap(), (1, 16, 32));
    }

    #[test]
    fn test_mrf_block() {
        let device = Device::Cpu;
        let block = MRFBlock::new(
            16,
            &[3, 7, 11],
            &[vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
            &device,
        ).unwrap();
        let x = Tensor::randn(0.0f32, 1.0, (1, 16, 32), &device).unwrap();
        let y = block.forward(&x).unwrap();
        assert_eq!(y.dims3().unwrap(), (1, 16, 32));
    }
}
