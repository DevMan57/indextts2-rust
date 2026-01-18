//! Audio resampling using rubato
//!
//! Provides high-quality sample rate conversion for audio processing.
//! Uses sinc interpolation for best quality.

use anyhow::{Context, Result};
use rubato::{
    Resampler as RubatoResampler,
    SincFixedIn,
    SincInterpolationType,
    SincInterpolationParameters,
    WindowFunction,
    calculate_cutoff,
};

/// Default chunk size for processing
const CHUNK_SIZE: usize = 1024;

/// Audio resampler using sinc interpolation
pub struct Resampler;

impl Resampler {
    /// Resample audio from one sample rate to another
    ///
    /// # Arguments
    /// * `samples` - Input audio samples (mono, f32)
    /// * `from_sr` - Source sample rate
    /// * `to_sr` - Target sample rate
    ///
    /// # Returns
    /// Resampled audio samples
    pub fn resample(samples: &[f32], from_sr: u32, to_sr: u32) -> Result<Vec<f32>> {
        if from_sr == to_sr {
            return Ok(samples.to_vec());
        }

        if samples.is_empty() {
            return Ok(vec![]);
        }

        // For short audio, use simple single-pass resampling
        if samples.len() <= CHUNK_SIZE * 2 {
            return Self::resample_simple(samples, from_sr, to_sr);
        }

        // For longer audio, use chunked processing
        Self::resample_chunked(samples, from_sr, to_sr)
    }

    /// Simple single-pass resampling for short audio
    fn resample_simple(samples: &[f32], from_sr: u32, to_sr: u32) -> Result<Vec<f32>> {
        let sinc_len = 256;
        let window = WindowFunction::BlackmanHarris2;
        let f_cutoff = calculate_cutoff(sinc_len, window);

        let params = SincInterpolationParameters {
            sinc_len,
            f_cutoff,
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 256,
            window,
        };

        let mut resampler = SincFixedIn::<f32>::new(
            to_sr as f64 / from_sr as f64,
            2.0,
            params,
            samples.len(),
            1, // mono
        ).context("Failed to create resampler")?;

        let input = vec![samples.to_vec()];
        let output = resampler.process(&input, None)
            .context("Resampling failed")?;

        Ok(output.into_iter().next().unwrap_or_default())
    }

    /// Chunked resampling for longer audio (more memory efficient)
    fn resample_chunked(samples: &[f32], from_sr: u32, to_sr: u32) -> Result<Vec<f32>> {
        let sinc_len = 128; // Slightly smaller for chunked processing
        let window = WindowFunction::Blackman2;
        let f_cutoff = calculate_cutoff(sinc_len, window);

        let params = SincInterpolationParameters {
            sinc_len,
            f_cutoff,
            interpolation: SincInterpolationType::Quadratic,
            oversampling_factor: 256,
            window,
        };

        let mut resampler = SincFixedIn::<f32>::new(
            to_sr as f64 / from_sr as f64,
            1.1, // Small headroom for ratio adjustments
            params,
            CHUNK_SIZE,
            1, // mono
        ).context("Failed to create chunked resampler")?;

        // Estimate output size
        let ratio = to_sr as f64 / from_sr as f64;
        let estimated_output_len = (samples.len() as f64 * ratio * 1.1) as usize;
        let mut output_samples = Vec::with_capacity(estimated_output_len);

        // Process in chunks
        let mut pos = 0;
        while pos + CHUNK_SIZE <= samples.len() {
            let chunk = &samples[pos..pos + CHUNK_SIZE];
            let input = vec![chunk.to_vec()];
            let output = resampler.process(&input, None)?;

            if let Some(out_chunk) = output.into_iter().next() {
                output_samples.extend(out_chunk);
            }

            pos += CHUNK_SIZE;
        }

        // Process remaining samples
        if pos < samples.len() {
            let remaining = &samples[pos..];
            let input = vec![remaining.to_vec()];
            let output = resampler.process_partial(Some(&input.iter().map(|v| v.as_slice()).collect::<Vec<_>>()), None)?;

            if let Some(out_chunk) = output.into_iter().next() {
                output_samples.extend(out_chunk);
            }
        }

        Ok(output_samples)
    }

    /// Get the delay introduced by the resampler (in output samples)
    pub fn get_delay(from_sr: u32, to_sr: u32, sinc_len: usize) -> usize {
        let ratio = to_sr as f64 / from_sr as f64;
        ((sinc_len as f64 / 2.0) * ratio) as usize
    }

    /// Resample to 16kHz (common for speech models like Wav2Vec)
    pub fn resample_to_16k(samples: &[f32], from_sr: u32) -> Result<Vec<f32>> {
        Self::resample(samples, from_sr, 16000)
    }

    /// Resample to 22050Hz (common for vocoders like BigVGAN)
    pub fn resample_to_22k(samples: &[f32], from_sr: u32) -> Result<Vec<f32>> {
        Self::resample(samples, from_sr, 22050)
    }

    /// Resample to 24kHz (common for some TTS systems)
    pub fn resample_to_24k(samples: &[f32], from_sr: u32) -> Result<Vec<f32>> {
        Self::resample(samples, from_sr, 24000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_same_rate_no_change() {
        let samples: Vec<f32> = (0..100).map(|i| (i as f32 * 0.01).sin()).collect();
        let result = Resampler::resample(&samples, 44100, 44100).unwrap();
        assert_eq!(result, samples);
    }

    #[test]
    fn test_empty_input() {
        let result = Resampler::resample(&[], 44100, 22050).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_downsample() {
        // Create a simple sine wave at 44100 Hz
        let samples: Vec<f32> = (0..4410)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();

        let result = Resampler::resample(&samples, 44100, 22050).unwrap();

        // Output should be approximately half the length
        assert!(result.len() > samples.len() / 3);
        assert!(result.len() < samples.len());
    }

    #[test]
    fn test_upsample() {
        let samples: Vec<f32> = (0..1000)
            .map(|i| (i as f32 * 0.01).sin())
            .collect();

        let result = Resampler::resample(&samples, 22050, 44100).unwrap();

        // Output should be approximately double the length
        assert!(result.len() > samples.len());
        assert!(result.len() < samples.len() * 3);
    }
}
