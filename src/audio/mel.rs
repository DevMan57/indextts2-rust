//! Mel spectrogram computation
//!
//! Computes mel-frequency spectrograms from audio samples.
//! Compatible with librosa-style mel spectrograms used in TTS systems.

use anyhow::Result;
use rustfft::{FftPlanner, num_complex::Complex};
use std::f32::consts::PI;
use std::sync::Arc;

/// Mel spectrogram computer
///
/// Configured to match IndexTTS2's mel spectrogram parameters:
/// - 80 mel bands
/// - 1024 FFT size
/// - 256 hop length
/// - 22050 Hz sample rate
pub struct MelSpectrogram {
    /// FFT size
    pub n_fft: usize,
    /// Hop length between frames
    pub hop_length: usize,
    /// Window length
    pub win_length: usize,
    /// Number of mel bands
    pub n_mels: usize,
    /// Sample rate
    pub sample_rate: u32,
    /// Minimum frequency
    pub fmin: f32,
    /// Maximum frequency (None = Nyquist)
    pub fmax: Option<f32>,
    /// Mel filterbank (precomputed)
    mel_filters: Vec<Vec<f32>>,
    /// Hann window (precomputed)
    window: Vec<f32>,
    /// FFT planner (cached)
    fft: Arc<dyn rustfft::Fft<f32>>,
}

impl MelSpectrogram {
    /// Create a new mel spectrogram computer
    ///
    /// # Arguments
    /// * `n_fft` - FFT size (typically 1024)
    /// * `hop_length` - Hop between frames (typically 256)
    /// * `win_length` - Window length (typically same as n_fft)
    /// * `n_mels` - Number of mel bands (typically 80)
    /// * `sample_rate` - Audio sample rate
    /// * `fmin` - Minimum frequency for mel scale
    /// * `fmax` - Maximum frequency (None = Nyquist)
    pub fn new(
        n_fft: usize,
        hop_length: usize,
        win_length: usize,
        n_mels: usize,
        sample_rate: u32,
        fmin: f32,
        fmax: Option<f32>,
    ) -> Self {
        let window = Self::hann_window(win_length);
        let fmax = fmax.unwrap_or(sample_rate as f32 / 2.0);
        let mel_filters = Self::mel_filterbank(n_fft, n_mels, sample_rate, fmin, fmax);

        // Pre-plan FFT
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n_fft);

        Self {
            n_fft,
            hop_length,
            win_length,
            n_mels,
            sample_rate,
            fmin,
            fmax: Some(fmax),
            mel_filters,
            window,
            fft,
        }
    }

    /// Create with IndexTTS2 default parameters (80 mel bands, 22050 Hz)
    pub fn new_default() -> Self {
        Self::new(
            1024,   // n_fft
            256,    // hop_length
            1024,   // win_length
            80,     // n_mels
            22050,  // sample_rate
            0.0,    // fmin
            None,   // fmax (Nyquist)
        )
    }

    /// Create from config
    pub fn from_config(
        sample_rate: u32,
        n_fft: usize,
        hop_length: usize,
        win_length: usize,
        n_mels: usize,
        fmin: f32,
    ) -> Self {
        Self::new(n_fft, hop_length, win_length, n_mels, sample_rate, fmin, None)
    }

    /// Compute mel spectrogram from audio samples
    ///
    /// # Arguments
    /// * `audio` - Audio samples (mono, f32, normalized to [-1, 1])
    ///
    /// # Returns
    /// Mel spectrogram as `Vec<Vec<f32>>` where shape is `[n_frames, n_mels]`
    pub fn compute(&self, audio: &[f32]) -> Result<Vec<Vec<f32>>> {
        let stft = self.stft(audio)?;
        // Use magnitude (not power) to match Python: torch.sqrt(spec.pow(2).sum(-1) + 1e-9)
        let magnitude_spec = self.magnitude_spectrum(&stft);
        let mel_spec = self.apply_mel_filters(&magnitude_spec);
        let log_mel = self.log_compress(&mel_spec);
        Ok(log_mel)
    }

    /// Compute mel spectrogram and return as transposed [n_mels, n_frames]
    ///
    /// This format is often required by neural network models
    pub fn compute_transposed(&self, audio: &[f32]) -> Result<Vec<Vec<f32>>> {
        let mel = self.compute(audio)?;
        Ok(self.transpose(&mel))
    }

    /// Transpose mel spectrogram from [n_frames, n_mels] to [n_mels, n_frames]
    fn transpose(&self, mel: &[Vec<f32>]) -> Vec<Vec<f32>> {
        if mel.is_empty() {
            return vec![];
        }
        let n_frames = mel.len();
        let n_mels = mel[0].len();

        let mut transposed = vec![vec![0.0; n_frames]; n_mels];
        for (i, frame) in mel.iter().enumerate() {
            for (j, &val) in frame.iter().enumerate() {
                transposed[j][i] = val;
            }
        }
        transposed
    }

    /// Short-time Fourier transform
    fn stft(&self, audio: &[f32]) -> Result<Vec<Vec<Complex<f32>>>> {
        // Handle empty input
        if audio.is_empty() {
            return Ok(Vec::new());
        }

        // Pad audio to center windows
        let pad_len = self.n_fft / 2;
        let padded_len = audio.len() + 2 * pad_len;
        let mut padded = vec![0.0f32; padded_len];
        padded[pad_len..pad_len + audio.len()].copy_from_slice(audio);

        // Reflect padding at boundaries
        for i in 0..pad_len {
            padded[pad_len - 1 - i] = audio[i.min(audio.len() - 1)];
            padded[pad_len + audio.len() + i] = audio[(audio.len() - 1).saturating_sub(i)];
        }

        let num_frames = (padded.len().saturating_sub(self.n_fft)) / self.hop_length + 1;
        let mut stft_frames = Vec::with_capacity(num_frames);

        // Reusable buffer for FFT
        let mut frame_buffer: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); self.n_fft];

        for i in 0..num_frames {
            let start = i * self.hop_length;

            // Fill frame buffer with windowed samples
            for j in 0..self.n_fft {
                let sample = if start + j < padded.len() {
                    padded[start + j]
                } else {
                    0.0
                };
                let window_val = if j < self.win_length {
                    self.window[j]
                } else {
                    0.0
                };
                frame_buffer[j] = Complex::new(sample * window_val, 0.0);
            }

            // Perform FFT
            self.fft.process(&mut frame_buffer);

            // Keep only positive frequencies
            stft_frames.push(frame_buffer[..self.n_fft / 2 + 1].to_vec());
        }

        Ok(stft_frames)
    }

    /// Compute magnitude spectrum from STFT
    ///
    /// Uses magnitude (|z| = sqrt(re² + im²)) NOT power (|z|²)
    /// This matches Python: torch.sqrt(spec.pow(2).sum(-1) + 1e-9)
    fn magnitude_spectrum(&self, stft: &[Vec<Complex<f32>>]) -> Vec<Vec<f32>> {
        stft.iter()
            .map(|frame| frame.iter().map(|c| c.norm()).collect())
            .collect()
    }

    /// Apply mel filterbank to magnitude spectrum
    fn apply_mel_filters(&self, mag_spec: &[Vec<f32>]) -> Vec<Vec<f32>> {
        mag_spec.iter()
            .map(|frame| {
                self.mel_filters.iter()
                    .map(|filter| {
                        filter.iter()
                            .zip(frame.iter())
                            .map(|(f, p)| f * p)
                            .sum()
                    })
                    .collect()
            })
            .collect()
    }

    /// Apply log compression (natural log, matching Python dynamic_range_compression_torch)
    fn log_compress(&self, mel_spec: &[Vec<f32>]) -> Vec<Vec<f32>> {
        // Floor value matching Python: torch.clamp(x, min=1e-5)
        const LOG_FLOOR: f32 = 1e-5;

        mel_spec.iter()
            .map(|frame| {
                frame.iter()
                    .map(|v| v.max(LOG_FLOOR).ln())
                    .collect()
            })
            .collect()
    }

    /// Create Hann window (periodic version for STFT)
    fn hann_window(size: usize) -> Vec<f32> {
        (0..size)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / size as f32).cos()))
            .collect()
    }

    /// Hz to Mel conversion (HTK formula)
    fn hz_to_mel(hz: f32) -> f32 {
        2595.0 * (1.0 + hz / 700.0).log10()
    }

    /// Mel to Hz conversion (HTK formula)
    fn mel_to_hz(mel: f32) -> f32 {
        700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
    }

    /// Create mel filterbank (triangular filters)
    fn mel_filterbank(n_fft: usize, n_mels: usize, sr: u32, fmin: f32, fmax: f32) -> Vec<Vec<f32>> {
        let n_freqs = n_fft / 2 + 1;
        let freq_bins: Vec<f32> = (0..n_freqs)
            .map(|i| i as f32 * sr as f32 / n_fft as f32)
            .collect();

        let mel_min = Self::hz_to_mel(fmin);
        let mel_max = Self::hz_to_mel(fmax);

        // Create mel points evenly spaced in mel scale
        let mel_points: Vec<f32> = (0..=n_mels + 1)
            .map(|i| Self::mel_to_hz(mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32))
            .collect();

        let mut filters = vec![vec![0.0; n_freqs]; n_mels];

        for i in 0..n_mels {
            let left = mel_points[i];
            let center = mel_points[i + 1];
            let right = mel_points[i + 2];

            for (j, &freq) in freq_bins.iter().enumerate() {
                if freq >= left && freq < center {
                    // Rising edge
                    filters[i][j] = (freq - left) / (center - left);
                } else if freq >= center && freq <= right {
                    // Falling edge
                    filters[i][j] = (right - freq) / (right - center);
                }
            }

            // Normalize filter (slaney normalization)
            let sum: f32 = filters[i].iter().sum();
            if sum > 0.0 {
                for val in filters[i].iter_mut() {
                    *val /= sum;
                }
            }
        }

        filters
    }

    /// Get number of frames for given audio length
    pub fn get_num_frames(&self, audio_len: usize) -> usize {
        let padded_len = audio_len + self.n_fft;
        (padded_len.saturating_sub(self.n_fft)) / self.hop_length + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mel_spectrogram_creation() {
        let mel = MelSpectrogram::new_default();
        assert_eq!(mel.n_mels, 80);
        assert_eq!(mel.n_fft, 1024);
        assert_eq!(mel.hop_length, 256);
    }

    #[test]
    fn test_mel_computation() {
        let mel = MelSpectrogram::new_default();

        // Create a simple test signal (1 second of 440Hz sine wave)
        let samples: Vec<f32> = (0..22050)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 22050.0).sin())
            .collect();

        let result = mel.compute(&samples).unwrap();

        // Check output dimensions
        assert!(!result.is_empty());
        assert_eq!(result[0].len(), 80); // 80 mel bands
    }

    #[test]
    fn test_hz_to_mel_conversion() {
        // Test known values
        assert!((MelSpectrogram::hz_to_mel(0.0) - 0.0).abs() < 0.01);
        assert!((MelSpectrogram::hz_to_mel(1000.0) - 1000.0).abs() < 50.0); // Approximately linear near 1kHz
    }

    #[test]
    fn test_hann_window() {
        let window = MelSpectrogram::hann_window(1024);
        assert_eq!(window.len(), 1024);

        // Hann window should be 0 at edges and ~1 at center
        assert!(window[0].abs() < 0.01);
        assert!(window[512] > 0.99);
    }

    #[test]
    fn test_empty_input() {
        let mel = MelSpectrogram::new_default();
        let result = mel.compute(&[]).unwrap();
        // Should handle empty input gracefully
        assert!(result.len() <= 1);
    }

    #[test]
    fn test_mel_output_range() {
        let mel = MelSpectrogram::new_default();
        // Create a normalized sine wave
        let samples: Vec<f32> = (0..22050)
            .map(|i| 0.5 * (2.0 * PI * 440.0 * i as f32 / 22050.0).sin())
            .collect();

        let result = mel.compute(&samples).unwrap();

        // All values should be finite (not NaN or infinite)
        for frame in &result {
            for &val in frame {
                assert!(val.is_finite(), "Mel value should be finite");
            }
        }
    }

    #[test]
    fn test_mel_frame_count() {
        let mel = MelSpectrogram::new_default();
        // 1 second of audio at 22050 Hz with hop_length 256
        let samples: Vec<f32> = vec![0.0; 22050];
        let result = mel.compute(&samples).unwrap();

        // Expected frames = ceil((samples - n_fft) / hop_length) + 1
        // For 22050 samples, 1024 n_fft, 256 hop: (22050 - 1024) / 256 + 1 ≈ 83 frames
        assert!(result.len() > 80);
        assert!(result.len() < 100);
    }
}
