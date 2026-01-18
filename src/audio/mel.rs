//! Mel spectrogram computation

use anyhow::Result;
use rustfft::{FftPlanner, num_complex::Complex};
use std::f32::consts::PI;

/// Mel spectrogram computer
pub struct MelSpectrogram {
    /// FFT size
    pub n_fft: usize,
    /// Hop length
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
    /// Mel filterbank
    mel_filters: Vec<Vec<f32>>,
    /// Hann window
    window: Vec<f32>,
}

impl MelSpectrogram {
    /// Create a new mel spectrogram computer
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
        }
    }
    
    /// Compute mel spectrogram from audio samples
    pub fn compute(&self, audio: &[f32]) -> Result<Vec<Vec<f32>>> {
        let stft = self.stft(audio)?;
        let power_spec = self.power_spectrum(&stft);
        let mel_spec = self.apply_mel_filters(&power_spec);
        let log_mel = self.log_compress(&mel_spec);
        Ok(log_mel)
    }
    
    /// Short-time Fourier transform
    fn stft(&self, audio: &[f32]) -> Result<Vec<Vec<Complex<f32>>>> {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(self.n_fft);
        
        let num_frames = (audio.len().saturating_sub(self.n_fft)) / self.hop_length + 1;
        let mut stft_frames = Vec::with_capacity(num_frames);
        
        for i in 0..num_frames {
            let start = i * self.hop_length;
            let mut frame: Vec<Complex<f32>> = (0..self.n_fft)
                .map(|j| {
                    let sample = if start + j < audio.len() {
                        audio[start + j]
                    } else {
                        0.0
                    };
                    let window_val = if j < self.win_length {
                        self.window[j]
                    } else {
                        0.0
                    };
                    Complex::new(sample * window_val, 0.0)
                })
                .collect();
            
            fft.process(&mut frame);
            stft_frames.push(frame[..self.n_fft / 2 + 1].to_vec());
        }
        
        Ok(stft_frames)
    }
    
    /// Compute power spectrum from STFT
    fn power_spectrum(&self, stft: &[Vec<Complex<f32>>]) -> Vec<Vec<f32>> {
        stft.iter()
            .map(|frame| frame.iter().map(|c| c.norm_sqr()).collect())
            .collect()
    }
    
    /// Apply mel filterbank to power spectrum
    fn apply_mel_filters(&self, power_spec: &[Vec<f32>]) -> Vec<Vec<f32>> {
        power_spec.iter()
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
    
    /// Apply log compression
    fn log_compress(&self, mel_spec: &[Vec<f32>]) -> Vec<Vec<f32>> {
        mel_spec.iter()
            .map(|frame| {
                frame.iter()
                    .map(|v| (v.max(1e-10)).ln())
                    .collect()
            })
            .collect()
    }
    
    /// Create Hann window
    fn hann_window(size: usize) -> Vec<f32> {
        (0..size)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / size as f32).cos()))
            .collect()
    }
    
    /// Hz to Mel conversion
    fn hz_to_mel(hz: f32) -> f32 {
        2595.0 * (1.0 + hz / 700.0).log10()
    }
    
    /// Mel to Hz conversion
    fn mel_to_hz(mel: f32) -> f32 {
        700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
    }
    
    /// Create mel filterbank
    fn mel_filterbank(n_fft: usize, n_mels: usize, sr: u32, fmin: f32, fmax: f32) -> Vec<Vec<f32>> {
        let n_freqs = n_fft / 2 + 1;
        let freq_bins: Vec<f32> = (0..n_freqs)
            .map(|i| i as f32 * sr as f32 / n_fft as f32)
            .collect();
        
        let mel_min = Self::hz_to_mel(fmin);
        let mel_max = Self::hz_to_mel(fmax);
        let mel_points: Vec<f32> = (0..n_mels + 2)
            .map(|i| Self::mel_to_hz(mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32))
            .collect();
        
        let mut filters = vec![vec![0.0; n_freqs]; n_mels];
        
        for i in 0..n_mels {
            let left = mel_points[i];
            let center = mel_points[i + 1];
            let right = mel_points[i + 2];
            
            for (j, &freq) in freq_bins.iter().enumerate() {
                if freq >= left && freq <= center {
                    filters[i][j] = (freq - left) / (center - left);
                } else if freq > center && freq <= right {
                    filters[i][j] = (right - freq) / (right - center);
                }
            }
        }
        
        filters
    }
}
