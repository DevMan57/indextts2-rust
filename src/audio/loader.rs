//! Audio file loading

use anyhow::{Context, Result};
use std::path::Path;

/// Audio loader that supports various formats
pub struct AudioLoader;

impl AudioLoader {
    /// Load audio from a file and return samples at the specified sample rate
    pub fn load<P: AsRef<Path>>(path: P, target_sr: u32) -> Result<(Vec<f32>, u32)> {
        let path = path.as_ref();
        
        // Use hound for WAV files
        if path.extension().map_or(false, |e| e == "wav") {
            return Self::load_wav(path, target_sr);
        }
        
        // TODO: Use symphonia for other formats
        Err(anyhow::anyhow!("Unsupported audio format: {:?}", path))
    }
    
    fn load_wav<P: AsRef<Path>>(path: P, target_sr: u32) -> Result<(Vec<f32>, u32)> {
        let reader = hound::WavReader::open(path.as_ref())
            .context("Failed to open WAV file")?;
        
        let spec = reader.spec();
        let sample_rate = spec.sample_rate;
        
        let samples: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Float => {
                reader.into_samples::<f32>()
                    .filter_map(Result::ok)
                    .collect()
            }
            hound::SampleFormat::Int => {
                let max_value = (1 << (spec.bits_per_sample - 1)) as f32;
                reader.into_samples::<i32>()
                    .filter_map(Result::ok)
                    .map(|s| s as f32 / max_value)
                    .collect()
            }
        };
        
        // Convert to mono if stereo
        let mono_samples = if spec.channels > 1 {
            samples
                .chunks(spec.channels as usize)
                .map(|chunk| chunk.iter().sum::<f32>() / chunk.len() as f32)
                .collect()
        } else {
            samples
        };
        
        // Resample if needed
        if sample_rate != target_sr {
            let resampled = super::Resampler::resample(&mono_samples, sample_rate, target_sr)?;
            Ok((resampled, target_sr))
        } else {
            Ok((mono_samples, sample_rate))
        }
    }
}
