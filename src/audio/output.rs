//! Audio output and playback

use anyhow::Result;
use std::path::Path;

/// Audio output handler
pub struct AudioOutput;

impl AudioOutput {
    /// Save audio samples to a WAV file
    pub fn save<P: AsRef<Path>>(samples: &[f32], sample_rate: u32, path: P) -> Result<()> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        
        let mut writer = hound::WavWriter::create(path.as_ref(), spec)?;
        
        for &sample in samples {
            let scaled = (sample * 32767.0).clamp(-32767.0, 32767.0) as i16;
            writer.write_sample(scaled)?;
        }
        
        writer.finalize()?;
        Ok(())
    }
    
    /// Save int16 samples directly
    pub fn save_int16<P: AsRef<Path>>(samples: &[i16], sample_rate: u32, path: P) -> Result<()> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        
        let mut writer = hound::WavWriter::create(path.as_ref(), spec)?;
        
        for &sample in samples {
            writer.write_sample(sample)?;
        }
        
        writer.finalize()?;
        Ok(())
    }
    
    /// Play audio through the default output device
    pub fn play(_samples: &[f32], _sample_rate: u32) -> Result<()> {
        // TODO: Implement using cpal/rodio
        Err(anyhow::anyhow!("Audio playback not yet implemented"))
    }
}
