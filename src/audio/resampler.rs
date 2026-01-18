//! Audio resampling using rubato

use anyhow::Result;
use rubato::{Resampler as RubatoResampler, SincFixedIn, SincInterpolationType, SincInterpolationParameters, WindowFunction};

/// Audio resampler
pub struct Resampler;

impl Resampler {
    /// Resample audio from one sample rate to another
    pub fn resample(samples: &[f32], from_sr: u32, to_sr: u32) -> Result<Vec<f32>> {
        if from_sr == to_sr {
            return Ok(samples.to_vec());
        }
        
        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 256,
            window: WindowFunction::BlackmanHarris2,
        };
        
        let mut resampler = SincFixedIn::<f32>::new(
            to_sr as f64 / from_sr as f64,
            2.0,
            params,
            samples.len(),
            1,
        )?;
        
        let input = vec![samples.to_vec()];
        let output = resampler.process(&input, None)?;
        
        Ok(output.into_iter().next().unwrap_or_default())
    }
}
