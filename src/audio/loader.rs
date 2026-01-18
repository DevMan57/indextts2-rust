//! Audio file loading
//!
//! Supports multiple audio formats via symphonia:
//! - WAV (PCM, float)
//! - MP3
//! - FLAC
//! - OGG/Vorbis

use anyhow::{Context, Result};
use std::path::Path;
use std::fs::File;

use symphonia::core::audio::{AudioBufferRef, Signal, SampleBuffer};
use symphonia::core::codecs::{CODEC_TYPE_NULL, DecoderOptions};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

/// Audio loader that supports various formats via symphonia
pub struct AudioLoader;

impl AudioLoader {
    /// Load audio from a file and return samples at the specified sample rate
    ///
    /// # Arguments
    /// * `path` - Path to the audio file
    /// * `target_sr` - Target sample rate (will resample if necessary)
    ///
    /// # Returns
    /// Tuple of (samples, sample_rate) where samples are mono f32 normalized to [-1, 1]
    pub fn load<P: AsRef<Path>>(path: P, target_sr: u32) -> Result<(Vec<f32>, u32)> {
        let path = path.as_ref();

        // Use hound for WAV files (faster and simpler)
        if path.extension().map_or(false, |e| e.eq_ignore_ascii_case("wav")) {
            return Self::load_wav(path, target_sr);
        }

        // Use symphonia for other formats
        Self::load_with_symphonia(path, target_sr)
    }

    /// Load audio using symphonia (supports MP3, FLAC, OGG, etc.)
    fn load_with_symphonia<P: AsRef<Path>>(path: P, target_sr: u32) -> Result<(Vec<f32>, u32)> {
        let path = path.as_ref();

        // Open the media source
        let src = File::open(path)
            .with_context(|| format!("Failed to open audio file: {:?}", path))?;

        // Create the media source stream
        let mss = MediaSourceStream::new(Box::new(src), Default::default());

        // Create a probe hint using the file's extension
        let mut hint = Hint::new();
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            hint.with_extension(ext);
        }

        // Use the default options for metadata and format readers
        let meta_opts: MetadataOptions = Default::default();
        let fmt_opts: FormatOptions = Default::default();

        // Probe the media source
        let probed = symphonia::default::get_probe()
            .format(&hint, mss, &fmt_opts, &meta_opts)
            .with_context(|| format!("Unsupported audio format: {:?}", path))?;

        // Get the instantiated format reader
        let mut format = probed.format;

        // Find the first audio track with a known codec
        let track = format
            .tracks()
            .iter()
            .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
            .ok_or_else(|| anyhow::anyhow!("No supported audio tracks found in {:?}", path))?;

        // Get track info
        let sample_rate = track.codec_params.sample_rate
            .ok_or_else(|| anyhow::anyhow!("Unknown sample rate"))?;
        let channels = track.codec_params.channels
            .map(|c| c.count())
            .unwrap_or(1);

        // Create a decoder for the track
        let dec_opts: DecoderOptions = Default::default();
        let mut decoder = symphonia::default::get_codecs()
            .make(&track.codec_params, &dec_opts)
            .with_context(|| "Unsupported codec")?;

        let track_id = track.id;

        // Decode all packets
        let mut all_samples: Vec<f32> = Vec::new();
        let mut sample_buf: Option<SampleBuffer<f32>> = None;

        loop {
            // Get the next packet
            let packet = match format.next_packet() {
                Ok(packet) => packet,
                Err(SymphoniaError::IoError(ref e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                    break; // End of stream
                }
                Err(SymphoniaError::ResetRequired) => {
                    // Track list changed, reinitialize if needed
                    break;
                }
                Err(e) => {
                    return Err(anyhow::anyhow!("Error reading packet: {}", e));
                }
            };

            // Skip packets from other tracks
            if packet.track_id() != track_id {
                continue;
            }

            // Decode the packet
            match decoder.decode(&packet) {
                Ok(decoded) => {
                    // Create sample buffer on first decode
                    if sample_buf.is_none() {
                        let spec = *decoded.spec();
                        let duration = decoded.capacity() as u64;
                        sample_buf = Some(SampleBuffer::new(duration, spec));
                    }

                    if let Some(ref mut buf) = sample_buf {
                        // Copy decoded samples to interleaved buffer
                        buf.copy_interleaved_ref(decoded);
                        all_samples.extend_from_slice(buf.samples());
                    }
                }
                Err(SymphoniaError::IoError(_)) | Err(SymphoniaError::DecodeError(_)) => {
                    // Skip corrupted packets
                    continue;
                }
                Err(e) => {
                    return Err(anyhow::anyhow!("Decode error: {}", e));
                }
            }
        }

        // Convert to mono if stereo
        let mono_samples = if channels > 1 {
            all_samples
                .chunks(channels)
                .map(|chunk| chunk.iter().sum::<f32>() / chunk.len() as f32)
                .collect()
        } else {
            all_samples
        };

        // Resample if needed
        if sample_rate != target_sr {
            let resampled = super::Resampler::resample(&mono_samples, sample_rate, target_sr)?;
            Ok((resampled, target_sr))
        } else {
            Ok((mono_samples, sample_rate))
        }
    }

    /// Load WAV files using hound (optimized for WAV)
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

    /// Load raw f32 samples from memory
    pub fn from_samples(samples: Vec<f32>, sample_rate: u32, target_sr: u32) -> Result<(Vec<f32>, u32)> {
        if sample_rate != target_sr {
            let resampled = super::Resampler::resample(&samples, sample_rate, target_sr)?;
            Ok((resampled, target_sr))
        } else {
            Ok((samples, sample_rate))
        }
    }

    /// Get audio file duration without fully decoding
    pub fn get_duration<P: AsRef<Path>>(path: P) -> Result<f64> {
        let path = path.as_ref();

        if path.extension().map_or(false, |e| e.eq_ignore_ascii_case("wav")) {
            let reader = hound::WavReader::open(path)?;
            let spec = reader.spec();
            let num_samples = reader.len() as f64;
            let duration = num_samples / spec.channels as f64 / spec.sample_rate as f64;
            return Ok(duration);
        }

        // For other formats, we'd need to probe with symphonia
        // This is a simplified implementation
        Err(anyhow::anyhow!("Duration detection not implemented for this format"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_loader_exists() {
        // Basic smoke test - AudioLoader struct exists
        let _loader = AudioLoader;
    }
}
