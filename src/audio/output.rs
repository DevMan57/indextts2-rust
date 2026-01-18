//! Audio output and playback
//!
//! Provides:
//! - WAV file saving
//! - Real-time audio playback via cpal
//! - Streaming audio output for TTS

use anyhow::{Context, Result};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::SampleFormat;

/// Audio output handler for saving and playback
pub struct AudioOutput;

impl AudioOutput {
    /// Save audio samples to a WAV file (16-bit PCM)
    ///
    /// # Arguments
    /// * `samples` - Audio samples (f32, normalized to [-1, 1])
    /// * `sample_rate` - Sample rate in Hz
    /// * `path` - Output file path
    pub fn save<P: AsRef<Path>>(samples: &[f32], sample_rate: u32, path: P) -> Result<()> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let mut writer = hound::WavWriter::create(path.as_ref(), spec)
            .with_context(|| format!("Failed to create WAV file: {:?}", path.as_ref()))?;

        for &sample in samples {
            let scaled = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
            writer.write_sample(scaled)?;
        }

        writer.finalize()?;
        Ok(())
    }

    /// Save audio samples to a WAV file (32-bit float)
    pub fn save_float<P: AsRef<Path>>(samples: &[f32], sample_rate: u32, path: P) -> Result<()> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };

        let mut writer = hound::WavWriter::create(path.as_ref(), spec)
            .with_context(|| format!("Failed to create WAV file: {:?}", path.as_ref()))?;

        for &sample in samples {
            writer.write_sample(sample)?;
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

    /// Play audio through the default output device (blocking)
    ///
    /// # Arguments
    /// * `samples` - Audio samples (f32, normalized to [-1, 1])
    /// * `sample_rate` - Sample rate in Hz
    pub fn play(samples: &[f32], sample_rate: u32) -> Result<()> {
        if samples.is_empty() {
            return Ok(());
        }

        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or_else(|| anyhow::anyhow!("No audio output device available"))?;

        // Get supported config
        let supported_configs = device
            .supported_output_configs()
            .context("Error querying audio configs")?;

        // Find a config that matches our sample rate, or use a close one
        let config = supported_configs
            .filter(|c| c.channels() == 1 || c.channels() == 2)
            .filter(|c| c.sample_format() == SampleFormat::F32)
            .find(|c| {
                c.min_sample_rate().0 <= sample_rate && c.max_sample_rate().0 >= sample_rate
            })
            .map(|c| c.with_sample_rate(cpal::SampleRate(sample_rate)))
            .or_else(|| {
                // Fallback: get default config
                device.default_output_config().ok()
            })
            .ok_or_else(|| anyhow::anyhow!("No suitable audio config found"))?;

        let channels = config.channels() as usize;

        // Create a shared buffer for playback
        let samples_arc = Arc::new(samples.to_vec());
        let position = Arc::new(Mutex::new(0usize));
        let finished = Arc::new(Mutex::new(false));

        let samples_clone = Arc::clone(&samples_arc);
        let position_clone = Arc::clone(&position);
        let finished_clone = Arc::clone(&finished);

        let stream = device.build_output_stream(
            &config.into(),
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                let mut pos = position_clone.lock().unwrap();
                let samples = &samples_clone;

                for frame in data.chunks_mut(channels) {
                    let sample = if *pos < samples.len() {
                        samples[*pos]
                    } else {
                        *finished_clone.lock().unwrap() = true;
                        0.0
                    };

                    // Fill all channels with the same sample (mono to stereo)
                    for s in frame.iter_mut() {
                        *s = sample;
                    }

                    if *pos < samples.len() {
                        *pos += 1;
                    }
                }
            },
            move |err| {
                eprintln!("Audio playback error: {}", err);
            },
            None,
        ).context("Failed to build output stream")?;

        stream.play().context("Failed to play audio stream")?;

        // Wait for playback to complete
        let duration_secs = samples.len() as f64 / sample_rate as f64;
        let wait_duration = Duration::from_secs_f64(duration_secs + 0.1); // Add small buffer

        let start = std::time::Instant::now();
        while start.elapsed() < wait_duration {
            if *finished.lock().unwrap() {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }

        // Small delay to ensure audio buffer is flushed
        std::thread::sleep(Duration::from_millis(50));

        Ok(())
    }

    /// Create a streaming audio player for real-time TTS output
    pub fn create_streaming_player(sample_rate: u32) -> Result<StreamingPlayer> {
        StreamingPlayer::new(sample_rate)
    }
}

/// Streaming audio player for real-time TTS output
pub struct StreamingPlayer {
    /// Sample rate
    sample_rate: u32,
    /// Audio buffer (ring buffer style)
    buffer: Arc<Mutex<Vec<f32>>>,
    /// Read position
    read_pos: Arc<Mutex<usize>>,
    /// Write position
    write_pos: Arc<Mutex<usize>>,
    /// Stream handle
    _stream: Option<cpal::Stream>,
    /// Is playing
    is_playing: Arc<Mutex<bool>>,
}

impl StreamingPlayer {
    /// Create a new streaming player
    pub fn new(sample_rate: u32) -> Result<Self> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or_else(|| anyhow::anyhow!("No audio output device available"))?;

        let config = device.default_output_config()
            .context("Failed to get default output config")?;

        let channels = config.channels() as usize;

        // Allocate a reasonable buffer size (2 seconds worth)
        let buffer_size = sample_rate as usize * 2;
        let buffer = Arc::new(Mutex::new(vec![0.0f32; buffer_size]));
        let read_pos = Arc::new(Mutex::new(0usize));
        let write_pos = Arc::new(Mutex::new(0usize));
        let is_playing = Arc::new(Mutex::new(false));

        let buffer_clone = Arc::clone(&buffer);
        let read_pos_clone = Arc::clone(&read_pos);
        let write_pos_clone = Arc::clone(&write_pos);

        let stream = device.build_output_stream(
            &config.into(),
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                let buffer = buffer_clone.lock().unwrap();
                let mut read = read_pos_clone.lock().unwrap();
                let write = write_pos_clone.lock().unwrap();

                for frame in data.chunks_mut(channels) {
                    let sample = if *read != *write {
                        let s = buffer[*read % buffer.len()];
                        *read = (*read + 1) % buffer.len();
                        s
                    } else {
                        0.0 // Buffer underrun - output silence
                    };

                    for s in frame.iter_mut() {
                        *s = sample;
                    }
                }
            },
            move |err| {
                eprintln!("Streaming playback error: {}", err);
            },
            None,
        ).context("Failed to build streaming output stream")?;

        Ok(Self {
            sample_rate,
            buffer,
            read_pos,
            write_pos,
            _stream: Some(stream),
            is_playing,
        })
    }

    /// Start playback
    pub fn play(&self) -> Result<()> {
        if let Some(ref stream) = self._stream {
            stream.play().context("Failed to start playback")?;
            *self.is_playing.lock().unwrap() = true;
        }
        Ok(())
    }

    /// Stop playback
    pub fn stop(&self) -> Result<()> {
        if let Some(ref stream) = self._stream {
            stream.pause().context("Failed to pause playback")?;
            *self.is_playing.lock().unwrap() = false;
        }
        Ok(())
    }

    /// Push samples to the buffer
    pub fn push_samples(&self, samples: &[f32]) {
        let mut buffer = self.buffer.lock().unwrap();
        let mut write = self.write_pos.lock().unwrap();

        for &sample in samples {
            let buf_len = buffer.len();
            buffer[*write % buf_len] = sample;
            *write = (*write + 1) % buf_len;
        }
    }

    /// Check if buffer has space for more samples
    pub fn buffer_space(&self) -> usize {
        let buffer = self.buffer.lock().unwrap();
        let read = *self.read_pos.lock().unwrap();
        let write = *self.write_pos.lock().unwrap();

        if write >= read {
            buffer.len() - (write - read)
        } else {
            read - write
        }
    }

    /// Get the sample rate
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Write samples to the buffer (alias for push_samples with Result)
    pub fn write(&self, samples: &[f32]) -> Result<()> {
        self.push_samples(samples);
        Ok(())
    }

    /// Wait for playback to complete (drain the buffer)
    pub fn drain(&self) -> Result<()> {
        // Wait until read catches up to write
        loop {
            let read = *self.read_pos.lock().unwrap();
            let write = *self.write_pos.lock().unwrap();
            
            if read == write {
                break;
            }
            
            // Small delay to avoid busy-waiting
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        
        // Extra delay for audio hardware buffer
        std::thread::sleep(std::time::Duration::from_millis(100));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_output_exists() {
        let _output = AudioOutput;
    }

    #[test]
    fn test_save_and_load_wav() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_audio_output.wav");

        // Create test samples (1 second of 440Hz sine wave)
        let samples: Vec<f32> = (0..22050)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 22050.0).sin() * 0.5)
            .collect();

        // Save
        AudioOutput::save(&samples, 22050, &path).unwrap();

        // Verify file exists
        assert!(path.exists());

        // Clean up
        std::fs::remove_file(&path).ok();
    }
}
