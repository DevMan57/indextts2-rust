//! Streaming synthesis for real-time audio output
//!
//! Provides chunk-based synthesis for low-latency TTS:
//! - Sentence-level chunking
//! - Asynchronous audio generation
//! - Real-time playback via audio output

use anyhow::Result;
use candle_core::Device;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;

use crate::text::TextNormalizer;
use crate::audio::StreamingPlayer;

/// Streaming configuration
#[derive(Clone)]
pub struct StreamingConfig {
    /// Maximum tokens per chunk
    pub max_tokens_per_chunk: usize,
    /// Audio buffer size in samples
    pub buffer_size: usize,
    /// Sample rate
    pub sample_rate: u32,
    /// Pre-buffer chunks before playback
    pub prebuffer_chunks: usize,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            max_tokens_per_chunk: 120,
            buffer_size: 4096,
            sample_rate: 22050,
            prebuffer_chunks: 2,
        }
    }
}

/// Audio chunk for streaming
#[derive(Clone)]
pub struct AudioChunk {
    /// Audio samples
    pub samples: Vec<f32>,
    /// Chunk index
    pub index: usize,
    /// Whether this is the last chunk
    pub is_final: bool,
    /// Original text for this chunk
    pub text: String,
}

impl AudioChunk {
    /// Create a new audio chunk
    pub fn new(samples: Vec<f32>, index: usize, is_final: bool, text: String) -> Self {
        Self {
            samples,
            index,
            is_final,
            text,
        }
    }

    /// Get duration in seconds
    pub fn duration(&self, sample_rate: u32) -> f32 {
        self.samples.len() as f32 / sample_rate as f32
    }
}

/// Streaming synthesizer for real-time audio generation
pub struct StreamingSynthesizer {
    device: Device,
    config: StreamingConfig,
    normalizer: TextNormalizer,

    // Channels for async communication
    text_sender: Option<Sender<String>>,
    audio_receiver: Option<Receiver<AudioChunk>>,

    // State
    is_running: bool,
    current_chunk: usize,
}

impl StreamingSynthesizer {
    /// Create a new streaming synthesizer
    pub fn new(device: &Device) -> Result<Self> {
        Self::with_config(StreamingConfig::default(), device)
    }

    /// Create with custom config
    pub fn with_config(config: StreamingConfig, device: &Device) -> Result<Self> {
        Ok(Self {
            device: device.clone(),
            config,
            normalizer: TextNormalizer::new(false),
            text_sender: None,
            audio_receiver: None,
            is_running: false,
            current_chunk: 0,
        })
    }

    /// Start streaming synthesis
    ///
    /// Segments the text and begins generating audio chunks.
    /// Returns a receiver for audio chunks.
    pub fn start(&mut self, text: &str) -> Result<Receiver<AudioChunk>> {
        let (chunk_sender, chunk_receiver) = channel::<AudioChunk>();

        // Normalize and split text into segments based on sentence boundaries
        let normalized = self.normalizer.normalize(text);
        let segments = split_into_segments(&normalized, self.config.max_tokens_per_chunk);

        let _device = self.device.clone();
        let _config = self.config.clone();

        // Spawn synthesis thread
        thread::spawn(move || {
            for (i, segment) in segments.iter().enumerate() {
                let is_final = i == segments.len() - 1;

                // Generate audio for segment
                // In real implementation, this would use the full pipeline
                // For now, generate placeholder audio
                let segment_duration_samples = segment.len() * 256; // Approximate
                let samples = vec![0.0f32; segment_duration_samples.max(1024)];

                let chunk = AudioChunk::new(
                    samples,
                    i,
                    is_final,
                    segment.clone(),
                );

                if chunk_sender.send(chunk).is_err() {
                    break; // Receiver dropped
                }
            }
        });

        self.is_running = true;
        self.current_chunk = 0;

        Ok(chunk_receiver)
    }

    /// Stop streaming synthesis
    pub fn stop(&mut self) {
        self.is_running = false;
        self.text_sender = None;
        self.audio_receiver = None;
    }

    /// Check if currently streaming
    pub fn is_running(&self) -> bool {
        self.is_running
    }

    /// Get current chunk index
    pub fn current_chunk(&self) -> usize {
        self.current_chunk
    }

    /// Stream and play audio directly
    ///
    /// Convenience method that handles playback automatically.
    pub fn stream_and_play(&mut self, text: &str) -> Result<()> {
        let receiver = self.start(text)?;
        let player = StreamingPlayer::new(self.config.sample_rate)?;

        // Prebuffer
        let mut prebuffer = Vec::new();
        for _ in 0..self.config.prebuffer_chunks {
            if let Ok(chunk) = receiver.recv() {
                prebuffer.extend(chunk.samples);
                if chunk.is_final {
                    break;
                }
            }
        }

        // Start playback with prebuffer
        player.write(&prebuffer)?;

        // Stream remaining chunks
        for chunk in receiver {
            player.write(&chunk.samples)?;
            self.current_chunk = chunk.index;

            if chunk.is_final {
                break;
            }
        }

        // Wait for playback to finish
        player.drain()?;
        self.is_running = false;

        Ok(())
    }

    /// Generate audio chunks synchronously
    ///
    /// Returns all chunks as a vector for non-streaming use.
    pub fn generate_all(&mut self, text: &str) -> Result<Vec<AudioChunk>> {
        let receiver = self.start(text)?;
        let mut chunks = Vec::new();

        for chunk in receiver {
            let is_final = chunk.is_final;
            chunks.push(chunk);
            if is_final {
                break;
            }
        }

        self.is_running = false;
        Ok(chunks)
    }

    /// Get concatenated audio from all chunks
    pub fn generate_audio(&mut self, text: &str) -> Result<Vec<f32>> {
        let chunks = self.generate_all(text)?;
        let total_len: usize = chunks.iter().map(|c| c.samples.len()).sum();

        let mut audio = Vec::with_capacity(total_len);
        for chunk in chunks {
            audio.extend(chunk.samples);
        }

        Ok(audio)
    }
}

/// Callback-based streaming interface
pub trait StreamingCallback: Send {
    /// Called when a new audio chunk is ready
    fn on_chunk(&mut self, chunk: &AudioChunk);

    /// Called when synthesis is complete
    fn on_complete(&mut self);

    /// Called on error
    fn on_error(&mut self, error: &str);
}

/// Stream with callback
pub fn stream_with_callback<C: StreamingCallback + 'static>(
    text: &str,
    device: &Device,
    mut callback: C,
) -> Result<()> {
    let mut synth = StreamingSynthesizer::new(device)?;
    let receiver = synth.start(text)?;

    thread::spawn(move || {
        for chunk in receiver {
            let is_final = chunk.is_final;
            callback.on_chunk(&chunk);

            if is_final {
                callback.on_complete();
                break;
            }
        }
    });

    Ok(())
}

/// Split text into segments based on sentence boundaries and max length
fn split_into_segments(text: &str, max_chars: usize) -> Vec<String> {
    let sentence_endings = ['.', '!', '?', '。', '！', '？'];
    let mut segments = Vec::new();
    let mut current = String::new();

    for c in text.chars() {
        current.push(c);

        // Check if we should split here
        let should_split = if current.len() >= max_chars {
            true
        } else if sentence_endings.contains(&c) && current.len() > 10 {
            true
        } else {
            false
        };

        if should_split {
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() {
                segments.push(trimmed);
            }
            current = String::new();
        }
    }

    // Add remaining text
    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        segments.push(trimmed);
    }

    if segments.is_empty() {
        segments.push(text.to_string());
    }

    segments
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_config_default() {
        let config = StreamingConfig::default();
        assert_eq!(config.max_tokens_per_chunk, 120);
        assert_eq!(config.sample_rate, 22050);
    }

    #[test]
    fn test_audio_chunk_new() {
        let chunk = AudioChunk::new(
            vec![0.0; 1000],
            0,
            false,
            "Hello".to_string(),
        );
        assert_eq!(chunk.index, 0);
        assert!(!chunk.is_final);
        assert_eq!(chunk.text, "Hello");
    }

    #[test]
    fn test_audio_chunk_duration() {
        let chunk = AudioChunk::new(vec![0.0; 22050], 0, false, "".to_string());
        let duration = chunk.duration(22050);
        assert!((duration - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_streaming_synthesizer_new() {
        let device = Device::Cpu;
        let synth = StreamingSynthesizer::new(&device).unwrap();
        assert!(!synth.is_running());
        assert_eq!(synth.current_chunk(), 0);
    }

    #[test]
    fn test_streaming_synthesizer_generate_all() {
        let device = Device::Cpu;
        let mut synth = StreamingSynthesizer::new(&device).unwrap();

        let chunks = synth.generate_all("Hello world. This is a test.").unwrap();
        assert!(!chunks.is_empty());

        // Last chunk should be marked as final
        let last = chunks.last().unwrap();
        assert!(last.is_final);
    }

    #[test]
    fn test_streaming_synthesizer_generate_audio() {
        let device = Device::Cpu;
        let mut synth = StreamingSynthesizer::new(&device).unwrap();

        let audio = synth.generate_audio("Hello world.").unwrap();
        assert!(!audio.is_empty());
    }
}
