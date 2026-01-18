//! Main inference pipeline for IndexTTS2
//!
//! Orchestrates all components for text-to-speech synthesis:
//! 1. Text normalization and tokenization
//! 2. Speaker/emotion encoding from reference audio
//! 3. GPT-based mel code generation
//! 4. Mel spectrogram synthesis via flow matching
//! 5. Waveform generation via BigVGAN vocoder

use anyhow::Result;
use candle_core::{Device, Tensor, DType};
use std::path::Path;

use crate::config::ModelConfig;
use crate::text::{TextNormalizer, TextTokenizer};
use crate::audio::{AudioLoader, Resampler, MelSpectrogram, AudioOutput};
use crate::models::semantic::{SemanticEncoder, SemanticCodec};
use crate::models::speaker::CAMPPlus;
use crate::models::emotion::EmotionMatrix;
use crate::models::gpt::{UnifiedVoice, GenerationConfig, generate};
use crate::models::s2mel::{LengthRegulator, DiffusionTransformer, FlowMatching};
use crate::models::vocoder::BigVGAN;

/// Inference configuration
#[derive(Clone)]
pub struct InferenceConfig {
    /// Generation temperature (0.0-1.0)
    pub temperature: f32,
    /// Top-k sampling (0 = disabled)
    pub top_k: usize,
    /// Top-p (nucleus) sampling (1.0 = disabled)
    pub top_p: f32,
    /// Repetition penalty
    pub repetition_penalty: f32,
    /// Maximum mel tokens to generate
    pub max_mel_tokens: usize,
    /// Number of flow matching steps
    pub flow_steps: usize,
    /// Classifier-free guidance rate
    pub cfg_rate: f32,
    /// Whether to use GPU
    pub use_gpu: bool,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: 50,
            top_p: 0.95,
            repetition_penalty: 1.1,
            max_mel_tokens: 1815,
            flow_steps: 25,
            cfg_rate: 0.7,
            use_gpu: false,
        }
    }
}

/// Result of inference
pub struct InferenceResult {
    /// Generated audio samples
    pub audio: Vec<f32>,
    /// Sample rate
    pub sample_rate: u32,
    /// Generated mel codes
    pub mel_codes: Vec<u32>,
    /// Generated mel spectrogram
    pub mel_spectrogram: Option<Tensor>,
}

impl InferenceResult {
    /// Save audio to file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        AudioOutput::save(&self.audio, self.sample_rate, path)
    }

    /// Get duration in seconds
    pub fn duration(&self) -> f32 {
        self.audio.len() as f32 / self.sample_rate as f32
    }
}

/// Main IndexTTS2 inference pipeline
pub struct IndexTTS2 {
    device: Device,
    config: ModelConfig,
    inference_config: InferenceConfig,

    // Text processing
    normalizer: TextNormalizer,
    tokenizer: TextTokenizer,

    // Audio processing
    mel_extractor: MelSpectrogram,
    resampler: Option<Resampler>,

    // Encoders
    semantic_encoder: SemanticEncoder,
    semantic_codec: SemanticCodec,
    speaker_encoder: CAMPPlus,
    emotion_matrix: Option<EmotionMatrix>,

    // Generation
    gpt: UnifiedVoice,

    // Synthesis
    length_regulator: LengthRegulator,
    dit: DiffusionTransformer,
    flow_matching: FlowMatching,
    vocoder: BigVGAN,
}

impl IndexTTS2 {
    /// Create a new IndexTTS2 instance from config file
    pub fn new<P: AsRef<Path>>(config_path: P) -> Result<Self> {
        Self::with_config(config_path, InferenceConfig::default())
    }

    /// Create with custom inference config
    pub fn with_config<P: AsRef<Path>>(
        config_path: P,
        inference_config: InferenceConfig,
    ) -> Result<Self> {
        let config = ModelConfig::load(&config_path)?;
        let model_dir = config_path.as_ref().parent()
            .unwrap_or(Path::new("."));

        let device = if inference_config.use_gpu {
            Device::cuda_if_available(0)?
        } else {
            Device::Cpu
        };

        // Initialize text processing
        let normalizer = TextNormalizer::new(false);
        let tokenizer_path = model_dir.join(&config.dataset.bpe_model);
        let tokenizer = if tokenizer_path.exists() {
            TextTokenizer::load(&tokenizer_path, normalizer.clone())?
        } else {
            // Create a fallback tokenizer with empty/default path
            TextTokenizer::load("tokenizer.json", normalizer.clone())
                .unwrap_or_else(|_| {
                    // If we can't load any tokenizer, we need to handle gracefully
                    panic!("No tokenizer found at {:?} or default location", tokenizer_path)
                })
        };

        // Initialize audio processing
        let fmax_f32 = config.s2mel.preprocess_params.spect_params.fmax
            .as_ref()
            .and_then(|s| s.parse::<f32>().ok());
        let mel_extractor = MelSpectrogram::new(
            config.s2mel.preprocess_params.spect_params.n_fft,
            config.s2mel.preprocess_params.spect_params.hop_length,
            config.s2mel.preprocess_params.spect_params.win_length,
            config.s2mel.preprocess_params.spect_params.n_mels,
            config.s2mel.preprocess_params.sr,
            config.s2mel.preprocess_params.spect_params.fmin,
            fmax_f32,
        );

        // Initialize encoders - use placeholder paths for now
        let w2v_stat_path = model_dir.join(&config.w2v_stat);
        let semantic_encoder = SemanticEncoder::load(&w2v_stat_path, None::<&std::path::PathBuf>, &device)?;
        let semantic_codec = SemanticCodec::new(&device)?;
        let speaker_encoder = CAMPPlus::new(&device)?;
        let emotion_matrix = Some(EmotionMatrix::new(&device)?);

        // Initialize generation model
        let gpt = UnifiedVoice::new(&device)?;

        // Initialize synthesis
        let length_regulator = LengthRegulator::new(&device)?;
        let dit = DiffusionTransformer::new(&device)?;
        let flow_matching = FlowMatching::new(&device);
        let vocoder = BigVGAN::new(&device)?;

        Ok(Self {
            device,
            config,
            inference_config,
            normalizer,
            tokenizer,
            mel_extractor,
            resampler: None,
            semantic_encoder,
            semantic_codec,
            speaker_encoder,
            emotion_matrix,
            gpt,
            length_regulator,
            dit,
            flow_matching,
            vocoder,
        })
    }

    /// Initialize all model weights
    pub fn load_weights<P: AsRef<Path>>(&mut self, model_dir: P) -> Result<()> {
        let model_dir = model_dir.as_ref();

        // Load semantic encoder weights if available
        let w2v_path = model_dir.join(&self.config.w2v_stat);
        if w2v_path.exists() {
            self.semantic_encoder.load_weights(&w2v_path)?;
        }
        // SemanticEncoder is initialized with placeholder weights by default

        // Load GPT
        let gpt_path = model_dir.join(&self.config.gpt_checkpoint);
        if gpt_path.exists() {
            self.gpt.load_weights(&gpt_path)?;
        } else {
            self.gpt.initialize_random()?;
        }

        // Load S2Mel (DiT)
        let s2mel_path = model_dir.join(&self.config.s2mel_checkpoint);
        if s2mel_path.exists() {
            self.dit.load_weights(&s2mel_path)?;
            self.length_regulator.load_weights(&s2mel_path)?;
        } else {
            self.dit.initialize_random()?;
            self.length_regulator.initialize_random()?;
        }

        // Load vocoder
        self.vocoder.initialize_random()?;

        // Load emotion matrix
        if let Some(ref mut emo) = self.emotion_matrix {
            let emo_path = model_dir.join(&self.config.emo_matrix);
            if emo_path.exists() {
                emo.load_weights(&emo_path)?;
            }
        }

        Ok(())
    }

    /// Perform text-to-speech inference
    ///
    /// # Arguments
    /// * `text` - Input text to synthesize
    /// * `speaker_audio` - Path to speaker reference audio
    ///
    /// # Returns
    /// * InferenceResult with generated audio
    pub fn infer<P: AsRef<Path>>(
        &mut self,
        text: &str,
        speaker_audio: P,
    ) -> Result<InferenceResult> {
        self.infer_with_emotion(text, speaker_audio, None)
    }

    /// Perform text-to-speech with emotion control
    ///
    /// # Arguments
    /// * `text` - Input text to synthesize
    /// * `speaker_audio` - Path to speaker reference audio
    /// * `emotion_audio` - Optional path to emotion reference audio
    ///
    /// # Returns
    /// * InferenceResult with generated audio
    pub fn infer_with_emotion<P: AsRef<Path>>(
        &mut self,
        text: &str,
        speaker_audio: P,
        emotion_audio: Option<P>,
    ) -> Result<InferenceResult> {
        // 1. Normalize and tokenize text
        let normalized = self.normalizer.normalize(text);
        let tokens = self.tokenizer.encode(&normalized)?;
        let text_ids = Tensor::new(&tokens[..], &self.device)?
            .unsqueeze(0)?; // Add batch dimension

        // 2. Load and process speaker reference
        let (speaker_samples, _sr) = AudioLoader::load(speaker_audio.as_ref(), 16000)?;
        let speaker_samples = Resampler::resample_to_16k(&speaker_samples, 16000)?;

        // Create mel features for speaker encoder (expects mel filterbank features)
        let speaker_mel_2d = self.mel_extractor.compute(&speaker_samples)?;
        // Flatten 2D mel [n_frames, n_mels] to 1D for Tensor::from_slice
        let speaker_mel: Vec<f32> = speaker_mel_2d.into_iter().flatten().collect();
        let n_frames = speaker_mel.len() / 80;
        let speaker_tensor = Tensor::from_slice(
            &speaker_mel,
            (1, n_frames, 80), // (batch, time, mel_bands)
            &self.device,
        )?;
        let speaker_emb = self.speaker_encoder.encode(&speaker_tensor)?;

        // Extract semantic features for conditioning (encode expects audio tensor)
        let audio_tensor = Tensor::from_slice(
            &speaker_samples,
            (1, speaker_samples.len()),
            &self.device,
        )?;
        let semantic_features = self.semantic_encoder.encode(&audio_tensor, None)?;
        let (semantic_codes, _) = self.semantic_codec.quantize(&semantic_features)?;

        // 3. Optional emotion processing
        let _emotion_emb = if let Some(emo_path) = emotion_audio {
            if let Some(ref _emo_matrix) = self.emotion_matrix {
                let (emo_samples, _) = AudioLoader::load(emo_path.as_ref(), 16000)?;
                let emo_mel_2d = self.mel_extractor.compute(&emo_samples)?;
                let emo_mel: Vec<f32> = emo_mel_2d.into_iter().flatten().collect();
                let emo_frames = emo_mel.len() / 80;
                let emo_tensor = Tensor::from_slice(
                    &emo_mel,
                    (1, emo_frames, 80),
                    &self.device,
                )?;
                // For now, use speaker embedding as emotion basis
                Some(self.speaker_encoder.encode(&emo_tensor)?)
            } else {
                None
            }
        } else {
            None
        };

        // 4. GPT generation - produce mel codes
        let gen_config = GenerationConfig {
            max_length: self.inference_config.max_mel_tokens,
            temperature: self.inference_config.temperature,
            top_k: self.inference_config.top_k,
            top_p: self.inference_config.top_p,
            repetition_penalty: self.inference_config.repetition_penalty,
            ..Default::default()
        };

        // Process conditioning through conformer/perceiver
        let conditioning = self.gpt.process_conditioning(&semantic_codes)?;

        let mel_codes = generate(
            &mut self.gpt,
            &text_ids,
            Some(&conditioning),
            &gen_config,
        )?;

        // 5. Length regulation - expand mel codes
        let mel_codes_tensor = Tensor::new(&mel_codes[..], &self.device)?
            .unsqueeze(0)?
            .to_dtype(DType::F32)?;

        let (content_features, _durations) = self.length_regulator.forward(
            &mel_codes_tensor.unsqueeze(2)?, // Add feature dim
            None,
        )?;

        // 6. Flow matching synthesis
        let (batch_size, seq_len, _) = content_features.dims3()?;
        let noise = self.flow_matching.sample_noise(&[batch_size, seq_len, 80])?;

        let mel_spec = self.flow_matching.sample(
            &self.dit,
            &noise,
            &content_features,
            Some(&speaker_emb),
        )?;

        // 7. Vocoder - mel to audio
        let mel_transposed = mel_spec.transpose(1, 2)?; // (batch, mel, time)
        let audio_tensor = self.vocoder.forward(&mel_transposed)?;

        let audio: Vec<f32> = audio_tensor.squeeze(0)?.squeeze(0)?.to_vec1()?;

        Ok(InferenceResult {
            audio,
            sample_rate: self.vocoder.sample_rate() as u32,
            mel_codes,
            mel_spectrogram: Some(mel_spec),
        })
    }

    /// Synthesize a single segment (internal use)
    fn synthesize_segment(
        &mut self,
        text: &str,
        speaker_emb: &Tensor,
        conditioning: &Tensor,
    ) -> Result<Vec<f32>> {
        // Tokenize
        let normalized = self.normalizer.normalize(text);
        let tokens = self.tokenizer.encode(&normalized)?;
        let text_ids = Tensor::new(&tokens[..], &self.device)?.unsqueeze(0)?;

        // Generate mel codes
        let gen_config = GenerationConfig {
            max_length: self.inference_config.max_mel_tokens,
            temperature: self.inference_config.temperature,
            ..Default::default()
        };

        let mel_codes = generate(&mut self.gpt, &text_ids, Some(conditioning), &gen_config)?;

        // Length regulation
        let mel_codes_tensor = Tensor::new(&mel_codes[..], &self.device)?
            .unsqueeze(0)?
            .to_dtype(DType::F32)?;

        let (content_features, _) = self.length_regulator.forward(
            &mel_codes_tensor.unsqueeze(2)?,
            None,
        )?;

        // Flow matching
        let (batch_size, seq_len, _) = content_features.dims3()?;
        let noise = self.flow_matching.sample_noise(&[batch_size, seq_len, 80])?;
        let mel_spec = self.flow_matching.sample(
            &self.dit,
            &noise,
            &content_features,
            Some(speaker_emb),
        )?;

        // Vocoder
        let mel_transposed = mel_spec.transpose(1, 2)?;
        let audio_tensor = self.vocoder.forward(&mel_transposed)?;

        audio_tensor.squeeze(0)?.squeeze(0)?.to_vec1().map_err(Into::into)
    }

    /// Get the device being used
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get sample rate
    pub fn sample_rate(&self) -> u32 {
        self.vocoder.sample_rate() as u32
    }

    /// Set inference temperature
    pub fn set_temperature(&mut self, temperature: f32) {
        self.inference_config.temperature = temperature;
    }

    /// Set top-k sampling
    pub fn set_top_k(&mut self, top_k: usize) {
        self.inference_config.top_k = top_k;
    }

    /// Set top-p sampling
    pub fn set_top_p(&mut self, top_p: f32) {
        self.inference_config.top_p = top_p;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inference_config_default() {
        let config = InferenceConfig::default();
        assert_eq!(config.temperature, 0.8);
        assert_eq!(config.top_k, 50);
        assert_eq!(config.flow_steps, 25);
    }

    #[test]
    fn test_inference_result_duration() {
        let result = InferenceResult {
            audio: vec![0.0; 22050],
            sample_rate: 22050,
            mel_codes: vec![],
            mel_spectrogram: None,
        };
        assert!((result.duration() - 1.0).abs() < 0.001);
    }
}
