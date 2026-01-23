//! Main inference pipeline for IndexTTS2
//!
//! Orchestrates all components for text-to-speech synthesis:
//! 1. Text normalization and tokenization
//! 2. Speaker/emotion encoding from reference audio
//! 3. GPT-based mel code generation
//! 4. Mel spectrogram synthesis via flow matching
//! 5. Waveform generation via BigVGAN vocoder

use anyhow::{Result, Context};
use candle_core::{Device, Tensor, DType, D};
use std::collections::HashSet;
use std::path::Path;

use crate::config::ModelConfig;
use crate::debug::WeightDiagnostics;
use crate::text::{TextNormalizer, TextTokenizer};
use crate::audio::{AudioLoader, Resampler, MelSpectrogram, AudioOutput};
use crate::models::semantic::{SemanticEncoder, SemanticCodec};
use crate::models::speaker::CAMPPlus;
use crate::models::emotion::EmotionMatrix;
use crate::models::gpt::{UnifiedVoice, GenerationConfig, generate, generate_with_hidden};
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
    /// Enable verbose weight loading diagnostics
    pub verbose_weights: bool,
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
            verbose_weights: false,
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
    #[allow(dead_code)]
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
        let mut diagnostics = WeightDiagnostics::new(self.inference_config.verbose_weights);

        // Load Wav2Vec-BERT full model if available
        if let Some(ref w2v_model) = self.config.w2v_model {
            let w2v_path = model_dir.join(w2v_model);
            if w2v_path.exists() {
                tracing::info!("Loading Wav2Vec-BERT model from {:?}...", w2v_path);

                // Enumerate tensors for diagnostics
                let tensors = diagnostics.load_safetensors(&w2v_path, "Wav2Vec-BERT", &self.device)?;
                let available_keys: Vec<String> = tensors.keys().cloned().collect();

                // Expected key patterns for Wav2Vec-BERT (actual naming from HuggingFace model)
                let expected_keys: HashSet<String> = [
                    "encoder.layers.0.self_attn.linear_k.weight",
                    "encoder.layers.0.self_attn.linear_v.weight",
                    "encoder.layers.0.self_attn.linear_q.weight",
                    "encoder.layers.0.self_attn.linear_out.weight",
                    "encoder.layers.0.conv_module.pointwise_conv1.weight",
                    "encoder.layers.0.conv_module.pointwise_conv2.weight",
                    "feature_projection.projection.weight",
                    "feature_projection.projection.bias",
                ].iter().map(|s| s.to_string()).collect();

                diagnostics.record_component(
                    "Wav2Vec-BERT",
                    &w2v_path.to_string_lossy(),
                    available_keys,
                    expected_keys,
                );

                self.semantic_encoder.load_weights(&w2v_path)
                    .with_context(|| format!("Failed to load Wav2Vec-BERT from {:?}", w2v_path))?;
            }
        }

        // Load GPT
        let gpt_path = model_dir.join(&self.config.gpt_checkpoint);
        if gpt_path.exists() {
            // Enumerate tensors for diagnostics
            let tensors = diagnostics.load_safetensors(&gpt_path, "GPT", &self.device)?;
            let available_keys: Vec<String> = tensors.keys().cloned().collect();

            // Expected key patterns for GPT (actual naming from IndexTTS checkpoint)
            let expected_keys: HashSet<String> = [
                "text_embedding.weight",
                "mel_embedding.weight",
                "final_norm.weight",
                "mel_head.weight",
                "gpt.h.0.attn.c_attn.weight",
                "gpt.h.0.attn.c_proj.weight",
                "gpt.h.0.mlp.c_fc.weight",
                "conditioning_encoder.encoders.0.self_attn.linear_k.weight",
            ].iter().map(|s| s.to_string()).collect();

            diagnostics.record_component(
                "GPT",
                &gpt_path.to_string_lossy(),
                available_keys,
                expected_keys,
            );

            self.gpt.load_weights(&gpt_path)
                .with_context(|| format!("Failed to load GPT weights from {:?}", gpt_path))?;
        } else {
            self.gpt.initialize_random()
                .context("Failed to initialize GPT with random weights")?;
        }

        // Load S2Mel (DiT)
        let s2mel_path = model_dir.join(&self.config.s2mel_checkpoint);
        if s2mel_path.exists() {
            // Enumerate tensors for diagnostics
            let tensors = diagnostics.load_safetensors(&s2mel_path, "DiT", &self.device)?;
            let available_keys: Vec<String> = tensors.keys().cloned().collect();

            // Expected key patterns for DiT (actual naming from IndexTTS s2mel checkpoint)
            let expected_keys: HashSet<String> = [
                "cfm.estimator.transformer.layers.0.feed_forward.w1.weight",
                "cfm.estimator.transformer.layers.0.feed_forward.w2.weight",
                "cfm.estimator.transformer.layers.0.attention.wq.weight",
                "cfm.estimator.x_embedder.weight_v",
                "cfm.estimator.t_embedder.mlp.0.weight",
                "length_regulator.model.0.weight",
            ].iter().map(|s| s.to_string()).collect();

            diagnostics.record_component(
                "DiT",
                &s2mel_path.to_string_lossy(),
                available_keys,
                expected_keys,
            );

            self.dit.load_weights(&s2mel_path)
                .with_context(|| format!("Failed to load DiT weights from {:?}", s2mel_path))?;
            self.length_regulator.load_weights(&s2mel_path)
                .with_context(|| format!("Failed to load LengthRegulator weights from {:?}", s2mel_path))?;
        } else {
            self.dit.initialize_random()
                .context("Failed to initialize DiT with random weights")?;
            self.length_regulator.initialize_random()
                .context("Failed to initialize LengthRegulator with random weights")?;
        }

        // Load BigVGAN vocoder
        if let Some(ref bigvgan_path) = self.config.bigvgan_checkpoint {
            let vocoder_path = model_dir.join(bigvgan_path);
            if vocoder_path.exists() {
                tracing::info!("Loading BigVGAN from {:?}...", vocoder_path);

                // Enumerate tensors for diagnostics
                let tensors = diagnostics.load_safetensors(&vocoder_path, "BigVGAN", &self.device)?;
                let available_keys: Vec<String> = tensors.keys().cloned().collect();

                // Expected key patterns for BigVGAN (actual naming from NVIDIA checkpoint)
                let expected_keys: HashSet<String> = [
                    "conv_pre.weight",
                    "conv_pre.bias",
                    "ups.0.weight",
                    "ups.0.bias",
                    "resblocks.0.convs1.0.weight",
                    "resblocks.0.convs1.0.bias",
                    "resblocks.0.activations.0.act.alpha",
                    "conv_post.weight",
                ].iter().map(|s| s.to_string()).collect();

                diagnostics.record_component(
                    "BigVGAN",
                    &vocoder_path.to_string_lossy(),
                    available_keys,
                    expected_keys,
                );

                self.vocoder.load_weights(&vocoder_path)
                    .with_context(|| format!("Failed to load BigVGAN weights from {:?}", vocoder_path))?;
            } else {
                self.vocoder.initialize_random()
                    .context("Failed to initialize BigVGAN with random weights")?;
            }
        } else {
            self.vocoder.initialize_random()
                .context("Failed to initialize BigVGAN with random weights")?;
        }

        // Initialize speaker encoder with random weights
        // Note: No CAMPPlus checkpoint available, using random initialization
        tracing::info!("Initializing speaker encoder (CAMPPlus)...");
        self.speaker_encoder.initialize_random()
            .context("Failed to initialize CAMPPlus speaker encoder")?;

        // Load emotion matrix
        if let Some(ref mut emo) = self.emotion_matrix {
            let emo_path = model_dir.join(&self.config.emo_matrix);
            if emo_path.exists() {
                emo.load_weights(&emo_path)
                    .with_context(|| format!("Failed to load emotion matrix from {:?}", emo_path))?;
            }
        }

        // Print final weight loading summary
        diagnostics.print_final_summary();

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
        let tokens = self.tokenizer.encode(&normalized)
            .context("Failed to tokenize input text")?;
        let text_ids = Tensor::new(&tokens[..], &self.device)
            .context("Failed to create text token tensor")?
            .unsqueeze(0)?; // Add batch dimension

        // 2. Load and process speaker reference
        let speaker_path = speaker_audio.as_ref();
        let (speaker_samples, _sr) = AudioLoader::load(speaker_path, 16000)
            .with_context(|| format!("Failed to load speaker audio from {:?}", speaker_path))?;
        let speaker_samples = Resampler::resample_to_16k(&speaker_samples, 16000)
            .context("Failed to resample speaker audio to 16kHz")?;

        // Create mel features for speaker encoder (expects mel filterbank features)
        let speaker_mel_2d = self.mel_extractor.compute(&speaker_samples)
            .context("Failed to compute mel spectrogram for speaker audio")?;
        // Flatten 2D mel [n_frames, n_mels] to 1D for Tensor::from_slice
        let speaker_mel: Vec<f32> = speaker_mel_2d.into_iter().flatten().collect();
        let n_frames = speaker_mel.len() / 80;
        let speaker_tensor = Tensor::from_slice(
            &speaker_mel,
            (1, n_frames, 80), // (batch, time, mel_bands)
            &self.device,
        ).context("Failed to create speaker mel tensor")?;
        let speaker_emb = self.speaker_encoder.encode(&speaker_tensor)
            .context("Failed to encode speaker embedding")?;

        // Extract semantic features for conditioning (encode expects audio tensor)
        let audio_tensor = Tensor::from_slice(
            &speaker_samples,
            (1, speaker_samples.len()),
            &self.device,
        ).context("Failed to create audio tensor for semantic encoding")?;
        let semantic_features = self.semantic_encoder.encode(&audio_tensor, None)
            .context("Failed to encode semantic features")?;
        let (_semantic_codes, _) = self.semantic_codec.quantize(&semantic_features)
            .context("Failed to quantize semantic features")?;

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

        // 4. GPT generation - produce mel codes AND hidden states
        let gen_config = GenerationConfig {
            max_length: self.inference_config.max_mel_tokens,
            temperature: self.inference_config.temperature,
            top_k: self.inference_config.top_k,
            top_p: self.inference_config.top_p,
            repetition_penalty: self.inference_config.repetition_penalty,
            ..Default::default()
        };

        // Process conditioning through conformer/perceiver
        let conditioning = self.gpt.process_conditioning(&speaker_tensor)?;

        // Generate mel codes AND capture hidden states (latent)
        // Python: codes, latent = gpt.inference_speech(...)
        let (mel_codes, hidden_states) = generate_with_hidden(
            &mut self.gpt,
            &text_ids,
            Some(&conditioning),
            &gen_config,
        )?;

        // 5. Compute S_infer = vq2emb(codes) + gpt_layer(latent)
        // This is the critical fix from the Python reference implementation

        // Debug: Check mel codes distribution
        eprintln!("DEBUG: mel_codes count={}, min={}, max={}",
            mel_codes.len(),
            mel_codes.iter().min().unwrap_or(&0),
            mel_codes.iter().max().unwrap_or(&0));

        // hidden_states: (1, num_codes, 1280) - GPT hidden states before lm_head
        let hs_mean: f32 = hidden_states.mean_all()?.to_scalar()?;
        let hs_var: f32 = hidden_states.var(D::Minus1)?.mean_all()?.to_scalar()?;
        eprintln!("DEBUG: hidden_states (latent) shape={:?}, mean={:.4}, var={:.4}",
            hidden_states.shape(), hs_mean, hs_var);

        // Step 1: gpt_layer(latent) - project hidden states: 1280 → 1024
        let latent_projected = self.length_regulator.project_gpt_embeddings(&hidden_states)?;
        let lp_mean: f32 = latent_projected.mean_all()?.to_scalar()?;
        let lp_var: f32 = latent_projected.var(D::Minus1)?.mean_all()?.to_scalar()?;
        eprintln!("DEBUG: gpt_layer(latent) shape={:?}, mean={:.4}, var={:.4}",
            latent_projected.shape(), lp_mean, lp_var);

        // Step 2: vq2emb(codes) - embed mel codes and project to 1024
        // Use GPT's mel_embedding to get 1280-dim, then project to 1024
        let mel_codes_tensor = Tensor::new(&mel_codes[..], &self.device)?
            .unsqueeze(0)?; // Shape: [1, seq_len]
        let mel_embeddings = self.gpt.embed_mel_codes(&mel_codes_tensor)?;
        let me_mean: f32 = mel_embeddings.mean_all()?.to_scalar()?;
        let me_var: f32 = mel_embeddings.var(D::Minus1)?.mean_all()?.to_scalar()?;
        eprintln!("DEBUG: vq2emb(codes) raw shape={:?}, mean={:.4}, var={:.4}",
            mel_embeddings.shape(), me_mean, me_var);

        // Project mel embeddings to 1024 (same as latent projection)
        let code_embeddings = self.length_regulator.project_gpt_embeddings(&mel_embeddings)?;
        let ce_mean: f32 = code_embeddings.mean_all()?.to_scalar()?;
        let ce_var: f32 = code_embeddings.var(D::Minus1)?.mean_all()?.to_scalar()?;
        eprintln!("DEBUG: vq2emb(codes) projected shape={:?}, mean={:.4}, var={:.4}",
            code_embeddings.shape(), ce_mean, ce_var);

        // Step 3: S_infer = vq2emb(codes) + gpt_layer(latent)
        // This is the key computation from Python!
        let s_infer = (&code_embeddings + &latent_projected)?;
        let si_mean: f32 = s_infer.mean_all()?.to_scalar()?;
        let si_var: f32 = s_infer.var(D::Minus1)?.mean_all()?.to_scalar()?;
        eprintln!("DEBUG: S_infer = vq2emb + latent, shape={:?}, mean={:.4}, var={:.4}",
            s_infer.shape(), si_mean, si_var);

        // Step 4: Process through length regulator (1024 → 512)
        let (content_features, _durations) = self.length_regulator.forward(
            &s_infer,
            None,
        )?;

        // Debug: Check content features
        let cf_mean: f32 = content_features.mean_all()?.to_scalar()?;
        let cf_var: f32 = content_features.var(D::Minus1)?.mean_all()?.to_scalar()?;
        eprintln!("DEBUG: content_features shape={:?}, mean={:.4}, var={:.4}",
            content_features.shape(), cf_mean, cf_var);

        // Debug: Check speaker embedding
        let spk_mean: f32 = speaker_emb.mean_all()?.to_scalar()?;
        let spk_var: f32 = speaker_emb.var(D::Minus1)?.mean_all()?.to_scalar()?;
        eprintln!("DEBUG: speaker_emb shape={:?}, mean={:.4}, var={:.4}",
            speaker_emb.shape(), spk_mean, spk_var);

        // Debug: Analyze speaker mel spectrogram (ground truth)
        {
            let spk_mel_2d = speaker_tensor.squeeze(0)?; // [n_frames, 80]
            let spk_mel_mean: f32 = spk_mel_2d.mean_all()?.to_scalar()?;
            let spk_mel_min: f32 = spk_mel_2d.flatten_all()?.min(0)?.to_scalar()?;
            let spk_mel_max: f32 = spk_mel_2d.flatten_all()?.max(0)?.to_scalar()?;
            let band_means: Vec<f32> = (0..80).map(|i| {
                spk_mel_2d.narrow(1, i, 1).unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap()
            }).collect();
            let low_bands: f32 = band_means[0..20].iter().sum::<f32>() / 20.0;
            let mid_bands: f32 = band_means[20..50].iter().sum::<f32>() / 30.0;
            let high_bands: f32 = band_means[50..80].iter().sum::<f32>() / 30.0;
            eprintln!("DEBUG: SPEAKER mel - mean={:.4}, range=[{:.4},{:.4}], bands: low={:.4} mid={:.4} high={:.4}",
                spk_mel_mean, spk_mel_min, spk_mel_max, low_bands, mid_bands, high_bands);
        }

        // 6. Flow matching synthesis
        let (batch_size, seq_len, _) = content_features.dims3()?;
        let noise = self.flow_matching.sample_noise(&[batch_size, seq_len, 80])?;

        // Mel normalization constants - the model was trained on normalized mels
        // These values center the log-mel range around 0 with reasonable variance
        // Note: Common TTS normalization uses mean=-5 to -7, std=4 to 5
        // Trying mean=-5.5, std=2 based on speaker mel distribution
        const MEL_MEAN: f64 = -5.5;  // Center of typical log-mel range
        const MEL_STD: f64 = 2.0;    // Typical dynamic range (try smaller for more aggressive normalization)

        // Create prompt_x from reference mel by padding/truncating to match seq_len
        let speaker_mel_len = speaker_tensor.dim(1)?;
        let prompt_x_raw = if speaker_mel_len >= seq_len {
            // Truncate: take first seq_len frames from speaker mel
            speaker_tensor.narrow(1, 0, seq_len)?
        } else {
            // Pad: repeat speaker mel to fill seq_len
            let repeat_factor = (seq_len + speaker_mel_len - 1) / speaker_mel_len;
            let repeated = speaker_tensor.repeat(&[1, repeat_factor, 1])?;
            repeated.narrow(1, 0, seq_len)?
        };

        // Normalize prompt_x: (mel - mean) / std
        let prompt_x = ((&prompt_x_raw - MEL_MEAN)? / MEL_STD)?;
        let prompt_norm_mean: f32 = prompt_x.mean_all()?.to_scalar()?;
        eprintln!("DEBUG: Normalized prompt_x mean: {:.4} (raw mean: {:.4})",
            prompt_norm_mean, speaker_tensor.mean_all()?.to_scalar::<f32>()?);

        // For pure TTS (not inpainting), use prompt_len=0 to generate entire sequence
        // The reference mel in prompt_x provides conditioning context but we generate all frames
        // Note: If doing audio continuation/inpainting, set prompt_len = speaker_mel_len.min(seq_len)
        let prompt_len = 0; // Pure TTS mode
        eprintln!("DEBUG: Flow matching with seq_len={}, speaker_mel_len={}, prompt_len={}",
            seq_len, speaker_mel_len, prompt_len);
        let mel_spec_normalized = self.flow_matching.sample(
            &self.dit,
            &noise,
            &prompt_x,
            &content_features,
            &speaker_emb,
            prompt_len,
        )?;

        // Denormalize output: mel = norm_mel * std + mean
        let mel_spec = ((&mel_spec_normalized * MEL_STD)? + MEL_MEAN)?;

        // Debug: Check mel spectrogram output
        let mel_mean: f32 = mel_spec.mean_all()?.to_scalar()?;
        let mel_var: f32 = mel_spec.var(D::Minus1)?.mean_all()?.to_scalar()?;
        let mel_min: f32 = mel_spec.flatten_all()?.min(0)?.to_scalar()?;
        let mel_max: f32 = mel_spec.flatten_all()?.max(0)?.to_scalar()?;
        eprintln!("DEBUG: mel_spec shape={:?}, mean={:.4}, var={:.4}, min={:.4}, max={:.4}",
            mel_spec.shape(), mel_mean, mel_var, mel_min, mel_max);

        // Detailed mel band analysis
        {
            let mel_2d = mel_spec.squeeze(0)?; // [seq_len, 80]
            let band_means: Vec<f32> = (0..80).map(|i| {
                mel_2d.narrow(1, i, 1).unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap()
            }).collect();
            let low_bands: f32 = band_means[0..20].iter().sum::<f32>() / 20.0;
            let mid_bands: f32 = band_means[20..50].iter().sum::<f32>() / 30.0;
            let high_bands: f32 = band_means[50..80].iter().sum::<f32>() / 30.0;
            eprintln!("DEBUG: mel band analysis - low(0-20)={:.4}, mid(20-50)={:.4}, high(50-80)={:.4}",
                low_bands, mid_bands, high_bands);
        }

        // 7. Compare generated mel with speaker mel (for debugging)
        let generated_mean: f32 = mel_spec.mean_all()?.to_scalar()?;
        let speaker_mel_mean: f32 = speaker_tensor.mean_all()?.to_scalar()?;
        eprintln!("DEBUG: Generated mel mean: {:.4}, Speaker mel mean: {:.4}, diff: {:.4}",
            generated_mean, speaker_mel_mean, speaker_mel_mean - generated_mean);

        // 8. Vocoder - mel to audio
        let mel_transposed = mel_spec.transpose(1, 2)?; // (batch, mel, time)
        let audio_tensor = self.vocoder.forward(&mel_transposed)?;

        // Debug: Check audio output
        let audio_mean: f32 = audio_tensor.mean_all()?.to_scalar()?;
        let audio_min: f32 = audio_tensor.flatten_all()?.min(0)?.to_scalar()?;
        let audio_max: f32 = audio_tensor.flatten_all()?.max(0)?.to_scalar()?;
        eprintln!("DEBUG: audio shape={:?}, mean={:.6}, min={:.4}, max={:.4}",
            audio_tensor.shape(), audio_mean, audio_min, audio_max);

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
        let prompt_x = Tensor::zeros((batch_size, seq_len, 80), DType::F32, &self.device)?;
        // Zero prompt region - prompt_len=0 when no reference mel
        let mel_spec = self.flow_matching.sample(
            &self.dit,
            &noise,
            &prompt_x,
            &content_features,
            speaker_emb,
            0, // No prompt region to zero
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
