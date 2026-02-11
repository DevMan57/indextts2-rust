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
use std::process::Command;

use crate::config::ModelConfig;
use crate::debug::WeightDiagnostics;
use crate::text::{TextNormalizer, TextTokenizer};
use crate::audio::{AudioLoader, Resampler, MelSpectrogram, AudioOutput};
use crate::models::semantic::{SemanticEncoder, SemanticCodec};
use crate::models::speaker::CAMPPlus;
use crate::models::emotion::{EmotionMatrix, EmotionControls};
use crate::models::gpt::{UnifiedVoice, GenerationConfig, generate_with_hidden};
use crate::models::s2mel::{LengthRegulator, DiffusionTransformer, FlowMatching, FlowMatchingConfig};
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
    /// Emotion blending alpha (0.0 - 1.0)
    pub emotion_alpha: f32,
    /// Emotion vector weights (8 values)
    pub emotion_vector: Option<Vec<f32>>,
    /// Use emotion text to derive vector
    pub use_emo_text: bool,
    /// Optional emotion text (defaults to main text)
    pub emotion_text: Option<String>,

    /// Maximum mel tokens to generate
    pub max_mel_tokens: usize,
    /// Number of flow matching steps
    pub flow_steps: usize,
    /// Classifier-free guidance rate
    pub cfg_rate: f32,
    /// Apply post-vocoder high-pass de-rumble filter
    pub de_rumble: bool,
    /// High-pass cutoff (Hz) used when de-rumble is enabled
    pub de_rumble_cutoff_hz: f32,
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
            emotion_alpha: 1.0,
            emotion_vector: None,
            use_emo_text: false,
            emotion_text: None,
            max_mel_tokens: 1815,
            flow_steps: 25,
            cfg_rate: 0.0,
            de_rumble: false,
            de_rumble_cutoff_hz: 140.0,
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
    model_dir: std::path::PathBuf,
    inference_config: InferenceConfig,

    // Text processing
    normalizer: TextNormalizer,
    tokenizer: TextTokenizer,

    // Audio processing
    mel_extractor: MelSpectrogram,
    speaker_fbank: MelSpectrogram,
    #[allow(dead_code)]
    resampler: Option<Resampler>,

    // Encoders
    semantic_encoder: SemanticEncoder,
    semantic_codec: SemanticCodec,
    speaker_encoder: CAMPPlus,
    emotion_matrix: Option<EmotionMatrix>,
    emotion_controls: Option<EmotionControls>,

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
            .unwrap_or(Path::new("."))
            .to_path_buf();

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

        // Speaker fbank extractor (16kHz, 25ms window, 10ms hop)
        let speaker_fbank = MelSpectrogram::new(
            400,
            160,
            400,
            80,
            16000,
            0.0,
            None,
        );

        // Initialize encoders - use placeholder paths for now
        let w2v_stat_path = model_dir.join(&config.w2v_stat);
        let semantic_encoder = SemanticEncoder::load(&w2v_stat_path, None::<&std::path::PathBuf>, &device)?;
        let semantic_codec = SemanticCodec::new(&device)?;
        let speaker_encoder = CAMPPlus::new(&device)?;
        let emotion_matrix = Some(EmotionMatrix::new(&device)?);
        let emotion_controls = None;

        // Initialize generation model
        let gpt = UnifiedVoice::from_gpt_config(&config.gpt, &device)?;

        // Initialize synthesis
        let length_regulator = LengthRegulator::new(&device)?;
        let dit = DiffusionTransformer::new(&device)?;
        let flow_matching = FlowMatching::with_config(
            FlowMatchingConfig {
                num_steps: inference_config.flow_steps.max(1),
                cfg_rate: inference_config.cfg_rate.max(0.0),
                use_cfg: inference_config.cfg_rate > 0.0,
                ..Default::default()
            },
            &device,
        );
        let vocoder = BigVGAN::new(&device)?;

        Ok(Self {
            device,
            config,
            model_dir: model_dir.clone(),
            inference_config,
            normalizer,
            tokenizer,
            mel_extractor,
            speaker_fbank,
            resampler: None,
            semantic_encoder,
            semantic_codec,
            speaker_encoder,
            emotion_matrix,
            emotion_controls,
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
            // Note: Uses combined wqkv.weight instead of separate wq/wk/wv
            let expected_keys: HashSet<String> = [
                "cfm.estimator.transformer.layers.0.feed_forward.w1.weight",
                "cfm.estimator.transformer.layers.0.feed_forward.w2.weight",
                "cfm.estimator.transformer.layers.0.attention.wqkv.weight",  // Combined QKV
                "cfm.estimator.transformer.layers.0.attention.wo.weight",    // Output projection
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

                // Expected key patterns for BigVGAN (NVIDIA checkpoint uses weight normalization)
                // Raw tensor names have .weight_g/.weight_v instead of .weight
                let expected_keys: HashSet<String> = [
                    "conv_pre.weight_g",
                    "conv_pre.weight_v",
                    "conv_pre.bias",
                    "ups.0.0.weight_g",  // Note: has extra .0. in path
                    "ups.0.0.weight_v",
                    "ups.0.0.bias",
                    "resblocks.0.convs1.0.weight_g",
                    "resblocks.0.convs1.0.weight_v",
                    "resblocks.0.convs1.0.bias",
                    "resblocks.0.activations.0.act.alpha",
                    "resblocks.0.activations.0.act.beta",
                    "conv_post.weight_g",
                    "conv_post.weight_v",
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

        // Load semantic codec quantizer (MaskGCT) for vq2emb
        let semantic_codec_path = model_dir.join("maskgct/semantic_codec/model.safetensors");
        if semantic_codec_path.exists() {
            self.semantic_codec.load_quantizer_weights(&semantic_codec_path)
                .with_context(|| format!("Failed to load semantic codec quantizer from {:?}", semantic_codec_path))?;
            eprintln!("Loaded MaskGCT semantic codec quantizer");
        } else {
            eprintln!("WARNING: MaskGCT semantic codec not found at {:?}, vq2emb will use random codebook", semantic_codec_path);
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

        // Load emotion + speaker matrices for emotion vector mixing
        let emo_path = model_dir.join(&self.config.emo_matrix);
        let spk_path = model_dir.join(&self.config.spk_matrix);
        if emo_path.exists() && spk_path.exists() {
            self.emotion_controls = Some(EmotionControls::load(
                &emo_path,
                &spk_path,
                self.config.emo_num.clone(),
                &self.device,
            )?);
            eprintln!("Loaded emotion + speaker matrices for emotion vectors");
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
        let speaker_path = speaker_audio.as_ref().to_path_buf();
        let mut emotion_audio = emotion_audio.map(|p| p.as_ref().to_path_buf());

        let (speaker_samples, _sr) = AudioLoader::load(&speaker_path, 16000)
            .with_context(|| format!("Failed to load speaker audio from {:?}", speaker_path))?;
        let speaker_samples = Resampler::resample_to_16k(&speaker_samples, 16000)
            .context("Failed to resample speaker audio to 16kHz")?;

        // Create mel features for prompt (ref mel)
        let speaker_mel_2d = self.mel_extractor.compute(&speaker_samples)
            .context("Failed to compute mel spectrogram for speaker audio")?;
        // Flatten 2D mel [n_frames, n_mels] to 1D for Tensor::from_slice
        let speaker_mel: Vec<f32> = speaker_mel_2d.into_iter().flatten().collect();
        let n_frames = speaker_mel.len() / 80;
        let speaker_mel_tensor = Tensor::from_slice(
            &speaker_mel,
            (1, n_frames, 80),
            &self.device,
        ).context("Failed to create speaker mel tensor")?;

        // Speaker style embedding (CAMPPlus) from fbank-like features
        let speaker_fbank = self.compute_speaker_fbank(&speaker_samples)?;
        let speaker_emb = self.speaker_encoder.encode(&speaker_fbank)
            .context("Failed to encode speaker embedding")?;

        // Extract semantic features for conditioning (encode expects audio tensor)
        let audio_tensor = Tensor::from_slice(
            &speaker_samples,
            (1, speaker_samples.len()),
            &self.device,
        ).context("Failed to create audio tensor for semantic encoding")?;
        let semantic_features = self.semantic_encoder.encode(&audio_tensor, None)
            .context("Failed to encode semantic features")?;
        // semantic_features: [B, T_semantic, 1024] - raw W2V-BERT hidden layer 17 features

        eprintln!("DEBUG: raw semantic features shape={:?}", semantic_features.shape());

        // Run through RepCodec encoder (VocosBackbone) + quantizer for proper S_ref
        // Python: _, S_ref = semantic_codec.quantize(feat)
        // This runs: feat -> VocosBackbone(12 ConvNeXt blocks) -> in_project -> VQ -> out_project
        let (s_ref, _semantic_codes) = self.semantic_codec.encode_and_quantize(&semantic_features)
            .context("Failed to encode and quantize semantic features through RepCodec")?;
        // s_ref: [B, T_semantic, 1024] - quantized embeddings in RepCodec's output space

        eprintln!("DEBUG: s_ref (RepCodec encoded+quantized) shape={:?}", s_ref.shape());
        let sr_mean: f32 = s_ref.mean_all()?.to_scalar()?;
        let sr_var: f32 = s_ref.var(candle_core::D::Minus1)?.mean_all()?.to_scalar()?;
        eprintln!("DEBUG: s_ref mean={:.4}, var={:.4}", sr_mean, sr_var);

        // 3. Optional emotion processing
        let mut emo_alpha = self.inference_config.emotion_alpha;
        let mut emo_vector = self.inference_config.emotion_vector.clone();
        let use_emo_text = self.inference_config.use_emo_text;
        let emo_text = self.inference_config.emotion_text.clone();

        if use_emo_text {
            let text_for_emo = emo_text.as_deref().unwrap_or(text);
            match self.infer_emotion_text(text_for_emo) {
                Ok(vec) => emo_vector = Some(vec),
                Err(e) => eprintln!("WARNING: emotion text inference failed, ignoring emo-text vector: {}", e),
            }
        }

        if use_emo_text || emo_vector.is_some() {
            emotion_audio = None;
        }

        if let Some(ref mut vec) = emo_vector {
            let scale = emo_alpha.clamp(0.0, 1.0);
            if (scale - 1.0).abs() > 1e-6 {
                for v in vec.iter_mut() {
                    *v = ((*v * scale * 10000.0).floor()) / 10000.0;
                }
            }
            *vec = normalize_emo_vec(vec.clone(), true);
        }

        if emotion_audio.is_none() {
            emotion_audio = Some(speaker_path.clone());
            emo_alpha = 1.0;
        }

        let emo_path = emotion_audio.as_ref().unwrap();
        let emo_features = if emo_path == &speaker_path {
            semantic_features.clone()
        } else {
            let (emo_samples, _sr) = AudioLoader::load(emo_path, 16000)?;
            let emo_samples = Resampler::resample_to_16k(&emo_samples, 16000)?;
            let emo_tensor = Tensor::from_slice(
                &emo_samples,
                (1, emo_samples.len()),
                &self.device,
            )?;
            self.semantic_encoder.encode(&emo_tensor, None)?
        };

        let mut emovec = match self.gpt.merge_emovec(&semantic_features, &emo_features, emo_alpha) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("WARNING: merge_emovec failed: {}", e);
                Tensor::zeros((1, self.config.gpt.model_dim), DType::F32, &self.device)?
            }
        };

        if let (Some(vec), Some(ref controls)) = (emo_vector.clone(), &self.emotion_controls) {
            let emovec_mat = controls.build_emovec_from_vector(&vec, &speaker_emb, false)?;
            let sum: f32 = vec.iter().sum();
            let base_scaled = (&emovec * (1.0 - sum) as f64)?;
            emovec = (&base_scaled + &emovec_mat)?;
        } else if emo_vector.is_some() && self.emotion_controls.is_none() {
            eprintln!(
                "WARNING: Emotion vector provided but emotion_controls are not loaded; vector will be ignored."
            );
        }

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
        let conditioning = self.gpt.process_conditioning(&semantic_features)?;
        let emovec = emovec.unsqueeze(1)?.broadcast_as(conditioning.shape())?;
        let conditioning = (&conditioning + &emovec)?;

        // Generate mel codes AND capture hidden states (latent)
        // Python: codes, latent = gpt.inference_speech(...)
        let (mel_codes, hidden_states) = self.generate_with_sampling_fallback(
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

        // Step 2: vq2emb(codes) - embed mel codes using semantic codec quantizer codebook
        // Python: S_infer = self.semantic_codec.quantizer.vq2emb(codes.unsqueeze(1))
        //         S_infer = S_infer.transpose(1, 2)
        // The semantic codec's vq2emb does: codebook lookup (8-dim) → proj_out (8 → 1024)
        let mel_codes_tensor = Tensor::new(&mel_codes[..], &self.device)?
            .unsqueeze(0)?; // Shape: [1, seq_len]
        let code_embeddings = self.semantic_codec.vq2emb(&mel_codes_tensor)?;  // → [1, T, 1024]
        let ce_mean: f32 = code_embeddings.mean_all()?.to_scalar()?;
        let ce_var: f32 = code_embeddings.var(D::Minus1)?.mean_all()?.to_scalar()?;
        eprintln!("DEBUG: vq2emb(codes) shape={:?}, mean={:.4}, var={:.4}",
            code_embeddings.shape(), ce_mean, ce_var);

        // Step 3: S_infer = vq2emb(codes) + gpt_layer(latent)
        // This is the key computation from Python!
        let s_infer = (&code_embeddings + &latent_projected)?;
        let si_mean: f32 = s_infer.mean_all()?.to_scalar()?;
        let si_var: f32 = s_infer.var(D::Minus1)?.mean_all()?.to_scalar()?;
        eprintln!("DEBUG: S_infer = vq2emb + latent, shape={:?}, mean={:.4}, var={:.4}",
            s_infer.shape(), si_mean, si_var);

        // Step 4: Process through length regulator (1024 → 512)
        // CRITICAL: Apply 1.72× temporal expansion ratio (from Python reference)
        // Each mel code needs to be expanded to ~1.72 mel frames
        let num_mel_codes = s_infer.dim(1)?;
        let target_len = ((num_mel_codes as f32) * 1.72).round() as usize;
        eprintln!("DEBUG: Length expansion: {} codes → {} frames (ratio 1.72×)", num_mel_codes, target_len);

        // Compute prompt_condition from S_ref (reference audio semantic features)
        // Python: prompt_condition = length_regulator(S_ref, ylens=ref_target_lengths)
        let ref_mel_len = n_frames; // number of mel frames from speaker audio
        let (prompt_condition, _) = self.length_regulator.forward(
            &s_ref,
            Some(&[ref_mel_len]),
        )?;
        let pc_mean: f32 = prompt_condition.mean_all()?.to_scalar()?;
        let pc_var: f32 = prompt_condition.var(candle_core::D::Minus1)?.mean_all()?.to_scalar()?;
        eprintln!("DEBUG: prompt_condition shape={:?}, mean={:.4}, var={:.4}",
            prompt_condition.shape(), pc_mean, pc_var);

        let (content_features, _durations) = self.length_regulator.forward(
            &s_infer,
            Some(&[target_len]),
        )?;

        // Debug: Check content features
        let cf_mean: f32 = content_features.mean_all()?.to_scalar()?;
        let cf_var: f32 = content_features.var(D::Minus1)?.mean_all()?.to_scalar()?;
        eprintln!("DEBUG: content_features shape={:?}, mean={:.4}, var={:.4}",
            content_features.shape(), cf_mean, cf_var);

        // Concatenate prompt_condition + cond along time dimension (dim=1)
        // Python: cat_condition = torch.cat([prompt_condition, cond], dim=1)
        let cat_condition = Tensor::cat(&[&prompt_condition, &content_features], 1)?;
        let total_len = cat_condition.dim(1)?;
        eprintln!("DEBUG: cat_condition shape={:?} (ref_mel={} + target={})",
            cat_condition.shape(), ref_mel_len, target_len);

        // Debug: Check speaker embedding
        let spk_mean: f32 = speaker_emb.mean_all()?.to_scalar()?;
        let spk_var: f32 = speaker_emb.var(D::Minus1)?.mean_all()?.to_scalar()?;
        eprintln!("DEBUG: speaker_emb shape={:?}, mean={:.4}, var={:.4}",
            speaker_emb.shape(), spk_mean, spk_var);

        // Debug: Analyze speaker mel spectrogram (ground truth)
        let (_speaker_mel_mean, _speaker_mel_min, _speaker_mel_max) = {
            let spk_mel_2d = speaker_mel_tensor.squeeze(0)?; // [n_frames, 80]
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
            (spk_mel_mean, spk_mel_min, spk_mel_max)
        };

        // 6. Flow matching synthesis
        // Python: z = torch.randn([B, 80, T_total])  where T_total = ref_mel_len + target_len
        let batch_size = 1usize;

        // Create noise in [B, C, T] format: [1, 80, total_len]
        let noise = self.flow_matching.sample_noise(&[batch_size, 80, total_len])?;

        // Create prompt_x: [B, 80, total_len] with first ref_mel_len frames from speaker mel
        // speaker_mel_tensor is [B, T, C] = [1, n_frames, 80]
        let prompt_mel_len = ref_mel_len.min(speaker_mel_tensor.dim(1)?); // Clamp to available mel frames
        let prompt_region = speaker_mel_tensor.narrow(1, 0, prompt_mel_len)?; // [B, prompt_mel_len, 80]

        // If ref_mel_len > available speaker mel frames, pad with zeros
        let prompt_tc = if prompt_mel_len < ref_mel_len {
            let padding = Tensor::zeros(
                (batch_size, ref_mel_len - prompt_mel_len, 80),
                DType::F32,
                &self.device,
            )?;
            Tensor::cat(&[prompt_region, padding], 1)?
        } else {
            prompt_region
        };
        // Append zeros for the generation region
        let gen_zeros = Tensor::zeros(
            (batch_size, target_len, 80),
            DType::F32,
            &self.device,
        )?;
        let prompt_x_tc = Tensor::cat(&[prompt_tc, gen_zeros], 1)?; // [B, total_len, 80]
        let prompt_x = prompt_x_tc.transpose(1, 2)?; // [B, 80, total_len]

        let prompt_len = ref_mel_len;
        eprintln!("DEBUG: Flow matching: total_len={}, ref_mel_len={}, target_len={}, prompt_len={}",
            total_len, ref_mel_len, target_len, prompt_len);

        // sample() expects [B, C, T] inputs and returns [B, C, T] output
        let mel_spec_ct = self.flow_matching.sample(
            &self.dit,
            &noise,           // [B, 80, total_len]
            &prompt_x,        // [B, 80, total_len]
            &cat_condition,   // [B, total_len, 512]
            &speaker_emb,     // [B, 192]
            prompt_len,       // = ref_mel_len
        )?;

        // Transpose output from [B, C, T] back to [B, T, C]
        let mel_spec_full = mel_spec_ct.transpose(1, 2)?;

        // CRITICAL: Strip prompt region from output (Python does this!)
        // Python: vc_target = vc_target[:, :, ref_mel.size(-1):]
        let mel_spec = mel_spec_full.narrow(1, prompt_len, target_len)?;
        eprintln!("DEBUG: Stripped prompt ({} frames), keeping {} generated frames",
            prompt_len, target_len);

        // Debug: Check mel spectrogram output
        let mel_mean: f32 = mel_spec.mean_all()?.to_scalar()?;
        let mel_var: f32 = mel_spec.var(D::Minus1)?.mean_all()?.to_scalar()?;
        let mel_min: f32 = mel_spec.flatten_all()?.min(0)?.to_scalar()?;
        let mel_max: f32 = mel_spec.flatten_all()?.max(0)?.to_scalar()?;
        eprintln!("DEBUG: mel_spec shape={:?}, mean={:.4}, var={:.4}, min={:.4}, max={:.4}",
            mel_spec.shape(), mel_mean, mel_var, mel_min, mel_max);

        // Detailed mel band analysis
        {
            let mel_2d = mel_spec.squeeze(0)?; // [target_len, 80]
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
        let speaker_mel_mean: f32 = speaker_mel_tensor.mean_all()?.to_scalar()?;
        eprintln!("DEBUG: Generated mel mean: {:.4}, Speaker mel mean: {:.4}, diff: {:.4}",
            generated_mean, speaker_mel_mean, speaker_mel_mean - generated_mean);

        // 8. Vocoder - mel to audio
        // Use raw DiT output like upstream (no extra mel shifting/scaling before BigVGAN).
        let mel_vocoder_input = mel_spec.clone();
        let mel_transposed = mel_vocoder_input.transpose(1, 2)?; // (batch, mel, time)
        let audio_tensor = self.vocoder.forward(&mel_transposed)?;

        // Debug: Check audio output
        let audio_mean: f32 = audio_tensor.mean_all()?.to_scalar()?;
        let audio_min: f32 = audio_tensor.flatten_all()?.min(0)?.to_scalar()?;
        let audio_max: f32 = audio_tensor.flatten_all()?.max(0)?.to_scalar()?;
        eprintln!("DEBUG: audio shape={:?}, mean={:.6}, min={:.4}, max={:.4}",
            audio_tensor.shape(), audio_mean, audio_min, audio_max);

        let mut audio: Vec<f32> = audio_tensor.squeeze(0)?.squeeze(0)?.to_vec1()?;

        // Safety guard: auto-apply de-rumble when waveform has strong DC bias.
        // This avoids the common "rumbling water" failure mode even without explicit CLI flags.
        let auto_derumble = !self.inference_config.de_rumble && audio_mean.abs() > 0.03;
        if self.inference_config.de_rumble || auto_derumble {
            let sr = self.vocoder.sample_rate() as f32;
            let cutoff_hz = if self.inference_config.de_rumble {
                self.inference_config.de_rumble_cutoff_hz.max(20.0)
            } else {
                140.0
            };
            audio = apply_high_pass(&audio, sr, cutoff_hz);
            if auto_derumble {
                eprintln!(
                    "INFO: Auto-applied de-rumble high-pass: mean={:.4}, cutoff_hz={:.1}",
                    audio_mean,
                    cutoff_hz
                );
            } else {
                eprintln!(
                    "INFO: Applied post-vocoder de-rumble high-pass: cutoff_hz={:.1}",
                    cutoff_hz
                );
            }
        }

        Ok(InferenceResult {
            audio,
            sample_rate: self.vocoder.sample_rate() as u32,
            mel_codes,
            mel_spectrogram: Some(mel_vocoder_input),
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

        // Generate mel codes AND capture hidden states for S_infer computation
        let (mel_codes, hidden_states) = self.generate_with_sampling_fallback(
            &text_ids,
            Some(conditioning),
            &gen_config,
        )?;

        // Compute S_infer = vq2emb(codes) + gpt_layer(latent) (same as main inference path)
        // Step 1: gpt_layer(latent) - project hidden states: 1280 → 1024
        let latent_projected = self.length_regulator.project_gpt_embeddings(&hidden_states)?;

        // Step 2: vq2emb(codes) - embed mel codes using semantic codec quantizer codebook
        let mel_codes_tensor = Tensor::new(&mel_codes[..], &self.device)?
            .unsqueeze(0)?; // Shape: [1, seq_len]
        let code_embeddings = self.semantic_codec.vq2emb(&mel_codes_tensor)?;  // → [1, T, 1024]

        // Step 3: S_infer = vq2emb(codes) + gpt_layer(latent)
        let s_infer = (&code_embeddings + &latent_projected)?;

        // Step 4: Process through length regulator with temporal expansion
        let num_mel_codes = s_infer.dim(1)?;
        let target_len = ((num_mel_codes as f32) * 1.72).round() as usize;

        let (content_features, _) = self.length_regulator.forward(
            &s_infer,
            Some(&[target_len]),
        )?;

        // Flow matching
        let (batch_size, seq_len, _) = content_features.dims3()?;
        // FlowMatching.sample expects [B, C, T] where C=80 mel channels.
        let noise = self.flow_matching.sample_noise(&[batch_size, 80, seq_len])?;
        let prompt_x = Tensor::zeros((batch_size, 80, seq_len), DType::F32, &self.device)?;
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
        let audio_tensor = self.vocoder.forward(&mel_spec)?;

        audio_tensor.squeeze(0)?.squeeze(0)?.to_vec1().map_err(Into::into)
    }

    fn generate_with_sampling_fallback(
        &mut self,
        text_ids: &Tensor,
        conditioning: Option<&Tensor>,
        gen_config: &GenerationConfig,
    ) -> Result<(Vec<u32>, Tensor)> {
        match generate_with_hidden(&mut self.gpt, text_ids, conditioning, gen_config) {
            Ok(out) => Ok(out),
            Err(err) => {
                let err_msg = format!("{err:#}");
                let is_cuda_invalid_value = err_msg.contains("CUDA_ERROR_INVALID_VALUE")
                    || err_msg.contains("DriverError(CUDA_ERROR_INVALID_VALUE");
                let needs_safe_sampling = gen_config.top_k != 0 || gen_config.top_p < 1.0;

                if is_cuda_invalid_value && needs_safe_sampling {
                    eprintln!(
                        "WARNING: GPU sampling failed ({}). Retrying once on GPU with safe sampling (top_k=0, top_p=1.0).",
                        err
                    );

                    let mut safe_config = gen_config.clone();
                    safe_config.top_k = 0;
                    safe_config.top_p = 1.0;

                    let retried = generate_with_hidden(
                        &mut self.gpt,
                        text_ids,
                        conditioning,
                        &safe_config,
                    )
                    .with_context(|| {
                        format!(
                            "safe GPU sampling retry failed after CUDA sampling error: {}",
                            err_msg
                        )
                    })?;

                    eprintln!("INFO: Safe GPU sampling retry succeeded.");
                    Ok(retried)
                } else {
                    Err(err)
                }
            }
        }
    }

    fn compute_speaker_fbank(&self, samples: &[f32]) -> Result<Tensor> {
        let mut fbank = self.speaker_fbank.compute(samples)?;
        if fbank.is_empty() {
            anyhow::bail!("Empty fbank output");
        }
        let n_frames = fbank.len();
        let n_mels = fbank[0].len();
        let mut means = vec![0.0f32; n_mels];
        for frame in &fbank {
            for (j, v) in frame.iter().enumerate() {
                means[j] += *v;
            }
        }
        for m in &mut means {
            *m /= n_frames as f32;
        }
        for frame in &mut fbank {
            for (j, v) in frame.iter_mut().enumerate() {
                *v -= means[j];
            }
        }
        let flat: Vec<f32> = fbank.into_iter().flatten().collect();
        Tensor::from_slice(&flat, (1, n_frames, n_mels), &self.device)
            .map_err(Into::into)
    }

    fn infer_emotion_text(&self, text: &str) -> Result<Vec<f32>> {
        let script = std::path::Path::new("scripts/qwen_emo.py");
        let script_path = if script.exists() {
            script.to_path_buf()
        } else {
            self.model_dir.parent().unwrap_or(&self.model_dir).join("scripts/qwen_emo.py")
        };
        let model_dir = self.model_dir.join(&self.config.qwen_emo_path);
        let output = Command::new("python")
            .arg(script_path)
            .arg("--model-dir")
            .arg(model_dir)
            .arg("--text")
            .arg(text)
            .output()
            .context("Failed to run qwen_emo.py")?;

        if !output.status.success() {
            anyhow::bail!("qwen_emo.py failed: {}", String::from_utf8_lossy(&output.stderr));
        }
        let stdout = String::from_utf8_lossy(&output.stdout);
        let vec: Vec<f32> = serde_json::from_str(stdout.trim())
            .context("Failed to parse emotion vector JSON")?;
        if vec.len() != 8 {
            anyhow::bail!("emotion vector from qwen_emo.py must have 8 values, got {}", vec.len());
        }
        Ok(vec)
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

fn normalize_emo_vec(mut emo_vector: Vec<f32>, apply_bias: bool) -> Vec<f32> {
    if apply_bias {
        let emo_bias = [0.9375, 0.875, 1.0, 1.0, 0.9375, 0.9375, 0.6875, 0.5625];
        for (v, b) in emo_vector.iter_mut().zip(emo_bias.iter()) {
            *v *= *b as f32;
        }
    }

    let sum: f32 = emo_vector.iter().sum();
    if sum > 0.8 {
        let scale = 0.8 / sum;
        for v in emo_vector.iter_mut() {
            *v *= scale;
        }
    }
    emo_vector
}

fn apply_high_pass(input: &[f32], sample_rate: f32, cutoff_hz: f32) -> Vec<f32> {
    if input.is_empty() || sample_rate <= 0.0 || cutoff_hz <= 0.0 {
        return input.to_vec();
    }

    let dt = 1.0 / sample_rate;
    let rc = 1.0 / (2.0 * std::f32::consts::PI * cutoff_hz);
    let alpha = rc / (rc + dt);

    let mut out = Vec::with_capacity(input.len());
    let mut prev_y = 0.0f32;
    let mut prev_x = input[0];
    for &x in input {
        let y = alpha * (prev_y + x - prev_x);
        out.push(y);
        prev_y = y;
        prev_x = x;
    }

    // Keep safe output headroom.
    let peak = out.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
    if peak > 0.98 {
        let scale = 0.98 / peak;
        for v in &mut out {
            *v *= scale;
        }
    }

    out
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





















