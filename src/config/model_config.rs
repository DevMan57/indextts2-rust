//! Model configuration types matching the IndexTTS2 config.yaml structure

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Root configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Dataset configuration
    pub dataset: DatasetConfig,
    
    /// GPT model configuration
    pub gpt: GptConfig,
    
    /// Semantic codec configuration
    pub semantic_codec: SemanticCodecConfig,
    
    /// Semantic-to-mel (S2Mel) configuration
    pub s2mel: S2MelConfig,
    
    /// GPT checkpoint filename
    pub gpt_checkpoint: String,
    
    /// Wav2Vec-BERT stats filename
    pub w2v_stat: String,

    /// Wav2Vec-BERT model filename (full model weights)
    #[serde(default)]
    pub w2v_model: Option<String>,

    /// S2Mel checkpoint filename
    pub s2mel_checkpoint: String,

    /// Emotion matrix filename
    pub emo_matrix: String,

    /// Speaker matrix filename
    pub spk_matrix: String,

    /// BigVGAN checkpoint filename
    #[serde(default)]
    pub bigvgan_checkpoint: Option<String>,
    
    /// Emotion class counts per category
    pub emo_num: Vec<usize>,
    
    /// Qwen emotion model path
    pub qwen_emo_path: String,
    
    /// Vocoder configuration
    pub vocoder: VocoderConfig,
    
    /// Model version
    #[serde(default = "default_version")]
    pub version: f32,
}

fn default_version() -> f32 {
    2.0
}

impl ModelConfig {
    /// Load configuration from a YAML file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref())
            .with_context(|| format!("Failed to read config file: {:?}", path.as_ref()))?;
        
        serde_yaml::from_str(&content)
            .with_context(|| "Failed to parse config YAML")
    }
    
    /// Get the full path to a checkpoint file
    pub fn checkpoint_path<P: AsRef<Path>>(&self, model_dir: P, filename: &str) -> std::path::PathBuf {
        model_dir.as_ref().join(filename)
    }
}

/// Dataset configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetConfig {
    /// BPE tokenizer model filename
    pub bpe_model: String,
    
    /// Audio sample rate
    pub sample_rate: u32,
    
    /// Whether to squeeze dimensions
    #[serde(default)]
    pub squeeze: bool,
    
    /// Mel spectrogram configuration
    pub mel: MelConfig,
}

/// Mel spectrogram configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MelConfig {
    /// Sample rate for mel computation
    pub sample_rate: u32,
    
    /// FFT size
    pub n_fft: usize,
    
    /// Hop length between frames
    pub hop_length: usize,
    
    /// Window length
    pub win_length: usize,
    
    /// Number of mel bands
    pub n_mels: usize,
    
    /// Minimum mel frequency
    #[serde(default)]
    pub mel_fmin: f32,
    
    /// Whether to normalize
    #[serde(default)]
    pub normalize: bool,
}

/// GPT model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GptConfig {
    /// Model dimension
    pub model_dim: usize,
    
    /// Maximum mel tokens
    pub max_mel_tokens: usize,
    
    /// Maximum text tokens
    pub max_text_tokens: usize,
    
    /// Number of attention heads
    pub heads: usize,
    
    /// Whether to use mel codes as input
    #[serde(default = "default_true")]
    pub use_mel_codes_as_input: bool,
    
    /// Mel length compression factor
    pub mel_length_compression: usize,
    
    /// Number of transformer layers
    pub layers: usize,
    
    /// Number of text tokens in vocabulary
    pub number_text_tokens: usize,
    
    /// Number of mel codes
    pub number_mel_codes: usize,
    
    /// Start mel token ID
    pub start_mel_token: usize,
    
    /// Stop mel token ID
    pub stop_mel_token: usize,
    
    /// Start text token ID
    pub start_text_token: usize,
    
    /// Stop text token ID
    pub stop_text_token: usize,
    
    /// Train solo embeddings
    #[serde(default)]
    pub train_solo_embeddings: bool,
    
    /// Conditioning type
    pub condition_type: String,
    
    /// Condition module configuration
    pub condition_module: ConditionModuleConfig,
    
    /// Emotion condition module configuration
    pub emo_condition_module: ConditionModuleConfig,
}

fn default_true() -> bool {
    true
}

/// Condition module (Conformer) configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConditionModuleConfig {
    /// Output dimension
    pub output_size: usize,
    
    /// Linear units in feed-forward
    pub linear_units: usize,
    
    /// Number of attention heads
    pub attention_heads: usize,
    
    /// Number of transformer blocks
    pub num_blocks: usize,
    
    /// Input layer type
    pub input_layer: String,
    
    /// Perceiver multiplier
    pub perceiver_mult: usize,
}

/// Semantic codec configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticCodecConfig {
    /// Codebook size
    pub codebook_size: usize,
    
    /// Hidden dimension
    pub hidden_size: usize,
    
    /// Codebook dimension
    pub codebook_dim: usize,
    
    /// Vocos decoder dimension
    pub vocos_dim: usize,
    
    /// Vocos intermediate dimension
    pub vocos_intermediate_dim: usize,
    
    /// Number of Vocos layers
    pub vocos_num_layers: usize,
}

/// S2Mel (Semantic-to-Mel) configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct S2MelConfig {
    /// Preprocessing parameters
    pub preprocess_params: S2MelPreprocessConfig,
    
    /// DiT type
    pub dit_type: String,
    
    /// Regression loss type
    pub reg_loss_type: String,
    
    /// Style encoder configuration
    pub style_encoder: StyleEncoderConfig,
    
    /// Length regulator configuration
    pub length_regulator: LengthRegulatorConfig,
    
    /// DiT (Diffusion Transformer) configuration
    #[serde(rename = "DiT")]
    pub dit: DiTConfig,
    
    /// WaveNet configuration
    pub wavenet: WaveNetConfig,
}

/// S2Mel preprocessing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct S2MelPreprocessConfig {
    /// Sample rate
    pub sr: u32,
    
    /// Spectrogram parameters
    pub spect_params: SpectParams,
}

/// Spectrogram parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectParams {
    /// FFT size
    pub n_fft: usize,
    
    /// Window length
    pub win_length: usize,
    
    /// Hop length
    pub hop_length: usize,
    
    /// Number of mel bands
    pub n_mels: usize,
    
    /// Minimum frequency
    #[serde(default)]
    pub fmin: f32,
    
    /// Maximum frequency (can be "None")
    #[serde(default)]
    pub fmax: Option<String>,
}

/// Style encoder configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleEncoderConfig {
    /// Embedding dimension
    pub dim: usize,
}

/// Length regulator configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LengthRegulatorConfig {
    /// Channel dimension
    pub channels: usize,
    
    /// Whether input is discrete
    pub is_discrete: bool,
    
    /// Input channels
    pub in_channels: usize,
    
    /// Content codebook size
    pub content_codebook_size: usize,
    
    /// Sampling ratios
    pub sampling_ratios: Vec<usize>,
    
    /// Whether to use vector quantization
    pub vector_quantize: bool,
    
    /// Number of codebooks
    pub n_codebooks: usize,
    
    /// Quantizer dropout rate
    pub quantizer_dropout: f32,
    
    /// Whether to use F0 conditioning
    pub f0_condition: bool,
    
    /// Number of F0 bins
    pub n_f0_bins: usize,
}

/// DiT (Diffusion Transformer) configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiTConfig {
    /// Hidden dimension
    pub hidden_dim: usize,
    
    /// Number of attention heads
    pub num_heads: usize,
    
    /// Number of layers (depth)
    pub depth: usize,
    
    /// Class dropout probability
    pub class_dropout_prob: f32,
    
    /// Block size for attention
    pub block_size: usize,
    
    /// Input channels (mel bands)
    pub in_channels: usize,
    
    /// Whether to use style conditioning
    pub style_condition: bool,
    
    /// Final layer type
    pub final_layer_type: String,
    
    /// Target type
    pub target: String,
    
    /// Content dimension
    pub content_dim: usize,
    
    /// Content codebook size
    pub content_codebook_size: usize,
    
    /// Content type
    pub content_type: String,
    
    /// F0 conditioning
    pub f0_condition: bool,
    
    /// Number of F0 bins
    pub n_f0_bins: usize,
    
    /// Number of content codebooks
    pub content_codebooks: usize,
    
    /// Whether model is causal
    pub is_causal: bool,
    
    /// Long skip connection
    pub long_skip_connection: bool,
    
    /// Zero prompt speech token
    pub zero_prompt_speech_token: bool,
    
    /// Time as token
    pub time_as_token: bool,
    
    /// Style as token
    pub style_as_token: bool,
    
    /// UViT skip connection
    pub uvit_skip_connection: bool,
    
    /// Add ResBlock in transformer
    pub add_resblock_in_transformer: bool,
}

/// WaveNet configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WaveNetConfig {
    /// Hidden dimension
    pub hidden_dim: usize,
    
    /// Number of layers
    pub num_layers: usize,
    
    /// Kernel size
    pub kernel_size: usize,
    
    /// Dilation rate
    pub dilation_rate: usize,
    
    /// Dropout probability
    pub p_dropout: f32,
    
    /// Style conditioning
    pub style_condition: bool,
}

/// Vocoder configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VocoderConfig {
    /// Vocoder type
    #[serde(rename = "type")]
    pub vocoder_type: String,
    
    /// Model name/path
    pub name: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_config_parse() {
        let yaml = r#"
dataset:
    bpe_model: bpe.model
    sample_rate: 24000
    squeeze: false
    mel:
        sample_rate: 24000
        n_fft: 1024
        hop_length: 256
        win_length: 1024
        n_mels: 100
        mel_fmin: 0
        normalize: false

gpt:
    model_dim: 1280
    max_mel_tokens: 1815
    max_text_tokens: 600
    heads: 20
    use_mel_codes_as_input: true
    mel_length_compression: 1024
    layers: 24
    number_text_tokens: 12000
    number_mel_codes: 8194
    start_mel_token: 8192
    stop_mel_token: 8193
    start_text_token: 0
    stop_text_token: 1
    train_solo_embeddings: false
    condition_type: "conformer_perceiver"
    condition_module:
        output_size: 512
        linear_units: 2048
        attention_heads: 8
        num_blocks: 6
        input_layer: "conv2d2"
        perceiver_mult: 2
    emo_condition_module:
        output_size: 512
        linear_units: 1024
        attention_heads: 4
        num_blocks: 4
        input_layer: "conv2d2"
        perceiver_mult: 2

semantic_codec:
    codebook_size: 8192
    hidden_size: 1024
    codebook_dim: 8
    vocos_dim: 384
    vocos_intermediate_dim: 2048
    vocos_num_layers: 12

s2mel:
    preprocess_params:
        sr: 22050
        spect_params:
            n_fft: 1024
            win_length: 1024
            hop_length: 256
            n_mels: 80
            fmin: 0
            fmax: "None"
    dit_type: "DiT"
    reg_loss_type: "l1"
    style_encoder:
        dim: 192
    length_regulator:
        channels: 512
        is_discrete: false
        in_channels: 1024
        content_codebook_size: 2048
        sampling_ratios: [1, 1, 1, 1]
        vector_quantize: false
        n_codebooks: 1
        quantizer_dropout: 0.0
        f0_condition: false
        n_f0_bins: 512
    DiT:
        hidden_dim: 512
        num_heads: 8
        depth: 13
        class_dropout_prob: 0.1
        block_size: 8192
        in_channels: 80
        style_condition: true
        final_layer_type: 'wavenet'
        target: 'mel'
        content_dim: 512
        content_codebook_size: 1024
        content_type: 'discrete'
        f0_condition: false
        n_f0_bins: 512
        content_codebooks: 1
        is_causal: false
        long_skip_connection: true
        zero_prompt_speech_token: false
        time_as_token: false
        style_as_token: false
        uvit_skip_connection: true
        add_resblock_in_transformer: false
    wavenet:
        hidden_dim: 512
        num_layers: 8
        kernel_size: 5
        dilation_rate: 1
        p_dropout: 0.2
        style_condition: true

gpt_checkpoint: gpt.pth
w2v_stat: wav2vec2bert_stats.pt
s2mel_checkpoint: s2mel.pth
emo_matrix: feat2.pt 
spk_matrix: feat1.pt
emo_num: [3, 17, 2, 8, 4, 5, 10, 24]
qwen_emo_path: qwen0.6bemo4-merge/ 
vocoder:
    type: "bigvgan"
    name: "nvidia/bigvgan_v2_22khz_80band_256x"
version: 2.0
"#;
        
        let config: ModelConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.gpt.model_dim, 1280);
        assert_eq!(config.gpt.layers, 24);
        assert_eq!(config.gpt.stop_mel_token, 8193);
    }
}
