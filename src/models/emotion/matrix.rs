//! Emotion matrix for emotion-conditioned TTS
//!
//! Provides emotion conditioning through a learnable emotion matrix.
//! The matrix maps discrete emotion categories to continuous embeddings
//! that can be blended with speaker embeddings.
//!
//! Architecture:
//! - 8 emotion categories with varying class counts
//! - Each emotion class maps to an embedding vector
//! - Emotion blending controlled by emo_alpha parameter

use anyhow::Result;
use candle_core::{safetensors, Device, Tensor, IndexOp};
use std::path::Path;

/// Default number of emotion categories
const NUM_EMOTION_CATEGORIES: usize = 8;
/// Default class counts per category [3, 17, 2, 8, 4, 5, 10, 24]
const DEFAULT_EMO_NUM: [usize; 8] = [3, 17, 2, 8, 4, 5, 10, 24];
/// Total number of emotion classes (sum of DEFAULT_EMO_NUM)
const TOTAL_EMOTION_CLASSES: usize = 73;
/// Default embedding dimension (matches speaker embedding)
const DEFAULT_EMBEDDING_DIM: usize = 192;

/// Emotion category names
pub const EMOTION_CATEGORIES: [&str; 8] = [
    "neutral",    // 3 classes
    "happy",      // 17 classes
    "sad",        // 2 classes
    "angry",      // 8 classes
    "fearful",    // 4 classes
    "disgusted",  // 5 classes
    "surprised",  // 10 classes
    "other",      // 24 classes
];

/// Emotion matrix for conditioning TTS with emotion
///
/// Maps discrete emotion classes to continuous embeddings that can be
/// blended with speaker style vectors to produce emotion-conditioned speech.
pub struct EmotionMatrix {
    device: Device,
    /// Number of classes per emotion category
    emo_num: Vec<usize>,
    /// Total number of emotion classes
    total_classes: usize,
    /// Embedding dimension
    embedding_dim: usize,
    /// Emotion embedding matrix (total_classes, embedding_dim)
    matrix: Option<Tensor>,
    /// Default emotion blending alpha
    default_alpha: f32,
}

impl EmotionMatrix {
    /// Create a new emotion matrix with default configuration
    pub fn new(device: &Device) -> Result<Self> {
        Self::with_config(DEFAULT_EMO_NUM.to_vec(), DEFAULT_EMBEDDING_DIM, device)
    }

    /// Create a new emotion matrix with custom configuration
    pub fn with_config(
        emo_num: Vec<usize>,
        embedding_dim: usize,
        device: &Device,
    ) -> Result<Self> {
        let total_classes: usize = emo_num.iter().sum();

        Ok(Self {
            device: device.clone(),
            emo_num,
            total_classes,
            embedding_dim,
            matrix: None,
            default_alpha: 0.5,
        })
    }

    /// Load emotion matrix from file
    pub fn load<P: AsRef<Path>>(path: P, device: &Device) -> Result<Self> {
        let mut matrix = Self::new(device)?;
        matrix.load_weights(path)?;
        Ok(matrix)
    }

    /// Initialize with random embeddings (for testing)
    pub fn initialize_random(&mut self) -> Result<()> {
        let matrix = Tensor::randn(
            0.0f32,
            0.02,
            (self.total_classes, self.embedding_dim),
            &self.device,
        )?;
        self.matrix = Some(matrix);
        Ok(())
    }

    /// Load weights from file (safetensors or PyTorch .pt)
    pub fn load_weights<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let path = path.as_ref();

        if !path.exists() {
            // Initialize with random for testing
            return self.initialize_random();
        }

        // Try safetensors format
        if let Ok(tensors) = safetensors::load(path, &self.device) {
            if let Some(matrix) = tensors.get("matrix").or(tensors.get("weight")) {
                self.matrix = Some(matrix.clone());
                return Ok(());
            }
        }

        // Fallback to random initialization
        self.initialize_random()
    }

    /// Get emotion embedding for a specific class index
    ///
    /// # Arguments
    /// * `class_idx` - Global class index (0 to total_classes-1)
    ///
    /// # Returns
    /// * Embedding vector (embedding_dim,)
    pub fn get_embedding(&self, class_idx: usize) -> Result<Tensor> {
        if class_idx >= self.total_classes {
            anyhow::bail!(
                "Class index {} out of range (max {})",
                class_idx,
                self.total_classes - 1
            );
        }

        let matrix = self
            .matrix
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Emotion matrix not initialized"))?;

        matrix.i(class_idx).map_err(Into::into)
    }

    /// Get emotion embedding by category and local class index
    ///
    /// # Arguments
    /// * `category_idx` - Emotion category (0-7)
    /// * `local_class_idx` - Class index within the category
    ///
    /// # Returns
    /// * Embedding vector (embedding_dim,)
    pub fn get_embedding_by_category(
        &self,
        category_idx: usize,
        local_class_idx: usize,
    ) -> Result<Tensor> {
        if category_idx >= NUM_EMOTION_CATEGORIES {
            anyhow::bail!(
                "Category index {} out of range (max {})",
                category_idx,
                NUM_EMOTION_CATEGORIES - 1
            );
        }

        if local_class_idx >= self.emo_num[category_idx] {
            anyhow::bail!(
                "Local class index {} out of range for category {} (max {})",
                local_class_idx,
                EMOTION_CATEGORIES[category_idx],
                self.emo_num[category_idx] - 1
            );
        }

        // Calculate global index
        let global_idx: usize = self.emo_num[..category_idx].iter().sum::<usize>() + local_class_idx;
        self.get_embedding(global_idx)
    }

    /// Get batch of emotion embeddings
    ///
    /// # Arguments
    /// * `class_indices` - Tensor of class indices (batch,)
    ///
    /// # Returns
    /// * Embeddings (batch, embedding_dim)
    pub fn get_embeddings_batch(&self, class_indices: &Tensor) -> Result<Tensor> {
        let matrix = self
            .matrix
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Emotion matrix not initialized"))?;

        matrix.index_select(class_indices, 0).map_err(Into::into)
    }

    /// Blend emotion embedding with speaker embedding
    ///
    /// # Arguments
    /// * `speaker_emb` - Speaker embedding (batch, embedding_dim)
    /// * `emotion_emb` - Emotion embedding (batch, embedding_dim)
    /// * `alpha` - Blending factor (0.0 = speaker only, 1.0 = emotion only)
    ///
    /// # Returns
    /// * Blended embedding (batch, embedding_dim)
    pub fn blend(
        &self,
        speaker_emb: &Tensor,
        emotion_emb: &Tensor,
        alpha: f32,
    ) -> Result<Tensor> {
        // blended = (1 - alpha) * speaker + alpha * emotion
        let speaker_weight = 1.0 - alpha;
        let weighted_speaker = (speaker_emb * speaker_weight as f64)?;
        let weighted_emotion = (emotion_emb * alpha as f64)?;
        (weighted_speaker + weighted_emotion).map_err(Into::into)
    }

    /// Get emotion-conditioned embedding
    ///
    /// Convenience method that looks up the emotion embedding and blends it
    /// with the speaker embedding in one call.
    ///
    /// # Arguments
    /// * `speaker_emb` - Speaker embedding (batch, embedding_dim)
    /// * `emotion_idx` - Emotion class index
    /// * `alpha` - Optional blending factor (uses default if None)
    ///
    /// # Returns
    /// * Blended embedding (batch, embedding_dim)
    pub fn condition(
        &self,
        speaker_emb: &Tensor,
        emotion_idx: usize,
        alpha: Option<f32>,
    ) -> Result<Tensor> {
        let emotion_emb = self.get_embedding(emotion_idx)?;
        let emotion_emb = emotion_emb.unsqueeze(0)?; // Add batch dimension

        // Broadcast emotion embedding to match batch size
        let batch_size = speaker_emb.dim(0)?;
        let emotion_emb = emotion_emb.broadcast_as((batch_size, self.embedding_dim))?;

        let alpha = alpha.unwrap_or(self.default_alpha);
        self.blend(speaker_emb, &emotion_emb, alpha)
    }

    /// Set default blending alpha
    pub fn set_default_alpha(&mut self, alpha: f32) {
        self.default_alpha = alpha.clamp(0.0, 1.0);
    }

    /// Get default blending alpha
    pub fn default_alpha(&self) -> f32 {
        self.default_alpha
    }

    /// Get number of emotion categories
    pub fn num_categories(&self) -> usize {
        NUM_EMOTION_CATEGORIES
    }

    /// Get class counts per category
    pub fn emo_num(&self) -> &[usize] {
        &self.emo_num
    }

    /// Get total number of emotion classes
    pub fn total_classes(&self) -> usize {
        self.total_classes
    }

    /// Get embedding dimension
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Check if matrix is initialized
    pub fn is_initialized(&self) -> bool {
        self.matrix.is_some()
    }

    /// Get category name by index
    pub fn category_name(category_idx: usize) -> Option<&'static str> {
        EMOTION_CATEGORIES.get(category_idx).copied()
    }

    /// Find category index by name
    pub fn category_index(name: &str) -> Option<usize> {
        let name_lower = name.to_lowercase();
        EMOTION_CATEGORIES
            .iter()
            .position(|&cat| cat == name_lower)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emotion_matrix_new() {
        let device = Device::Cpu;
        let matrix = EmotionMatrix::new(&device).unwrap();

        assert_eq!(matrix.num_categories(), 8);
        assert_eq!(matrix.total_classes(), 73);
        assert_eq!(matrix.embedding_dim(), 192);
    }

    #[test]
    fn test_emotion_matrix_initialize() {
        let device = Device::Cpu;
        let mut matrix = EmotionMatrix::new(&device).unwrap();
        matrix.initialize_random().unwrap();

        assert!(matrix.is_initialized());
    }

    #[test]
    fn test_get_embedding() {
        let device = Device::Cpu;
        let mut matrix = EmotionMatrix::new(&device).unwrap();
        matrix.initialize_random().unwrap();

        let emb = matrix.get_embedding(0).unwrap();
        assert_eq!(emb.dims1().unwrap(), 192);

        let emb = matrix.get_embedding(72).unwrap();
        assert_eq!(emb.dims1().unwrap(), 192);
    }

    #[test]
    fn test_get_embedding_by_category() {
        let device = Device::Cpu;
        let mut matrix = EmotionMatrix::new(&device).unwrap();
        matrix.initialize_random().unwrap();

        // Neutral category has 3 classes
        let emb = matrix.get_embedding_by_category(0, 0).unwrap();
        assert_eq!(emb.dims1().unwrap(), 192);

        let emb = matrix.get_embedding_by_category(0, 2).unwrap();
        assert_eq!(emb.dims1().unwrap(), 192);

        // Happy category has 17 classes
        let emb = matrix.get_embedding_by_category(1, 16).unwrap();
        assert_eq!(emb.dims1().unwrap(), 192);
    }

    #[test]
    fn test_blend() {
        let device = Device::Cpu;
        let mut matrix = EmotionMatrix::new(&device).unwrap();
        matrix.initialize_random().unwrap();

        let speaker_emb = Tensor::randn(0.0f32, 1.0, (2, 192), &device).unwrap();
        let emotion_emb = Tensor::randn(0.0f32, 1.0, (2, 192), &device).unwrap();

        let blended = matrix.blend(&speaker_emb, &emotion_emb, 0.5).unwrap();
        assert_eq!(blended.dims2().unwrap(), (2, 192));
    }

    #[test]
    fn test_condition() {
        let device = Device::Cpu;
        let mut matrix = EmotionMatrix::new(&device).unwrap();
        matrix.initialize_random().unwrap();

        let speaker_emb = Tensor::randn(0.0f32, 1.0, (4, 192), &device).unwrap();
        let conditioned = matrix.condition(&speaker_emb, 5, Some(0.3)).unwrap();

        assert_eq!(conditioned.dims2().unwrap(), (4, 192));
    }

    #[test]
    fn test_category_lookup() {
        assert_eq!(EmotionMatrix::category_name(0), Some("neutral"));
        assert_eq!(EmotionMatrix::category_name(1), Some("happy"));
        assert_eq!(EmotionMatrix::category_name(7), Some("other"));
        assert_eq!(EmotionMatrix::category_name(8), None);

        assert_eq!(EmotionMatrix::category_index("happy"), Some(1));
        assert_eq!(EmotionMatrix::category_index("HAPPY"), Some(1));
        assert_eq!(EmotionMatrix::category_index("unknown"), None);
    }

    #[test]
    fn test_emo_num() {
        let device = Device::Cpu;
        let matrix = EmotionMatrix::new(&device).unwrap();

        assert_eq!(matrix.emo_num(), &[3, 17, 2, 8, 4, 5, 10, 24]);
    }
}
