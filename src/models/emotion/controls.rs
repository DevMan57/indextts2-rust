//! Emotion control helpers (matrix + speaker matrix).
//!
//! Mirrors Python reference behavior for emotion-vector mixing.

use anyhow::{Context, Result};
use candle_core::{safetensors, Device, IndexOp, Tensor, D};
use std::path::Path;

/// Emotion/speaker matrix helper for building emotion vectors from user controls.
pub struct EmotionControls {
    emo_num: Vec<usize>,
    emo_matrix: Vec<Tensor>, // per-category [num_classes, dim]
    spk_matrix: Vec<Tensor>, // per-category [num_classes, dim]
}

impl EmotionControls {
    /// Load emotion and speaker matrices from checkpoint files.
    pub fn load<P: AsRef<Path>>(
        emo_path: P,
        spk_path: P,
        emo_num: Vec<usize>,
        device: &Device,
    ) -> Result<Self> {
        let emo_tensor = load_matrix_tensor(emo_path.as_ref(), device)
            .context("Failed to load emotion matrix")?;
        let spk_tensor = load_matrix_tensor(spk_path.as_ref(), device)
            .context("Failed to load speaker matrix")?;

        let emo_matrix = split_matrix(&emo_tensor, &emo_num)?;
        let spk_matrix = split_matrix(&spk_tensor, &emo_num)?;

        Ok(Self { emo_num, emo_matrix, spk_matrix })
    }

    /// Return configured class counts per emotion category.
    pub fn emo_num(&self) -> &[usize] {
        &self.emo_num
    }

    /// Build emotion vector embedding from weight vector and style embedding.
    /// Returns emovec_mat: (1, dim)
    pub fn build_emovec_from_vector(
        &self,
        weight_vector: &[f32],
        style: &Tensor, // (1, dim)
        use_random: bool,
    ) -> Result<Tensor> {
        if weight_vector.len() != self.emo_num.len() {
            anyhow::bail!(
                "Emotion vector length mismatch: got {}, expected {}",
                weight_vector.len(),
                self.emo_num.len()
            );
        }

        let mut picked: Vec<Tensor> = Vec::with_capacity(self.emo_num.len());
        for (idx, spk_mat) in self.spk_matrix.iter().enumerate() {
            let class_index = if use_random {
                (rand::random::<u32>() as usize) % self.emo_num[idx]
            } else {
                find_most_similar_cosine(style, spk_mat)?
            };
            let emo_mat = &self.emo_matrix[idx];
            let emo_vec = emo_mat.i(class_index)?;
            picked.push(emo_vec.unsqueeze(0)?);
        }

        let emo_matrix = Tensor::cat(&picked.iter().collect::<Vec<_>>(), 0)?;
        let weight = Tensor::from_slice(weight_vector, (1, weight_vector.len()), style.device())?;
        let emovec_mat = weight.matmul(&emo_matrix)?;
        Ok(emovec_mat)
    }
}

fn load_matrix_tensor(path: &Path, device: &Device) -> Result<Tensor> {
    let tensors = safetensors::load(path, device)
        .with_context(|| format!("Failed to load safetensors at {:?}", path))?;
    if let Some(t) = tensors.get("matrix").or(tensors.get("weight")).or(tensors.get("data")) {
        Ok(t.clone())
    } else {
        anyhow::bail!("No 'matrix' or 'weight' tensor found in {:?}", path)
    }
}

fn split_matrix(matrix: &Tensor, emo_num: &[usize]) -> Result<Vec<Tensor>> {
    let total: usize = emo_num.iter().sum();
    let (rows, _dim) = matrix.dims2()?;
    if rows != total {
        anyhow::bail!("Matrix rows {} do not match emo_num sum {}", rows, total);
    }

    let mut out = Vec::with_capacity(emo_num.len());
    let mut offset = 0usize;
    for &n in emo_num {
        let slice = matrix.i(offset..offset + n)?;
        out.push(slice);
        offset += n;
    }
    Ok(out)
}

fn find_most_similar_cosine(query: &Tensor, matrix: &Tensor) -> Result<usize> {
    // query: (1, dim), matrix: (num, dim)
    let query = query.squeeze(0)?;
    let query_norm = query.sqr()?.sum(D::Minus1)?.sqrt()?;
    let matrix_norm = matrix.sqr()?.sum(D::Minus1)?.sqrt()?;
    let dot = matrix.matmul(&query.unsqueeze(1)?)?.squeeze(1)?;

    // Candle does not reliably scalar-broadcast for mul([N], []), so use a host scalar.
    let query_norm_scalar = query_norm.to_scalar::<f32>()?;
    let denom = (matrix_norm * query_norm_scalar as f64)?;
    let sim = dot.broadcast_div(&denom)?;
    let idx = sim.argmax(D::Minus1)?;
    Ok(idx.to_vec0::<u32>()? as usize)
}



