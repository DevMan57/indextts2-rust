//! Utility functions and helpers for IndexTTS2
//!
//! This module provides common utilities used across the crate.

use std::env;
use std::fs;
use std::path::PathBuf;

/// Tensor utilities
pub mod tensor_utils {
    use candle_core::{DType, Device, Result, Tensor};

    /// Create a causal mask as u8 tensor (1 = attend, 0 = mask)
    ///
    /// For autoregressive generation, position i can attend to positions <= i
    pub fn create_causal_mask_u8(
        query_len: usize,
        key_len: usize,
        device: &Device,
    ) -> Result<Tensor> {
        let start_pos = key_len.saturating_sub(query_len);
        let mut mask_data = vec![0u8; query_len * key_len];

        for q in 0..query_len {
            for k in 0..key_len {
                if k <= (start_pos + q) {
                    mask_data[q * key_len + k] = 1;
                }
            }
        }

        let mask = Tensor::from_slice(&mask_data, (query_len, key_len), device)?;
        mask.unsqueeze(0)?.unsqueeze(0)
    }

    /// Convert u8 mask to f32 mask for attention (1.0 = attend, -inf = mask)
    pub fn mask_to_attention_bias(mask: &Tensor) -> Result<Tensor> {
        // Where mask is 1, return 0.0; where mask is 0, return -inf
        let mask_f32 = mask.to_dtype(DType::F32)?;
        let neg_inf = Tensor::new(f32::NEG_INFINITY, mask.device())?;
        let _zeros = Tensor::zeros_like(&mask_f32)?;
        
        // (1 - mask) * -inf: 0 where mask=1, -inf where mask=0
        let inv_mask = (Tensor::ones_like(&mask_f32)? - &mask_f32)?;
        let bias = inv_mask.broadcast_mul(&neg_inf.broadcast_as(inv_mask.shape())?)?;
        
        Ok(bias)
    }
}

/// String utilities
pub mod string_utils {
    /// Join a vector of strings with a separator
    pub fn join_strings(strings: &[String], separator: &str) -> String {
        strings.join(separator)
    }
}

/// Optional parity dump helpers.
///
/// Enable by setting `INDEXTTS2_PARITY_DIR` to a writable directory.
/// Files are written as `<name>.bin` + `<name>.json` metadata.
pub mod parity_dump {
    use super::*;
    use candle_core::{DType, Tensor};
    use serde_json::json;
    use std::io::Write;

    fn base_dir() -> Option<PathBuf> {
        let value = env::var("INDEXTTS2_PARITY_DIR").ok()?;
        if value.trim().is_empty() {
            return None;
        }
        Some(PathBuf::from(value))
    }

    fn write_blob(name: &str, dtype: &str, shape: &[usize], bytes: &[u8]) -> std::io::Result<()> {
        let Some(dir) = base_dir() else {
            return Ok(());
        };
        fs::create_dir_all(&dir)?;
        let bin_path = dir.join(format!("{name}.bin"));
        let meta_path = dir.join(format!("{name}.json"));

        let mut f = fs::File::create(&bin_path)?;
        f.write_all(bytes)?;

        let meta = json!({
            "name": name,
            "dtype": dtype,
            "shape": shape,
            "numel": shape.iter().product::<usize>(),
            "bin": bin_path.file_name().and_then(|s| s.to_str()).unwrap_or_default(),
        });
        fs::write(meta_path, serde_json::to_vec_pretty(&meta)?)?;
        Ok(())
    }

    /// Dump a tensor as f32 binary + JSON metadata.
    pub fn dump_tensor_f32(name: &str, tensor: &Tensor) {
        let res = (|| -> anyhow::Result<()> {
            let t = if tensor.dtype() == DType::F32 {
                tensor.clone()
            } else {
                tensor.to_dtype(DType::F32)?
            };
            let shape = t.dims().to_vec();
            let values: Vec<f32> = t.flatten_all()?.to_vec1()?;
            let mut bytes = Vec::with_capacity(values.len() * 4);
            for v in values {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            write_blob(name, "f32", &shape, &bytes)?;
            Ok(())
        })();
        if let Err(e) = res {
            eprintln!("WARNING: parity dump failed for {name}: {e}");
        }
    }

    /// Dump a tensor as i64 binary + JSON metadata.
    pub fn dump_tensor_i64(name: &str, tensor: &Tensor) {
        let res = (|| -> anyhow::Result<()> {
            let t = if tensor.dtype() == DType::I64 {
                tensor.clone()
            } else {
                tensor.to_dtype(DType::I64)?
            };
            let shape = t.dims().to_vec();
            let values: Vec<i64> = t.flatten_all()?.to_vec1()?;
            let mut bytes = Vec::with_capacity(values.len() * 8);
            for v in values {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            write_blob(name, "i64", &shape, &bytes)?;
            Ok(())
        })();
        if let Err(e) = res {
            eprintln!("WARNING: parity dump failed for {name}: {e}");
        }
    }

    /// Dump raw u32 slice as binary + metadata.
    pub fn dump_u32_slice(name: &str, values: &[u32]) {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        if let Err(e) = write_blob(name, "u32", &[values.len()], &bytes) {
            eprintln!("WARNING: parity dump failed for {name}: {e}");
        }
    }

    /// Dump a scalar usize as a tiny text file.
    pub fn dump_usize(name: &str, value: usize) {
        let Some(dir) = base_dir() else {
            return;
        };
        if let Err(e) = fs::create_dir_all(&dir)
            .and_then(|_| fs::write(dir.join(format!("{name}.txt")), value.to_string()))
        {
            eprintln!("WARNING: parity dump failed for {name}: {e}");
        }
    }

    /// Dump a scalar f32 as a tiny text file.
    pub fn dump_f32(name: &str, value: f32) {
        let Some(dir) = base_dir() else {
            return;
        };
        if let Err(e) = fs::create_dir_all(&dir)
            .and_then(|_| fs::write(dir.join(format!("{name}.txt")), value.to_string()))
        {
            eprintln!("WARNING: parity dump failed for {name}: {e}");
        }
    }
}
