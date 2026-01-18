//! NPY file loading for golden data comparison
//!
//! Loads NumPy .npy files for validation against Python outputs.

use anyhow::{Context, Result};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

/// NPY array data
#[derive(Debug, Clone)]
pub struct NpyArray {
    /// Shape of the array
    pub shape: Vec<usize>,
    /// Data type string (e.g., "<f4", "<i8")
    pub dtype: String,
    /// Raw data bytes
    pub data: Vec<u8>,
}

impl NpyArray {
    /// Get total number of elements
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get as f32 vector (assumes float32 dtype)
    pub fn as_f32(&self) -> Result<Vec<f32>> {
        if !self.dtype.contains("f4") && !self.dtype.contains("float32") {
            anyhow::bail!("Expected float32, got {}", self.dtype);
        }

        let floats: Vec<f32> = self.data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        Ok(floats)
    }

    /// Get as i64 vector (assumes int64 dtype)
    pub fn as_i64(&self) -> Result<Vec<i64>> {
        if !self.dtype.contains("i8") && !self.dtype.contains("int64") {
            anyhow::bail!("Expected int64, got {}", self.dtype);
        }

        let ints: Vec<i64> = self.data
            .chunks_exact(8)
            .map(|b| i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
            .collect();

        Ok(ints)
    }

    /// Get as i32 vector (assumes int32 dtype)
    pub fn as_i32(&self) -> Result<Vec<i32>> {
        if !self.dtype.contains("i4") && !self.dtype.contains("int32") {
            anyhow::bail!("Expected int32, got {}", self.dtype);
        }

        let ints: Vec<i32> = self.data
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        Ok(ints)
    }
}

/// Load an NPY file
pub fn load_npy<P: AsRef<Path>>(path: P) -> Result<NpyArray> {
    let path = path.as_ref();
    let file = File::open(path)
        .with_context(|| format!("Failed to open NPY file: {:?}", path))?;
    let mut reader = BufReader::new(file);

    // Read magic number (6 bytes: \x93NUMPY)
    let mut magic = [0u8; 6];
    reader.read_exact(&mut magic)?;
    if &magic[..] != b"\x93NUMPY" {
        anyhow::bail!("Invalid NPY magic number");
    }

    // Read version (2 bytes)
    let mut version = [0u8; 2];
    reader.read_exact(&mut version)?;
    let major = version[0];
    let _minor = version[1];

    // Read header length
    let header_len = if major == 1 {
        let mut len_bytes = [0u8; 2];
        reader.read_exact(&mut len_bytes)?;
        u16::from_le_bytes(len_bytes) as usize
    } else {
        let mut len_bytes = [0u8; 4];
        reader.read_exact(&mut len_bytes)?;
        u32::from_le_bytes(len_bytes) as usize
    };

    // Read header
    let mut header_bytes = vec![0u8; header_len];
    reader.read_exact(&mut header_bytes)?;
    let header = String::from_utf8_lossy(&header_bytes);

    // Parse header (simple parsing for common cases)
    let dtype = parse_dtype(&header)?;
    let shape = parse_shape(&header)?;

    // Calculate data size
    let elem_size = match dtype.as_str() {
        s if s.contains("f4") => 4,
        s if s.contains("f8") => 8,
        s if s.contains("i4") => 4,
        s if s.contains("i8") => 8,
        s if s.contains("i2") => 2,
        s if s.contains("u1") => 1,
        _ => anyhow::bail!("Unsupported dtype: {}", dtype),
    };
    let data_size = shape.iter().product::<usize>() * elem_size;

    // Read data
    let mut data = vec![0u8; data_size];
    reader.read_exact(&mut data)?;

    Ok(NpyArray { shape, dtype, data })
}

/// Load NPY file as f32 vector
pub fn load_npy_f32<P: AsRef<Path>>(path: P) -> Result<(Vec<f32>, Vec<usize>)> {
    let arr = load_npy(path)?;
    let data = arr.as_f32()?;
    Ok((data, arr.shape))
}

/// Load NPY file as i64 vector
pub fn load_npy_i64<P: AsRef<Path>>(path: P) -> Result<(Vec<i64>, Vec<usize>)> {
    let arr = load_npy(path)?;
    let data = arr.as_i64()?;
    Ok((data, arr.shape))
}

/// Parse dtype from NPY header
fn parse_dtype(header: &str) -> Result<String> {
    // Look for 'descr': '<f4' or similar
    let start = header.find("'descr'")
        .or_else(|| header.find("\"descr\""))
        .ok_or_else(|| anyhow::anyhow!("No descr in header"))?;

    let rest = &header[start..];
    let colon = rest.find(':').ok_or_else(|| anyhow::anyhow!("No colon after descr"))?;
    let after_colon = &rest[colon + 1..];

    // Find the dtype string
    let quote_start = after_colon.find(['\'', '"'])
        .ok_or_else(|| anyhow::anyhow!("No dtype string"))?;
    let quote_char = after_colon.chars().nth(quote_start).unwrap();
    let dtype_start = quote_start + 1;
    let dtype_end = after_colon[dtype_start..].find(quote_char)
        .ok_or_else(|| anyhow::anyhow!("Unclosed dtype string"))?;

    Ok(after_colon[dtype_start..dtype_start + dtype_end].to_string())
}

/// Parse shape from NPY header
fn parse_shape(header: &str) -> Result<Vec<usize>> {
    // Look for 'shape': (1, 2, 3) or similar
    let start = header.find("'shape'")
        .or_else(|| header.find("\"shape\""))
        .ok_or_else(|| anyhow::anyhow!("No shape in header"))?;

    let rest = &header[start..];
    let paren_start = rest.find('(')
        .ok_or_else(|| anyhow::anyhow!("No shape tuple"))?;
    let paren_end = rest.find(')')
        .ok_or_else(|| anyhow::anyhow!("Unclosed shape tuple"))?;

    let shape_str = &rest[paren_start + 1..paren_end];

    // Parse comma-separated integers
    let shape: Result<Vec<usize>> = shape_str
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| {
            s.trim()
                .parse::<usize>()
                .map_err(|e| anyhow::anyhow!("Invalid shape element: {}", e))
        })
        .collect();

    shape
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_dtype() {
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (1, 2, 3)}";
        let dtype = parse_dtype(header).unwrap();
        assert_eq!(dtype, "<f4");
    }

    #[test]
    fn test_parse_shape() {
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (1, 2, 3)}";
        let shape = parse_shape(header).unwrap();
        assert_eq!(shape, vec![1, 2, 3]);
    }

    #[test]
    fn test_parse_shape_1d() {
        let header = "{'descr': '<i8', 'fortran_order': False, 'shape': (100,)}";
        let shape = parse_shape(header).unwrap();
        assert_eq!(shape, vec![100]);
    }

    #[test]
    fn test_npy_array_len() {
        let arr = NpyArray {
            shape: vec![2, 3, 4],
            dtype: "<f4".to_string(),
            data: vec![0; 2 * 3 * 4 * 4],
        };
        assert_eq!(arr.len(), 24);
    }

    #[test]
    fn test_npy_array_as_f32() {
        let data: Vec<u8> = vec![0.0f32, 1.0, 2.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let arr = NpyArray {
            shape: vec![3],
            dtype: "<f4".to_string(),
            data,
        };

        let floats = arr.as_f32().unwrap();
        assert_eq!(floats, vec![0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_npy_array_as_i64() {
        let data: Vec<u8> = vec![1i64, 2, 3]
            .iter()
            .flat_map(|i| i.to_le_bytes())
            .collect();

        let arr = NpyArray {
            shape: vec![3],
            dtype: "<i8".to_string(),
            data,
        };

        let ints = arr.as_i64().unwrap();
        assert_eq!(ints, vec![1, 2, 3]);
    }
}
