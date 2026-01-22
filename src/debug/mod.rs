//! Debug and validation utilities
//!
//! Tools for comparing Rust implementation against Python reference:
//! - NPY file loading
//! - Tensor comparison with tolerance
//! - Layer-by-layer validation
//! - Weight loading diagnostics

mod validator;
mod npy_loader;
mod weight_diagnostics;

pub use validator::{Validator, ValidationResult, ValidationConfig};
pub use npy_loader::{load_npy, load_npy_f32, load_npy_i64, NpyArray};
pub use weight_diagnostics::{WeightDiagnostics, ComponentReport};
