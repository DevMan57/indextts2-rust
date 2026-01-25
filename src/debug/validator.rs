//! Tensor validation against golden reference data
//!
//! Compares Rust outputs to Python golden data with configurable tolerances.

use anyhow::{Context, Result};
use candle_core::Tensor;
use std::path::Path;

use super::npy_loader::{load_npy_f32, load_npy_i64};

/// Validation configuration
#[derive(Clone)]
pub struct ValidationConfig {
    /// Absolute tolerance for float comparisons
    pub atol: f32,
    /// Relative tolerance for float comparisons
    pub rtol: f32,
    /// Whether to print detailed diffs
    pub verbose: bool,
    /// Maximum number of differences to print
    pub max_diffs: usize,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            atol: 1e-4,
            rtol: 1e-3,
            verbose: false,
            max_diffs: 10,
        }
    }
}

/// Result of validation
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Name of validated component
    pub name: String,
    /// Whether validation passed
    pub passed: bool,
    /// Shape matches
    pub shape_match: bool,
    /// Expected shape
    pub expected_shape: Vec<usize>,
    /// Actual shape
    pub actual_shape: Vec<usize>,
    /// Maximum absolute difference
    pub max_abs_diff: f32,
    /// Mean absolute difference
    pub mean_abs_diff: f32,
    /// Number of elements that differ beyond tolerance
    pub num_diffs: usize,
    /// Total number of elements
    pub total_elements: usize,
    /// Error message if any
    pub error: Option<String>,
}

impl ValidationResult {
    /// Create a passing result
    pub fn pass(name: &str, shape: Vec<usize>) -> Self {
        Self {
            name: name.to_string(),
            passed: true,
            shape_match: true,
            expected_shape: shape.clone(),
            actual_shape: shape,
            max_abs_diff: 0.0,
            mean_abs_diff: 0.0,
            num_diffs: 0,
            total_elements: 0,
            error: None,
        }
    }

    /// Create a failing result
    pub fn fail(name: &str, error: &str) -> Self {
        Self {
            name: name.to_string(),
            passed: false,
            shape_match: false,
            expected_shape: vec![],
            actual_shape: vec![],
            max_abs_diff: 0.0,
            mean_abs_diff: 0.0,
            num_diffs: 0,
            total_elements: 0,
            error: Some(error.to_string()),
        }
    }

    /// Summary string
    pub fn summary(&self) -> String {
        if self.passed {
            format!(
                "[PASS] {} - shape {:?}, max_diff={:.2e}, mean_diff={:.2e}",
                self.name, self.actual_shape, self.max_abs_diff, self.mean_abs_diff
            )
        } else if let Some(ref err) = self.error {
            format!("[FAIL] {} - {}", self.name, err)
        } else {
            format!(
                "[FAIL] {} - shape mismatch: expected {:?}, got {:?}, {} diffs of {} elements",
                self.name, self.expected_shape, self.actual_shape,
                self.num_diffs, self.total_elements
            )
        }
    }
}

/// Validator for comparing Rust outputs to Python golden data
pub struct Validator {
    golden_dir: std::path::PathBuf,
    config: ValidationConfig,
    results: Vec<ValidationResult>,
}

impl Validator {
    /// Create a new validator
    pub fn new<P: AsRef<Path>>(golden_dir: P) -> Self {
        Self::with_config(golden_dir, ValidationConfig::default())
    }

    /// Create with custom config
    pub fn with_config<P: AsRef<Path>>(golden_dir: P, config: ValidationConfig) -> Self {
        Self {
            golden_dir: golden_dir.as_ref().to_path_buf(),
            config,
            results: Vec::new(),
        }
    }

    /// Validate a tensor against golden data
    pub fn validate_tensor(&mut self, name: &str, actual: &Tensor, subdir: &str) -> Result<ValidationResult> {
        let golden_path = self.golden_dir.join(subdir).join(format!("{}.npy", name));

        if !golden_path.exists() {
            let result = ValidationResult::fail(name, &format!("Golden file not found: {:?}", golden_path));
            self.results.push(result.clone());
            return Ok(result);
        }

        // Load golden data
        let (golden_data, golden_shape) = load_npy_f32(&golden_path)
            .with_context(|| format!("Failed to load golden data: {:?}", golden_path))?;

        // Get actual data
        let actual_shape: Vec<usize> = actual.dims().to_vec();
        let actual_data: Vec<f32> = actual.flatten_all()?.to_vec1()?;

        // Compare shapes
        let shape_match = golden_shape == actual_shape;

        if !shape_match {
            let result = ValidationResult {
                name: name.to_string(),
                passed: false,
                shape_match: false,
                expected_shape: golden_shape,
                actual_shape,
                max_abs_diff: 0.0,
                mean_abs_diff: 0.0,
                num_diffs: 0,
                total_elements: golden_data.len(),
                error: Some("Shape mismatch".to_string()),
            };
            self.results.push(result.clone());
            return Ok(result);
        }

        // Compare values
        let (max_abs_diff, mean_abs_diff, num_diffs, diffs) =
            self.compare_values(&golden_data, &actual_data);

        let passed = num_diffs == 0;

        if self.config.verbose && !passed {
            println!("Differences in {}:", name);
            for (i, (idx, expected, actual, diff)) in diffs.iter().enumerate() {
                if i >= self.config.max_diffs {
                    println!("  ... and {} more", diffs.len() - i);
                    break;
                }
                println!("  [{}] expected={:.6}, actual={:.6}, diff={:.2e}",
                        idx, expected, actual, diff);
            }
        }

        let result = ValidationResult {
            name: name.to_string(),
            passed,
            shape_match: true,
            expected_shape: golden_shape,
            actual_shape,
            max_abs_diff,
            mean_abs_diff,
            num_diffs,
            total_elements: golden_data.len(),
            error: None,
        };

        self.results.push(result.clone());
        Ok(result)
    }

    /// Validate integer array (e.g., tokens, mel codes)
    pub fn validate_i64(&mut self, name: &str, actual: &[i64], subdir: &str) -> Result<ValidationResult> {
        let golden_path = self.golden_dir.join(subdir).join(format!("{}.npy", name));

        if !golden_path.exists() {
            let result = ValidationResult::fail(name, &format!("Golden file not found: {:?}", golden_path));
            self.results.push(result.clone());
            return Ok(result);
        }

        let (golden_data, golden_shape) = load_npy_i64(&golden_path)?;

        // Compare
        let shape_match = golden_data.len() == actual.len();
        let num_diffs = golden_data.iter()
            .zip(actual.iter())
            .filter(|(a, b)| a != b)
            .count();

        let passed = shape_match && num_diffs == 0;

        let result = ValidationResult {
            name: name.to_string(),
            passed,
            shape_match,
            expected_shape: golden_shape,
            actual_shape: vec![actual.len()],
            max_abs_diff: num_diffs as f32,
            mean_abs_diff: num_diffs as f32 / actual.len().max(1) as f32,
            num_diffs,
            total_elements: golden_data.len(),
            error: None,
        };

        self.results.push(result.clone());
        Ok(result)
    }

    /// Compare float values with tolerance
    fn compare_values(&self, expected: &[f32], actual: &[f32]) -> (f32, f32, usize, Vec<(usize, f32, f32, f32)>) {
        let mut max_diff = 0.0f32;
        let mut sum_diff = 0.0f32;
        let mut diffs = Vec::new();

        for (i, (&e, &a)) in expected.iter().zip(actual.iter()).enumerate() {
            let diff = (e - a).abs();
            max_diff = max_diff.max(diff);
            sum_diff += diff;

            // Check if beyond tolerance
            let tol = self.config.atol + self.config.rtol * e.abs();
            if diff > tol {
                diffs.push((i, e, a, diff));
            }
        }

        let mean_diff = sum_diff / expected.len().max(1) as f32;
        (max_diff, mean_diff, diffs.len(), diffs)
    }

    /// Get all validation results
    pub fn results(&self) -> &[ValidationResult] {
        &self.results
    }

    /// Check if all validations passed
    pub fn all_passed(&self) -> bool {
        self.results.iter().all(|r| r.passed)
    }

    /// Print summary
    pub fn print_summary(&self) {
        println!("\n=== Validation Summary ===\n");

        let passed = self.results.iter().filter(|r| r.passed).count();
        let total = self.results.len();

        for result in &self.results {
            println!("{}", result.summary());
        }

        println!("\nTotal: {}/{} passed", passed, total);

        if passed == total {
            println!("✓ All validations passed!");
        } else {
            println!("✗ {} validation(s) failed", total - passed);
        }
    }

    /// Clear results
    pub fn clear(&mut self) {
        self.results.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_config_default() {
        let config = ValidationConfig::default();
        assert_eq!(config.atol, 1e-4);
        assert_eq!(config.rtol, 1e-3);
    }

    #[test]
    fn test_validation_result_pass() {
        let result = ValidationResult::pass("test", vec![1, 2, 3]);
        assert!(result.passed);
        assert!(result.shape_match);
    }

    #[test]
    fn test_validation_result_fail() {
        let result = ValidationResult::fail("test", "error message");
        assert!(!result.passed);
        assert!(result.error.is_some());
    }

    #[test]
    fn test_validation_result_summary() {
        let result = ValidationResult::pass("test", vec![1, 2, 3]);
        let summary = result.summary();
        assert!(summary.contains("[PASS]"));
        assert!(summary.contains("test"));
    }

    #[test]
    fn test_validator_compare_values() {
        let config = ValidationConfig {
            atol: 1e-4,
            rtol: 1e-3,
            verbose: false,
            max_diffs: 10,
        };
        let validator = Validator::with_config(".", config);

        // Identical values
        let expected = vec![1.0, 2.0, 3.0];
        let actual = vec![1.0, 2.0, 3.0];
        let (max_diff, _mean_diff, num_diffs, _) = validator.compare_values(&expected, &actual);
        assert_eq!(max_diff, 0.0);
        assert_eq!(num_diffs, 0);

        // Small differences within tolerance
        let actual = vec![1.0001, 2.0001, 3.0001];
        let (max_diff, _, num_diffs, _) = validator.compare_values(&expected, &actual);
        assert!(max_diff < 0.001);
        assert_eq!(num_diffs, 0);

        // Large differences
        let actual = vec![1.1, 2.1, 3.1];
        let (_, _, num_diffs, _) = validator.compare_values(&expected, &actual);
        assert_eq!(num_diffs, 3);
    }
}
