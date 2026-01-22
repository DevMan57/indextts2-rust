//! Weight loading diagnostics for exposing tensor name mismatches
//!
//! When loading safetensors files, mismatched tensor names cause silent
//! fallback to random weights. This module provides visibility into what
//! tensors are available vs. expected for each model component.

use anyhow::Result;
use candle_core::{safetensors, Device, Tensor};
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// Diagnostic report for a single component's weight loading
#[derive(Debug, Clone)]
pub struct ComponentReport {
    /// Name of the component (e.g., "Conformer", "DiT")
    pub component_name: String,
    /// Path to the safetensors file
    pub file_path: String,
    /// All keys available in the safetensors file
    pub available_keys: Vec<String>,
    /// Keys the component tried to load
    pub expected_keys: HashSet<String>,
    /// Keys that were found (intersection)
    pub found_keys: HashSet<String>,
    /// Keys that were expected but missing
    pub missing_keys: HashSet<String>,
    /// Keys in file but not expected (extra/unused)
    pub extra_keys: HashSet<String>,
}

impl ComponentReport {
    /// Calculate the ratio of found keys to expected keys
    pub fn success_rate(&self) -> f32 {
        if self.expected_keys.is_empty() {
            return 1.0;
        }
        self.found_keys.len() as f32 / self.expected_keys.len() as f32
    }

    /// Print a summary of the component's weight loading status
    pub fn print_summary(&self) {
        eprintln!("\n=== {} Weight Loading ===", self.component_name);
        eprintln!("  File: {}", self.file_path);
        eprintln!(
            "  Available in file: {} tensors",
            self.available_keys.len()
        );
        eprintln!(
            "  Expected: {} | Found: {} | Missing: {}",
            self.expected_keys.len(),
            self.found_keys.len(),
            self.missing_keys.len()
        );

        if !self.missing_keys.is_empty() {
            eprintln!("  MISSING:");
            for key in self.missing_keys.iter().take(10) {
                eprintln!("    - {}", key);
            }
            if self.missing_keys.len() > 10 {
                eprintln!("    ... and {} more", self.missing_keys.len() - 10);
            }
        }
    }
}

/// Weight loading wrapper with diagnostics
///
/// Tracks expected vs. found tensors across all model components
/// and provides visibility into silent fallbacks.
pub struct WeightDiagnostics {
    /// Whether to print verbose output
    verbose: bool,
    /// Reports for each component
    reports: Vec<ComponentReport>,
}

impl WeightDiagnostics {
    /// Create a new diagnostics instance
    ///
    /// # Arguments
    /// * `verbose` - If true, print detailed tensor information during loading
    pub fn new(verbose: bool) -> Self {
        Self {
            verbose,
            reports: Vec::new(),
        }
    }

    /// Load safetensors and enumerate all keys
    ///
    /// # Arguments
    /// * `path` - Path to the safetensors file
    /// * `component_name` - Name of the component being loaded (for logging)
    /// * `device` - Device to load tensors on
    ///
    /// # Returns
    /// HashMap of tensor name to Tensor
    pub fn load_safetensors<P: AsRef<Path>>(
        &mut self,
        path: P,
        component_name: &str,
        device: &Device,
    ) -> Result<HashMap<String, Tensor>> {
        let path = path.as_ref();
        let tensors = safetensors::load(path, device)?;

        let available_keys: Vec<String> = tensors.keys().cloned().collect();

        if self.verbose {
            eprintln!(
                "\n[{}] Loaded {} tensors from {:?}",
                component_name,
                available_keys.len(),
                path
            );
            let display_count = available_keys.len().min(5);
            eprintln!("  First {} keys: {:?}", display_count, &available_keys[..display_count]);
        }

        Ok(tensors)
    }

    /// Record expected vs found keys for a component
    ///
    /// Call this after attempting to load weights to track which tensors
    /// were found and which are missing.
    ///
    /// # Arguments
    /// * `component_name` - Name of the component
    /// * `file_path` - Path to the safetensors file
    /// * `available_keys` - All keys present in the file
    /// * `expected_keys` - Keys the component tried to load
    pub fn record_component(
        &mut self,
        component_name: &str,
        file_path: &str,
        available_keys: Vec<String>,
        expected_keys: HashSet<String>,
    ) {
        let available_set: HashSet<String> = available_keys.iter().cloned().collect();

        let found_keys: HashSet<String> = expected_keys
            .intersection(&available_set)
            .cloned()
            .collect();

        let missing_keys: HashSet<String> = expected_keys
            .difference(&found_keys)
            .cloned()
            .collect();

        let extra_keys: HashSet<String> = available_set
            .difference(&expected_keys)
            .cloned()
            .collect();

        let report = ComponentReport {
            component_name: component_name.to_string(),
            file_path: file_path.to_string(),
            available_keys,
            expected_keys,
            found_keys,
            missing_keys: missing_keys.clone(),
            extra_keys,
        };

        // Print summary if verbose or if there are missing keys
        if self.verbose || !missing_keys.is_empty() {
            report.print_summary();
        }

        self.reports.push(report);
    }

    /// Print final summary of all component weight loading
    pub fn print_final_summary(&self) {
        eprintln!("\n=== Weight Loading Summary ===");
        for report in &self.reports {
            let status = if report.missing_keys.is_empty() {
                "OK"
            } else {
                "MISSING"
            };
            eprintln!(
                "  [{}] {}: {:.0}% loaded ({}/{})",
                status,
                report.component_name,
                report.success_rate() * 100.0,
                report.found_keys.len(),
                report.expected_keys.len()
            );
        }
    }

    /// Get all reports
    pub fn reports(&self) -> &[ComponentReport] {
        &self.reports
    }

    /// Check if all components loaded successfully (no missing keys)
    pub fn all_loaded(&self) -> bool {
        self.reports.iter().all(|r| r.missing_keys.is_empty())
    }

    /// Get total number of missing tensors across all components
    pub fn total_missing(&self) -> usize {
        self.reports.iter().map(|r| r.missing_keys.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_component_report_success_rate() {
        let report = ComponentReport {
            component_name: "Test".to_string(),
            file_path: "test.safetensors".to_string(),
            available_keys: vec!["a".to_string(), "b".to_string()],
            expected_keys: HashSet::from(["a".to_string(), "b".to_string(), "c".to_string()]),
            found_keys: HashSet::from(["a".to_string(), "b".to_string()]),
            missing_keys: HashSet::from(["c".to_string()]),
            extra_keys: HashSet::new(),
        };

        assert!((report.success_rate() - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_component_report_success_rate_empty() {
        let report = ComponentReport {
            component_name: "Test".to_string(),
            file_path: "test.safetensors".to_string(),
            available_keys: vec![],
            expected_keys: HashSet::new(),
            found_keys: HashSet::new(),
            missing_keys: HashSet::new(),
            extra_keys: HashSet::new(),
        };

        assert!((report.success_rate() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_weight_diagnostics_new() {
        let diag = WeightDiagnostics::new(true);
        assert!(diag.reports.is_empty());
        assert!(diag.all_loaded());
        assert_eq!(diag.total_missing(), 0);
    }

    #[test]
    fn test_weight_diagnostics_record_component() {
        let mut diag = WeightDiagnostics::new(false);

        diag.record_component(
            "TestComponent",
            "test.safetensors",
            vec!["a".to_string(), "b".to_string()],
            HashSet::from(["a".to_string(), "c".to_string()]),
        );

        assert_eq!(diag.reports.len(), 1);
        assert_eq!(diag.reports[0].found_keys.len(), 1);
        assert_eq!(diag.reports[0].missing_keys.len(), 1);
        assert!(!diag.all_loaded());
        assert_eq!(diag.total_missing(), 1);
    }
}
