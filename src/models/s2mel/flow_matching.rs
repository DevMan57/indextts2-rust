//! Flow Matching (CFM) for mel spectrogram synthesis
//!
//! Implements Conditional Flow Matching with:
//! - Euler ODE solver for iterative denoising
//! - Classifier-free guidance (CFG)
//! - Optimal transport interpolation path

use anyhow::{Context, Result};
use candle_core::{Device, Tensor, DType, D};
use std::path::Path;

use super::dit::DiffusionTransformer;

/// Flow Matching configuration
#[derive(Clone)]
pub struct FlowMatchingConfig {
    /// Number of inference steps
    pub num_steps: usize,
    /// Classifier-free guidance rate
    pub cfg_rate: f32,
    /// Minimum timestep (sigma_min)
    pub sigma_min: f32,
    /// Whether to use CFG
    pub use_cfg: bool,
}

impl Default for FlowMatchingConfig {
    fn default() -> Self {
        Self {
            num_steps: 25,
            cfg_rate: 0.7,
            sigma_min: 1e-4,
            use_cfg: true,
        }
    }
}

/// Flow Matching sampler for mel generation
pub struct FlowMatching {
    device: Device,
    config: FlowMatchingConfig,
}

impl FlowMatching {
    /// Create with default config
    pub fn new(device: &Device) -> Self {
        Self::with_config(FlowMatchingConfig::default(), device)
    }

    /// Create with custom config
    pub fn with_config(config: FlowMatchingConfig, device: &Device) -> Self {
        Self {
            device: device.clone(),
            config,
        }
    }

    /// Compute velocity for flow matching
    ///
    /// In CFM, the velocity field v(x, t) pushes samples from noise to data.
    /// The optimal transport path is: x_t = (1-t) * x_0 + t * x_1
    /// where x_0 is noise and x_1 is target
    ///
    /// # Arguments
    /// * `model` - DiT model for velocity prediction
    /// * `x` - Current state (batch, seq_len, mel_channels)
    /// * `t` - Current timestep (batch,)
    /// * `content` - Content conditioning
    /// * `style` - Optional style conditioning
    ///
    /// # Returns
    /// * Predicted velocity (batch, seq_len, mel_channels)
    fn compute_velocity(
        &self,
        model: &DiffusionTransformer,
        x: &Tensor,
        t: &Tensor,
        content: &Tensor,
        style: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Model predicts velocity directly
        model.forward(x, t, content, style)
    }

    /// Compute velocity with classifier-free guidance
    fn compute_velocity_cfg(
        &self,
        model: &DiffusionTransformer,
        x: &Tensor,
        t: &Tensor,
        content: &Tensor,
        style: Option<&Tensor>,
    ) -> Result<Tensor> {
        if !self.config.use_cfg || self.config.cfg_rate == 1.0 {
            return self.compute_velocity(model, x, t, content, style);
        }

        // Conditional velocity
        let v_cond = self.compute_velocity(model, x, t, content, style)?;

        // Unconditional velocity (no style)
        let v_uncond = self.compute_velocity(model, x, t, content, None)?;

        // CFG: v = v_uncond + cfg_rate * (v_cond - v_uncond)
        let cfg = self.config.cfg_rate as f64;
        let diff = (&v_cond - &v_uncond)?;
        (&v_uncond + (diff * cfg)?).map_err(Into::into)
    }

    /// Euler step for ODE integration
    ///
    /// x_{t+dt} = x_t + dt * v(x_t, t)
    fn euler_step(
        &self,
        model: &DiffusionTransformer,
        x: &Tensor,
        t: f32,
        dt: f32,
        content: &Tensor,
        style: Option<&Tensor>,
    ) -> Result<Tensor> {
        let batch_size = x.dim(0)?;

        // Create timestep tensor
        let t_tensor = Tensor::from_slice(
            &vec![t; batch_size],
            (batch_size,),
            &self.device,
        )?;

        // Compute velocity
        let v = self.compute_velocity_cfg(model, x, &t_tensor, content, style)?;

        // Euler integration
        (x + (v * dt as f64)?).map_err(Into::into)
    }

    /// Sample mel spectrogram using flow matching
    ///
    /// Integrates the ODE from t=0 (noise) to t=1 (data)
    ///
    /// # Arguments
    /// * `model` - DiT model
    /// * `noise` - Initial noise (batch, seq_len, mel_channels)
    /// * `content` - Content features (batch, seq_len, content_dim)
    /// * `style` - Optional style embedding (batch, style_dim)
    ///
    /// # Returns
    /// * Generated mel spectrogram (batch, seq_len, mel_channels)
    pub fn sample(
        &self,
        model: &DiffusionTransformer,
        noise: &Tensor,
        content: &Tensor,
        style: Option<&Tensor>,
    ) -> Result<Tensor> {
        let num_steps = self.config.num_steps;
        let dt = 1.0 / num_steps as f32;

        let mut x = noise.clone();

        // Euler integration from t=0 to t=1
        for step in 0..num_steps {
            let t = step as f32 / num_steps as f32;
            x = self.euler_step(model, &x, t, dt, content, style)?;
        }

        Ok(x)
    }

    /// Sample with adaptive step size (Heun's method)
    ///
    /// More accurate than Euler but requires 2 function evaluations per step
    pub fn sample_heun(
        &self,
        model: &DiffusionTransformer,
        noise: &Tensor,
        content: &Tensor,
        style: Option<&Tensor>,
    ) -> Result<Tensor> {
        let num_steps = self.config.num_steps;
        let dt = 1.0 / num_steps as f32;
        let batch_size = noise.dim(0)?;

        let mut x = noise.clone();

        for step in 0..num_steps {
            let t = step as f32 / num_steps as f32;
            let t_next = (step + 1) as f32 / num_steps as f32;

            // First velocity evaluation
            let t_tensor = Tensor::from_slice(
                &vec![t; batch_size],
                (batch_size,),
                &self.device,
            )?;
            let v1 = self.compute_velocity_cfg(model, &x, &t_tensor, content, style)?;

            // Euler prediction
            let x_pred = (&x + (&v1 * dt as f64)?)?;

            // Second velocity evaluation at predicted point
            let t_next_tensor = Tensor::from_slice(
                &vec![t_next; batch_size],
                (batch_size,),
                &self.device,
            )?;
            let v2 = self.compute_velocity_cfg(model, &x_pred, &t_next_tensor, content, style)?;

            // Heun's correction: x = x + dt * (v1 + v2) / 2
            let v_avg = ((&v1 + &v2)? * 0.5)?;
            x = (&x + (v_avg * dt as f64)?)?;
        }

        Ok(x)
    }

    /// Generate initial noise
    pub fn sample_noise(&self, shape: &[usize]) -> Result<Tensor> {
        Tensor::randn(0.0f32, 1.0, shape, &self.device).map_err(Into::into)
    }

    /// Compute training loss (for reference)
    ///
    /// CFM loss: ||v_theta(x_t, t) - (x_1 - x_0)||^2
    ///
    /// # Arguments
    /// * `model` - DiT model
    /// * `x0` - Noise samples
    /// * `x1` - Target mel spectrograms
    /// * `content` - Content conditioning
    /// * `style` - Style conditioning
    ///
    /// # Returns
    /// * MSE loss
    pub fn compute_loss(
        &self,
        model: &DiffusionTransformer,
        x0: &Tensor,
        x1: &Tensor,
        content: &Tensor,
        style: Option<&Tensor>,
    ) -> Result<Tensor> {
        let batch_size = x0.dim(0)?;

        // Sample random timesteps
        let t_vals: Vec<f32> = (0..batch_size)
            .map(|_| rand::random::<f32>())
            .collect();
        let t = Tensor::from_slice(&t_vals, (batch_size,), &self.device)?;

        // Interpolate: x_t = (1-t) * x0 + t * x1
        let t_expanded = t.unsqueeze(1)?.unsqueeze(2)?;
        let t_expanded = t_expanded.broadcast_as(x0.shape())?;
        let one_minus_t = (1.0 - &t_expanded)?;
        let x_t = (one_minus_t.mul(x0)? + t_expanded.mul(x1)?)?;

        // Target velocity: x1 - x0
        let target = (x1 - x0)?;

        // Predicted velocity
        let pred = model.forward(&x_t, &t, content, style)?;

        // MSE loss
        let diff = (&pred - &target)?;
        let sq = diff.sqr()?;
        sq.mean_all().map_err(Into::into)
    }

    /// Get number of steps
    pub fn num_steps(&self) -> usize {
        self.config.num_steps
    }

    /// Get CFG rate
    pub fn cfg_rate(&self) -> f32 {
        self.config.cfg_rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flow_matching_config_default() {
        let config = FlowMatchingConfig::default();
        assert_eq!(config.num_steps, 25);
        assert!((config.cfg_rate - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_flow_matching_new() {
        let device = Device::Cpu;
        let fm = FlowMatching::new(&device);
        assert_eq!(fm.num_steps(), 25);
        assert!((fm.cfg_rate() - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_sample_noise() {
        let device = Device::Cpu;
        let fm = FlowMatching::new(&device);

        let noise = fm.sample_noise(&[2, 100, 80]).unwrap();
        assert_eq!(noise.dims(), &[2, 100, 80]);
    }

    #[test]
    fn test_flow_matching_sample() {
        let device = Device::Cpu;

        // Create DiT model (uninitialized returns input)
        let dit = DiffusionTransformer::new(&device).unwrap();

        // Use fewer steps for faster test
        let config = FlowMatchingConfig {
            num_steps: 5,
            cfg_rate: 0.7,
            sigma_min: 1e-4,
            use_cfg: false, // Disable CFG for simpler test
        };
        let fm = FlowMatching::with_config(config, &device);

        let noise = fm.sample_noise(&[1, 50, 80]).unwrap();
        let content = Tensor::randn(0.0f32, 1.0, (1, 50, 512), &device).unwrap();

        let mel = fm.sample(&dit, &noise, &content, None).unwrap();
        assert_eq!(mel.dims3().unwrap(), (1, 50, 80));
    }

    #[test]
    fn test_flow_matching_sample_with_style() {
        let device = Device::Cpu;

        let mut dit = DiffusionTransformer::new(&device).unwrap();
        dit.initialize_random().unwrap();

        let config = FlowMatchingConfig {
            num_steps: 3,
            cfg_rate: 0.7,
            sigma_min: 1e-4,
            use_cfg: true,
        };
        let fm = FlowMatching::with_config(config, &device);

        let noise = fm.sample_noise(&[1, 20, 80]).unwrap();
        let content = Tensor::randn(0.0f32, 1.0, (1, 20, 512), &device).unwrap();
        let style = Tensor::randn(0.0f32, 1.0, (1, 192), &device).unwrap();

        let mel = fm.sample(&dit, &noise, &content, Some(&style)).unwrap();
        let (batch, len, channels) = mel.dims3().unwrap();
        assert_eq!(batch, 1);
        assert_eq!(len, 20);
        assert_eq!(channels, 80);
    }

    #[test]
    fn test_flow_matching_sample_heun() {
        let device = Device::Cpu;

        let dit = DiffusionTransformer::new(&device).unwrap();

        let config = FlowMatchingConfig {
            num_steps: 3,
            cfg_rate: 0.5,
            sigma_min: 1e-4,
            use_cfg: false,
        };
        let fm = FlowMatching::with_config(config, &device);

        let noise = fm.sample_noise(&[1, 30, 80]).unwrap();
        let content = Tensor::randn(0.0f32, 1.0, (1, 30, 512), &device).unwrap();

        let mel = fm.sample_heun(&dit, &noise, &content, None).unwrap();
        assert_eq!(mel.dims3().unwrap(), (1, 30, 80));
    }

    #[test]
    fn test_compute_loss() {
        let device = Device::Cpu;

        let mut dit = DiffusionTransformer::new(&device).unwrap();
        dit.initialize_random().unwrap();

        let fm = FlowMatching::new(&device);

        let x0 = Tensor::randn(0.0f32, 1.0, (2, 10, 80), &device).unwrap();
        let x1 = Tensor::randn(0.0f32, 1.0, (2, 10, 80), &device).unwrap();
        let content = Tensor::randn(0.0f32, 1.0, (2, 10, 512), &device).unwrap();

        let loss = fm.compute_loss(&dit, &x0, &x1, &content, None).unwrap();
        let loss_val: f32 = loss.to_scalar().unwrap();

        // Loss should be positive
        assert!(loss_val > 0.0);
    }
}
