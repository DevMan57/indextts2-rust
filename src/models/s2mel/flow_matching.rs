//! Flow Matching (CFM) for mel spectrogram synthesis
//!
//! Implements Conditional Flow Matching with:
//! - Euler ODE solver for iterative denoising
//! - Classifier-free guidance (CFG)
//! - Optimal transport interpolation path

use anyhow::Result;
use candle_core::{Device, DType, Tensor};
use crate::utils::parity_dump;

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
    /// * `x` - Current state (batch, 80, time) - [B, C, T] format (Python API)
    /// * `prompt_x` - Reference mel (batch, 80, time) - [B, C, T] format
    /// * `t` - Current timestep (batch,)
    /// * `cond` - Semantic conditioning (batch, time, 512) - [B, T, C] format
    /// * `style` - Speaker style (batch, 192)
    ///
    /// # Returns
    /// * Predicted velocity (batch, 80, time) - [B, C, T] format
    fn compute_velocity(
        &self,
        model: &DiffusionTransformer,
        x: &Tensor,
        prompt_x: &Tensor,
        t: &Tensor,
        cond: &Tensor,
        style: &Tensor,
    ) -> Result<Tensor> {
        // DiT expects [B, T, C] format internally, so transpose inputs
        let x_tc = x.transpose(1, 2)?;            // [B, C, T] -> [B, T, C]
        let prompt_x_tc = prompt_x.transpose(1, 2)?;  // [B, C, T] -> [B, T, C]

        // Model predicts velocity in [B, T, C] format
        let v_tc = model.forward(&x_tc, &prompt_x_tc, t, cond, style)?;

        // Transpose output back to [B, C, T]
        v_tc.transpose(1, 2).map_err(Into::into)
    }

    /// Compute velocity with classifier-free guidance
    ///
    /// Python formula: dphi_dt = (1 + cfg_rate) * v_cond - cfg_rate * v_uncond
    /// This is equivalent to: v_cond + cfg_rate * (v_cond - v_uncond)
    fn compute_velocity_cfg(
        &self,
        model: &DiffusionTransformer,
        x: &Tensor,
        prompt_x: &Tensor,
        t: &Tensor,
        cond: &Tensor,
        style: &Tensor,
    ) -> Result<Tensor> {
        if !self.config.use_cfg || self.config.cfg_rate == 0.0 {
            return self.compute_velocity(model, x, prompt_x, t, cond, style);
        }

        // Conditional velocity (with style and conditioning)
        let v_cond = self.compute_velocity(model, x, prompt_x, t, cond, style)?;

        // Unconditional velocity (zero style, conditioning, AND prompt_x for CFG)
        // Python: stacked_prompt_x = torch.cat([prompt_x, torch.zeros_like(prompt_x)], dim=0)
        // So the unconditional path gets zeros for prompt_x as well
        let zero_style = Tensor::zeros_like(style)?;
        let zero_cond = Tensor::zeros_like(cond)?;
        let zero_prompt_x = Tensor::zeros_like(prompt_x)?;
        let v_uncond = self.compute_velocity(model, x, &zero_prompt_x, t, &zero_cond, &zero_style)?;

        // CFG formula from Python: (1 + rate) * v_cond - rate * v_uncond
        // = v_cond + rate * (v_cond - v_uncond)
        let cfg = self.config.cfg_rate as f64;
        let scaled_cond = (&v_cond * (1.0 + cfg))?;
        let scaled_uncond = (&v_uncond * cfg)?;
        (scaled_cond - scaled_uncond).map_err(Into::into)
    }

    /// Euler step for ODE integration
    ///
    /// x_{t+dt} = x_t + dt * v(x_t, t)
    fn euler_step(
        &self,
        model: &DiffusionTransformer,
        x: &Tensor,
        prompt_x: &Tensor,
        t: f32,
        dt: f32,
        cond: &Tensor,
        style: &Tensor,
    ) -> Result<Tensor> {
        let batch_size = x.dim(0)?;

        // Create timestep tensor
        let t_tensor = Tensor::from_slice(
            &vec![t; batch_size],
            (batch_size,),
            &self.device,
        )?;

        // Compute velocity
        let v = self.compute_velocity_cfg(model, x, prompt_x, &t_tensor, cond, style)?;

        // Standard Euler integration: x_{t+dt} = x_t + dt * v
        // NO velocity scaling - matches Python: x = x + dt * dphi_dt
        (x + (v * dt as f64)?).map_err(Into::into)
    }

    /// Sample mel spectrogram using flow matching
    ///
    /// Integrates the ODE from t=0 (noise) to t=1 (data)
    /// Uses [B, C, T] format (batch, 80, time) to match Python API
    ///
    /// # Arguments
    /// * `model` - DiT model
    /// * `noise` - Initial noise (batch, 80, time) - [B, C, T] format
    /// * `prompt_x` - Reference mel (batch, 80, time) - [B, C, T] format
    /// * `cond` - Semantic conditioning (batch, time, 512) - [B, T, C] format
    /// * `style` - Speaker style (batch, 192)
    /// * `prompt_len` - Number of frames in prompt region (to be preserved from prompt_x)
    ///
    /// # Returns
    /// * Generated mel spectrogram (batch, 80, time) - [B, C, T] format
    pub fn sample(
        &self,
        model: &DiffusionTransformer,
        noise: &Tensor,
        prompt_x: &Tensor,
        cond: &Tensor,
        style: &Tensor,
        prompt_len: usize,
    ) -> Result<Tensor> {
        let num_steps = self.config.num_steps;
        let dt = 1.0 / num_steps as f32;
        // [B, C, T] format: time is dimension 2
        let time_len = noise.dim(2)?;

        parity_dump::dump_tensor_f32("rust_dit_input_noise", noise);
        parity_dump::dump_tensor_f32("rust_dit_input_prompt_x", prompt_x);
        parity_dump::dump_tensor_f32("rust_dit_input_cond", cond);
        parity_dump::dump_tensor_f32("rust_dit_input_style", style);
        parity_dump::dump_usize("rust_dit_prompt_len", prompt_len);
        parity_dump::dump_usize("rust_dit_num_steps", num_steps);
        parity_dump::dump_f32("rust_dit_cfg_rate", self.config.cfg_rate);

        let mut x = noise.clone();

        // Zero out prompt region initially (Python: x[..., :prompt_len] = 0)
        // In [B, C, T] format, we narrow on dimension 2 (time)
        if prompt_len > 0 && prompt_len < time_len {
            // Create mask: zeros for prompt region, keep generation region
            let zeros = Tensor::zeros((1, 80, prompt_len), DType::F32, &self.device)?;
            let gen_part = x.narrow(2, prompt_len, time_len - prompt_len)?;
            x = Tensor::cat(&[zeros, gen_part], 2)?;
        }

        // Euler integration from t=0 to t=1
        for step in 0..num_steps {
            let t = step as f32 / num_steps as f32;
            x = self.euler_step(model, &x, prompt_x, t, dt, cond, style)?;

            if step < 3 {
                let name = format!("rust_dit_step_{step:02}");
                parity_dump::dump_tensor_f32(&name, &x);
            }

            // Zero out prompt region after each step (Python: x[..., :prompt_len] = 0)
            // The prompt region is preserved from prompt_x, not generated
            // In [B, C, T] format, we narrow on dimension 2 (time)
            if prompt_len > 0 && prompt_len < time_len {
                let zeros = Tensor::zeros((1, 80, prompt_len), DType::F32, &self.device)?;
                let gen_part = x.narrow(2, prompt_len, time_len - prompt_len)?;
                x = Tensor::cat(&[zeros, gen_part], 2)?;
            }
        }

        // Copy prompt region from prompt_x into the final output
        // Python: final_mel[..., :prompt_len] = prompt[..., :prompt_len]
        if prompt_len > 0 && prompt_len < time_len {
            let prompt_part = prompt_x.narrow(2, 0, prompt_len)?;
            let gen_part = x.narrow(2, prompt_len, time_len - prompt_len)?;
            x = Tensor::cat(&[prompt_part, gen_part], 2)?;
        }

        Ok(x)
    }

    /// Sample with adaptive step size (Heun's method)
    ///
    /// More accurate than Euler but requires 2 function evaluations per step
    /// Uses [B, C, T] format (batch, 80, time) to match Python API
    ///
    /// # Arguments
    /// * `model` - DiT model
    /// * `noise` - Initial noise (batch, 80, time) - [B, C, T] format
    /// * `prompt_x` - Reference mel (batch, 80, time) - [B, C, T] format
    /// * `cond` - Semantic conditioning (batch, time, 512) - [B, T, C] format
    /// * `style` - Speaker style (batch, 192)
    ///
    /// # Returns
    /// * Generated mel spectrogram (batch, 80, time) - [B, C, T] format
    pub fn sample_heun(
        &self,
        model: &DiffusionTransformer,
        noise: &Tensor,
        prompt_x: &Tensor,
        cond: &Tensor,
        style: &Tensor,
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
            let v1 = self.compute_velocity_cfg(model, &x, prompt_x, &t_tensor, cond, style)?;

            // Euler prediction
            let x_pred = (&x + (&v1 * dt as f64)?)?;

            // Second velocity evaluation at predicted point
            let t_next_tensor = Tensor::from_slice(
                &vec![t_next; batch_size],
                (batch_size,),
                &self.device,
            )?;
            let v2 = self.compute_velocity_cfg(model, &x_pred, prompt_x, &t_next_tensor, cond, style)?;

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
    /// Uses [B, C, T] format (batch, 80, time) to match Python API
    ///
    /// # Arguments
    /// * `model` - DiT model
    /// * `x0` - Noise samples (batch, 80, time) - [B, C, T] format
    /// * `x1` - Target mel spectrograms (batch, 80, time) - [B, C, T] format
    /// * `prompt_x` - Reference mel (batch, 80, time) - [B, C, T] format
    /// * `cond` - Semantic conditioning (batch, time, 512) - [B, T, C] format
    /// * `style` - Speaker style (batch, 192)
    ///
    /// # Returns
    /// * MSE loss
    pub fn compute_loss(
        &self,
        model: &DiffusionTransformer,
        x0: &Tensor,
        x1: &Tensor,
        prompt_x: &Tensor,
        cond: &Tensor,
        style: &Tensor,
    ) -> Result<Tensor> {
        let batch_size = x0.dim(0)?;

        // Sample random timesteps
        let t_vals: Vec<f32> = (0..batch_size)
            .map(|_| rand::random::<f32>())
            .collect();
        let t = Tensor::from_slice(&t_vals, (batch_size,), &self.device)?;

        // Interpolate in [B, C, T] space: x_t = (1-t) * x0 + t * x1
        // t shape: (B,) -> need to expand to (B, 1, 1) for broadcasting
        let t_expanded = t.unsqueeze(1)?.unsqueeze(2)?;
        let t_expanded = t_expanded.broadcast_as(x0.shape())?;
        let one_minus_t = (1.0 - &t_expanded)?;
        let x_t = (one_minus_t.mul(x0)? + t_expanded.mul(x1)?)?;

        // Target velocity in [B, C, T]: x1 - x0
        let target = (x1 - x0)?;

        // Transpose to [B, T, C] for DiT
        let x_t_tc = x_t.transpose(1, 2)?;
        let prompt_x_tc = prompt_x.transpose(1, 2)?;

        // Predicted velocity in [B, T, C]
        let pred_tc = model.forward(&x_t_tc, &prompt_x_tc, &t, cond, style)?;

        // Transpose prediction back to [B, C, T] for loss computation
        let pred = pred_tc.transpose(1, 2)?;

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

        // [B, C, T] format: (batch, 80, time)
        let noise = fm.sample_noise(&[2, 80, 100]).unwrap();
        assert_eq!(noise.dims(), &[2, 80, 100]);
    }

    #[test]
    fn test_flow_matching_sample() {
        let device = Device::Cpu;

        // Create DiT model with random weights
        let mut dit = DiffusionTransformer::new(&device).unwrap();
        dit.initialize_random().unwrap();

        // Use fewer steps for faster test
        let config = FlowMatchingConfig {
            num_steps: 3,
            cfg_rate: 0.7,
            sigma_min: 1e-4,
            use_cfg: false, // Disable CFG for simpler test
        };
        let fm = FlowMatching::with_config(config, &device);

        // [B, C, T] format for noise and prompt_x
        let noise = fm.sample_noise(&[1, 80, 50]).unwrap();
        let prompt_x = Tensor::zeros((1, 80, 50), DType::F32, &device).unwrap();
        // cond stays [B, T, C] as DiT expects this for conditioning
        let cond = Tensor::randn(0.0f32, 1.0, (1, 50, 512), &device).unwrap();
        let style = Tensor::randn(0.0f32, 1.0, (1, 192), &device).unwrap();

        let mel = fm.sample(&dit, &noise, &prompt_x, &cond, &style, 0).unwrap();
        // Output is [B, C, T] = (1, 80, 50)
        assert_eq!(mel.dims3().unwrap(), (1, 80, 50));
    }

    #[test]
    fn test_flow_matching_sample_with_cfg() {
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

        // [B, C, T] format for noise and prompt_x
        let noise = fm.sample_noise(&[1, 80, 20]).unwrap();
        let prompt_x = Tensor::zeros((1, 80, 20), DType::F32, &device).unwrap();
        // cond stays [B, T, C]
        let cond = Tensor::randn(0.0f32, 1.0, (1, 20, 512), &device).unwrap();
        let style = Tensor::randn(0.0f32, 1.0, (1, 192), &device).unwrap();

        let mel = fm.sample(&dit, &noise, &prompt_x, &cond, &style, 5).unwrap();
        // Output is [B, C, T]
        let (batch, channels, time) = mel.dims3().unwrap();
        assert_eq!(batch, 1);
        assert_eq!(channels, 80);
        assert_eq!(time, 20);
    }

    #[test]
    fn test_flow_matching_sample_heun() {
        let device = Device::Cpu;

        let mut dit = DiffusionTransformer::new(&device).unwrap();
        dit.initialize_random().unwrap();

        let config = FlowMatchingConfig {
            num_steps: 3,
            cfg_rate: 0.5,
            sigma_min: 1e-4,
            use_cfg: false,
        };
        let fm = FlowMatching::with_config(config, &device);

        // [B, C, T] format for noise and prompt_x
        let noise = fm.sample_noise(&[1, 80, 30]).unwrap();
        let prompt_x = Tensor::zeros((1, 80, 30), DType::F32, &device).unwrap();
        // cond stays [B, T, C]
        let cond = Tensor::randn(0.0f32, 1.0, (1, 30, 512), &device).unwrap();
        let style = Tensor::randn(0.0f32, 1.0, (1, 192), &device).unwrap();

        let mel = fm.sample_heun(&dit, &noise, &prompt_x, &cond, &style).unwrap();
        // Output is [B, C, T]
        assert_eq!(mel.dims3().unwrap(), (1, 80, 30));
    }

    #[test]
    fn test_compute_loss() {
        let device = Device::Cpu;

        let mut dit = DiffusionTransformer::new(&device).unwrap();
        dit.initialize_random().unwrap();

        let fm = FlowMatching::new(&device);

        // compute_loss uses [B, C, T] format for x0, x1, prompt_x
        let x0 = Tensor::randn(0.0f32, 1.0, (2, 80, 10), &device).unwrap();
        let x1 = Tensor::randn(0.0f32, 1.0, (2, 80, 10), &device).unwrap();
        let prompt_x = Tensor::zeros((2, 80, 10), DType::F32, &device).unwrap();
        // cond stays [B, T, C]
        let cond = Tensor::randn(0.0f32, 1.0, (2, 10, 512), &device).unwrap();
        let style = Tensor::randn(0.0f32, 1.0, (2, 192), &device).unwrap();

        let loss = fm.compute_loss(&dit, &x0, &x1, &prompt_x, &cond, &style).unwrap();
        let loss_val: f32 = loss.to_scalar().unwrap();

        // Loss should be positive
        assert!(loss_val > 0.0);
    }
}
