//! Weight loading diagnostic tool
//!
//! Loads each safetensors checkpoint and reports which tensor keys are present,
//! compared against what the Rust model code expects to load.
//!
//! Usage: cargo run --release --no-default-features --bin diagnose_weights

use anyhow::Result;
use candle_core::{safetensors, Device};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

fn load_and_list_keys(path: &Path, device: &Device) -> Result<BTreeMap<String, Vec<usize>>> {
    let tensors = safetensors::load(path, device)?;
    let mut keys = BTreeMap::new();
    for (name, tensor) in &tensors {
        keys.insert(name.clone(), tensor.dims().to_vec());
    }
    Ok(keys)
}

fn check_keys(
    component: &str,
    checkpoint_keys: &BTreeMap<String, Vec<usize>>,
    expected_keys: &[String],
) {
    println!("\n{}", "=".repeat(60));
    println!("  {} Weight Diagnosis", component);
    println!("{}", "=".repeat(60));

    let mut found = 0;
    let mut missing = Vec::new();

    for key in expected_keys {
        if checkpoint_keys.contains_key(key) {
            found += 1;
        } else {
            missing.push(key.clone());
        }
    }

    // Find unconsumed keys (in checkpoint but not expected)
    let expected_set: BTreeSet<_> = expected_keys.iter().collect();
    let unconsumed: Vec<_> = checkpoint_keys
        .keys()
        .filter(|k| !expected_set.contains(k))
        .collect();

    println!("  Total expected: {}", expected_keys.len());
    println!("  Found:          {}", found);
    println!("  MISSING:        {}", missing.len());
    println!("  Unconsumed:     {}", unconsumed.len());

    if !missing.is_empty() {
        println!("\n  MISSING tensors (Rust expects but checkpoint lacks):");
        for key in &missing {
            println!("    - {}", key);
        }
    }

    if !unconsumed.is_empty() {
        println!("\n  UNCONSUMED tensors (checkpoint has but Rust ignores):");
        for key in &unconsumed {
            if let Some(shape) = checkpoint_keys.get(*key) {
                println!("    + {} {:?}", key, shape);
            }
        }
    }
}

/// Build the list of keys that wav2vec_bert.rs attempts to load
fn wav2vec_bert_expected_keys() -> Vec<String> {
    let mut keys = Vec::new();

    // Feature projection
    keys.push("feature_projection.layer_norm.weight".into());
    keys.push("feature_projection.layer_norm.bias".into());
    keys.push("feature_projection.projection.weight".into());
    keys.push("feature_projection.projection.bias".into());

    // 24 encoder layers
    for i in 0..24 {
        let prefix = format!("encoder.layers.{}", i);

        // Self attention
        let attn = format!("{}.self_attn", prefix);
        for proj in &["linear_q", "linear_k", "linear_v", "linear_out"] {
            keys.push(format!("{}.{}.weight", attn, proj));
            keys.push(format!("{}.{}.bias", attn, proj));
        }
        // Distance embedding
        keys.push(format!("{}.distance_embedding.weight", attn));

        // Layer norms (self_attn_layer_norm, final_layer_norm)
        keys.push(format!("{}.self_attn_layer_norm.weight", prefix));
        keys.push(format!("{}.self_attn_layer_norm.bias", prefix));
        keys.push(format!("{}.final_layer_norm.weight", prefix));
        keys.push(format!("{}.final_layer_norm.bias", prefix));

        // FFN1 (intermediate dense + output dense)
        keys.push(format!("{}.ffn1.intermediate_dense.weight", prefix));
        keys.push(format!("{}.ffn1.intermediate_dense.bias", prefix));
        keys.push(format!("{}.ffn1.output_dense.weight", prefix));
        keys.push(format!("{}.ffn1.output_dense.bias", prefix));
        keys.push(format!("{}.ffn1_layer_norm.weight", prefix));
        keys.push(format!("{}.ffn1_layer_norm.bias", prefix));

        // Conv module
        let conv = format!("{}.conv_module", prefix);
        keys.push(format!("{}.layer_norm.weight", conv));
        keys.push(format!("{}.layer_norm.bias", conv));
        keys.push(format!("{}.pointwise_conv1.weight", conv));
        keys.push(format!("{}.depthwise_conv.weight", conv));
        keys.push(format!("{}.depthwise_layer_norm.weight", conv));
        keys.push(format!("{}.depthwise_layer_norm.bias", conv));
        keys.push(format!("{}.pointwise_conv2.weight", conv));
    }

    keys
}

/// Build the list of keys that conformer.rs attempts to load from gpt.safetensors
fn conformer_expected_keys() -> Vec<String> {
    let mut keys = Vec::new();

    // The code checks for conditioning_encoder.encoders.{i}.self_attn.linear_q.weight
    // to count layers, then loads each block
    for i in 0..6 {
        let prefix = format!("conditioning_encoder.encoders.{}", i);

        // Self attention
        let attn = format!("{}.self_attn", prefix);
        for proj in &["linear_q", "linear_k", "linear_v", "linear_out"] {
            keys.push(format!("{}.{}.weight", attn, proj));
            keys.push(format!("{}.{}.bias", attn, proj));
        }
        // linear_pos
        keys.push(format!("{}.linear_pos.weight", attn));
        // pos_bias_u, pos_bias_v
        keys.push(format!("{}.pos_bias_u", attn));
        keys.push(format!("{}.pos_bias_v", attn));

        // Norms
        keys.push(format!("{}.norm_mha.weight", prefix));
        keys.push(format!("{}.norm_mha.bias", prefix));
        keys.push(format!("{}.norm_ff.weight", prefix));
        keys.push(format!("{}.norm_ff.bias", prefix));
        keys.push(format!("{}.norm_conv.weight", prefix));
        keys.push(format!("{}.norm_conv.bias", prefix));
        keys.push(format!("{}.norm_final.weight", prefix));
        keys.push(format!("{}.norm_final.bias", prefix));

        // Feed forward
        keys.push(format!("{}.feed_forward.w_1.weight", prefix));
        keys.push(format!("{}.feed_forward.w_1.bias", prefix));
        keys.push(format!("{}.feed_forward.w_2.weight", prefix));
        keys.push(format!("{}.feed_forward.w_2.bias", prefix));

        // Conv module
        let conv = format!("{}.conv_module", prefix);
        keys.push(format!("{}.pointwise_conv1.weight", conv));
        keys.push(format!("{}.depthwise_conv.weight", conv));
        keys.push(format!("{}.depthwise_conv.bias", conv));
        keys.push(format!("{}.batch_norm.weight", conv));
        keys.push(format!("{}.batch_norm.bias", conv));
        keys.push(format!("{}.batch_norm.running_mean", conv));
        keys.push(format!("{}.batch_norm.running_var", conv));
        keys.push(format!("{}.pointwise_conv2.weight", conv));
    }

    // Input embedding (conv layer for mel -> hidden)
    keys.push("conditioning_encoder.embedding.weight".into());
    keys.push("conditioning_encoder.embedding.bias".into());

    keys
}

/// Build the list of keys that perceiver.rs attempts to load from gpt.safetensors
fn perceiver_expected_keys() -> Vec<String> {
    let mut keys = Vec::new();

    keys.push("perceiver_encoder.latents".into());
    keys.push("perceiver_encoder.proj_context.weight".into());
    keys.push("perceiver_encoder.proj_context.bias".into());
    keys.push("perceiver_encoder.norm.gamma".into());

    // 2 layers
    for i in 0..2 {
        let prefix = format!("perceiver_encoder.layers.{}", i);
        // Cross attention (index 0)
        keys.push(format!("{}.0.to_q.weight", prefix));
        keys.push(format!("{}.0.to_kv.weight", prefix));
        keys.push(format!("{}.0.to_out.weight", prefix));
        // FFN (index 1)
        keys.push(format!("{}.1.0.weight", prefix));
        keys.push(format!("{}.1.0.bias", prefix));
        keys.push(format!("{}.1.2.weight", prefix));
        keys.push(format!("{}.1.2.bias", prefix));
    }

    keys
}

/// Build the list of keys that dit.rs attempts to load from s2mel.safetensors
///
/// This must match what dit.rs load_weights() actually looks for:
/// - AdaLayerNorm loads {prefix}.norm.weight + {prefix}.project_layer.weight/bias
/// - skip_linear, conv1, res_projection use plain weight (not weight_v/weight_g)
/// - conv2 uses plain weight [80, 512, 1]
/// - WaveNet uses flat key structure: wavenet.in_layers.{i}.conv.conv.*
/// - WaveNet cond_layer is shared (not per-layer)
/// - Each block has skip_in_linear.weight/bias
fn dit_expected_keys() -> Vec<String> {
    let mut keys = Vec::new();
    let prefix = "cfm.estimator";

    // x_embedder (weight normalized)
    keys.push(format!("{}.x_embedder.weight_v", prefix));
    keys.push(format!("{}.x_embedder.weight_g", prefix));
    keys.push(format!("{}.x_embedder.bias", prefix));

    // cond_projection
    keys.push(format!("{}.cond_projection.weight", prefix));
    keys.push(format!("{}.cond_projection.bias", prefix));

    // cond_x_merge_linear
    keys.push(format!("{}.cond_x_merge_linear.weight", prefix));
    keys.push(format!("{}.cond_x_merge_linear.bias", prefix));

    // t_embedder
    keys.push(format!("{}.t_embedder.freqs", prefix));
    keys.push(format!("{}.t_embedder.mlp.0.weight", prefix));
    keys.push(format!("{}.t_embedder.mlp.0.bias", prefix));
    keys.push(format!("{}.t_embedder.mlp.2.weight", prefix));
    keys.push(format!("{}.t_embedder.mlp.2.bias", prefix));

    // cond_embedder
    keys.push(format!("{}.cond_embedder.weight", prefix));

    // 13 transformer blocks
    for i in 0..13 {
        let bp = format!("{}.transformer.layers.{}", prefix, i);
        // Attention
        keys.push(format!("{}.attention.wqkv.weight", bp));
        keys.push(format!("{}.attention.wo.weight", bp));
        // Feed-forward (SwiGLU)
        keys.push(format!("{}.feed_forward.w1.weight", bp));
        keys.push(format!("{}.feed_forward.w2.weight", bp));
        keys.push(format!("{}.feed_forward.w3.weight", bp));
        // AdaLayerNorm for attention (norm.weight + project_layer.weight/bias)
        keys.push(format!("{}.attention_norm.norm.weight", bp));
        keys.push(format!("{}.attention_norm.project_layer.weight", bp));
        keys.push(format!("{}.attention_norm.project_layer.bias", bp));
        // AdaLayerNorm for FFN (norm.weight + project_layer.weight/bias)
        keys.push(format!("{}.ffn_norm.norm.weight", bp));
        keys.push(format!("{}.ffn_norm.project_layer.weight", bp));
        keys.push(format!("{}.ffn_norm.project_layer.bias", bp));
        // UViT skip connection linear
        keys.push(format!("{}.skip_in_linear.weight", bp));
        keys.push(format!("{}.skip_in_linear.bias", bp));
    }

    // transformer.norm (loaded as simple LayerNorm, only uses norm.weight)
    keys.push(format!("{}.transformer.norm.norm.weight", prefix));

    // skip_linear (plain weight, not weight-normalized)
    keys.push(format!("{}.skip_linear.weight", prefix));
    keys.push(format!("{}.skip_linear.bias", prefix));

    // conv1 (plain weight, not weight-normalized)
    keys.push(format!("{}.conv1.weight", prefix));
    keys.push(format!("{}.conv1.bias", prefix));

    // t_embedder2
    keys.push(format!("{}.t_embedder2.freqs", prefix));
    keys.push(format!("{}.t_embedder2.mlp.0.weight", prefix));
    keys.push(format!("{}.t_embedder2.mlp.0.bias", prefix));
    keys.push(format!("{}.t_embedder2.mlp.2.weight", prefix));
    keys.push(format!("{}.t_embedder2.mlp.2.bias", prefix));

    // WaveNet - shared cond_layer (single layer, not per-wavenet-layer)
    keys.push(format!("{}.wavenet.cond_layer.conv.conv.weight_v", prefix));
    keys.push(format!("{}.wavenet.cond_layer.conv.conv.weight_g", prefix));
    keys.push(format!("{}.wavenet.cond_layer.conv.conv.bias", prefix));

    // WaveNet - 8 layers with flat key structure
    for i in 0..8 {
        // in_layers: wavenet.in_layers.{i}.conv.conv.*
        keys.push(format!("{}.wavenet.in_layers.{}.conv.conv.weight_v", prefix, i));
        keys.push(format!("{}.wavenet.in_layers.{}.conv.conv.weight_g", prefix, i));
        keys.push(format!("{}.wavenet.in_layers.{}.conv.conv.bias", prefix, i));
        // res_skip_layers: wavenet.res_skip_layers.{i}.conv.conv.*
        keys.push(format!("{}.wavenet.res_skip_layers.{}.conv.conv.weight_v", prefix, i));
        keys.push(format!("{}.wavenet.res_skip_layers.{}.conv.conv.weight_g", prefix, i));
        keys.push(format!("{}.wavenet.res_skip_layers.{}.conv.conv.bias", prefix, i));
    }

    // res_projection (plain weight, not weight-normalized)
    keys.push(format!("{}.res_projection.weight", prefix));
    keys.push(format!("{}.res_projection.bias", prefix));

    // final_layer
    keys.push(format!("{}.final_layer.adaLN_modulation.1.weight", prefix));
    keys.push(format!("{}.final_layer.adaLN_modulation.1.bias", prefix));
    keys.push(format!("{}.final_layer.linear.weight_v", prefix));
    keys.push(format!("{}.final_layer.linear.weight_g", prefix));
    keys.push(format!("{}.final_layer.linear.bias", prefix));

    // conv2 output projection (plain weight [80, 512, 1], not weight-normalized)
    keys.push(format!("{}.conv2.weight", prefix));
    keys.push(format!("{}.conv2.bias", prefix));

    keys
}

fn main() -> Result<()> {
    let device = Device::Cpu;
    let checkpoints_dir = Path::new("checkpoints");

    println!("IndexTTS2 Weight Loading Diagnostic");
    println!("===================================\n");

    // 1. wav2vec_bert.safetensors
    let wav2vec_path = checkpoints_dir.join("wav2vec_bert.safetensors");
    if wav2vec_path.exists() {
        let keys = load_and_list_keys(&wav2vec_path, &device)?;
        println!("wav2vec_bert.safetensors: {} tensors", keys.len());
        println!("All keys:");
        for (name, shape) in &keys {
            println!("  {} {:?}", name, shape);
        }
        let expected = wav2vec_bert_expected_keys();
        check_keys("Wav2Vec-BERT", &keys, &expected);
    } else {
        println!("WARNING: wav2vec_bert.safetensors not found!");
    }

    // 2. gpt.safetensors
    let gpt_path = checkpoints_dir.join("gpt.safetensors");
    if gpt_path.exists() {
        let keys = load_and_list_keys(&gpt_path, &device)?;
        println!("\ngpt.safetensors: {} tensors", keys.len());
        println!("All keys:");
        for (name, shape) in &keys {
            println!("  {} {:?}", name, shape);
        }
        let conformer_expected = conformer_expected_keys();
        check_keys("Conformer", &keys, &conformer_expected);

        let perceiver_expected = perceiver_expected_keys();
        check_keys("Perceiver", &keys, &perceiver_expected);
    } else {
        println!("WARNING: gpt.safetensors not found!");
    }

    // 3. s2mel.safetensors
    let s2mel_path = checkpoints_dir.join("s2mel.safetensors");
    if s2mel_path.exists() {
        let keys = load_and_list_keys(&s2mel_path, &device)?;
        println!("\ns2mel.safetensors: {} tensors", keys.len());
        println!("All keys:");
        for (name, shape) in &keys {
            println!("  {} {:?}", name, shape);
        }
        let dit_expected = dit_expected_keys();
        check_keys("DiT", &keys, &dit_expected);
    } else {
        println!("WARNING: s2mel.safetensors not found!");
    }

    // 4. bigvgan.safetensors (reference)
    let bigvgan_path = checkpoints_dir.join("bigvgan.safetensors");
    if bigvgan_path.exists() {
        let keys = load_and_list_keys(&bigvgan_path, &device)?;
        println!("\nbigvgan.safetensors: {} tensors (REFERENCE - WORKING)", keys.len());
    }

    Ok(())
}
