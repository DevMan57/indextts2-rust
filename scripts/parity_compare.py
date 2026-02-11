#!/usr/bin/env python3
"""
Compare Rust parity dumps against Python reference tensors.

Expected Rust dumps are generated when INDEXTTS2_PARITY_DIR is set during Rust inference.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import types
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf


def patch_torch_load_weights_only() -> None:
    original_load = torch.load

    def patched_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return original_load(*args, **kwargs)

    torch.load = patched_load


def patch_transformers_compat() -> None:
    # index-tts custom generation utils expects this symbol in some transformers versions.
    try:
        import transformers.cache_utils as cache_utils
        import transformers.generation.candidate_generator as cand
        import transformers.generation.configuration_utils as gen_cfg
    except Exception:
        return
    if not hasattr(cache_utils, "QuantizedCacheConfig"):
        class QuantizedCacheConfig:  # pragma: no cover - compatibility shim
            pass
        cache_utils.QuantizedCacheConfig = QuantizedCacheConfig
    if not hasattr(cand, "_crop_past_key_values"):
        def _crop_past_key_values(past_key_values, *args, **kwargs):
            return past_key_values
        cand._crop_past_key_values = _crop_past_key_values
    if not hasattr(cand, "_prepare_attention_mask"):
        def _prepare_attention_mask(model_kwargs, *args, **kwargs):
            return model_kwargs
        cand._prepare_attention_mask = _prepare_attention_mask
    if not hasattr(cand, "_prepare_token_type_ids"):
        def _prepare_token_type_ids(model_kwargs, *args, **kwargs):
            return model_kwargs
        cand._prepare_token_type_ids = _prepare_token_type_ids
    if not hasattr(gen_cfg, "NEED_SETUP_CACHE_CLASSES_MAPPING"):
        gen_cfg.NEED_SETUP_CACHE_CLASSES_MAPPING = {}
    if not hasattr(gen_cfg, "QUANT_BACKEND_CLASSES_MAPPING"):
        gen_cfg.QUANT_BACKEND_CLASSES_MAPPING = {}


def load_dump(dump_dir: Path, name: str) -> np.ndarray:
    meta_path = dump_dir / f"{name}.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing dump metadata: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    bin_path = dump_dir / meta["bin"]
    dtype = meta["dtype"]
    if dtype == "f32":
        arr = np.fromfile(bin_path, dtype=np.float32)
    elif dtype == "i64":
        arr = np.fromfile(bin_path, dtype=np.int64)
    elif dtype == "u32":
        arr = np.fromfile(bin_path, dtype=np.uint32)
    else:
        raise ValueError(f"Unsupported dump dtype: {dtype}")
    return arr.reshape(meta["shape"])


def load_scalar_text(dump_dir: Path, name: str, cast):
    path = dump_dir / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing scalar dump: {path}")
    return cast(path.read_text(encoding="utf-8").strip())


def strip_boundary_text_tokens_for_prepare_inputs(
    rust_text_ids: np.ndarray,
    start_text_token: int,
    stop_text_token: int,
) -> np.ndarray:
    arr = rust_text_ids.astype(np.int64, copy=False)
    if arr.ndim == 2 and arr.shape[0] == 1:
        seq = arr[0]
    elif arr.ndim == 1:
        seq = arr
    else:
        raise ValueError(f"Unexpected rust_text_ids shape: {arr.shape}")

    if seq.size >= 2 and int(seq[0]) == start_text_token and int(seq[-1]) == stop_text_token:
        seq = seq[1:-1]
    return np.expand_dims(seq, axis=0)


def compare(name: str, rust: np.ndarray, py: np.ndarray) -> dict:
    if rust.shape != py.shape and rust.size == py.size:
        rust = rust.reshape(-1)
        py = py.reshape(-1)
    if rust.shape != py.shape:
        return {
            "name": name,
            "ok": False,
            "shape_match": False,
            "rust_shape": list(rust.shape),
            "python_shape": list(py.shape),
        }
    diff = np.abs(rust.astype(np.float64) - py.astype(np.float64))
    return {
        "name": name,
        "ok": True,
        "shape_match": True,
        "rust_shape": list(rust.shape),
        "python_shape": list(py.shape),
        "max_abs_diff": float(diff.max(initial=0.0)),
        "mean_abs_diff": float(diff.mean() if diff.size else 0.0),
    }


def align_python_attention_seq(
    name: str,
    rust: np.ndarray,
    py: np.ndarray,
    cond_len: int,
    text_len: int,
) -> np.ndarray:
    # Python prepare_gpt_inputs adds start/stop text tokens; Rust prefill currently doesn't.
    # For attention K/V and score tensors, strip those two extra positions so we can compare
    # against Rust's exact prefill sequence.
    if re.fullmatch(r"gpt_step\d+_block\d{2}_(k|v)", name):
        kind = "kv"
    elif re.fullmatch(r"gpt_step\d+_block\d{2}_attn_scores_(pre|post)_mask", name):
        kind = "scores"
    else:
        return py

    if kind == "kv" and py.ndim == 4 and rust.ndim == 4 and py.shape[2] == rust.shape[2] + 2:
        py_len = py.shape[2]
        keep = list(range(cond_len)) + list(range(cond_len + 1, cond_len + 1 + text_len)) + [py_len - 1]
        if len(keep) == rust.shape[2] and max(keep, default=-1) < py_len:
            return py[:, :, keep, :]

    if kind == "scores" and py.ndim == 4 and rust.ndim == 4 and py.shape[3] == rust.shape[3] + 2:
        py_len = py.shape[3]
        keep = list(range(cond_len)) + list(range(cond_len + 1, cond_len + 1 + text_len)) + [py_len - 1]
        if len(keep) == rust.shape[3] and max(keep, default=-1) < py_len:
            return py[:, :, :, keep]

    return py


def dump_exists(dump_dir: Path, name: str) -> bool:
    return (dump_dir / f"{name}.json").exists()


def discover_rust_logged_gpt_steps(dump_dir: Path) -> dict[int, list[str]]:
    logged: dict[int, list[str]] = {}
    for meta_path in dump_dir.glob("rust_gpt_logits_step*.json"):
        stem = meta_path.stem
        if stem == "rust_gpt_logits_step_last":
            continue
        m = re.fullmatch(r"rust_gpt_logits_step(\d+)", stem)
        if m is None:
            continue
        step_idx = int(m.group(1))
        if step_idx == 0:
            continue
        logged.setdefault(step_idx, []).append(stem)

    last_name = "rust_gpt_logits_step_last"
    last_idx_name = "rust_gpt_logits_step_last_index"
    last_idx_path = dump_dir / f"{last_idx_name}.txt"
    if dump_exists(dump_dir, last_name) and last_idx_path.exists():
        last_idx = load_scalar_text(dump_dir, last_idx_name, int)
        logged.setdefault(last_idx, []).append(last_name)

    return logged


def extract_cache_seq_len(past_key_values) -> int | None:
    if not past_key_values:
        return None
    first_layer = past_key_values[0]
    if first_layer is None or len(first_layer) == 0:
        return None
    key = first_layer[0]
    if key is None or key.ndim < 3:
        return None
    return int(key.shape[2])


def run_gpt_forward_with_trace(
    gpt,
    input_ids,
    attention_mask,
    trace_blocks: int,
    step_idx: int,
    past_key_values=None,
):
    traces: dict[str, np.ndarray] = {}
    hooks = []
    step_tag = f"step{step_idx}"

    def to_last_token_3d(tensor: torch.Tensor) -> np.ndarray:
        t = tensor.detach()
        if t.ndim == 1:
            t = t.unsqueeze(0).unsqueeze(0)
        elif t.ndim == 2:
            t = t.unsqueeze(1)
        elif t.ndim >= 3:
            t = t[:, -1:, :]
        return t.cpu().numpy()

    def save_pre(name: str):
        def hook(_module, inputs):
            if not inputs:
                return
            traces[name] = to_last_token_3d(inputs[0])

        return hook

    def save_fwd(name: str, tuple_index: int | None = None):
        def hook(_module, _inputs, output):
            value = output[tuple_index] if tuple_index is not None else output
            traces[name] = to_last_token_3d(value)

        return hook

    def save_pre_once(name: str):
        def hook(_module, inputs):
            if name in traces or not inputs:
                return
            traces[name] = to_last_token_3d(inputs[0])

        return hook

    def save_fwd_once(name: str, tuple_index: int | None = None):
        def hook(_module, _inputs, output):
            if name in traces:
                return
            value = output[tuple_index] if tuple_index is not None else output
            traces[name] = to_last_token_3d(value)

        return hook

    def to_numpy(tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy()

    def resolve_past_kv(layer_past, layer_idx: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if layer_past is None:
            return None, None
        if isinstance(layer_past, (tuple, list)):
            if len(layer_past) >= 2 and torch.is_tensor(layer_past[0]) and torch.is_tensor(layer_past[1]):
                return layer_past[0], layer_past[1]
            if layer_idx < len(layer_past):
                candidate = layer_past[layer_idx]
                if (
                    isinstance(candidate, (tuple, list))
                    and len(candidate) >= 2
                    and torch.is_tensor(candidate[0])
                    and torch.is_tensor(candidate[1])
                ):
                    return candidate[0], candidate[1]
        key_cache = getattr(layer_past, "key_cache", None)
        value_cache = getattr(layer_past, "value_cache", None)
        if (
            isinstance(key_cache, (tuple, list))
            and isinstance(value_cache, (tuple, list))
            and layer_idx < len(key_cache)
            and layer_idx < len(value_cache)
        ):
            key_tensor = key_cache[layer_idx]
            value_tensor = value_cache[layer_idx]
            if torch.is_tensor(key_tensor) and torch.is_tensor(value_tensor):
                return key_tensor, value_tensor
        return None, None

    block_count = min(trace_blocks, len(gpt.inference_model.transformer.h))
    for block_idx in range(block_count):
        block = gpt.inference_model.transformer.h[block_idx]
        prefix = f"py_gpt_{step_tag}_block_{block_idx:02}"
        hooks.append(block.register_forward_pre_hook(save_pre(f"{prefix}_input")))
        hooks.append(block.ln_1.register_forward_hook(save_fwd(f"{prefix}_ln1")))
        hooks.append(block.ln_2.register_forward_hook(save_fwd(f"{prefix}_ln2")))
        hooks.append(block.register_forward_hook(save_fwd(f"{prefix}_out", tuple_index=0)))

    final_norm_module = gpt.inference_model.transformer.ln_f
    norm_module = gpt.inference_model.lm_head[0]
    linear_module = gpt.inference_model.lm_head[1]

    # Transformer final LayerNorm (matches Rust final norm before lm_head path).
    hooks.append(final_norm_module.register_forward_pre_hook(save_pre(f"py_gpt_{step_tag}_pre_final_ln")))
    hooks.append(final_norm_module.register_forward_hook(save_fwd(f"py_gpt_{step_tag}_post_final_ln")))

    # Keep optional lm_head norm tracing for debugging (not part of default comparison set).
    hooks.append(norm_module.register_forward_pre_hook(save_pre_once(f"py_gpt_{step_tag}_lm_norm_pre")))
    hooks.append(norm_module.register_forward_hook(save_fwd_once(f"py_gpt_{step_tag}_lm_norm_post")))
    hooks.append(linear_module.register_forward_pre_hook(save_pre(f"py_gpt_{step_tag}_pre_lm_head")))
    hooks.append(linear_module.register_forward_hook(save_fwd(f"py_gpt_{step_tag}_post_lm_head")))

    block0_attn = gpt.inference_model.transformer.h[0].attn
    original_block0_attn_forward = block0_attn.forward

    def wrapped_block0_attn_forward(self, hidden_states, *f_args, **f_kwargs):
        past_key_value = f_kwargs.get("past_key_value")
        if past_key_value is None:
            past_key_value = f_kwargs.get("layer_past")
        if past_key_value is None and len(f_args) > 0:
            past_key_value = f_args[0]

        attention_mask_local = f_kwargs.get("attention_mask")
        if attention_mask_local is None and len(f_args) > 2:
            attention_mask_local = f_args[2]

        encoder_hidden_states = f_kwargs.get("encoder_hidden_states")
        if encoder_hidden_states is None and len(f_args) > 4:
            encoder_hidden_states = f_args[4]
        encoder_attention_mask = f_kwargs.get("encoder_attention_mask")
        if encoder_attention_mask is None and len(f_args) > 5:
            encoder_attention_mask = f_args[5]

        if encoder_hidden_states is not None and hasattr(self, "q_attn"):
            query_states = self.q_attn(hidden_states)
            key_states, value_states = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask_for_scores = encoder_attention_mask
        else:
            query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
            attention_mask_for_scores = attention_mask_local

        shape_q = (*query_states.shape[:-1], -1, self.head_dim)
        shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
        query_states = query_states.view(shape_q).transpose(1, 2)
        key_states = key_states.view(shape_kv).transpose(1, 2)
        value_states = value_states.view(shape_kv).transpose(1, 2)
        layer_idx = int(getattr(self, "layer_idx", 0) or 0)
        past_key, past_value = resolve_past_kv(past_key_value, layer_idx)
        if past_key is not None and past_value is not None:
            key_states = torch.cat((past_key, key_states), dim=-2)
            value_states = torch.cat((past_value, value_states), dim=-2)

        query_last = query_states[:, :, -1:, :]
        traces[f"py_gpt_{step_tag}_block_00_q"] = to_numpy(query_last)
        traces[f"py_gpt_{step_tag}_block_00_k"] = to_numpy(key_states)
        traces[f"py_gpt_{step_tag}_block_00_v"] = to_numpy(value_states)

        attn_scores = torch.matmul(query_states, key_states.transpose(-1, -2))
        if self.scale_attn_weights:
            attn_scores = attn_scores / torch.full(
                [],
                value_states.size(-1) ** 0.5,
                dtype=attn_scores.dtype,
                device=attn_scores.device,
            )
        if self.scale_attn_by_inverse_layer_idx:
            attn_scores = attn_scores / float(self.layer_idx + 1)
        traces[f"py_gpt_{step_tag}_block_00_attn_scores_pre_mask"] = to_numpy(attn_scores[:, :, -1:, :])

        attn_scores_post_mask = attn_scores
        if not self.is_cross_attention:
            key_length = key_states.size(-2)
            query_length = query_states.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            mask_value = torch.full(
                [],
                torch.finfo(attn_scores_post_mask.dtype).min,
                dtype=attn_scores_post_mask.dtype,
                device=attn_scores_post_mask.device,
            )
            attn_scores_post_mask = torch.where(
                causal_mask,
                attn_scores_post_mask.to(attn_scores_post_mask.dtype),
                mask_value,
            )

        if attention_mask_for_scores is not None:
            attn_mask = attention_mask_for_scores[:, :, :, : key_states.shape[-2]]
            attn_scores_post_mask = attn_scores_post_mask + attn_mask
        traces[f"py_gpt_{step_tag}_block_00_attn_scores_post_mask"] = to_numpy(attn_scores_post_mask[:, :, -1:, :])

        attn_probs = torch.softmax(attn_scores_post_mask, dim=-1)
        attn_probs = attn_probs.type(value_states.dtype)
        attn_output = torch.matmul(attn_probs, value_states).transpose(1, 2)
        traces[f"py_gpt_{step_tag}_block_00_attn_out_pre_out_proj"] = to_numpy(
            attn_output[:, -1:, :].reshape(attn_output.shape[0], 1, -1)
        )

        return original_block0_attn_forward(hidden_states, *f_args, **f_kwargs)

    block0_attn.forward = types.MethodType(wrapped_block0_attn_forward, block0_attn)

    try:
        model_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": True,
            "return_dict": True,
        }
        if past_key_values is not None:
            model_kwargs["past_key_values"] = past_key_values
        with torch.no_grad():
            out = gpt.inference_model(**model_kwargs)
    finally:
        block0_attn.forward = original_block0_attn_forward
        for hook in hooks:
            hook.remove()

    return out, traces


def resolve_python_checkpoint(model_dir: Path, configured_name: str) -> Path:
    configured = model_dir / configured_name
    if configured.suffix == ".safetensors":
        alt = configured.with_suffix(".pth")
        if alt.exists():
            return alt
    return configured


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Rust tensors to Python reference")
    parser.add_argument("--dump-dir", default="debug/parity", help="Rust parity dump directory")
    parser.add_argument("--python-repo", default=r"C:\AI\index-tts", help="Path to python index-tts repo")
    parser.add_argument("--config", default="checkpoints/config.yaml", help="Path to config.yaml")
    parser.add_argument("--model-dir", default="checkpoints", help="Model dir used by config paths")
    parser.add_argument("--device", default="cuda:0", help="Torch device (e.g., cuda:0 or cpu)")
    parser.add_argument("--trace-blocks", type=int, default=24, help="Number of GPT blocks to trace")
    parser.add_argument("--jump-abs-threshold", type=float, default=0.15, help="Absolute mean diff threshold for first large jump")
    parser.add_argument("--jump-rel-factor", type=float, default=1.8, help="Relative jump factor vs previous block mean diff")
    args = parser.parse_args()

    dump_dir = Path(args.dump_dir)
    if not dump_dir.exists():
        raise FileNotFoundError(f"Dump dir not found: {dump_dir}")

    patch_transformers_compat()
    patch_torch_load_weights_only()
    sys.path.insert(0, str(Path(args.python_repo)))
    from indextts.gpt.model_v2 import UnifiedVoice
    from indextts.s2mel.modules.commons import MyModel, load_checkpoint2
    from indextts.utils.checkpoint import load_checkpoint

    cfg = OmegaConf.load(args.config)
    model_dir = Path(args.model_dir)

    requested_device = torch.device(args.device)
    if requested_device.type.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable for python reference; falling back to cpu")
        requested_device = torch.device("cpu")

    print(f"[python] loading models on {requested_device}...")

    gpt_ckpt = resolve_python_checkpoint(model_dir, str(cfg.gpt_checkpoint))
    s2mel_ckpt = resolve_python_checkpoint(model_dir, str(cfg.s2mel_checkpoint))

    gpt = UnifiedVoice(**cfg.gpt, use_accel=False)
    load_checkpoint(gpt, str(gpt_ckpt))
    gpt = gpt.to(requested_device).eval()
    gpt.post_init_gpt2_config(use_deepspeed=False, kv_cache=True, half=False)

    s2mel = MyModel(cfg.s2mel, use_gpt_latent=True)
    s2mel, _, _, _ = load_checkpoint2(
        s2mel,
        None,
        str(s2mel_ckpt),
        load_only_params=True,
        ignore_modules=[],
        is_distributed=False,
    )
    s2mel = s2mel.to(requested_device).eval()
    s2mel.models["cfm"].estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

    results: list[dict] = []

    # GPT logits parity (step 0)
    rust_text_ids = load_dump(dump_dir, "rust_gpt_text_ids")
    rust_conditioning = load_dump(dump_dir, "rust_gpt_conditioning")
    rust_logits = load_dump(dump_dir, "rust_gpt_logits_step0")
    rust_text_len = load_scalar_text(dump_dir, "rust_gpt_text_len", int)
    rust_cond_len = load_scalar_text(dump_dir, "rust_gpt_cond_len", int)

    text_ids_for_prepare = strip_boundary_text_tokens_for_prepare_inputs(
        rust_text_ids,
        int(cfg.gpt.start_text_token),
        int(cfg.gpt.stop_text_token),
    )
    text_ids = torch.from_numpy(text_ids_for_prepare).to(requested_device)
    conditioning = torch.from_numpy(rust_conditioning.astype(np.float32)).to(requested_device)

    input_ids, inputs_embeds, attention_mask = gpt.prepare_gpt_inputs(conditioning, text_ids)
    gpt.inference_model.store_mel_emb(inputs_embeds)
    traced_blocks = min(max(args.trace_blocks, 1), len(gpt.inference_model.transformer.h))
    out, gpt_step0_traces = run_gpt_forward_with_trace(
        gpt,
        input_ids,
        attention_mask,
        traced_blocks,
        step_idx=0,
    )
    gpt_step1_traces: dict[str, np.ndarray] = {}
    py_logits = out.logits[:, -1, :].detach().cpu().numpy()

    results.append(compare("gpt_logits_step0", rust_logits, py_logits))

    # Optional multi-step GPT logits parity (step1 / stepN) and cache growth.
    py_logits_by_step: dict[int, np.ndarray] = {0: py_logits}
    py_cache_len_per_step: list[int] = []
    step0_cache_len = extract_cache_seq_len(out.past_key_values)
    if step0_cache_len is not None:
        py_cache_len_per_step.append(step0_cache_len)

    rust_logged_gpt_steps = discover_rust_logged_gpt_steps(dump_dir)
    rust_generated_tokens = (
        load_dump(dump_dir, "rust_gpt_generated_tokens")
        if dump_exists(dump_dir, "rust_gpt_generated_tokens")
        else None
    )
    rust_generated_tokens_list = (
        [int(x) for x in rust_generated_tokens.reshape(-1).tolist()]
        if rust_generated_tokens is not None
        else []
    )
    rust_cache_len_per_step = (
        load_dump(dump_dir, "rust_gpt_cache_len_per_step")
        if dump_exists(dump_dir, "rust_gpt_cache_len_per_step")
        else None
    )

    max_step_needed = max(rust_logged_gpt_steps.keys(), default=0)
    if rust_cache_len_per_step is not None:
        max_step_needed = max(max_step_needed, int(rust_cache_len_per_step.size) - 1)

    if max_step_needed > 0:
        if not rust_generated_tokens_list:
            print("[warn] rust_gpt_generated_tokens missing/empty; skipping multi-step GPT parity")
        else:
            past_key_values = out.past_key_values
            running_attention_mask = attention_mask
            for step in range(1, max_step_needed + 1):
                token_idx = step - 1
                if token_idx >= len(rust_generated_tokens_list):
                    print(
                        f"[warn] cannot compute python gpt step {step}: "
                        f"need generated token index {token_idx}, only {len(rust_generated_tokens_list)} available"
                    )
                    break

                next_input = torch.tensor(
                    [[rust_generated_tokens_list[token_idx]]],
                    device=requested_device,
                    dtype=input_ids.dtype,
                )
                next_mask = torch.ones(
                    (running_attention_mask.shape[0], 1),
                    dtype=running_attention_mask.dtype,
                    device=running_attention_mask.device,
                )
                running_attention_mask = torch.cat([running_attention_mask, next_mask], dim=1)
                if step == 1:
                    out_step, gpt_step1_traces = run_gpt_forward_with_trace(
                        gpt,
                        next_input,
                        running_attention_mask,
                        traced_blocks,
                        step_idx=1,
                        past_key_values=past_key_values,
                    )
                else:
                    with torch.no_grad():
                        out_step = gpt.inference_model(
                            input_ids=next_input,
                            attention_mask=running_attention_mask,
                            past_key_values=past_key_values,
                            use_cache=True,
                            return_dict=True,
                        )
                past_key_values = out_step.past_key_values
                py_logits_by_step[step] = out_step.logits[:, -1, :].detach().cpu().numpy()
                cache_len = extract_cache_seq_len(past_key_values)
                if cache_len is not None:
                    py_cache_len_per_step.append(cache_len)

    for step_idx in sorted(rust_logged_gpt_steps):
        rust_names = sorted(rust_logged_gpt_steps[step_idx])
        py_arr = py_logits_by_step.get(step_idx)
        if py_arr is None:
            print(f"[warn] missing python logits for step {step_idx}, skipping logged rust step dumps")
            continue
        for rust_name in rust_names:
            rust_arr = load_dump(dump_dir, rust_name)
            result_name = (
                "gpt_logits_step_last"
                if rust_name == "rust_gpt_logits_step_last"
                else f"gpt_logits_step{step_idx}"
            )
            cmp = compare(result_name, rust_arr, py_arr)
            cmp["step_index"] = step_idx
            cmp["rust_dump"] = rust_name
            results.append(cmp)

    if rust_cache_len_per_step is not None and py_cache_len_per_step:
        py_cache_arr = np.asarray(py_cache_len_per_step, dtype=np.uint32)
        results.append(compare("gpt_cache_len_per_step", rust_cache_len_per_step, py_cache_arr))

    all_gpt_traces = dict(gpt_step0_traces)
    all_gpt_traces.update(gpt_step1_traces)

    gpt_trace_pairs: list[tuple[str, str, str]] = []
    for step_idx in (0, 1):
        step_tag = f"step{step_idx}"
        gpt_trace_pairs.extend(
            [
                (
                    f"gpt_{step_tag}_block00_q",
                    f"rust_gpt_{step_tag}_block_00_q",
                    f"py_gpt_{step_tag}_block_00_q",
                ),
                (
                    f"gpt_{step_tag}_block00_k",
                    f"rust_gpt_{step_tag}_block_00_k",
                    f"py_gpt_{step_tag}_block_00_k",
                ),
                (
                    f"gpt_{step_tag}_block00_v",
                    f"rust_gpt_{step_tag}_block_00_v",
                    f"py_gpt_{step_tag}_block_00_v",
                ),
                (
                    f"gpt_{step_tag}_block00_attn_scores_pre_mask",
                    f"rust_gpt_{step_tag}_block_00_attn_scores_pre_mask",
                    f"py_gpt_{step_tag}_block_00_attn_scores_pre_mask",
                ),
                (
                    f"gpt_{step_tag}_block00_attn_scores_post_mask",
                    f"rust_gpt_{step_tag}_block_00_attn_scores_post_mask",
                    f"py_gpt_{step_tag}_block_00_attn_scores_post_mask",
                ),
                (
                    f"gpt_{step_tag}_block00_attn_out_pre_out_proj",
                    f"rust_gpt_{step_tag}_block_00_attn_out_pre_out_proj",
                    f"py_gpt_{step_tag}_block_00_attn_out_pre_out_proj",
                ),
            ]
        )
        for block_idx in range(traced_blocks):
            gpt_trace_pairs.extend(
                [
                    (
                        f"gpt_{step_tag}_block{block_idx:02}_input",
                        f"rust_gpt_{step_tag}_block_{block_idx:02}_input",
                        f"py_gpt_{step_tag}_block_{block_idx:02}_input",
                    ),
                    (
                        f"gpt_{step_tag}_block{block_idx:02}_ln1",
                        f"rust_gpt_{step_tag}_block_{block_idx:02}_ln1",
                        f"py_gpt_{step_tag}_block_{block_idx:02}_ln1",
                    ),
                    (
                        f"gpt_{step_tag}_block{block_idx:02}_ln2",
                        f"rust_gpt_{step_tag}_block_{block_idx:02}_ln2",
                        f"py_gpt_{step_tag}_block_{block_idx:02}_ln2",
                    ),
                    (
                        f"gpt_{step_tag}_block{block_idx:02}_out",
                        f"rust_gpt_{step_tag}_block_{block_idx:02}_out",
                        f"py_gpt_{step_tag}_block_{block_idx:02}_out",
                    ),
                ]
            )
        gpt_trace_pairs.extend(
            [
                (
                    f"gpt_{step_tag}_pre_final_ln",
                    f"rust_gpt_{step_tag}_pre_final_ln",
                    f"py_gpt_{step_tag}_pre_final_ln",
                ),
                (
                    f"gpt_{step_tag}_post_final_ln",
                    f"rust_gpt_{step_tag}_post_final_ln",
                    f"py_gpt_{step_tag}_post_final_ln",
                ),
                (
                    f"gpt_{step_tag}_pre_lm_head",
                    f"rust_gpt_{step_tag}_pre_lm_head",
                    f"py_gpt_{step_tag}_pre_lm_head",
                ),
                (
                    f"gpt_{step_tag}_post_lm_head",
                    f"rust_gpt_{step_tag}_post_lm_head",
                    f"py_gpt_{step_tag}_post_lm_head",
                ),
            ]
        )

    block_out_means: dict[int, float] = {}
    for result_name, rust_name, py_name in gpt_trace_pairs:
        if not dump_exists(dump_dir, rust_name):
            continue
        rust_arr = load_dump(dump_dir, rust_name)
        py_arr = all_gpt_traces.get(py_name)
        if py_arr is None:
            print(f"[warn] missing python trace '{py_name}', skipping {result_name}")
            continue
        py_arr = align_python_attention_seq(
            result_name,
            rust_arr,
            py_arr,
            rust_cond_len,
            rust_text_len,
        )
        cmp = compare(result_name, rust_arr, py_arr)
        results.append(cmp)
        if (
            result_name.startswith("gpt_step0_block")
            and result_name.endswith("_out")
            and cmp.get("shape_match")
        ):
            block_part = result_name.split("_", maxsplit=3)[2]
            block_idx = int(block_part.replace("block", ""))
            block_out_means[block_idx] = float(cmp.get("mean_abs_diff", 0.0))
        if re.fullmatch(r"gpt_step\d+_block00_(k|v)", result_name) and rust_arr.ndim == 4 and py_arr.ndim == 4:
            rust_last = rust_arr[:, :, -1:, :]
            py_last = py_arr[:, :, -1:, :]
            results.append(compare(f"{result_name}_last_token", rust_last, py_last))

    if block_out_means:
        print("\nGPT block output mean_abs_diff:")
        for idx in sorted(block_out_means):
            print(f"  block {idx:02}: {block_out_means[idx]:.6e}")

        sorted_blocks = sorted(block_out_means)
        jump = None
        for idx in sorted_blocks:
            if idx <= 1:
                continue
            prev_idx = idx - 1
            if prev_idx not in block_out_means:
                continue
            prev_mean = block_out_means[prev_idx]
            curr_mean = block_out_means[idx]
            rel = curr_mean / max(prev_mean, 1e-12)
            if curr_mean >= args.jump_abs_threshold and rel >= args.jump_rel_factor:
                jump = (idx, prev_mean, curr_mean, rel)
                break
        if jump is None:
            print("First large jump: none detected with current thresholds")
        else:
            idx, prev_mean, curr_mean, rel = jump
            print(
                "First large jump: "
                f"block {idx:02} (prev={prev_mean:.6e}, curr={curr_mean:.6e}, rel={rel:.3f}x)"
            )

    # DiT first 3 Euler steps parity
    rust_noise = load_dump(dump_dir, "rust_dit_input_noise")
    rust_prompt_x_full = load_dump(dump_dir, "rust_dit_input_prompt_x")
    rust_cond = load_dump(dump_dir, "rust_dit_input_cond")
    rust_style = load_dump(dump_dir, "rust_dit_input_style")
    prompt_len = load_scalar_text(dump_dir, "rust_dit_prompt_len", int)
    num_steps = load_scalar_text(dump_dir, "rust_dit_num_steps", int)
    cfg_rate = load_scalar_text(dump_dir, "rust_dit_cfg_rate", float)

    noise = torch.from_numpy(rust_noise.astype(np.float32)).to(requested_device)
    prompt_x_full = torch.from_numpy(rust_prompt_x_full.astype(np.float32)).to(requested_device)
    cond = torch.from_numpy(rust_cond.astype(np.float32)).to(requested_device)
    style = torch.from_numpy(rust_style.astype(np.float32)).to(requested_device)

    cfm = s2mel.models["cfm"]
    x = noise.clone()
    total_len = int(x.shape[-1])
    x_lens = torch.LongTensor([total_len]).to(requested_device)
    t_span = torch.linspace(0, 1, num_steps + 1, device=requested_device)

    prompt = prompt_x_full[:, :, :prompt_len] if prompt_len > 0 else torch.zeros_like(x[:, :, :0])
    prompt_x = torch.zeros_like(x)
    if prompt_len > 0:
        prompt_x[..., :prompt_len] = prompt[..., :prompt_len]
        x[..., :prompt_len] = 0

    t = t_span[0]
    with torch.no_grad():
        for step in range(1, len(t_span)):
            dt = t_span[step] - t_span[step - 1]
            if cfg_rate > 0:
                stacked_prompt_x = torch.cat([prompt_x, torch.zeros_like(prompt_x)], dim=0)
                stacked_style = torch.cat([style, torch.zeros_like(style)], dim=0)
                stacked_mu = torch.cat([cond, torch.zeros_like(cond)], dim=0)
                stacked_x = torch.cat([x, x], dim=0)
                stacked_t = torch.cat([t.unsqueeze(0), t.unsqueeze(0)], dim=0)
                stacked_dphi_dt = cfm.estimator(
                    stacked_x, stacked_prompt_x, x_lens, stacked_t, stacked_style, stacked_mu
                )
                dphi_dt, cfg_dphi_dt = stacked_dphi_dt.chunk(2, dim=0)
                dphi_dt = (1.0 + cfg_rate) * dphi_dt - cfg_rate * cfg_dphi_dt
            else:
                dphi_dt = cfm.estimator(x, prompt_x, x_lens, t.unsqueeze(0), style, cond)

            x = x + dt * dphi_dt

            idx = step - 1
            if idx < 3:
                py_step = x.detach().cpu().numpy()
                rust_step = load_dump(dump_dir, f"rust_dit_step_{idx:02}")
                results.append(compare(f"dit_step_{idx:02}", rust_step, py_step))

            t = t + dt
            if step < len(t_span) - 1 and prompt_len > 0:
                x[:, :, :prompt_len] = 0

    print("\nParity Summary:")
    for item in results:
        if not item.get("shape_match", False):
            print(
                f"  [FAIL] {item['name']}: shape mismatch rust={item['rust_shape']} python={item['python_shape']}"
            )
            continue
        print(
            f"  [OK] {item['name']}: max_abs_diff={item['max_abs_diff']:.6e}, "
            f"mean_abs_diff={item['mean_abs_diff']:.6e}"
        )

    out_path = dump_dir / "parity_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved detailed results to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
