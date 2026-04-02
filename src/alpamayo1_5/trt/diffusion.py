# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from contextlib import nullcontext

import torch
import torch.nn as nn

from alpamayo1_5.trt.prefix_cache import PrefixKVCache

logger = logging.getLogger(__name__)


class StaticKVDiffusionStepModule(nn.Module):
    def __init__(
        self,
        action_in_proj: nn.Module,
        expert: nn.Module,
        action_out_proj: nn.Module,
        n_diffusion_tokens: int,
        action_space_dims: tuple[int, ...],
        num_layers: int,
    ):
        super().__init__()
        self.action_in_proj = action_in_proj
        self.expert = expert
        self.action_out_proj = action_out_proj
        self.n_diffusion_tokens = n_diffusion_tokens
        self.action_space_dims = action_space_dims
        self.num_layers = num_layers

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        prefix_k: torch.Tensor,
        prefix_v: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        future_token_embeds = self.action_in_proj(x, t)
        if future_token_embeds.dim() == 2:
            future_token_embeds = future_token_embeds.view(batch_size, self.n_diffusion_tokens, -1)

        past_key_values = PrefixKVCache(prefix_k, prefix_v)
        expert_out = self.expert(
            inputs_embeds=future_token_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            use_cache=False,
        )
        last_hidden = expert_out.last_hidden_state[:, -self.n_diffusion_tokens :]
        return self.action_out_proj(last_hidden).view(-1, *self.action_space_dims)


def _build_diffusion_module(model: nn.Module, dtype: torch.dtype, device: str) -> tuple[StaticKVDiffusionStepModule, dict]:
    n_diffusion_tokens = model.action_space.get_action_space_dims()[0]
    action_space_dims = model.action_space.get_action_space_dims()
    expert_cfg = model.expert.config
    num_layers = expert_cfg.num_hidden_layers
    num_kv_heads = expert_cfg.num_key_value_heads
    head_dim = expert_cfg.head_dim

    model.expert.config._attn_implementation = "sdpa"
    module = (
        StaticKVDiffusionStepModule(
            action_in_proj=model.action_in_proj,
            expert=model.expert,
            action_out_proj=model.action_out_proj,
            n_diffusion_tokens=n_diffusion_tokens,
            action_space_dims=action_space_dims,
            num_layers=num_layers,
        )
        .to(device=device, dtype=dtype)
        .eval()
    )

    cfg = dict(
        n_diffusion_tokens=n_diffusion_tokens,
        action_space_dims=action_space_dims,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
    return module, cfg


def _make_sample_inputs(
    cfg: dict,
    min_prefix_len: int,
    max_prefix_len: int,
    dtype: torch.dtype,
    device: str,
    batch_size: int,
) -> tuple[tuple, tuple, callable]:
    n_diffusion_tokens = cfg["n_diffusion_tokens"]
    action_space_dims = cfg["action_space_dims"]
    num_layers = cfg["num_layers"]
    num_kv_heads = cfg["num_kv_heads"]
    head_dim = cfg["head_dim"]
    bsz = int(batch_size)
    if bsz <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    opt_prefix_len = (min_prefix_len + max_prefix_len) // 2

    def make_inputs(prefix_len: int) -> list[torch.Tensor]:
        return [
            torch.randn(bsz, *action_space_dims, dtype=dtype, device=device),
            torch.zeros(bsz, 1, 1, dtype=dtype, device=device),
            torch.zeros(num_layers, bsz, num_kv_heads, prefix_len, head_dim, dtype=dtype, device=device),
            torch.zeros(num_layers, bsz, num_kv_heads, prefix_len, head_dim, dtype=dtype, device=device),
            torch.arange(n_diffusion_tokens, device=device).unsqueeze(0).unsqueeze(0).expand(3, bsz, -1).clone(),
            torch.zeros(bsz, 1, n_diffusion_tokens, prefix_len + n_diffusion_tokens, dtype=dtype, device=device),
        ]

    sample_inputs = tuple(make_inputs(opt_prefix_len))
    prefix_dim = torch.export.Dim("prefix_len", min=min_prefix_len, max=max_prefix_len)
    mask_dim = prefix_dim + n_diffusion_tokens
    dynamic_shapes = (
        None,
        None,
        {3: prefix_dim},
        {3: prefix_dim},
        None,
        {3: mask_dim},
    )
    return sample_inputs, dynamic_shapes, make_inputs


def _export_diffusion_module(
    module: nn.Module,
    sample_inputs: tuple,
    dynamic_shapes: tuple,
) -> "torch.export.ExportedProgram":
    with torch.no_grad():

        return torch.export._trace._export(
            module,
            sample_inputs,
            dynamic_shapes=dynamic_shapes,
            strict=False,
            prefer_deferred_runtime_asserts_over_guards=True,
        )


def _build_trt_input_specs(make_inputs: callable, min_prefix_len: int, opt_prefix_len: int, max_prefix_len: int):
    import torch_tensorrt

    min_inputs = make_inputs(min_prefix_len)
    opt_inputs = make_inputs(opt_prefix_len)
    max_inputs = make_inputs(max_prefix_len)
    return [
        torch_tensorrt.Input(
            min_shape=t_min.shape,
            opt_shape=t_opt.shape,
            max_shape=t_max.shape,
            dtype=t_min.dtype,
        )
        for t_min, t_opt, t_max in zip(min_inputs, opt_inputs, max_inputs)
    ]


def compile_diffusion_step_no_cache(
    model: nn.Module,
    max_prefix_len: int,
    min_prefix_len: int = 1,
    batch_size: int = 1,
    device: str = "cuda",
    offload_module_to_cpu: bool = False,
    debug: bool = False,
    accuracy_check: bool = True,
) -> nn.Module:
    import torch_tensorrt
    # batch_size = 1

    dtype = torch.float16
    module, cfg = _build_diffusion_module(model, dtype, device)

    sample_inputs, dynamic_shapes, make_inputs = _make_sample_inputs(
        cfg, min_prefix_len, max_prefix_len, dtype, device, batch_size
    )
    ref_output = None
    if accuracy_check:
        with torch.no_grad():
            ref_output = module(*sample_inputs)

    exported = _export_diffusion_module(module, sample_inputs, dynamic_shapes)
    opt_prefix_len = (min_prefix_len + max_prefix_len) // 2
    trt_input_specs = _build_trt_input_specs(make_inputs, min_prefix_len, opt_prefix_len, max_prefix_len)
    trt_settings = {
        "use_explicit_typing": True,
        "use_fp32_acc": True,
        "truncate_double": True,
        "min_block_size": 1,
        "use_python_runtime": True,
        "debug": debug,
        "allow_complex_guards_as_runtime_asserts": True,
        "offload_module_to_cpu": offload_module_to_cpu,
    }
    with torch_tensorrt.dynamo.Debugger() if debug else nullcontext():
        trt_step = torch_tensorrt.dynamo.compile(exported, inputs=trt_input_specs, **trt_settings)

    if accuracy_check and ref_output is not None:
        with torch.no_grad():
            trt_output = trt_step(*sample_inputs)
        max_diff = torch.abs(ref_output.float() - trt_output.float()).max().item()
        mean_diff = torch.abs(ref_output.float() - trt_output.float()).mean().item()
        logger.info("Diffusion TRT check max|delta|=%.6f mean|delta|=%.6f", max_diff, mean_diff)

    model._trt_diffusion_step_no_cache = trt_step
    return trt_step


def save_diffusion_engine(
    model: nn.Module,
    path: str,
    max_prefix_len: int,
    min_prefix_len: int = 1,
    batch_size: int = 1,
    device: str = "cuda",
) -> bool:
    import torch_tensorrt

    from alpamayo1_5.trt.engine_io import save_trt_engine

    dtype = torch.float16
    module, cfg = _build_diffusion_module(model, dtype, device)
    sample_inputs, dynamic_shapes, make_inputs = _make_sample_inputs(
        cfg, min_prefix_len, max_prefix_len, dtype, device, batch_size
    )
    try:
        exported = _export_diffusion_module(module, sample_inputs, dynamic_shapes)
    except Exception as e:
        logger.error("All export methods failed: %s", e)
        return False

    opt_prefix_len = (min_prefix_len + max_prefix_len) // 2
    trt_input_specs = _build_trt_input_specs(make_inputs, min_prefix_len, opt_prefix_len, max_prefix_len)
    trt_settings = {
        "truncate_double": True,
        "min_block_size": 1,
        "use_explicit_typing": True,
        "use_fp32_acc": True,
        "immutable_weights": True,
    }
    try:
        engine_bytes = torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine(
            exported, inputs=trt_input_specs, **trt_settings
        )
    except Exception as e:
        logger.error("TRT serialization failed: %s", e)
        return False

    metadata = {
        "component": "diffusion",
        "save_format": "raw_trt_engine",
        "precision": "FP16",
        "min_prefix_len": min_prefix_len,
        "max_prefix_len": max_prefix_len,
        "n_diffusion_tokens": cfg["n_diffusion_tokens"],
        "action_space_dims": list(cfg["action_space_dims"]),
        "num_layers": cfg["num_layers"],
        "num_kv_heads": cfg["num_kv_heads"],
        "head_dim": cfg["head_dim"],
    }
    save_trt_engine(engine_bytes, path, metadata)
    return True
