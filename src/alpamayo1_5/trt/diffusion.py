# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from contextlib import nullcontext
from types import MethodType

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


def _disable_action_in_proj_first_linear_quantizers(action_in_proj: nn.Module) -> None:
    """
    This patch is specifically for the error in AutoQuant (fp8/nvfp4) quantization. 
    Disable block quantizers on the 60-wide Fourier input that TRT cannot tile by 16.
    """
    encoder = getattr(action_in_proj, "encoder", None)
    trunk = getattr(encoder, "trunk", None)
    if trunk is None or len(trunk) == 0:
        return

    first_layer = trunk[0]
    disabled = []
    for quantizer_name in ("input_quantizer", "weight_quantizer"):
        quantizer = getattr(first_layer, quantizer_name, None)
        disable = getattr(quantizer, "disable", None)
        if callable(disable):
            disable()
            disabled.append(quantizer_name)

    if disabled:
        logger.info(
            "Disabled action_in_proj.encoder.trunk.0 %s before TRT diffusion compile",
            ", ".join(disabled),
        )


def _patch_expert_o_proj_static_last_dim(expert: nn.Module) -> None:
    """
    This patch is specifically for the error in AutoQuant (fp8/nvfp4) quantization. 
    Make expert attention o_proj inputs expose a static hidden dimension to TRT.
    """
    patched = 0
    for layer in getattr(expert, "layers", []):
        self_attn = getattr(layer, "self_attn", None)
        o_proj = getattr(self_attn, "o_proj", None)
        if o_proj is None or getattr(o_proj, "_trt_static_last_dim_patched", False):
            continue

        in_features = getattr(o_proj, "in_features", None)
        if in_features is None:
            weight = getattr(o_proj, "weight", None)
            if weight is None or len(getattr(weight, "shape", ())) < 2:
                logger.debug("Skipping expert o_proj static-dim patch; cannot infer in_features for %s", o_proj)
                continue
            in_features = int(weight.shape[1])
        in_features = int(in_features)

        o_proj._trt_original_forward = o_proj.forward
        o_proj._trt_static_o_proj_in_features = in_features

        def _static_last_dim_forward(self, input: torch.Tensor, *args, **kwargs):
            target_shape = tuple(input.shape[:-1]) + (self._trt_static_o_proj_in_features,)
            input = input.reshape(target_shape).contiguous()
            return self._trt_original_forward(input, *args, **kwargs)

        o_proj.forward = MethodType(_static_last_dim_forward, o_proj)
        o_proj._trt_static_last_dim_patched = True
        patched += 1

    logger.info("Patched %d expert self_attn.o_proj module(s) with static last-dim reshape", patched)


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
    max_bsz = int(batch_size)
    if max_bsz <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    opt_prefix_len = (min_prefix_len + max_prefix_len) // 2

    def make_inputs(prefix_len: int, bsz: int | None = None) -> list[torch.Tensor]:
        b = max_bsz if bsz is None else int(bsz)
        return [
            torch.randn(b, *action_space_dims, dtype=dtype, device=device),
            torch.zeros(b, 1, 1, dtype=dtype, device=device),
            torch.zeros(num_layers, b, num_kv_heads, prefix_len, head_dim, dtype=dtype, device=device),
            torch.zeros(num_layers, b, num_kv_heads, prefix_len, head_dim, dtype=dtype, device=device),
            torch.arange(n_diffusion_tokens, device=device).unsqueeze(0).unsqueeze(0).expand(3, b, -1).clone(),
            torch.zeros(b, 1, n_diffusion_tokens, prefix_len + n_diffusion_tokens, dtype=dtype, device=device),
        ]

    sample_inputs = tuple(make_inputs(opt_prefix_len))
    prefix_dim = torch.export.Dim("prefix_len", min=min_prefix_len, max=max_prefix_len)
    mask_dim = prefix_dim + n_diffusion_tokens
    # Declare `batch` as dynamic when max_bsz > 1; otherwise torch.export
    # bakes in the example batch size as a static dim and TRT rejects
    # runtime batches that differ (e.g. compiled bsz=2 but runtime bsz=1)
    # with: "Static dimension mismatch ... Set [...,1,...] Expected [...,2,...]".
    # When max_bsz == 1 we leave batch static (matches the original behaviour
    # and avoids declaring a degenerate Dim with min == max).
    if max_bsz > 1:
        batch_dim = torch.export.Dim("batch", min=1, max=max_bsz)
        dynamic_shapes = (
            {0: batch_dim},                  # x
            {0: batch_dim},                  # t
            {1: batch_dim, 3: prefix_dim},   # prefix_k
            {1: batch_dim, 3: prefix_dim},   # prefix_v
            {1: batch_dim},                  # position_ids
            {0: batch_dim, 3: mask_dim},     # attention_mask
        )
    else:
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


def _build_trt_input_specs(
    make_inputs: callable,
    min_prefix_len: int,
    opt_prefix_len: int,
    max_prefix_len: int,
    max_batch_size: int,
):
    import torch_tensorrt

    # min: smallest batch (1) and smallest prefix
    # opt: midpoint of both (favoured by the TRT optimizer)
    # max: largest batch and largest prefix
    min_bsz = 1
    max_bsz = int(max_batch_size)
    opt_bsz = max(min_bsz, (min_bsz + max_bsz) // 2)
    min_inputs = make_inputs(min_prefix_len, bsz=min_bsz)
    opt_inputs = make_inputs(opt_prefix_len, bsz=opt_bsz)
    max_inputs = make_inputs(max_prefix_len, bsz=max_bsz)
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
    # _disable_action_in_proj_first_linear_quantizers(module.action_in_proj)
    # _patch_expert_o_proj_static_last_dim(module.expert)

    sample_inputs, dynamic_shapes, make_inputs = _make_sample_inputs(
        cfg, min_prefix_len, max_prefix_len, dtype, device, batch_size
    )
    ref_output = None
    if accuracy_check:
        with torch.no_grad():
            ref_output = module(*sample_inputs)

    exported = _export_diffusion_module(module, sample_inputs, dynamic_shapes)
    opt_prefix_len = (min_prefix_len + max_prefix_len) // 2
    trt_input_specs = _build_trt_input_specs(
        make_inputs, min_prefix_len, opt_prefix_len, max_prefix_len, batch_size
    )
    trt_settings = {
        "use_explicit_typing": True,
        "use_fp32_acc": True,
        "truncate_double": True,
        "min_block_size": 1,
        "use_python_runtime": True,
        "debug": debug,
        "allow_complex_guards_as_runtime_asserts": True,
        "offload_module_to_cpu": offload_module_to_cpu,
        "decompose_attention": False,
        # "require_full_compilation": True,
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
    trt_input_specs = _build_trt_input_specs(
        make_inputs, min_prefix_len, opt_prefix_len, max_prefix_len, batch_size
    )
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
