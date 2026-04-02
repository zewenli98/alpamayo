# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_orig_qwen3vl_attn_forward = None


def _patch_qwen3vl_vision_attention() -> bool:
    global _orig_qwen3vl_attn_forward

    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLVisionAttention,
            apply_rotary_pos_emb_vision,
            eager_attention_forward,
        )
    except ImportError:
        logger.warning("Could not import Qwen3VL attention modules - patch not applied")
        return False

    if _orig_qwen3vl_attn_forward is not None:
        return True

    _orig_qwen3vl_attn_forward = Qwen3VLVisionAttention.forward

    def _static_lengths_forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        static_lengths = getattr(self, "_static_lengths", None)
        if static_lengths is None or self.config._attn_implementation == "flash_attention_2":
            return _orig_qwen3vl_attn_forward(
                self,
                hidden_states,
                cu_seqlens,
                rotary_pos_emb=rotary_pos_emb,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        attention_interface = eager_attention_forward
        if self.config._attn_implementation in ALL_ATTENTION_FUNCTIONS:
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        splits = [torch.split(t, static_lengths, dim=2) for t in (query_states, key_states, value_states)]
        attn_outputs = [
            attention_interface(
                self,
                q,
                k,
                v,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                is_causal=False,
                **kwargs,
            )[0]
            for q, k, v in zip(*splits)
        ]

        attn_output = torch.cat(attn_outputs, dim=1).reshape(seq_length, -1).contiguous()
        return self.proj(attn_output)

    Qwen3VLVisionAttention.forward = _static_lengths_forward
    logger.info("Patched Qwen3VLVisionAttention.forward for static lengths")
    return True


class VisualFixedGrid(nn.Module):
    def __init__(self, visual: nn.Module, grid_thw: torch.Tensor):
        super().__init__()
        self.visual = visual.eval()

        with torch.no_grad():
            pos_embeds = self.visual.fast_pos_embed_interpolate(grid_thw)

            rotary_pos_emb = self.visual.rot_pos_emb(grid_thw)
            seq_len = pos_embeds.shape[0]
            rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)

            cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
                dim=0, dtype=torch.int32
            )
            cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

            static_lengths = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cpu().tolist()
            self._static_lengths = [int(x) for x in static_lengths]
            for blk in self.visual.blocks:
                blk.attn._static_lengths = self._static_lengths

        self.register_buffer("pos_embeds", pos_embeds, persistent=False)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)
        self.register_buffer("cu_seqlens", cu_seqlens, persistent=False)
        logger.info("VisualFixedGrid: %s tokens, static_lengths=%s", seq_len, self._static_lengths)

    def forward(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        hidden_states = self.visual.patch_embed(hidden_states)
        torch._check(hidden_states.shape[0] != 0)
        hidden_states = hidden_states + self.pos_embeds.to(hidden_states.dtype)

        position_embeddings = (
            self.cos.to(hidden_states.dtype),
            self.sin.to(hidden_states.dtype),
        )

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.visual.blocks):
            hidden_states = blk(hidden_states, cu_seqlens=self.cu_seqlens, position_embeddings=position_embeddings)
            if layer_num in self.visual.deepstack_visual_indexes:
                idx = self.visual.deepstack_visual_indexes.index(layer_num)
                deepstack_feature_lists.append(self.visual.deepstack_merger_list[idx](hidden_states))

        hidden_states = self.visual.merger(hidden_states)
        return hidden_states, deepstack_feature_lists


class _PixelOnlyWrapper(nn.Module):
    def __init__(self, vfg: VisualFixedGrid):
        super().__init__()
        self.vfg = vfg

    def forward(self, pixel_values: torch.Tensor):
        return self.vfg(pixel_values, grid_thw=None)


class _RepeatCollapseVisionWrapper(nn.Module):
    def __init__(
        self,
        trt_model: nn.Module,
        base_pixel_rows: int,
        base_grid_rows: int,
    ):
        super().__init__()
        self.trt_model = trt_model
        self.base_pixel_rows = int(base_pixel_rows)
        self.base_grid_rows = int(base_grid_rows)

    @staticmethod
    def _blocks_identical(x: torch.Tensor, repeat_factor: int) -> bool:
        if repeat_factor <= 1:
            return True
        block = x.shape[0] // repeat_factor
        first = x[:block]
        for i in range(1, repeat_factor):
            cur = x[i * block : (i + 1) * block]
            if not torch.equal(first, cur):
                return False
        return True

    @staticmethod
    def _repeat_first_dim(x: torch.Tensor, repeat_factor: int) -> torch.Tensor:
        if repeat_factor <= 1:
            return x
        return x.repeat((repeat_factor,) + (1,) * (x.dim() - 1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if grid_thw is None:
            return self.trt_model(hidden_states, None)

        total_pixel_rows = int(hidden_states.shape[0])
        total_grid_rows = int(grid_thw.shape[0])

        can_factor = (
            self.base_pixel_rows > 0
            and self.base_grid_rows > 0
            and total_pixel_rows % self.base_pixel_rows == 0
            and total_grid_rows % self.base_grid_rows == 0
        )
        if not can_factor:
            return self.trt_model(hidden_states, None)

        repeat_factor_pixel = total_pixel_rows // self.base_pixel_rows
        repeat_factor_grid = total_grid_rows // self.base_grid_rows
        if repeat_factor_pixel <= 1 or repeat_factor_pixel != repeat_factor_grid:
            return self.trt_model(hidden_states, None)

        repeat_factor = int(repeat_factor_pixel)
        if not self._blocks_identical(grid_thw, repeat_factor):
            return self.trt_model(hidden_states, None)
        if not self._blocks_identical(hidden_states, repeat_factor):
            return self.trt_model(hidden_states, None)

        base_hidden = hidden_states[: self.base_pixel_rows]
        trt_out = self.trt_model(base_hidden, None)
        main_out = self._repeat_first_dim(trt_out[0], repeat_factor)
        deepstack_out = [self._repeat_first_dim(x, repeat_factor) for x in trt_out[1]]
        return main_out, deepstack_out


def _prepare_vision_module(
    visual_model: nn.Module,
    model_inputs: dict[str, Any],
    device: str,
) -> tuple[nn.Module, torch.Tensor, torch.Tensor]:
    dtype = torch.float16

    visual_model.config.attn_implementation = "sdpa"
    visual_model.config._attn_implementation = "sdpa"
    visual_model.config.use_cache = False
    visual_model = visual_model.to(dtype=dtype, device=device).eval()

    pixel_values = model_inputs["tokenized_data"]["pixel_values"].to(dtype=dtype, device=device)
    image_grid_thw = model_inputs["tokenized_data"]["image_grid_thw"].to(device=device)

    wrapped = VisualFixedGrid(visual_model, image_grid_thw).to(device).eval()
    return wrapped, pixel_values, image_grid_thw


def _export_vision_module(module: nn.Module, inputs: tuple) -> "torch.export.ExportedProgram":
    try:
        ep = torch.export.export(module, args=inputs, strict=False)
        logger.info("Export succeeded")
        return ep
    except Exception as e:
        logger.warning("Standard export failed (%s), trying _trace._export...", e)
        ep = torch.export._trace._export(
            module,
            args=inputs,
            strict=False,
            prefer_deferred_runtime_asserts_over_guards=True,
        )
        logger.info("Trace export succeeded")
        return ep


def compile_vision_model(
    visual_model: nn.Module,
    model_inputs: dict[str, Any],
    device: str = "cuda",
    debug: bool = False,
    offload_module_to_cpu: bool = False,
) -> nn.Module | None:
    import torch_tensorrt

    if not _patch_qwen3vl_vision_attention():
        logger.error("Failed to patch vision attention - aborting")
        return None

    wrapped, pixel_values, image_grid_thw = _prepare_vision_module(visual_model, model_inputs, device)
    logger.info("  pixel_values shape: %s", pixel_values.shape)
    logger.info("  image_grid_thw: %s", image_grid_thw)

    inputs = (pixel_values, None)
    trt_settings = {
        "truncate_double": True,
        "min_block_size": 1,
        "use_python_runtime": True,
        "immutable_weights": True,
        "offload_module_to_cpu": offload_module_to_cpu,
        "use_explicit_typing": True,
        "use_fp32_acc": True,
    }

    try:
        ep = _export_vision_module(wrapped, inputs)
    except Exception as e:
        logger.error("All export methods failed: %s", e)
        return None

    try:
        with (
            torch_tensorrt.dynamo.Debugger(log_level="debug", engine_builder_monitor=False)
            if debug
            else nullcontext()
        ):
            trt_model = torch_tensorrt.dynamo.compile(ep, inputs, **trt_settings)
    except Exception as e:
        logger.error("TRT compilation failed: %s", e)
        return None

    if offload_module_to_cpu:
        wrapped = wrapped.to(device=device, dtype=torch.float16).eval()
    wrapped_trt_model = _RepeatCollapseVisionWrapper(
        trt_model=trt_model,
        base_pixel_rows=int(pixel_values.shape[0]),
        base_grid_rows=int(image_grid_thw.shape[0]),
    ).eval()

    with torch.no_grad():
        torch_out = wrapped(*inputs)
        trt_out = wrapped_trt_model(*inputs)
    main_diff = torch.abs(torch_out[0].float() - trt_out[0].float())
    logger.info("  max|delta|=%.6f mean|delta|=%.6f", main_diff.max().item(), main_diff.mean().item())
    return wrapped_trt_model


def compile_and_replace_vision_model(
    model: nn.Module,
    model_inputs: dict[str, Any],
    device: str = "cuda",
    debug: bool = False,
    offload_module_to_cpu: bool = False,
) -> bool:
    compiled = compile_vision_model(
        model.vlm.model.visual,
        model_inputs,
        device=device,
        debug=debug,
        offload_module_to_cpu=offload_module_to_cpu,
    )
    if compiled is None:
        logger.error("Vision model compilation failed")
        return False

    model.vlm.model.visual.forward = compiled.forward
    model._trt_vision_model = compiled
    logger.info("Vision model replaced with TRT-compiled version")
    return True


def save_vision_engine(
    visual_model: nn.Module,
    model_inputs: dict[str, Any],
    path: str,
    device: str = "cuda",
    offload_module_to_cpu: bool = False,
) -> bool:
    import torch_tensorrt

    from alpamayo1_5.trt.engine_io import save_trt_engine

    if not _patch_qwen3vl_vision_attention():
        logger.error("Failed to patch vision attention - aborting")
        return False

    wrapped_base, pixel_values, image_grid_thw = _prepare_vision_module(visual_model, model_inputs, device)
    wrapped = _PixelOnlyWrapper(wrapped_base).to(device).eval()
    inputs = (pixel_values,)

    trt_settings = {
        "truncate_double": True,
        "min_block_size": 1,
        "use_explicit_typing": True,
        "use_fp32_acc": True,
        "immutable_weights": True,
        "offload_module_to_cpu": offload_module_to_cpu,
    }

    try:
        ep = _export_vision_module(wrapped, inputs)
    except Exception as e:
        logger.error("All export methods failed: %s", e)
        return False

    trt_input_spec = torch_tensorrt.Input.from_tensor(pixel_values)
    try:
        engine_bytes = torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine(
            ep, inputs=(trt_input_spec,), **trt_settings
        )
    except Exception as e:
        logger.error("TRT serialization failed: %s", e)
        return False

    metadata = {
        "component": "vision",
        "precision": "FP16",
        "pixel_values_shape": list(pixel_values.shape),
        "pixel_values_dtype": str(pixel_values.dtype),
        "image_grid_thw": image_grid_thw.tolist(),
    }
    save_trt_engine(engine_bytes, path, metadata)
    logger.info("Vision engine saved to %s", path)
    return True
