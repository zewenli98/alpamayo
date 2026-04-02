# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from alpamayo1_5.trt.lm_with_cache import compile_vlm_lm_trt_with_cache

logger = logging.getLogger(__name__)


def compile_vlm_lm_trt(
    model: nn.Module,
    max_seq_len: int = 4096,
    precision: str = "BF16",
    device: str = "cuda",
    debug: bool = False,
    accuracy_check: bool = False,
) -> nn.Module:
    del precision
    logger.info(
        "compile_vlm_lm_trt delegates to compile_vlm_lm_trt_with_cache for Alpamayo1.5"
    )
    return compile_vlm_lm_trt_with_cache(
        model=model,
        max_seq_len=max_seq_len,
        max_prefix_len=max_seq_len,
        batch_size=2,
        device=device,
        offload_module_to_cpu=False,
        debug=debug,
        accuracy_check=accuracy_check,
    )


def generate_alpamayo_with_static_cache(
    model: nn.Module,
    trt_backbone: nn.Module,
    input_ids: torch.LongTensor,
    tokenized_data: dict,
    eos_token_id: int,
    max_new_tokens: int = 256,
    top_p: float = 0.98,
    temperature: float = 0.6,
    num_return_sequences: int = 1,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    del trt_backbone, eos_token_id, device, dtype
    generation_config = model.vlm.generation_config
    generation_config.top_p = top_p
    generation_config.temperature = temperature
    generation_config.do_sample = True
    generation_config.num_return_sequences = num_return_sequences
    generation_config.max_new_tokens = max_new_tokens
    generation_config.return_dict_in_generate = True
    generation_config.top_k = None
    generation_config.pad_token_id = model.tokenizer.pad_token_id
    return model.vlm.generate(input_ids=input_ids, generation_config=generation_config, **tokenized_data)
