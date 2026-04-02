# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import logging
import pathlib
import time
from typing import Any

import einops
import numpy as np
import torch
import torch.nn as nn
from transformers import StoppingCriteriaList
from transformers.generation.logits_process import LogitsProcessorList

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class VisionTRTEngineWrapper:
    def __init__(self, runner):
        self.runner = runner

    def forward(self, hidden_states: torch.Tensor, grid_thw=None):
        outputs = self.runner(hidden_states)
        return outputs[0], outputs[1:]

    def __call__(self, hidden_states: torch.Tensor, grid_thw=None):
        return self.forward(hidden_states, grid_thw)


class DiffusionTRTEngineWrapper:
    def __init__(self, runner):
        self.runner = runner

    def __call__(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        prefix_k: torch.Tensor,
        prefix_v: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        result = self.runner(x, t, prefix_k, prefix_v, position_ids, attention_mask)
        if isinstance(result, (list, tuple)):
            return result[0]
        return result


def load_test_data(clip_id: str, t0_us: int = 5_100_000) -> tuple[dict, list]:
    from alpamayo1_5 import helper
    from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset

    data = load_physical_aiavdataset(clip_id, t0_us=t0_us)
    messages = helper.create_message(
        frames=data["image_frames"].flatten(0, 1), camera_indices=data["camera_indices"]
    )
    return data, messages


def prepare_model_inputs(model, data: dict, messages: list, device: str = "cuda") -> callable:
    from alpamayo1_5 import helper

    processor = helper.get_processor(model.tokenizer)

    def create_inputs():
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )
        model_inputs = {
            "tokenized_data": inputs,
            "ego_history_xyz": data["ego_history_xyz"].clone(),
            "ego_history_rot": data["ego_history_rot"].clone(),
        }
        return helper.to_device(model_inputs, device)

    return create_inputs


def compute_trajectory_metrics(pred_xyz: torch.Tensor, gt_xyz: torch.Tensor) -> dict[str, float]:
    gt_xy = gt_xyz.cpu()[0, 0, :, :2].T.numpy()
    pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
    ade_per_sample = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
    fde_per_sample = np.linalg.norm(pred_xy[:, :, -1] - gt_xy[:, -1], axis=1)
    return {
        "min_ade": float(ade_per_sample.min()),
        "mean_ade": float(ade_per_sample.mean()),
        "min_fde": float(fde_per_sample.min()),
        "mean_fde": float(fde_per_sample.mean()),
    }


def measure_prefix_seq_len(model, create_inputs_fn: callable, seed: int = 42) -> int:
    from alpamayo1_5.models.alpamayo1_5 import ExpertLogitsProcessor
    from alpamayo1_5.models.token_utils import StopAfterEOS, to_special_token

    torch.cuda.manual_seed_all(seed)
    _inputs = create_inputs_fn()
    _tokenized = _inputs["tokenized_data"]
    _input_ids = _tokenized.pop("input_ids")
    _input_ids = model.fuse_traj_tokens(
        _input_ids,
        {"ego_history_xyz": _inputs["ego_history_xyz"], "ego_history_rot": _inputs["ego_history_rot"]},
    )
    _eos_id = model.tokenizer.convert_tokens_to_ids(to_special_token("traj_future_start"))

    gen_cfg = model.vlm.generation_config
    gen_cfg.top_p = 0.98
    gen_cfg.temperature = 0.6
    gen_cfg.do_sample = True
    gen_cfg.num_return_sequences = 1
    gen_cfg.max_new_tokens = 256
    gen_cfg.output_logits = True
    gen_cfg.return_dict_in_generate = True
    gen_cfg.top_k = None
    gen_cfg.pad_token_id = model.tokenizer.pad_token_id

    with torch.no_grad():
        vlm_out = model.vlm.generate(
            input_ids=_input_ids,
            generation_config=gen_cfg,
            stopping_criteria=StoppingCriteriaList([StopAfterEOS(eos_token_id=_eos_id)]),
            logits_processor=LogitsProcessorList(
                [
                    ExpertLogitsProcessor(
                        traj_token_offset=model.config.traj_token_start_idx,
                        traj_vocab_size=model.config.traj_vocab_size,
                    )
                ]
            ),
            **_tokenized,
        )
    return int(vlm_out.past_key_values.get_seq_length())


def save_engines(
    model: nn.Module,
    model_inputs: dict[str, Any],
    engine_dir: str | pathlib.Path,
    max_prefix_len: int,
) -> bool:
    from alpamayo1_5.trt.diffusion import save_diffusion_engine
    from alpamayo1_5.trt.vision import save_vision_engine

    engine_dir = pathlib.Path(engine_dir)
    engine_dir.mkdir(parents=True, exist_ok=True)
    vision_path = engine_dir / "vision.trt"
    diffusion_path = engine_dir / "diffusion.trt"

    ok_vision = save_vision_engine(model.vlm.model.visual, model_inputs, path=str(vision_path), device="cuda")
    if not ok_vision:
        return False
    ok_diffusion = save_diffusion_engine(
        model,
        path=str(diffusion_path),
        max_prefix_len=max_prefix_len,
        min_prefix_len=1,
        batch_size=1,
        device="cuda",
    )
    return bool(ok_diffusion)


def load_vision_engine(engine_dir: str | pathlib.Path, device: str = "cuda") -> VisionTRTEngineWrapper:
    from alpamayo1_5.trt.engine_io import TRTEngineRunner

    vision_path = pathlib.Path(engine_dir) / "vision.trt"
    if not vision_path.exists():
        raise FileNotFoundError(f"Vision engine not found: {vision_path}")
    return VisionTRTEngineWrapper(TRTEngineRunner(vision_path, device=device))


def load_engines(
    engine_dir: str | pathlib.Path,
    device: str = "cuda",
) -> tuple[VisionTRTEngineWrapper, DiffusionTRTEngineWrapper]:
    from alpamayo1_5.trt.engine_io import TRTEngineRunner

    engine_dir = pathlib.Path(engine_dir)
    vision_path = engine_dir / "vision.trt"
    diffusion_path = engine_dir / "diffusion.trt"
    if not vision_path.exists():
        raise FileNotFoundError(f"Vision engine not found: {vision_path}")
    if not diffusion_path.exists():
        raise FileNotFoundError(f"Diffusion engine not found: {diffusion_path}")
    return (
        VisionTRTEngineWrapper(TRTEngineRunner(vision_path, device=device)),
        DiffusionTRTEngineWrapper(TRTEngineRunner(diffusion_path, device=device)),
    )


def run_inference_pure_trt(
    model,
    create_inputs_fn: callable,
    trt_vision,
    trt_diffusion,
    seed: int = 42,
    num_traj_samples: int = 1,
    max_generation_length: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, dict, float]:
    from alpamayo1_5.models.alpamayo1_5 import ExpertLogitsProcessor
    from alpamayo1_5.models.token_utils import (
        StopAfterEOS,
        extract_text_tokens,
        replace_padding_after_eos,
        to_special_token,
    )

    torch.cuda.manual_seed_all(seed)
    model_inputs = create_inputs_fn()
    dtype = torch.bfloat16
    device = "cuda"
    start_time = time.perf_counter()

    with torch.autocast("cuda", dtype=dtype):
        ego_history_xyz = model_inputs["ego_history_xyz"]
        ego_history_rot = model_inputs["ego_history_rot"]
        bsz, _, _, _ = ego_history_xyz.shape
        tokenized_data = model_inputs["tokenized_data"]
        input_ids = tokenized_data.pop("input_ids")
        input_ids = model.fuse_traj_tokens(
            input_ids, {"ego_history_xyz": ego_history_xyz, "ego_history_rot": ego_history_rot}
        )

        original_vision_forward = None
        if trt_vision is not None:
            original_vision_forward = model.vlm.model.visual.forward
            model.vlm.model.visual.forward = trt_vision.forward

        eos_token_id = model.tokenizer.convert_tokens_to_ids(to_special_token("traj_future_start"))
        generation_config = model.vlm.generation_config
        generation_config.top_p = 0.98
        generation_config.temperature = 0.6
        generation_config.do_sample = True
        generation_config.num_return_sequences = num_traj_samples
        generation_config.max_new_tokens = max_generation_length
        generation_config.output_logits = True
        generation_config.return_dict_in_generate = True
        generation_config.top_k = None
        generation_config.pad_token_id = model.tokenizer.pad_token_id

        stopping_criteria = StoppingCriteriaList([StopAfterEOS(eos_token_id=eos_token_id)])
        logits_processor = LogitsProcessorList(
            [
                ExpertLogitsProcessor(
                    traj_token_offset=model.config.traj_token_start_idx,
                    traj_vocab_size=model.config.traj_vocab_size,
                )
            ]
        )
        vlm_outputs = model.vlm.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            stopping_criteria=stopping_criteria,
            logits_processor=logits_processor,
            **tokenized_data,
        )
        vlm_outputs.rope_deltas = model.vlm.model.rope_deltas
        if original_vision_forward is not None:
            model.vlm.model.visual.forward = original_vision_forward

        vlm_outputs.sequences = replace_padding_after_eos(
            token_ids=vlm_outputs.sequences,
            eos_token_id=eos_token_id,
            pad_token_id=model.tokenizer.pad_token_id,
        )

        b_star = vlm_outputs.sequences.shape[0]
        traj_future_start_mask = vlm_outputs.sequences == eos_token_id
        has_traj_future_start = traj_future_start_mask.any(dim=1)
        traj_future_start_positions = traj_future_start_mask.int().argmax(dim=1)
        last_token_positions = torch.full((b_star,), vlm_outputs.sequences.shape[1] - 1, device=device)
        valid_token_pos_id = torch.where(has_traj_future_start, traj_future_start_positions, last_token_positions)
        offset = valid_token_pos_id + 1

        n_diffusion_tokens = model.action_space.get_action_space_dims()[0]
        prompt_cache = vlm_outputs.past_key_values
        prefill_seq_len = prompt_cache.get_seq_length()

        position_ids = torch.arange(n_diffusion_tokens, device=device)
        position_ids = einops.repeat(position_ids, "l -> 3 b l", b=b_star).clone()
        position_ids = position_ids + (vlm_outputs.rope_deltas + offset[:, None]).to(device)

        neg_inf = torch.finfo(torch.float32).min
        attention_mask = torch.zeros(
            b_star, 1, n_diffusion_tokens, prefill_seq_len + n_diffusion_tokens, dtype=torch.float32, device=device
        )
        for i in range(b_star):
            attention_mask[i, :, :, offset[i] : -n_diffusion_tokens] = neg_inf
        attention_mask = attention_mask.to(dtype)

        prefix_k = torch.stack([layer.keys for layer in prompt_cache.layers], dim=0)
        prefix_v = torch.stack([layer.values for layer in prompt_cache.layers], dim=0)

        forward_kwargs = {}
        if model.config.expert_non_causal_attention:
            forward_kwargs["is_causal"] = False

        if trt_diffusion is not None:

            def step_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
                return trt_diffusion(x.to(dtype), t.to(dtype), prefix_k, prefix_v, position_ids, attention_mask)

        else:

            def step_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
                b = x.shape[0]
                future_token_embeds = model.action_in_proj(x, t)
                if future_token_embeds.dim() == 2:
                    future_token_embeds = future_token_embeds.view(b, n_diffusion_tokens, -1)
                expert_out = model.expert(
                    inputs_embeds=future_token_embeds,
                    position_ids=position_ids,
                    past_key_values=prompt_cache,
                    attention_mask=attention_mask,
                    use_cache=True,
                    **forward_kwargs,
                )
                prompt_cache.crop(prefill_seq_len)
                last_hidden = expert_out.last_hidden_state[:, -n_diffusion_tokens:]
                return model.action_out_proj(last_hidden).view(-1, *model.action_space.get_action_space_dims())

        total_batch = bsz * num_traj_samples
        sampled_action = model.diffusion.sample(
            batch_size=total_batch, step_fn=step_fn, device=device, return_all_steps=False
        )

        hist_xyz_rep = einops.repeat(ego_history_xyz[:, -1], "b ... -> (b n) ...", n=num_traj_samples)
        hist_rot_rep = einops.repeat(ego_history_rot[:, -1], "b ... -> (b n) ...", n=num_traj_samples)
        pred_xyz, pred_rot = model.action_space.action_to_traj(sampled_action, hist_xyz_rep, hist_rot_rep)
        pred_xyz = einops.rearrange(pred_xyz, "(b ns nj) ... -> b ns nj ...", ns=1, nj=num_traj_samples)
        pred_rot = einops.rearrange(pred_rot, "(b ns nj) ... -> b ns nj ...", ns=1, nj=num_traj_samples)

        extra = extract_text_tokens(model.tokenizer, vlm_outputs.sequences)
        for k in extra:
            extra[k] = np.array(extra[k]).reshape([input_ids.shape[0], 1, num_traj_samples])

    return pred_xyz, pred_rot, extra, time.perf_counter() - start_time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TRT engine test for Alpamayo1.5")
    parser.add_argument("--model_path", type=str, default="nvidia/Alpamayo-1.5-10B")
    parser.add_argument("--clip_id", type=str, default="030c760c-ae38-49aa-9ad8-f5650a545d26")
    parser.add_argument("--engine-dir", type=str, default="/tmp/alpamayo1_5_engines")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--infer", action="store_true")
    parser.add_argument("--vision-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    do_save = args.save or (not args.save and not args.infer)
    do_infer = args.infer or (not args.save and not args.infer)

    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

    model = Alpamayo1_5.from_pretrained(args.model_path, dtype=torch.bfloat16).to("cuda")
    model.eval()
    data, messages = load_test_data(args.clip_id)
    gt_xyz = data["ego_future_xyz"]
    create_inputs_fn = prepare_model_inputs(model, data, messages, device="cuda")
    model_inputs = create_inputs_fn()

    max_prefix_len = None
    if do_save:
        max_prefix_len = measure_prefix_seq_len(model, create_inputs_fn, seed=args.seed)
        ok = save_engines(model, model_inputs, engine_dir=args.engine_dir, max_prefix_len=max_prefix_len)
        if not ok:
            return 1

    pred_xyz_pt, _, _, pt_time = run_inference_pure_trt(
        model, create_inputs_fn, trt_vision=None, trt_diffusion=None, seed=args.seed
    )
    pt_metrics = compute_trajectory_metrics(pred_xyz_pt, gt_xyz)
    logger.info("PyTorch no-cache minADE: %.4f (%.1fms)", pt_metrics["min_ade"], pt_time * 1000)

    if not do_infer:
        return 0

    if args.vision_only:
        trt_vision = load_vision_engine(args.engine_dir, device="cuda")
        trt_diffusion = None
    else:
        trt_vision, trt_diffusion = load_engines(args.engine_dir, device="cuda")

    pred_xyz_trt, _, extra_trt, trt_time = run_inference_pure_trt(
        model, create_inputs_fn, trt_vision=trt_vision, trt_diffusion=trt_diffusion, seed=args.seed
    )
    trt_metrics = compute_trajectory_metrics(pred_xyz_trt, gt_xyz)
    logger.info("TRT minADE: %.4f (%.1fms)", trt_metrics["min_ade"], trt_time * 1000)
    logger.info("TRT CoC: %s...", extra_trt["cot"][0][0, 0][:120])

    diff = torch.abs(pred_xyz_pt.cpu().float() - pred_xyz_trt.cpu().float())
    ade_diff = abs(pt_metrics["min_ade"] - trt_metrics["min_ade"])
    logger.info(
        "comparison max_diff=%.6f mean_diff=%.6f ade_diff=%.4f",
        diff.max().item(),
        diff.mean().item(),
        ade_diff,
    )
    return 0 if ade_diff < 0.15 else 1


if __name__ == "__main__":
    raise SystemExit(main())
