# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import logging
import time
from typing import Any, Callable

import numpy as np
import torch

from alpamayo1_5.trt.compile_trt import (
    compile_trt_modules,
    run_inference_trt,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_test_data(clip_id: str, t0_us: int = 5_100_000) -> tuple[dict, list]:
    from alpamayo1_5 import helper
    from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset

    data = load_physical_aiavdataset(clip_id, t0_us=t0_us)
    messages = helper.create_message(
        frames=data["image_frames"].flatten(0, 1), camera_indices=data["camera_indices"]
    )
    return data, messages


def prepare_model_inputs(
    model, data: dict, messages: list, device: str = "cuda"
) -> Callable[[], dict[str, Any]]:
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


def run_inference_pytorch(
    model,
    create_inputs_fn: callable,
    seed: int = 42,
    num_traj_samples: int = 1,
    max_generation_length: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, dict, float]:
    torch.cuda.manual_seed_all(seed)
    model_inputs = create_inputs_fn()
    start_time = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=num_traj_samples,
            max_generation_length=max_generation_length,
            return_extra=True,
        )
    return pred_xyz, pred_rot, extra, time.perf_counter() - start_time


def benchmark_inference(
    run_fn: callable,
    num_runs: int = 5,
    warmup_runs: int = 2,
    label: str = "Model",
) -> float:
    logger.info("Benchmarking %s...", label)
    for _ in range(warmup_runs):
        run_fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        _, _, _, elapsed = run_fn()
        torch.cuda.synchronize()
        times.append(elapsed)
    return sum(times) / len(times)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TRT test for Alpamayo1.5")
    parser.add_argument("--model_path", type=str, default="nvidia/Alpamayo-1.5-10B")
    parser.add_argument("--clip_id", type=str, default="030c760c-ae38-49aa-9ad8-f5650a545d26")
    parser.add_argument("--offload_module_to_cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-trt", action="store_true", help="Skip TRT compilation")
    parser.add_argument("--quick", action="store_true", help="Skip full TRT inference")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark-runs", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data, messages = load_test_data(args.clip_id)
    gt_xyz = data["ego_future_xyz"]

    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

    model = Alpamayo1_5.from_pretrained(args.model_path, dtype=torch.bfloat16).to("cuda")
    model.eval()
    create_inputs_fn = prepare_model_inputs(model, data, messages, device="cuda")

    pred_xyz_pytorch, _, extra_pytorch, pytorch_time = run_inference_pytorch(model, create_inputs_fn, seed=args.seed)
    pytorch_metrics = compute_trajectory_metrics(pred_xyz_pytorch, gt_xyz)
    logger.info("PyTorch minADE: %.4f m", pytorch_metrics["min_ade"])
    logger.info("PyTorch CoC: %s...", extra_pytorch["cot"][0][0, 0][:120])

    if args.skip_trt:
        return 0

    pred_xyz_nocache, _, _, nocache_time = run_inference_trt(
        model,
        create_inputs_fn,
        trt_vision=None,
        trt_lm=None,
        trt_diffusion=None,
        seed=args.seed,
    )
    nocache_metrics = compute_trajectory_metrics(pred_xyz_nocache, gt_xyz)
    logger.info("PyTorch no-cache minADE: %.4f m", nocache_metrics["min_ade"])

    if args.quick:
        return 0

    trt_vision, trt_lm, trt_diffusion, prefix_seq_len = compile_trt_modules(
        model,
        create_inputs_fn,
        seed=args.seed,
        offload_module_to_cpu=args.offload_module_to_cpu,
        max_generation_length=256,
    )

    pred_xyz_trt, _, extra_trt, trt_time = run_inference_trt(
        model,
        create_inputs_fn,
        trt_vision=trt_vision,
        trt_lm=trt_lm,
        trt_diffusion=trt_diffusion,
        seed=args.seed,
    )
    trt_metrics = compute_trajectory_metrics(pred_xyz_trt, gt_xyz)
    logger.info("TRT minADE: %.4f m", trt_metrics["min_ade"])
    logger.info("TRT CoC: %s...", extra_trt["cot"][0][0, 0][:120])
    logger.info("max_prefix_len (TRT): %d", prefix_seq_len)
    logger.info(
        "timing (single run) pytorch=%.2fms nocache=%.2fms trt=%.2fms",
        pytorch_time * 1000,
        nocache_time * 1000,
        trt_time * 1000,
    )

    if args.benchmark:
        pytorch_avg = benchmark_inference(
            run_fn=lambda: run_inference_pytorch(model, create_inputs_fn, seed=args.seed),
            num_runs=args.benchmark_runs,
            label="PyTorch KV-cache",
        )
        nocache_avg = benchmark_inference(
            run_fn=lambda: run_inference_trt(model, create_inputs_fn, None, None, None, seed=args.seed),
            num_runs=args.benchmark_runs,
            label="PyTorch no-cache",
        )
        trt_avg = benchmark_inference(
            run_fn=lambda: run_inference_trt(
                model, create_inputs_fn, trt_vision, trt_lm, trt_diffusion, seed=args.seed
            ),
            num_runs=args.benchmark_runs,
            label="Full TRT",
        )
        logger.info(
            "benchmark avg (ms): pytorch=%.2f nocache=%.2f trt=%.2f",
            pytorch_avg * 1000,
            nocache_avg * 1000,
            trt_avg * 1000,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
