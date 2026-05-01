import json
import logging
import os

import huggingface_hub
import torch
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

try:
    import modelopt.torch.quantization as mtq  # noqa: F401f

    assert torch.ops.tensorrt.quantize_op.default
except Exception:
    logger.warning("Unable to import quantization op. Please install modelopt library")

from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.nn.modules.tensor_quantizer import TensorQuantizer
from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor
from modelopt.torch.utils.dataset_utils import (
    create_forward_loop,
    get_dataset_dataloader,
)
from safetensors import safe_open

# FP8 E4M3 format has a maximum representable value of 448.0
MAX_BOUND_FP8 = 448.0
# Additional scaling factor for NVFP4
MAX_BOUND_NVFP4 = 6.0


# def _disable_modelopt_fp8_cuda_extension() -> None:
#     """
#     Disable ModelOpt FP8 CUDA extension and force eager FP8 fake-quant path.

#     This avoids known illegal-memory-access failures in some environments when
#     modelopt_cuda_ext_fp8 kernels are exercised during export/compile flows.
#     """
#     try:
#         import modelopt.torch.quantization.extensions as mtq_ext
#         import modelopt.torch.quantization.tensor_quant as mtq_tq
#     except Exception:
#         return

#     if getattr(mtq_ext, "_alpamayo_fp8_ext_disabled", False):
#         return

#     def _no_fp8_ext(raise_if_failed: bool = False):
#         return None

#     mtq_ext.get_cuda_ext_fp8 = _no_fp8_ext
#     mtq_tq.get_cuda_ext_fp8 = _no_fp8_ext
#     mtq_ext._alpamayo_fp8_ext_disabled = True
#     logger.info("Disabled modelopt FP8 CUDA extension; using eager FP8 fake quantization")


def quantize_model(model, args, tokenizer=None, calibration_forward_loop=None):
    """
    Quantize a PyTorch model using ModelOpt post-training quantization (PTQ).

    This function applies quantization to reduce model precision for faster inference
    while maintaining acceptable accuracy. It uses calibration data generated from
    the provided tokenizer to determine optimal quantization parameters.

    Supported quantization formats:
        - fp8: 8-bit floating point quantization
        - nvfp4: 4-bit NVIDIA floating point quantization
    Args:
        model: PyTorch model to quantize. Must be in evaluation mode.
        args: Command line arguments containing quant_format and debug.
        tokenizer: Hugging Face tokenizer for creating calibration data.
            Required only when `calibration_forward_loop` is not provided.
        calibration_forward_loop: Optional callable taking `model` and running
            calibration forward passes. Use this for non-text modules whose
            forward signature is not compatible with dataset_utils batches.

    Returns:
        Quantized model
    """
    # Create calibration forward loop. For standard text models we can build
    # it from tokenizer-based data, but vision modules often need custom args.
    if calibration_forward_loop is None:
        if tokenizer is None:
            raise ValueError(
                "tokenizer must be provided when calibration_forward_loop is None"
            )
        calib_dataloader = get_dataset_dataloader(
            tokenizer=tokenizer,
            batch_size=32,
            num_samples=512,
            device="cuda:0",
        )
        calibrate_loop = create_forward_loop(dataloader=calib_dataloader)
    else:
        calibrate_loop = calibration_forward_loop
    if args.quant_format == "int8":
        if args.quant_algo == "smoothquant":
            if args.weight_only:
                raise RuntimeError(
                    "SmoothQuant is supported for weight-and-activation quantization, weight-only flag should not be set"
                )
            quant_cfg = mtq.INT8_SMOOTHQUANT_CFG
        elif args.weight_only:
            quant_cfg = mtq.INT8_WEIGHT_ONLY_CFG
        else:
            raise RuntimeError(
                f"Unsupported args.quant_algo: {args.quant_algo} and args.weight_only: {args.weight_only} for int8 quantization"
            )
    elif args.quant_format == "fp8":
        # _disable_modelopt_fp8_cuda_extension()
        if args.weight_only:
            quant_cfg = mtq.FP8_2D_BLOCKWISE_WEIGHT_ONLY_CFG
        else:
            quant_cfg = mtq.FP8_DEFAULT_CFG
    elif args.quant_format == "nvfp4":
        quant_cfg = mtq.NVFP4_DEFAULT_CFG
        quant_cfg["quant_cfg"]["*action_in_proj.encoder.trunk.0.input_quantizer"] = {
            "enable": False
        }
        quant_cfg["quant_cfg"]["*action_in_proj.encoder.trunk.0.weight_quantizer"] = {
            "enable": False
        }
    elif args.quant_format == "w4a8_nvfp4_fp8":
        quant_cfg = mtq.W4A8_NVFP4_FP8_CFG
    else:
        raise RuntimeError("Unsupported quantization format")

    model = mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)
    # For weight-only PTQ, fold quantized weights back into module parameters and
    # disable weight quantizers. This avoids runtime quantize_op on exported graphs,
    # which can otherwise fail TRT conversion due to lifted fake scale tensors.
    if args.debug:
        print("================== quantize_model summary ==================")
        mtq.print_quant_summary(model)
    
    if args.weight_only:
        mtq.fold_weight(model)

    return model
