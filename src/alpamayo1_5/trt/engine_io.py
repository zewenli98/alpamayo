# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import pathlib
from typing import Any

import torch

logger = logging.getLogger(__name__)

_TRT_TO_TORCH_DTYPE: dict[Any, torch.dtype] | None = None


def _trt_dtype_to_torch(trt_dtype) -> torch.dtype:
    import tensorrt as trt

    global _TRT_TO_TORCH_DTYPE
    if _TRT_TO_TORCH_DTYPE is None:
        _TRT_TO_TORCH_DTYPE = {
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.HALF: torch.float16,
            trt.DataType.BF16: torch.bfloat16,
            trt.DataType.INT8: torch.int8,
            trt.DataType.INT32: torch.int32,
            trt.DataType.INT64: torch.int64,
            trt.DataType.BOOL: torch.bool,
        }
    return _TRT_TO_TORCH_DTYPE.get(trt_dtype, torch.float32)


def save_trt_engine(engine_bytes: bytes, path: str | pathlib.Path, metadata: dict | None = None) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(engine_bytes)
    logger.info("TRT engine written to %s (%.1f MB)", path, len(engine_bytes) / 1024 / 1024)

    sidecar = path.with_suffix(path.suffix + ".json")
    with open(sidecar, "w") as f:
        json.dump(metadata or {}, f, indent=2)
    logger.info("Engine metadata written to %s", sidecar)


def load_trt_engine_metadata(path: str | pathlib.Path) -> dict:
    sidecar = pathlib.Path(str(path) + ".json")
    if sidecar.exists():
        with open(sidecar) as f:
            return json.load(f)
    return {}


class TRTEngineRunner:
    def __init__(
        self,
        path: str | pathlib.Path,
        device: str = "cuda",
        stream: torch.cuda.Stream | None = None,
    ):
        import tensorrt as trt

        self.path = pathlib.Path(path)
        self.device = torch.device(device)
        self._stream = stream
        self.metadata = load_trt_engine_metadata(self.path)

        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)
        with open(self.path, "rb") as f:
            engine_bytes = f.read()
        self.engine: trt.ICudaEngine = runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TRT engine from {self.path}")

        self.context: trt.IExecutionContext = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TRT execution context")

        self.input_names: list[str] = []
        self.output_names: list[str] = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

        self._output_cache: dict[str, torch.Tensor] = {}

    @property
    def _cuda_stream(self) -> int:
        if self._stream is not None:
            return self._stream.cuda_stream
        return torch.cuda.current_stream(self.device).cuda_stream

    def _get_output_tensor(self, name: str, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
        cached = self._output_cache.get(name)
        if cached is None or cached.shape != torch.Size(shape) or cached.dtype != dtype:
            self._output_cache[name] = torch.empty(shape, dtype=dtype, device=self.device)
        return self._output_cache[name]

    def __call__(self, *inputs: torch.Tensor) -> list[torch.Tensor]:
        if len(inputs) != len(self.input_names):
            raise ValueError(f"Expected {len(self.input_names)} inputs, got {len(inputs)}")

        for name, tensor in zip(self.input_names, inputs):
            tensor = tensor.contiguous().to(self.device)
            self.context.set_tensor_address(name, tensor.data_ptr())
            self.context.set_input_shape(name, tuple(tensor.shape))

        output_tensors: list[torch.Tensor] = []
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = _trt_dtype_to_torch(self.engine.get_tensor_dtype(name))
            out = self._get_output_tensor(name, shape, dtype)
            self.context.set_tensor_address(name, out.data_ptr())
            output_tensors.append(out)

        self.context.execute_async_v3(self._cuda_stream)
        torch.cuda.current_stream(self.device).synchronize()
        return output_tensors
