# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
TensorRT compilation subpackage for Alpamayo1.5.

All components use torch.export + torch_tensorrt.dynamo.compile (eager, no lazy
recompile). MTTM (MutableTorchTensorRTModule) is not used.
"""

from alpamayo1_5.trt.diffusion import (
    compile_diffusion_step_no_cache,
    save_diffusion_engine,
)
from alpamayo1_5.trt.engine_io import (
    TRTEngineRunner,
    save_trt_engine,
)
from alpamayo1_5.trt.lm import (
    compile_vlm_lm_trt,
    generate_alpamayo_with_static_cache,
)
from alpamayo1_5.trt.vision import (
    compile_and_replace_vision_model,
    compile_vision_model,
    save_vision_engine,
)

__all__ = [
    "compile_vision_model",
    "compile_and_replace_vision_model",
    "save_vision_engine",
    "compile_vlm_lm_trt",
    "generate_alpamayo_with_static_cache",
    "compile_diffusion_step_no_cache",
    "save_diffusion_engine",
    "save_trt_engine",
    "TRTEngineRunner",
]
