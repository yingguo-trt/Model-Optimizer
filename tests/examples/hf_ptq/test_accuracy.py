# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Accuracy regression tests for published quantized model checkpoints.

Each case evaluates two models on the same benchmark suite via nv-eval
(see ``tests/_test_utils/nv_eval_runner``):

    1. The published quantized checkpoint (e.g. ``nvidia/Llama-3.1-8B-Instruct-FP8``)
    2. Its unquantized counterpart (e.g. ``meta-llama/Llama-3.1-8B-Instruct``)

The quantized score must not drop more than ``max_drop_from_baseline`` below
the baseline on any task; otherwise the test fails.

``quantized_model`` and ``baseline_model`` accept both Hugging Face ids and
local filesystem paths — the underlying inference backends and nv-eval
treat them identically.

Requires ``NV_EVAL_DIR`` to point at a directory containing ``launch_eval.py``
(the nv-eval launcher entry point).
"""

import os
from dataclasses import dataclass

import pytest
import torch
from _test_utils import nv_eval_runner

pytestmark = pytest.mark.release


@dataclass
class AccuracyCase:
    quantized_model: str  # HF id or local path
    baseline_model: str  # HF id or local path
    backend: str
    tensor_parallel_size: int
    mini_sm: int
    tasks: tuple[str, ...]
    max_drop_from_baseline: float = 0.05
    reasoning: bool = False
    max_new_tokens: int = 16384

    @property
    def test_id(self) -> str:
        name = os.path.basename(self.quantized_model.rstrip("/"))
        return f"{name}_{self.backend}"


CASES: list[AccuracyCase] = [
    AccuracyCase(
        quantized_model="nvidia/Llama-3.1-8B-Instruct-FP8",
        baseline_model="meta-llama/Llama-3.1-8B-Instruct",
        backend="vllm",
        tensor_parallel_size=1,
        mini_sm=89,
        tasks=(
            "nemo_skills.ns_mmlu_pro",
            "nemo_skills.ns_gpqa",
        ),
    ),
    AccuracyCase(
        quantized_model="nvidia/Phi-4-reasoning-plus-FP8",
        baseline_model="microsoft/Phi-4-reasoning-plus",
        backend="vllm",
        tensor_parallel_size=1,
        mini_sm=89,
        tasks=(
            "nemo_skills.ns_gpqa",
            "nemo_skills.ns_aime2025",
        ),
        reasoning=True,
        max_new_tokens=64000,
    ),
]


def _idfn(case: AccuracyCase) -> str:
    return case.test_id


def _skip_if_unsupported(case: AccuracyCase) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        pytest.skip("CUDA is not available")
    sm = torch.cuda.get_device_capability()
    if sm < (case.mini_sm // 10, case.mini_sm % 10):
        pytest.skip(f"requires sm{case.mini_sm} or higher; have sm{sm[0]}{sm[1]}")
    if torch.cuda.device_count() < case.tensor_parallel_size:
        pytest.skip(f"requires at least {case.tensor_parallel_size} GPUs")


@pytest.mark.parametrize("case", CASES, ids=_idfn)
def test_accuracy(case: AccuracyCase, record_property):
    _skip_if_unsupported(case)

    print(f"\n[test_accuracy] quantized={case.quantized_model}")
    print(f"[test_accuracy] baseline ={case.baseline_model}")
    print(
        f"[test_accuracy] backend={case.backend} "
        f"tp={case.tensor_parallel_size} tasks={list(case.tasks)}"
    )

    baseline_scores = nv_eval_runner.run(
        case.baseline_model,
        backend=case.backend,
        tasks=list(case.tasks),
        tensor_parallel_size=case.tensor_parallel_size,
        reasoning=case.reasoning,
        max_new_tokens=case.max_new_tokens,
    )
    quantized_scores = nv_eval_runner.run(
        case.quantized_model,
        backend=case.backend,
        tasks=list(case.tasks),
        tensor_parallel_size=case.tensor_parallel_size,
        reasoning=case.reasoning,
        max_new_tokens=case.max_new_tokens,
    )

    failures: list[str] = []
    for task in case.tasks:
        baseline = baseline_scores[task]
        quantized = quantized_scores[task]
        drop = baseline - quantized

        record_property(f"{task}.baseline", baseline)
        record_property(f"{task}.quantized", quantized)
        record_property(f"{task}.drop", drop)
        record_property(f"{task}.threshold", case.max_drop_from_baseline)

        status = "PASS" if drop <= case.max_drop_from_baseline else "FAIL"
        print(
            f"[test_accuracy] {task}: baseline={baseline:.4f} quantized={quantized:.4f} "
            f"drop={drop:+.4f} threshold={case.max_drop_from_baseline} {status}"
        )
        if drop > case.max_drop_from_baseline:
            failures.append(
                f"{task}: drop {drop:.4f} exceeds threshold {case.max_drop_from_baseline}"
            )

    if failures:
        pytest.fail("\n".join(failures))
