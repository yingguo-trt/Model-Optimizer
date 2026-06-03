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
"""Accuracy regression tests for published quantized checkpoints via nv-eval."""

import concurrent.futures
import os
from dataclasses import dataclass
from typing import Literal

import pytest
import torch
from _test_utils import nv_eval_runner

pytestmark = pytest.mark.release

ExecutionMode = Literal["auto", "parallel", "sequential"]


@dataclass(frozen=True)
class TaskSpec:
    name: str
    max_drop_from_baseline: float = 0.05

    @property
    def test_id(self) -> str:
        return self.name.split(".")[-1].replace("-", "_")


@dataclass(frozen=True)
class AccuracyCase:
    quantized_model: str
    baseline_model: str
    backend: str
    tensor_parallel_size: int
    mini_sm: int
    tasks: tuple[TaskSpec, ...]
    mode: ExecutionMode = "auto"
    reasoning: bool = False
    max_new_tokens: int = 16384
    parallelism: int = 64
    max_num_tokens: int = 8192
    temperature: float = 0.0
    top_p: float = 1.0e-5
    kv_cache_free_gpu_memory_fraction: float = 0.9
    cuda_visible_devices: str | None = None

    @property
    def test_id(self) -> str:
        name = os.path.basename(self.quantized_model.rstrip("/"))
        tasks = "_".join(task.test_id for task in self.tasks)
        return f"{name}_{self.backend}_tp{self.tensor_parallel_size}_{self.mode}_{tasks}"

    @property
    def task_names(self) -> list[str]:
        return [task.name for task in self.tasks]


CASES: list[AccuracyCase] = [
    # Ungated smoke gate while Llama/Phi coverage waits on HF access.
    AccuracyCase(
        quantized_model="nvidia/Qwen3-8B-FP8",
        baseline_model="Qwen/Qwen3-8B",
        backend="vllm",
        tensor_parallel_size=1,
        mini_sm=89,
        tasks=(
            # mmlu_pro + gpqa + aime25 smoke-tested already (logs saved); re-enable when needed.
            # TaskSpec("nemo_skills.ns_mmlu_pro"),
            # TaskSpec("nemo_skills.ns_gpqa"),
            # TaskSpec("nemo_skills.aime25"),
            TaskSpec("nemo_skills.ifbench"),
        ),
        reasoning=True,
        max_new_tokens=32768,  # fits Qwen3-8B 40960 ctx with ~8K headroom for the gpqa prompt
        temperature=0.6,  # Qwen3 thinking-mode recommended sampling (avg-of-N tasks)
        top_p=0.95,
    ),
    # TODO: re-enable once the CI HuggingFace token has been granted gated-dataset
    # access for Idavidrein/gpqa and the AIME 2025 dataset.
    # AccuracyCase(
    #     quantized_model="nvidia/Phi-4-reasoning-plus-FP8",
    #     baseline_model="microsoft/Phi-4-reasoning-plus",
    #     backend="vllm",
    #     tensor_parallel_size=1,
    #     mini_sm=89,
    #     tasks=(
    #         TaskSpec("nemo_skills.ns_gpqa"),
    #         TaskSpec("nemo_skills.ns_aime2025"),
    #     ),
    #     reasoning=True,
    #     max_new_tokens=64000,
    # ),
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


def _gpu_pool(case: AccuracyCase) -> list[str]:
    if case.cuda_visible_devices:
        ids = [d.strip() for d in case.cuda_visible_devices.split(",") if d.strip()]
        return list(dict.fromkeys(ids))
    return [str(i) for i in range(torch.cuda.device_count())]


def _parallel_device_split(pool: list[str], tp: int) -> tuple[str, str] | None:
    if len(pool) < 2 * tp:
        return None
    return ",".join(pool[:tp]), ",".join(pool[tp : 2 * tp])


def _execution_split(case: AccuracyCase) -> tuple[list[str], tuple[str, str] | None]:
    pool = _gpu_pool(case)
    if len(pool) < case.tensor_parallel_size:
        pytest.skip(f"requires {case.tensor_parallel_size} GPUs in pool {pool}")
    if case.mode not in ("auto", "parallel", "sequential"):
        pytest.fail(f"unsupported accuracy execution mode: {case.mode!r}")
    if case.mode == "sequential":
        return pool, None

    split = _parallel_device_split(pool, case.tensor_parallel_size)
    if case.mode == "parallel" and split is None:
        pytest.skip(
            f"mode=parallel needs {2 * case.tensor_parallel_size} GPUs in pool {pool}"
        )
    return pool, split


@pytest.mark.parametrize("case", CASES, ids=_idfn)
def test_accuracy(case: AccuracyCase, record_property):
    _skip_if_unsupported(case)

    pool, split = _execution_split(case)
    run_parallel = split is not None

    print(f"\n[test_accuracy] quantized={case.quantized_model}")
    print(f"[test_accuracy] baseline ={case.baseline_model}")
    print(
        f"[test_accuracy] backend={case.backend} tp={case.tensor_parallel_size} "
        f"mode={case.mode} run={'parallel' if run_parallel else 'sequential'} "
        f"tasks={case.task_names}"
    )

    record_property("case.mode", case.mode)
    record_property("case.run", "parallel" if run_parallel else "sequential")
    record_property("case.backend", case.backend)
    record_property("case.tensor_parallel_size", case.tensor_parallel_size)
    record_property("case.parallelism", case.parallelism)
    record_property("case.max_new_tokens", case.max_new_tokens)
    record_property("case.max_num_tokens", case.max_num_tokens)
    record_property("case.temperature", case.temperature)
    record_property("case.top_p", case.top_p)
    record_property(
        "case.kv_cache_free_gpu_memory_fraction", case.kv_cache_free_gpu_memory_fraction
    )
    record_property("case.gpu_pool", ",".join(pool))

    def _run(label: str, model_path: str, devices: str | None) -> dict[str, float]:
        try:
            return nv_eval_runner.run(
                model_path,
                backend=case.backend,
                tasks=case.task_names,
                tensor_parallel_size=case.tensor_parallel_size,
                reasoning=case.reasoning,
                max_new_tokens=case.max_new_tokens,
                parallelism=case.parallelism,
                max_num_tokens=case.max_num_tokens,
                temperature=case.temperature,
                top_p=case.top_p,
                kv_cache_free_gpu_memory_fraction=case.kv_cache_free_gpu_memory_fraction,
                cuda_visible_devices=devices,
            )
        except Exception as exc:
            raise RuntimeError(f"{label} eval failed for {model_path}") from exc

    def _record_baseline(scores: dict[str, float]) -> None:
        for task in case.tasks:
            record_property(f"{task.name}.baseline", scores[task.name])

    if run_parallel:
        baseline_devices, quantized_devices = split
        scores: dict[str, dict[str, float]] = {}
        errors: list[Exception] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool_exec:
            futures = {
                pool_exec.submit(
                    _run, "baseline", case.baseline_model, baseline_devices
                ): "baseline",
                pool_exec.submit(
                    _run, "quantized", case.quantized_model, quantized_devices
                ): "quantized",
            }
            for future in concurrent.futures.as_completed(futures):
                label = futures[future]
                try:
                    scores[label] = future.result()
                except Exception as exc:
                    errors.append(exc)
                else:
                    if label == "baseline":
                        _record_baseline(scores[label])
        # raise after both finish, not in-loop: let the slow side reach its own teardown
        if errors:
            raise errors[0]
        baseline_scores = scores["baseline"]
        quantized_scores = scores["quantized"]
    else:
        # normalized pool (same as the parallel split), None when unpinned
        seq_devices = ",".join(pool) if case.cuda_visible_devices else None
        baseline_scores = _run("baseline", case.baseline_model, seq_devices)
        _record_baseline(baseline_scores)
        quantized_scores = _run("quantized", case.quantized_model, seq_devices)

    failures: list[str] = []
    for task in case.tasks:
        baseline = baseline_scores[task.name]
        quantized = quantized_scores[task.name]
        drop = baseline - quantized

        record_property(f"{task.name}.quantized", quantized)
        record_property(f"{task.name}.drop", drop)
        record_property(f"{task.name}.threshold", task.max_drop_from_baseline)

        status = "PASS" if drop <= task.max_drop_from_baseline else "FAIL"
        print(
            f"[test_accuracy] {task.name}: baseline={baseline:.4f} quantized={quantized:.4f} "
            f"drop={drop:+.4f} threshold={task.max_drop_from_baseline} {status}"
        )
        if drop > task.max_drop_from_baseline:
            failures.append(
                f"{task.name}: drop {drop:.4f} exceeds threshold {task.max_drop_from_baseline}"
            )

    if failures:
        pytest.fail("\n".join(failures))
