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
"""Run nv-eval launcher and collect benchmark scores for accuracy tests."""

import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

# task -> (group_key, metric_key, score_key) inside results.yml
TASK_INFO: dict[str, tuple[str, str, str]] = {
    "nemo_skills.ns_mmlu_pro": ("mmlu-pro", "pass@1", "symbolic_correct"),
    "nemo_skills.ns_gpqa": ("gpqa", "pass@1[avg-of-8]", "symbolic_correct"),
    "nemo_skills.aime25": ("aime25", "pass@1[avg-of-8]", "symbolic_correct"),
    "ns_scicode": ("scicode", "pass@1[avg-of-8]", "subtask_accuracy"),
    "ns_ifbench": ("ifbench", "pass@1[avg-of-8]", "prompt_loose_accuracy"),
    "ns_aa_lcr": ("aalcr", "pass@1", "judge_correct"),
}

_INVOCATION_ID_RE = re.compile(r"to check status: nv-eval status (\S+)")
_SERVER_NAME_RE = re.compile(r"--name (llm_server_\d+)")


def _is_ci() -> bool:
    return any(
        os.environ.get(name)
        for name in ("CI", "JENKINS_URL", "BUILD_URL", "BUILD_NUMBER", "JOB_NAME")
    )


def _skip_or_fail(message: str) -> None:
    if _is_ci():
        pytest.fail(message)
    pytest.skip(message)


def _get_nv_eval_dir() -> Path:
    raw = os.environ.get("NV_EVAL_DIR")
    if not raw:
        _skip_or_fail("NV_EVAL_DIR is not set; cannot run nv-eval accuracy tests")
    path = Path(raw)
    if not (path / "launch_eval.py").exists():
        _skip_or_fail(f"launch_eval.py not found under NV_EVAL_DIR={path}")
    return path


def run(
    model_path: str,
    backend: str,
    tasks: list[str],
    tensor_parallel_size: int,
    max_new_tokens: int = 16384,
    parallelism: int = 64,
    reasoning: bool = False,
    max_num_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    kv_cache_free_gpu_memory_fraction: float | None = None,
    cuda_visible_devices: str | None = None,
    docker_image: str | None = None,
    extra_llm_args: str | None = None,
    extra_args: list[str] | None = None,
) -> dict[str, float]:
    """Evaluate ``model_path`` on ``tasks`` and return ``{task: score}``.

    Local setup gaps skip the test; CI setup gaps fail it.
    """
    for task in tasks:
        if task not in TASK_INFO:
            _skip_or_fail(f"task {task!r} missing from nv_eval_runner.TASK_INFO")

    nv_eval_dir = _get_nv_eval_dir()

    cmd = [
        sys.executable,
        "launch_eval.py",
        "--model-path",
        model_path,
        "--backend",
        backend,
        "--tp-size",
        str(tensor_parallel_size),
        "--task-list",
        ",".join(tasks),
        "--max-new-tokens",
        str(max_new_tokens),
        "--parallelism",
        str(parallelism),
        "--wait",  # block until the eval finishes; local executor is otherwise fire-and-forget
    ]
    if reasoning:
        cmd.append("--reasoning")
    if max_num_tokens is not None:
        cmd.extend(["--max-num-tokens", str(max_num_tokens)])
    if temperature is not None:
        cmd.extend(["--t", str(temperature)])
    if top_p is not None:
        cmd.extend(["--top-p", str(top_p)])
    if kv_cache_free_gpu_memory_fraction is not None:
        cmd.extend(
            [
                "--kv-cache-free-gpu-memory-fraction",
                str(kv_cache_free_gpu_memory_fraction),
            ]
        )
    if cuda_visible_devices is not None:
        cmd.extend(["--cuda-visible-devices", cuda_visible_devices])
    if docker_image is not None:
        cmd.extend(["--docker-image", docker_image])
    if extra_llm_args is not None:
        cmd.extend(["--extra-llm-args", extra_llm_args])
    if extra_args:
        cmd.extend(extra_args)

    print(f"[nv_eval_runner] cwd={nv_eval_dir}")
    print(f"[nv_eval_runner] cmd={' '.join(shlex.quote(c) for c in cmd)}")

    proc = subprocess.run(
        cmd,
        cwd=nv_eval_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    print(proc.stdout, end="", flush=True)
    try:
        if proc.returncode != 0:
            raise RuntimeError(f"launch_eval.py exited with code {proc.returncode}")

        invocation_id = _extract_invocation_id(proc.stdout)
        return _collect_scores(nv_eval_dir, invocation_id, tasks)
    finally:
        _teardown_servers(proc.stdout)


def _extract_invocation_id(stdout: str) -> str:
    m = _INVOCATION_ID_RE.search(stdout)
    if not m:
        raise RuntimeError("could not find invocation_id in launch_eval.py stdout")
    return m.group(1)


def _teardown_servers(launch_stdout: str) -> None:
    """Best-effort cleanup for server containers started by launch_eval.py."""
    for name in dict.fromkeys(_SERVER_NAME_RE.findall(launch_stdout)):
        result = subprocess.run(
            ["docker", "rm", "-f", name],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            print(f"[nv_eval_runner] tore down server container {name}")


def _dig(node, keys: list, results_yml: Path):
    """Walk ``keys`` into ``node``; on a miss, report which key and what keys exist there."""
    for i, key in enumerate(keys):
        if not isinstance(node, dict) or key not in node:
            where = ".".join(map(str, keys[:i])) or "<root>"
            available = sorted(node.keys()) if isinstance(node, dict) else f"<{type(node).__name__}>"
            raise RuntimeError(
                f"{results_yml}: missing {key!r} under {where}; available there: {available}"
            )
        node = node[key]
    return node


def _collect_scores(nv_eval_dir: Path, invocation_id: str, tasks: list[str]) -> dict[str, float]:
    import yaml

    results_root = nv_eval_dir / "nv_eval_results"
    run_dirs = sorted(p for p in results_root.rglob(f"*-{invocation_id}") if p.is_dir())
    if not run_dirs:
        raise RuntimeError(
            f"no run directory matched *-{invocation_id} under {results_root}"
        )
    run_dir = run_dirs[0]

    scores: dict[str, float] = {}
    for idx, task in enumerate(tasks):
        group_key, metric_key, score_key = TASK_INFO[task]
        candidate_results = [
            run_dir / f"{task}.{idx}" / "artifacts" / "results.yml",
            run_dir / task / "artifacts" / "results.yml",
        ]
        results_yml = next(
            (path for path in candidate_results if path.exists()), candidate_results[0]
        )
        if not results_yml.exists():
            checked = "\n".join(f"  {path}" for path in candidate_results)
            tree = "\n".join(
                f"  {p.relative_to(run_dir)}" for p in sorted(run_dir.rglob("*")) if p.is_file()
            )
            raise RuntimeError(
                f"results.yml missing for task {task!r}: {results_yml}\n"
                f"checked paths:\n{checked}\n"
                f"files actually written under {run_dir}:\n{tree or '  (none — eval produced no output)'}"
            )
        with open(results_yml) as f:
            data = yaml.safe_load(f)
        path = ["results", "groups", group_key, "metrics", metric_key, "scores", score_key, "value"]
        value = float(_dig(data, path, results_yml))
        # Compare all drops as fractions; nemo-skills emits percentages.
        if value > 1.0:
            value /= 100.0
        scores[task] = value
    return scores
