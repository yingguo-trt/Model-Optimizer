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
"""Wrapper around the internal nv-eval framework for accuracy regression tests.

Shells out to ``launch_eval.py`` from
``Model-Optimizer-Internal/examples/nv_eval`` and parses ``results.yml`` to
extract numeric benchmark scores.

The nv-eval directory is resolved from the ``NV_EVAL_DIR`` environment
variable; tests that depend on this wrapper skip if it isn't set.
"""

import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

# Mirrors the relevant rows of Model-Optimizer-Internal/examples/nv_eval/benchmarks.py.
# Key   = task name as passed to launch_eval.py --task-list (also the result subdir name).
# Value = (group_key, metric_key, score_key) — path inside results.yml under results.groups.
TASK_INFO: dict[str, tuple[str, str, str]] = {
    "nemo_skills.ns_mmlu_pro": ("mmlu-pro", "pass@1", "symbolic_correct"),
    "nemo_skills.ns_gpqa": ("gpqa", "pass@1[avg-of-8]", "symbolic_correct"),
    "ns_scicode": ("scicode", "pass@1[avg-of-8]", "subtask_accuracy"),
    "ns_ifbench": ("ifbench", "pass@1[avg-of-8]", "prompt_loose_accuracy"),
    "ns_aa_lcr": ("aalcr", "pass@1", "judge_correct"),
}

_INVOCATION_ID_RE = re.compile(r"to check status: nv-eval status (\S+)")


def _get_nv_eval_dir() -> Path:
    raw = os.environ.get("NV_EVAL_DIR")
    if not raw:
        pytest.skip("NV_EVAL_DIR is not set; cannot run nv-eval accuracy tests")
    path = Path(raw)
    if not (path / "launch_eval.py").exists():
        pytest.skip(f"launch_eval.py not found under NV_EVAL_DIR={path}")
    return path


def run(
    model_path: str,
    backend: str,
    tasks: list[str],
    tensor_parallel_size: int,
    max_new_tokens: int = 16384,
    parallelism: int = 64,
    extra_args: list[str] | None = None,
) -> dict[str, float]:
    """Evaluate ``model_path`` on ``tasks`` and return ``{task: score}``.

    Skips the calling test if ``NV_EVAL_DIR`` is missing or any task is unknown.
    """
    for task in tasks:
        if task not in TASK_INFO:
            pytest.skip(f"task {task!r} missing from nv_eval_runner.TASK_INFO")

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
    ]
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
    if proc.returncode != 0:
        raise RuntimeError(f"launch_eval.py exited with code {proc.returncode}")

    invocation_id = _extract_invocation_id(proc.stdout)
    return _collect_scores(nv_eval_dir, invocation_id, tasks)


def _extract_invocation_id(stdout: str) -> str:
    m = _INVOCATION_ID_RE.search(stdout)
    if not m:
        raise RuntimeError("could not find invocation_id in launch_eval.py stdout")
    return m.group(1)


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
    for task in tasks:
        group_key, metric_key, score_key = TASK_INFO[task]
        results_yml = run_dir / task / "artifacts" / "results.yml"
        if not results_yml.exists():
            raise RuntimeError(f"results.yml missing for task {task!r}: {results_yml}")
        with open(results_yml) as f:
            data = yaml.safe_load(f)
        try:
            value = data["results"]["groups"][group_key]["metrics"][metric_key]["scores"][
                score_key
            ]["value"]
        except (KeyError, TypeError) as e:
            raise RuntimeError(
                f"score path not found in {results_yml}: "
                f"results.groups.{group_key}.metrics.{metric_key}.scores.{score_key}: {e}"
            ) from e
        scores[task] = float(value)
    return scores
