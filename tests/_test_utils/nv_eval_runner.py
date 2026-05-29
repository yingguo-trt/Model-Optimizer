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
"""Wrapper around the nv-eval launcher for accuracy regression tests.

Shells out to ``launch_eval.py`` (resolved from the ``NV_EVAL_DIR`` env var)
and parses ``results.yml`` to extract numeric benchmark scores.

Tests that depend on this wrapper skip if ``NV_EVAL_DIR`` is unset or the
directory does not contain a ``launch_eval.py`` following the nv-eval CLI
contract.
"""

import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

# Maps each task to the path inside results.yml where its score lives:
#   results.groups.<group_key>.metrics.<metric_key>.scores.<score_key>.value
# Key   = task name as passed to launch_eval.py --task-list (also the result subdir name).
# Value = (group_key, metric_key, score_key).
TASK_INFO: dict[str, tuple[str, str, str]] = {
    "nemo_skills.ns_mmlu_pro": ("mmlu-pro", "pass@1", "symbolic_correct"),
    "nemo_skills.ns_gpqa": ("gpqa", "pass@1[avg-of-8]", "symbolic_correct"),
    "nemo_skills.ns_aime2025": ("aime25", "pass@1[avg-of-64]", "symbolic_correct"),
    "ns_scicode": ("scicode", "pass@1[avg-of-8]", "subtask_accuracy"),
    "ns_ifbench": ("ifbench", "pass@1[avg-of-8]", "prompt_loose_accuracy"),
    "ns_aa_lcr": ("aalcr", "pass@1", "judge_correct"),
}

_INVOCATION_ID_RE = re.compile(r"to check status: nv-eval status (\S+)")
# launch_eval.py launches the inference server with `docker run ... --name llm_server_<port>`
# and echoes that command; we parse the name from stdout to tear the container down.
_SERVER_NAME_RE = re.compile(r"--name (llm_server_\d+)")


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
    reasoning: bool = False,
    extra_args: list[str] | None = None,
) -> dict[str, float]:
    """Evaluate ``model_path`` on ``tasks`` and return ``{task: score}``.

    ``reasoning=True`` enables the launcher's reasoning mode (thinking tokens
    enabled, longer generation budgets recommended). Pass a larger
    ``max_new_tokens`` (e.g. 64000) when enabling reasoning.

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
    if reasoning:
        cmd.append("--reasoning")
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
        # launch_eval.py removes the server container on its failure paths but
        # NOT on the success path, leaving it running and holding ~90% of GPU
        # memory, which starves the next model's server. Tear down whatever this
        # invocation started, regardless of pass/fail. See design review §9.1.
        _teardown_servers(proc.stdout)


def _extract_invocation_id(stdout: str) -> str:
    m = _INVOCATION_ID_RE.search(stdout)
    if not m:
        raise RuntimeError("could not find invocation_id in launch_eval.py stdout")
    return m.group(1)


def _teardown_servers(launch_stdout: str) -> None:
    """Force-remove the inference-server container(s) launch_eval.py started.

    launch_eval.py only tears the server container down on its failure paths; on
    the success path it leaves the container running, holding ~90% of GPU memory
    and starving the next model's server. We parse the container name from the
    launcher's own ``docker run --name`` line and force-remove it. Best-effort
    and idempotent: removing an already-gone container is a harmless no-op.
    """
    for name in dict.fromkeys(_SERVER_NAME_RE.findall(launch_stdout)):
        result = subprocess.run(
            ["docker", "rm", "-f", name],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            print(f"[nv_eval_runner] tore down server container {name}")


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
