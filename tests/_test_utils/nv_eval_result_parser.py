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
"""Shared nv-eval results.yml parsing.

Pytest-free so the OSS pytest (test_accuracy) and the modelopt-jenkins harness import
the SAME task->metric mapping and scoring, instead of each grepping logs or duplicating
the path into results.yml.
"""
from __future__ import annotations

from pathlib import Path

# task -> (group_key, metric_key, score_key) inside results.yml
TASK_INFO: dict[str, tuple[str, str, str]] = {
    "nemo_skills.mmlu-pro": ("mmlu-pro", "pass@1", "symbolic_correct"),
    "nemo_skills.ns_gpqa": ("gpqa", "pass@1[avg-of-8]", "symbolic_correct"),
    "nemo_skills.aime25": ("aime25", "pass@1[avg-of-8]", "symbolic_correct"),
    "ns_scicode": ("scicode", "pass@1[avg-of-8]", "subtask_accuracy"),
    "nemo_skills.ifbench": ("ifbench", "pass@1[avg-of-8]", "prompt_loose_accuracy"),
    "nemo_skills.ifeval": ("ifeval", "pass@1[avg-of-8]", "prompt_loose_accuracy"),
    "nemo_skills.gsm8k": ("gsm8k", "pass@1[avg-of-4]", "symbolic_correct"),
    "ns_aa_lcr": ("aalcr", "pass@1", "judge_correct"),
}


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


def collect_scores(results_root, invocation_id: str, task_names: list[str]) -> dict[str, float]:
    """Return ``{task: score}`` for ``task_names`` from the run under ``results_root``.

    ``results_root`` is the directory nv-eval writes run dirs into (``nv_eval_results``
    for the local executor; the shared output_dir for slurm/pyxis).
    """
    import yaml

    results_root = Path(results_root)
    run_dirs = sorted(p for p in results_root.rglob(f"*-{invocation_id}") if p.is_dir())
    if not run_dirs:
        raise RuntimeError(f"no run directory matched *-{invocation_id} under {results_root}")
    run_dir = run_dirs[0]

    scores: dict[str, float] = {}
    for idx, task in enumerate(task_names):
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
