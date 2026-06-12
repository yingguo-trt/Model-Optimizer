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
"""Shared accuracy gate.

The drop is measured same-GPU/same-backend (baseline - quantized); absolute scores are
not portable across chips, so only the drop is gated. ``evaluate_drop`` is the primitive
both the OSS pytest and the harness use; ``evaluate`` builds the per-metric verdict rows
the harness writes to PostgreSQL.
"""
from __future__ import annotations

MAX_DROP_FROM_BASELINE = "max_drop_from_baseline"


def evaluate_drop(baseline: float, quantized: float, threshold_value: float) -> tuple[float, bool]:
    """Return ``(drop, passed)`` where ``drop = baseline - quantized``."""
    drop = baseline - quantized
    return drop, drop <= threshold_value


def evaluate(case, baseline_scores: dict[str, float], quantized_scores: dict[str, float]) -> list[dict]:
    """Per (task, metric) verdict rows for a harness ``cases.Case``.

    ``delta_value = quantized - baseline`` (signed; ``drop = -delta``). Only
    ``max_drop_from_baseline`` is supported today; other threshold types raise so a
    new gate semantics is a conscious addition, not a silent pass.
    """
    serving = getattr(case, "serving", None)
    rows: list[dict] = []
    for task in case.tasks:
        if task.threshold_type != MAX_DROP_FROM_BASELINE:
            raise ValueError(
                f"{task.task_name}: unsupported threshold_type {task.threshold_type!r}"
            )
        baseline = baseline_scores[task.task_name]
        quantized = quantized_scores[task.task_name]
        drop, passed = evaluate_drop(baseline, quantized, task.threshold_value)
        rows.append({
            "task_name": task.task_name,
            "metric_name": task.metric_name,
            "metric_value": quantized,
            "baseline_value": baseline,
            "delta_value": quantized - baseline,
            "threshold_type": task.threshold_type,
            "threshold_value": task.threshold_value,
            "metric_status": "passed" if passed else "failed",
            "num_repeats": None,
            "temperature": getattr(serving, "temperature", None),
            "top_p": getattr(serving, "top_p", None),
            "max_new_tokens": getattr(serving, "max_new_tokens", None),
            "extra": {},
        })
    return rows
