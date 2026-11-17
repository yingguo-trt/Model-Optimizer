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

import base64
import copy
import io
import os
import subprocess
import sys
import tempfile
import traceback

import pytest
import torch

from .deploy_config import engine_tp_kwargs

# Cache for available backends detection (computed once at import time)
_AVAILABLE_BACKENDS = None


def get_available_backends():
    """Detect which backends are available in the current environment.

    Returns:
        set: A set of available backend names ('trtllm', 'vllm', 'sglang')
    """
    global _AVAILABLE_BACKENDS
    if _AVAILABLE_BACKENDS is not None:
        return _AVAILABLE_BACKENDS

    available = set()

    try:
        import tensorrt_llm  # noqa: F401

        available.add("trtllm")
    except ImportError:
        pass

    try:
        import vllm  # noqa: F401

        available.add("vllm")
    except ImportError:
        pass

    try:
        import sglang  # noqa: F401

        available.add("sglang")
    except ImportError:
        pass

    _AVAILABLE_BACKENDS = available
    print(f"[deploy_utils] Detected available backends: {available}")
    return _AVAILABLE_BACKENDS


def _run_trtllm_deploy(
    model_id: str,
    tensor_parallel_size: int,
    mini_sm: int,
    base_model: str,
    engine_kwargs: dict | None = None,
    sampling_kwargs: dict | None = None,
    **request_kwargs,
) -> None:
    """Top-level entry for subprocess: run TensorRT-LLM deploy in a child process."""
    try:
        deployer = ModelDeployer(
            backend="trtllm",
            model_id=model_id,
            tensor_parallel_size=tensor_parallel_size,
            mini_sm=mini_sm,
            base_model=base_model,
            engine_kwargs=engine_kwargs,
            sampling_kwargs=sampling_kwargs,
            **request_kwargs,
        )
        deployer._deploy_trtllm_impl()
    except Exception:
        traceback.print_exc()
        pytest.fail(traceback.format_exc())


def _run_vllm_deploy(
    model_id: str,
    tensor_parallel_size: int,
    mini_sm: int,
    base_model: str,
    engine_kwargs: dict | None = None,
    sampling_kwargs: dict | None = None,
    **request_kwargs,
) -> None:
    """Top-level entry for subprocess: run vLLM deploy in a child process."""
    try:
        deployer = ModelDeployer(
            backend="vllm",
            model_id=model_id,
            tensor_parallel_size=tensor_parallel_size,
            mini_sm=mini_sm,
            base_model=base_model,
            engine_kwargs=engine_kwargs,
            sampling_kwargs=sampling_kwargs,
            **request_kwargs,
        )
        deployer._deploy_vllm_impl()
    except Exception:
        traceback.print_exc()
        pytest.fail(traceback.format_exc())


def _run_sglang_deploy(
    model_id: str,
    tensor_parallel_size: int,
    mini_sm: int,
    base_model: str,
    engine_kwargs: dict | None = None,
    sampling_kwargs: dict | None = None,
    **request_kwargs,
) -> None:
    """Top-level entry for subprocess: run SGLang deploy in a child process."""
    try:
        deployer = ModelDeployer(
            backend="sglang",
            model_id=model_id,
            tensor_parallel_size=tensor_parallel_size,
            mini_sm=mini_sm,
            base_model=base_model,
            engine_kwargs=engine_kwargs,
            sampling_kwargs=sampling_kwargs,
            **request_kwargs,
        )
        deployer._deploy_sglang_impl()
    except Exception:
        traceback.print_exc()
        pytest.fail(traceback.format_exc())


def _run_deploy_via_subprocess(
    backend: str,
    model_id: str,
    tensor_parallel_size: int,
    mini_sm: int,
    base_model: str,
    engine_kwargs: dict,
    sampling_kwargs: dict,
    **request_kwargs,
) -> None:
    """Run deploy in a subprocess and print its stdout/stderr so pytest capture=tee-sys captures to DB."""
    tests_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_root = os.path.dirname(tests_dir)
    env = {
        **os.environ,
        "PYTHONPATH": project_root + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }
    code = f"""import sys
sys.path.insert(0, {tests_dir!r})
if __name__ == '__main__':
    from _test_utils.deploy_utils import _run_{backend}_deploy
    _run_{backend}_deploy(
        {model_id!r}, {tensor_parallel_size}, {mini_sm}, {base_model!r},
        {engine_kwargs!r}, {sampling_kwargs!r}, **{request_kwargs!r}
    )
"""
    if backend == "trtllm":
        mpirun = os.environ.get("MPIRUN", "mpirun")
        cmd = [
            mpirun,
            "--allow-run-as-root",
            "--oversubscribe",
            "-n",
            "1",
            sys.executable,
        ]
    else:
        cmd = [sys.executable, "-c", code]

    if backend == "trtllm":
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            tmp_path = f.name
        try:
            result = subprocess.run(
                [*cmd, tmp_path],
                cwd=tests_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        finally:
            os.unlink(tmp_path)
    else:
        result = subprocess.run(
            cmd,
            cwd=tests_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if result.stdout:
        print(result.stdout, end="", flush=True)
    result.check_returncode()


# Common test prompts for all backends
COMMON_PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]


class ModelDeployer:
    def __init__(
        self,
        backend: str = "trtllm",
        model_id: str = "",
        tensor_parallel_size: int = 1,
        mini_sm: int = 89,
        base_model: str = "",
        engine_kwargs: dict | None = None,
        sm_family: int | None = None,
        sampling_kwargs: dict | None = None,
        input_config: dict | None = None,
        chat_kwargs: dict | None = None,
        require_speculation: bool = False,
        sm_versions: list[int] | None = None,
    ):
        """
        Initialize the ModelDeployer.

        Args:
            backend: The backend to use ('vllm', 'trtllm', or 'sglang')
            model_id: Path to the model
            tensor_parallel_size: Tensor parallel size for distributed inference
            mini_sm: Minimum SM (Streaming Multiprocessor) requirement for the model
            engine_kwargs: resolved native engine parameters from the case and run YAML.
                Set model and TP through model_id and tensor_parallel_size only.
            sm_family: if set, skip unless the GPU belongs to this SM family (100 for SM100 to
                SM103), for a backend path that checks the family rather than a minimum SM
            sampling_kwargs: resolved native backend sampling parameters.
            input_config: optional prompt and local image for a configured request.
            chat_kwargs: native vLLM chat-call options, separate from sampling parameters.
            require_speculation: require evidence of actual draft execution during generation.
            sm_versions: exact CUDA capabilities allowed by the selected backend recipe.
        """
        self.backend = backend
        self.model_id = model_id
        self.tensor_parallel_size = tensor_parallel_size
        self.mini_sm = mini_sm
        self.base_model = base_model
        self.engine_kwargs = dict(engine_kwargs or {})
        self.sm_family = sm_family
        self.sampling_kwargs = dict(sampling_kwargs or {})
        self.input_config = dict(input_config or {})
        self.chat_kwargs = dict(chat_kwargs or {})
        self.require_speculation = require_speculation
        self.sm_versions = sm_versions
        if (self.input_config or self.chat_kwargs) and backend != "vllm":
            raise ValueError("Image inputs and chat_kwargs currently require the vLLM adapter")
        if {"messages", "sampling_params"} & self.chat_kwargs.keys():
            raise ValueError("Set input_config and sampling_kwargs instead of chat-call inputs")
        if require_speculation:
            if not base_model or backend not in ("vllm", "sglang"):
                raise ValueError("Speculative execution checks require a target and vLLM or SGLang")
            if backend == "vllm" and self.engine_kwargs.get("disable_log_stats") is not False:
                raise ValueError("Speculative execution checks require disable_log_stats: false")
        if backend == "sglang" and self.engine_kwargs.get("enable_layernorm_sp"):
            if self.engine_kwargs.get("ep_size", tensor_parallel_size) != tensor_parallel_size:
                raise ValueError("Layernorm sequence parallelism requires EP=TP; override both")
        reserved_keys = sorted(
            {"model", "model_path", "tensor_parallel_size", "tp_size"} & self.engine_kwargs.keys()
        )
        if reserved_keys:
            raise ValueError(
                f"Set model_id and tensor_parallel_size instead of {reserved_keys} in "
                "engine_kwargs so resource checks and engine arguments stay consistent."
            )

    def _engine_arguments(self):
        """Bind runtime model paths and TP without supplying test parameter defaults."""
        is_draft = bool(self.base_model) or "eagle" in self.model_id.lower()
        model_key = "model_path" if self.backend == "sglang" else "model"
        kwargs = {
            model_key: self.base_model if is_draft else self.model_id,
            **engine_tp_kwargs(self.backend, self.tensor_parallel_size),
            **copy.deepcopy(self.engine_kwargs),
        }
        if is_draft:
            # test_eagle may replace the checkpoint ID with its local path after collection.
            if self.backend == "sglang":
                if self.require_speculation and kwargs.get("speculative_algorithm") != "DFLASH":
                    raise ValueError(
                        "This draft execution check requires speculative_algorithm: DFLASH"
                    )
                if "speculative_draft_model_path" in kwargs:
                    raise ValueError("Set the draft checkpoint through model_id, not engine_kwargs")
                if not kwargs.get("speculative_algorithm"):
                    raise ValueError(
                        "Draft cases require speculative_algorithm to load the drafter"
                    )
                kwargs["speculative_draft_model_path"] = self.model_id
            else:
                spec_config = kwargs.get("speculative_config")
                if not isinstance(spec_config, dict):
                    raise ValueError("Draft cases require speculative_config to load the drafter")
                if self.require_speculation and spec_config.get("method") != "dflash":
                    raise ValueError(
                        "This draft execution check requires speculative_config.method: dflash"
                    )
                model_field = "model" if self.backend == "vllm" else "speculative_model_dir"
                model_fields = (
                    {"model"}
                    if self.backend == "vllm"
                    else {"speculative_model_dir", "speculative_model"}
                )
                if model_fields & spec_config.keys():
                    raise ValueError("Set the draft checkpoint through model_id, not engine_kwargs")
                spec_config[model_field] = self.model_id
        print(f"Engine arguments ({self.backend}): {kwargs!r}", flush=True)
        return kwargs

    def run(self):
        """Run the deployment based on the specified backend."""
        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            pytest.skip("CUDA is not available")
            return
        if torch.cuda.get_device_capability() < (
            self.mini_sm // 10,
            self.mini_sm % 10,
        ):
            pytest.skip(reason=f"Requires sm{self.mini_sm} or higher")
            return
        if self.sm_family is not None and (
            torch.cuda.get_device_capability()[0] != self.sm_family // 10
        ):
            pytest.skip(reason=f"Requires an sm{self.sm_family // 10}x GPU")
            return
        if self.sm_versions is not None:
            major, minor = torch.cuda.get_device_capability()
            if major * 10 + minor not in self.sm_versions:
                pytest.skip(f"Selected backend recipe requires SM in {self.sm_versions}")

        if torch.cuda.device_count() < self.tensor_parallel_size:
            pytest.skip(reason=f"Requires at least {self.tensor_parallel_size} GPUs")
            return

        print(f"Deploying model: {self.model_id} with backend: {self.backend}")
        print(f"Tensor parallel size: {self.tensor_parallel_size}")
        print(f"Sampling arguments: {self.sampling_kwargs!r}", flush=True)
        if self.input_config:
            print(f"Input configuration: {self.input_config!r}", flush=True)
        if self.chat_kwargs:
            print(f"Chat arguments: {self.chat_kwargs!r}", flush=True)
        # Use subprocess + capture so pytest capture=tee-sys (and DB plugins) see deploy output.
        if self.backend in ("vllm", "trtllm", "sglang"):
            _run_deploy_via_subprocess(
                backend=self.backend,
                model_id=self.model_id,
                tensor_parallel_size=self.tensor_parallel_size,
                mini_sm=self.mini_sm,
                base_model=self.base_model,
                engine_kwargs=self.engine_kwargs,
                sampling_kwargs=self.sampling_kwargs,
                input_config=self.input_config,
                chat_kwargs=self.chat_kwargs,
                require_speculation=self.require_speculation,
            )
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _deploy_trtllm_impl(self):
        """Run TensorRT-LLM deploy (used by subprocess in run())."""
        from tensorrt_llm import LLM, SamplingParams
        from tensorrt_llm.llmapi import CudaGraphConfig, EagleDecodingConfig, KvCacheConfig

        sampling_params = SamplingParams(**self.sampling_kwargs)
        llm_kwargs = self._engine_arguments()
        # Preserve the typed configs used by this suite; YAML supplies their field values.
        for field, config_type in (
            ("kv_cache_config", KvCacheConfig),
            ("cuda_graph_config", CudaGraphConfig),
        ):
            if isinstance(llm_kwargs.get(field), dict):
                llm_kwargs[field] = config_type(**llm_kwargs[field])
        if "eagle" in self.model_id.lower() and isinstance(
            llm_kwargs.get("speculative_config"), dict
        ):
            llm_kwargs["speculative_config"] = EagleDecodingConfig(
                **llm_kwargs["speculative_config"]
            )
        llm = LLM(**llm_kwargs)

        # Deferred import: transformers is a heavy optional dependency not always present.
        from transformers import AutoTokenizer

        tokenizer_model = (
            self.base_model
            if self.base_model or "eagle" in self.model_id.lower()
            else self.model_id
        )
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, trust_remote_code=True)
        prompts = COMMON_PROMPTS
        if tokenizer.chat_template is not None:
            prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for prompt in COMMON_PROMPTS
            ]

        outputs = llm.generate(prompts, sampling_params)
        assert len(outputs) == len(COMMON_PROMPTS), (
            f"Expected {len(COMMON_PROMPTS)} outputs, got {len(outputs)}"
        )
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text
            assert isinstance(generated_text, str), f"Output {i} text is not a string"
            assert generated_text.strip(), f"Output {i} generated empty text"
            print(f"Model: {self.model_id}")
            print(f"Input prompt: {COMMON_PROMPTS[i]!r}")
            print(f"Rendered prompt: {output.prompt!r}")
            print(f"Generated text: {generated_text!r}")
            print("-" * 50)
        del llm

    def _deploy_vllm_impl(self):
        """Run vLLM deploy (used by subprocess in run())."""
        from vllm import LLM, SamplingParams

        llm = LLM(**self._engine_arguments())
        sampling_params = SamplingParams(**self.sampling_kwargs)
        prompts = [self.input_config["prompt"]] if self.input_config else COMMON_PROMPTS
        conversations = [[{"role": "user", "content": p}] for p in prompts]
        if "image" in self.input_config:
            # Pillow is needed only for the optional vision-language cases.
            from PIL import Image

            with Image.open(self.input_config["image"]) as image, io.BytesIO() as buffer:
                image.convert("RGB").save(buffer, format="PNG")
                data = base64.b64encode(buffer.getvalue()).decode("ascii")
            conversations = [
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{data}"},
                            },
                            {"type": "text", "text": prompts[0]},
                        ],
                    }
                ]
            ]
        before = self._vllm_draft_counts(llm) if self.require_speculation else {}
        outputs = llm.chat(conversations, sampling_params, **self.chat_kwargs)
        if self.require_speculation:
            after = self._vllm_draft_counts(llm)
            counts = {key: after[key] - before[key] for key in before}
            print(f"Speculative execution counts (generation delta): {counts!r}", flush=True)
            assert all(value > 0 for value in counts.values()), (
                f"Generation did not exercise the draft checkpoint: {counts}"
            )

        assert len(outputs) == len(prompts), f"Expected {len(prompts)} outputs, got {len(outputs)}"

        for i, output in enumerate(outputs):
            assert hasattr(output, "outputs"), f"Output {i} missing 'outputs' attribute"
            assert len(output.outputs) > 0, f"Output {i} has no generated text"
            assert hasattr(output.outputs[0], "text"), f"Output {i} missing 'text' attribute"
            assert isinstance(output.outputs[0].text, str), f"Output {i} text is not a string"
            generated_text = output.outputs[0].text
            assert generated_text.strip(), f"Output {i} generated empty text"

            print(f"Model: {self.model_id}")
            print(f"Input prompt: {prompts[i]!r}")
            print(f"Rendered prompt: {getattr(output, 'prompt', None)!r}")
            print(f"Generated text: {generated_text!r}")
            print("-" * 50)
        del llm

    @staticmethod
    def _vllm_draft_counts(llm):
        names = ("vllm:spec_decode_num_drafts", "vllm:spec_decode_num_draft_tokens")
        counts = dict.fromkeys(names, 0)
        for metric in llm.get_metrics():
            if metric.name in counts:
                counts[metric.name] += metric.value
        return counts

    def _deploy_sglang_impl(self):
        """Run SGLang deploy (used by subprocess in run())."""
        import sglang as sgl

        if "eagle" in self.model_id.lower() and "nemotron" in self.base_model.lower():
            # Nemotron hybrid (Mamba+attention) requires extra_buffer scheduler
            # strategy and SGLANG_ENABLE_SPEC_V2 for radix cache compatibility
            # with speculative decoding
            os.environ["SGLANG_ENABLE_SPEC_V2"] = "1"
        llm = sgl.Engine(**self._engine_arguments())
        try:
            # Deferred import: transformers is a heavy optional dependency not always present.
            from transformers import AutoTokenizer

            tokenizer_model = (
                self.base_model
                if self.base_model or "eagle" in self.model_id.lower()
                else self.model_id
            )
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, trust_remote_code=True)
            if tokenizer.chat_template is not None:
                prompts = [
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": p}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for p in COMMON_PROMPTS
                ]
            else:
                prompts = COMMON_PROMPTS

            generation_kwargs = (
                {"sampling_params": self.sampling_kwargs} if self.sampling_kwargs else {}
            )
            outputs = llm.generate(prompts, **generation_kwargs)
            if self.require_speculation:
                names = ("spec_verify_ct", "spec_num_proposed_drafts")
                counts = {
                    name: sum(output.get("meta_info", {}).get(name, 0) for output in outputs)
                    for name in names
                }
                print(f"Speculative execution counts: {counts!r}", flush=True)
                assert all(value > 0 for value in counts.values()), (
                    f"Generation did not exercise the draft checkpoint: {counts}"
                )
            assert len(outputs) == len(COMMON_PROMPTS), (
                f"Expected {len(COMMON_PROMPTS)} outputs, got {len(outputs)}"
            )
            for i, output in enumerate(outputs):
                if isinstance(output, dict):
                    generated_text = output["text"]
                elif isinstance(output, str):
                    generated_text = output
                else:
                    raise TypeError(f"Output {i} has unexpected type {type(output)}: {output!r}")
                assert isinstance(generated_text, str), f"Output {i} text is not a string"
                assert generated_text.strip(), f"Output {i} generated empty text"
                print(f"Model: {self.model_id}")
                print(f"Input prompt: {COMMON_PROMPTS[i]!r}")
                print(f"Rendered prompt: {prompts[i]!r}")
                print(f"Generated text: {generated_text!r}")
                print("-" * 50)
        finally:
            llm.shutdown()
        del llm
