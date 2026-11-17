# Model Optimizer End-to-End Tests

This directory contains end-to-end tests for the Model Optimizer's [examples](../../examples/).

## Adding new tests

To add a test for a new example, create a new example directory in `tests/examples` following the existing examples as guidance.
Make sure to use as small models and less data as possible to keep the tests fast. Unless needed, have tests finish under 15 minutes.

## Running the tests

To run a test, start from the recommended docker image from our [installation docs](https://nvidia.github.io/Model-Optimizer/getting_started/2_installation.html).
Then mount your local modelopt directory to `/workspace/Model-Optimizer` and run this from the root of the repository.

```bash
cd /workspace/Model-Optimizer
pip install -e ".[all,dev-test]"
pytest tests/examples/$TEST
```

### Hugging Face checkpoint deployment

`hf_ptq/deploy_cases/<family>/<checkpoint>.yaml` contains the deployment case
defaults. Each file describes one exact checkpoint and keeps its configured
vLLM, SGLang and TensorRT-LLM backends together. Top-level keys match functions
in `hf_ptq/test_deploy.py`, each containing a case list; one checkpoint can appear
under multiple test functions when their setup differs.
Add model cases there instead of adding Python parameter lists. This test
configuration is separate from the checkpoint's Hugging Face `config.json`.

Each backend's `engine_kwargs` lists the initialization parameters explicitly
set by that case; `sampling_kwargs` lists its generation parameters. vLLM's
`chat_kwargs` contains request options such as `chat_template_kwargs`, separate
from `SamplingParams`. For example,
the [GLM-5.1 configuration](hf_ptq/deploy_cases/glm/glm-5.1-nvfp4.yaml) sets
vLLM's `max_model_len` and SGLang's `context_length` to 4096. Edit the case YAML
to change its checked-in defaults, or use a run YAML for a particular run.
Options omitted from these maps use the installed engine's defaults; the Python
adapter does not supply context-length, quantization or sampling defaults.

The catalog is discovered once per pytest collection, with files sorted by
path. Existing node IDs and default parameter values are preserved, including
`<model_id>_<backend>` parameter IDs for `-k` filtering. Case order follows the
sorted files rather than the former Python list order.

Use an optional run YAML to override settings for the selected cases:

```yaml
run:
  backend: vllm
  tensor_parallel_size: 4
backends:
  vllm:
    engine_kwargs:
      max_model_len: 4096
    sampling_kwargs:
      temperature: 0
      max_tokens: 32
```

```bash
pytest tests/examples/hf_ptq/test_deploy.py --run-release \
  -k Llama-3.1-8B-Instruct-FP8 --deploy-config /path/to/deploy-run.yaml
```

`--deploy-backend` and `--deploy-tp` take precedence over YAML `run` settings.
Without overrides, case defaults and installed-backend discovery remain in
effect. An explicitly selected backend is collected even if its import is
broken, so missing engine support cannot silently remove those cases.

TP belongs to the run: the resource check and engine constructor use the same
resolved value (`tensor_parallel_size` for vLLM/TensorRT-LLM, `tp_size` for
SGLang). The GPU allocation must cover it; GPU memory and model/engine support
still determine whether deployment succeeds. Model identity and TP cannot be
overridden inside `engine_kwargs`.
For image generation, TensorRT-LLM's VisualGen adapter binds the same resolved
TP to `parallel_config.tp_size` and checks the complete worker count, including
CFG and Ulysses parallelism, against the allocation.

Each backend accepts its native `engine_kwargs` and `sampling_kwargs`; vLLM also
accepts `chat_kwargs`. Overrides
apply to every selected case of that backend. Maps merge by parameter name;
nested parameter values replace the complete value, and explicit YAML `null`
is preserved. TensorRT-LLM's `kv_cache_config` and `cuda_graph_config` mappings,
and its EAGLE `speculative_config`, are converted to the corresponding native
configuration objects before initialization.

Model paths and TP are bound at runtime. For paired draft checkpoints, use `base_model` for the
target and `model_id` for the drafter; the adapter supplies the draft path after
any local-checkpoint preparation. Do not duplicate that path inside
`engine_kwargs` or disable speculation: an EAGLE case must load its drafter.
The subprocess logs the resolved engine arguments before
initialization, including values from run overrides. With SGLang layernorm
sequence parallelism, override EP together with TP; their values must match.

`require_speculation: true` checks actual DFlash execution, in addition to
nonempty generation. The vLLM adapter requires `disable_log_stats: false` and
positive per-generation draft counter deltas. SGLang checks the batch's verify
and proposed-draft counts. Missing or zero counters fail the test; there is no
acceptance-rate or performance threshold. These cases retain the DFlash method
when run overrides are applied, so n-gram speculation cannot substitute for the
published draft checkpoint.

For vLLM image-text requests, set `input_config.prompt` and `input_config.image`
in the case YAML. Image paths are relative to that YAML file and must exist at
collection time. The adapter embeds the local image in the chat request; it
does not fetch an image from the network. A prompt without an image selects a
single text request. Cases without `input_config` retain the existing text batch.

Image-generation checkpoints use `task: image-generation` in the same catalog.
The [Qwen-Image case](hf_ptq/deploy_cases/qwen/qwen-image-nvfp4.yaml) pins its HF
`revision`, `pipeline_class` and backend `engine_version`. Its independent
VisualGen adapter downloads the published pipeline, checks its static NVFP4
metadata and scales, then generates and decodes an RGB image of the configured
size. It does not run ModelOpt quantization or export. Engine and sampling
overrides use the same `--deploy-config` interface as the text cases; select
image cases separately because VisualGen and LLM parameters differ.

```bash
pytest tests/examples/hf_ptq/test_deploy.py --run-release \
  -k Qwen-Image-NVFP4 --deploy-backend trtllm \
  --basetemp=test_artifacts/deploy_hf/pytest
```

The image, logs and resolved request are saved in the case's pytest temporary
directory. The command above keeps them in a directory suitable for archiving.

The newly added release cases have the following initial deployment recipes:

| Checkpoint | Backend | Image | Default TP |
| --- | --- | --- | --- |
| Muse-Spark-Open-816B-A42B-DFlash2 | SGLang | `lmsysorg/sglang:dev-muse-spark-open` | 4 (EP4) |
| NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16 | vLLM | `vllm/vllm-openai:v0.32.0` | 2 |
| NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 | vLLM | `vllm/vllm-openai:v0.32.0` | 1 |
| NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-LiLiCorr | vLLM | `vllm/vllm-openai:nightly-cf5cea2da37d3dfed27d2e0aa3041c3e553c2d57` | 1 |
| Step-5-Preview-NVFP4 | vLLM | `vllm/vllm-openai:step-5-preview` | 4 |
| Qwen-Image-NVFP4 | TRT-LLM VisualGen | `nvcr.io/nvidia/tensorrt-llm/release:1.4.0rc3` | 1 |

Each YAML links its upstream recipe. These configurations await GPU validation
in this suite; image tags alone are not execution evidence. The LiLiCorr recipe
allows data center Blackwell and DGX Spark candidates with
`sm_versions: [100, 103, 121]`.
This does not qualify every device with that capability. Muse's layernorm SP
recipe needs TP4/EP4 or TP8/EP8 with NVLink/NVSwitch connectivity. The 8192-token
context and 128-token generation limits are bounded functional workloads, not
model capacity limits. Other engines listed by a model card remain separate
coverage work until their corresponding cases are implemented and executed.

The run YAML customizes the checked-in cases; it is not another model catalog.
Existing GPU/SM skips, checkpoint preparation and shared ModelOpt pytest
dependencies still apply. Collection alone does not validate model execution.

These integration tests load the checkpoint and generate a batch through the
backend's Python API, including image input for the new VL cases. They require
one non-empty text output per prompt, or a decodable image of the configured size
for image-generation cases;
they do not measure model accuracy or exercise an HTTP serving endpoint.
GPU qualification requires actual passing cases, with the source revision,
backend image, checkpoint revision, resolved configuration and pytest results
recorded together. Configuration unit tests and collection previews do not
provide that execution evidence.

### Legacy Diffusers ONNX/TensorRT tests

Automatic example CI excludes `diffusers/test_diffusers_onnx_trt.py` while retaining
HF export, quantization, and PyTorch inference coverage. The legacy ONNX/TensorRT
tests remain available for manual validation during the deprecation transition:

```bash
pytest tests/examples/diffusers/test_diffusers_onnx_trt.py
```

## Environment variables

The following environment variables can be set to control the behavior of the tests:

- `MODELOPT_LOCAL_MODEL_ROOT`: If set, the tests will use the local model directory instead of downloading the model from the internet. Default is not set, which means the model will be downloaded.
