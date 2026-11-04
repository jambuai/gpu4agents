# Where gpu.yaml's model-specific values come from

Every value in `model:`/`compute.gpu:`/`runtime:` that depends on the model
you deploy (VRAM, dtype, context length, tool-call parser, `trust_remote_code`)
has a source. This page lists those sources and the command that reads them.

## `gpu inspect-model`

```bash
gpu inspect-model Qwen/Qwen2.5-7B-Instruct
```

```text
Qwen/Qwen2.5-7B-Instruct @ main
  architecture   Qwen2ForCausalLM (model_type: qwen2)
  parameters     7.62B (BF16)
  weight size    ~15.2 GB -> suggested min_vram_gb: 22
  context        native 32768
  tool calling   hermes

Sources
  https://huggingface.co/api/models/Qwen/Qwen2.5-7B-Instruct
  https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/raw/main/config.json

Suggested gpu_runtime: snippet
gpu_runtime:
  model:
    id: Qwen/Qwen2.5-7B-Instruct

  compute:
    gpu:
      min_vram_gb: 22  # estimate, see notes
      count: 1

  runtime:
    engine: vllm
    context_length: 32768
    tool_calling: auto  # resolves to hermes automatically
```

Paste the snippet under `gpu_runtime:` in your `gpu.yaml` and adjust
`lifecycle`/`health`/`budget` to get a starting config. Add `--json` for
machine-readable output, `--revision` for a non-`main` branch/tag, and
`--hf-token-env` if the repo is gated and your token lives under a different
env var than `HF_TOKEN`.

The output is a starting point. The command never calls the provider API and
never boots anything. It reads two public HuggingFace documents and does
arithmetic. Confirm the numbers with `gpu offers`, which finds available
capacity at that size, and with a real `gpu setup`, which is the only way to
know that a model boots and serves.

## The two documents, and what each field feeds

### 1. `https://huggingface.co/api/models/<repo>`: the Hub API

The on-disk shape of the checkpoint. It is read from the safetensors file
headers, so it is more reliable than what the model's author declares in
prose.

| JSON field | Feeds | Notes |
| --- | --- | --- |
| `safetensors.total` | `compute.gpu.min_vram_gb` (via weight size) | Total parameter count. Absent for GGUF-only repos, which you size manually. |
| `safetensors.parameters` | `runtime.dtype`, byte width for the VRAM estimate | `{"BF16": N, "F8_E4M3": M}` style breakdown. `inspect-model` picks the dtype with the most parameters, since a few embedding/norm tensors often stay bf16 even in an FP8 release. |
| `gated` | whether you need `model.hf_token_env` | `false` / `"auto"` / `"manual"`. Any non-`false` value means a token is required. |
| `tags`, `pipeline_tag` | sanity check | Confirms it is a text-generation checkpoint before you provision a GPU for it. |

### 2. `https://huggingface.co/<repo>/raw/<revision>/config.json`: the model's own HF config

The architecture's declaration of itself. `transformers` and vLLM read it to
know how to load the model.

| JSON field | Feeds | Notes |
| --- | --- | --- |
| `architectures` | recognizing the model family | e.g. `Qwen3_5ForConditionalGeneration`. Cross-reference against [vLLM's supported models](https://docs.vllm.ai/en/latest/models/supported_models/). An unlisted architecture fails to load regardless of VRAM. |
| `model_type` | hybrid-attention detection | gpu flags known linear-attention families (`qwen3_5`/`qwen3_6`/`qwen3_8`, `jamba`, `mamba`, `bamba`, `deltanet`/`gdn` in the string, ...). These need an Ampere or newer GPU for their Triton/FLA kernels. In a live run, a Turing card (Quadro RTX 8000) failed to boot vLLM at all for a `qwen3_5` model, and an Ampere card (RTX A6000) worked immediately. |
| `torch_dtype` | `runtime.dtype`, VRAM estimate (fallback) | Used only when the Hub API's `safetensors.parameters` is absent. When both exist, the API's on-disk dtype is preferred, since a repo can ship a different dtype than it trained in. |
| `quantization_config.quant_method` | VRAM estimate, `runtime.dtype: auto` | `fp8`/`awq`/`gptq`/`bitsandbytes_4bit` change the byte width per parameter; when present, set `runtime.dtype: auto` and let vLLM read the quantization from the checkpoint instead of forcing a dtype that fights it. |
| `max_position_embeddings` | `runtime.context_length` | The model's native window. `inspect-model` suggests `min(native, 32768)`. The native number is often a YaRN-extended marketing ceiling (e.g. 1,048,576), and allocating KV cache for the full window is neither necessary nor cheap for typical agentic/tool-use turns. |
| `auto_map` | `model.trust_remote_code` | Its presence, regardless of value, is the signal `transformers` uses to know a repo ships custom modeling code. If it is present, you need `trust_remote_code: true`. If it is absent, you may still need it: some very new architectures need it during a transition period even with native library support. |
| `rope_scaling` | context length ceiling sanity-check | Confirms whether a large `max_position_embeddings` is native or extended (YaRN/linear/dynamic). Extended windows cost more KV cache per token than the base architecture would suggest. |

### Tool-call parser: a third, unwritten source

No machine-readable field says "use `--tool-call-parser X`". The parser comes
from a hand-maintained table in `gpu4agents/core/tool_parsers.py`, matched
against vLLM's [tool calling
docs](https://docs.vllm.ai/en/latest/features/tool_calling/) and sometimes
against live tests when the docs don't cover a model family yet. The Qwen3.5+
`qwen3_xml` entry came from a live test where `hermes` left the model's tool
call as inert text in `content` instead of a parsed `tool_calls` array.

Consequences:

- It only matches on repo name substrings (`"qwen3.5"`, `"llama-3"`, ...).
  A fine-tune with a custom name (`empero-ai/Qwythos-9B-v2`, based on Qwen3.5)
  is not recognized even though the underlying format is known. Set
  `runtime.tool_call_parser` explicitly once you know the base model.
- It can be wrong or incomplete for new models. If `inspect-model` reports no
  parser guessed, or an explicit parser you set does not produce a
  `tool_calls` array, the fix is the same. Pick a candidate parser (start from
  the model's base-architecture family, e.g. Qwen3.8 → try what worked for
  Qwen3.5), restart the server with `--enable-auto-tool-choice
  --tool-call-parser <candidate>`, and send one real tool-calling request.
  A working parser puts the call in `message.tool_calls`, not `message.content`.
  See [`docs/ADDING_A_PROVIDER.md`](ADDING_A_PROVIDER.md) for how the fingerprint
  interacts with runtime-only changes like this if you're doing it through
  `gpu setup` rather than by hand.
- When you do confirm one empirically, add it to `_TOOL_PARSER_HINTS` in
  `gpu4agents/core/tool_parsers.py` (a one-line PR) so `tool_calling: auto`
  picks it up for everyone next time, the same way the Qwen3.5+ finding did.

## Don't write a new gpu.yaml per model - use the catalog

For each new model it is tempting to copy the whole file, since the provider,
lifecycle and budget stay the same and only `model:`/`compute.gpu:` differ.
After a few models a repo has `jambu.qwythos.yaml`, `jambu.model-b.yaml`,
`jambu.model-c.yaml`..., duplicated again in every sub-project directory that
wants one. Changing a lifecycle default then means editing all of them.

Instead, keep what varies by model in the catalog and what varies by project
in the active `gpu.yaml`, and reference the catalog entry by name.

```yaml
# gpu.models.yaml - one entry per model, shared by every project in the
# repo (found by walking up from a project's gpu.yaml, same as gpu.yaml
# itself is found - one catalog can sit at the repo root and serve every
# sub-project under it).
qwythos-9b:
  model: {id: empero-ai/Qwythos-9B-v2, trust_remote_code: true}
  compute: {gpu: {min_vram_gb: 40, name_filter: A6000}}
  runtime: {tool_calling: "on", tool_call_parser: qwen3_xml}

qwen3.8-27b-uncensored:
  model: {id: orcarouter/Qwen3.8-27B-Uncensored-FP8}
  compute: {gpu: {min_vram_gb: 44, name_filter: A6000}}
```

```yaml
# gpu.yaml - just this project's policy + which model
gpu_runtime:
  version: 1
  model_profile: qwythos-9b   # the only thing that changes to switch models
  lifecycle: {idle_timeout: 15m, max_lifetime: 6h}
  budget: {max_hourly_cost_usd: 1.50}
```

The project's own fields always win over the profile's. A profile is a set of
defaults, so a project can override one leaf (`compute.gpu.min_vram_gb: 80`,
say) without redeclaring the whole model.

`gpu inspect-model <repo> --add-profile <name>` writes straight into the
catalog, creating it if it doesn't exist, instead of printing a snippet to
paste into a new file. `gpu profiles` lists what's in the catalog.

This is the same base+overlay pattern as docker-compose's `extends`, Helm
`values.yaml`, or Kustomize overlays. One place defines the models, and each
project names the one it uses and what it changes.

A full standalone file still makes sense for a one-off experiment nobody else
will reuse, or for a config in a different project that will never share this
repo's catalog. Trying a new model is usually a catalog case.

## Dynamic / un-modeled vLLM flags

`gpu.yaml`'s schema models the fields that affect lifecycle, cost, and
provider portability. It does not try to cover every vLLM flag, since vLLM adds
flags faster than a fixed contract could track. For everything else, `runtime:`
has two escape hatches, applied in this order. The later one wins on conflict,
matching how vLLM's own argparse takes the last occurrence of a repeated flag:

```yaml
runtime:
  # Structured: {flag: value} -> --flag value. Handles the common shapes:
  #   true            -> a bare flag                    --enable-prefix-caching
  #   false           -> the negated flag                --no-enforce-eager
  #   a dict/list      -> JSON-encoded (several vLLM flags take JSON)
  #                                          --limit-mm-per-prompt '{"image":0}'
  #   anything else    -> --flag str(value)              --max-num-seqs 64
  vllm_args:
    max_num_seqs: 64
    enable-prefix-caching: true

  # Raw argv, appended last, verbatim - for anything the structured form
  # can't express (repeated flags, exotic syntax).
  extra_args: ["--kv-cache-dtype", "fp8"]
```

You can set a flag that gpu already manages through another field (`--model`,
`--port`, `--enable-auto-tool-choice`, ...) via `vllm_args`. It wins, since it
is applied after those. `gpu validate` warns about it, because the config field
meant to control that behavior (e.g. `runtime.tool_calling`) then silently stops
reflecting what is running.
