# Where jambu.yaml's model-specific values come from

Every value in `model:`/`compute.gpu:`/`runtime:` that depends on *which model*
you're deploying (VRAM, dtype, context length, tool-call parser,
`trust_remote_code`) has a real source. This is that source, and the command
that reads it for you.

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

Paste the snippet under `gpu_runtime:` in your `jambu.yaml`, adjust
`lifecycle`/`health`/`budget` for your situation, and you have a starting
config. Add `--json` for machine-readable output, `--revision` for a
non-`main` branch/tag, `--hf-token-env` if the repo is gated and your token
lives under a different env var than `HF_TOKEN`.

**This is a starting point, not an oracle.** It never talks to Vast.ai and
never boots anything — it only reads two public HuggingFace documents and does
arithmetic. Confirm the real number with `gpu offers` (finds actual
capacity at that size) and a real `gpu setup` (the only way to know a
model genuinely boots and serves).

## The two documents, and exactly what each field feeds

### 1. `https://huggingface.co/api/models/<repo>` — the Hub API

The **real, on-disk** shape of the checkpoint — more reliable than anything in
prose, because it's read from the actual safetensors file headers, not
declared by the model's author.

| JSON field | Feeds | Notes |
| --- | --- | --- |
| `safetensors.total` | `compute.gpu.min_vram_gb` (via weight size) | Total parameter count. Absent for GGUF-only repos — size manually if so. |
| `safetensors.parameters` | `runtime.dtype`, byte width for the VRAM estimate | `{"BF16": N, "F8_E4M3": M}` style breakdown. `inspect-model` picks the dtype with the most parameters — a few embedding/norm tensors often stay bf16 even in an FP8 release. |
| `gated` | whether you need `model.hf_token_env` | `false` / `"auto"` / `"manual"` — any non-`false` value means a real token is required, not just recommended. |
| `tags`, `pipeline_tag` | sanity check | Confirms it's actually a text-generation checkpoint before you provision a GPU for it. |

### 2. `https://huggingface.co/<repo>/raw/<revision>/config.json` — the model's own HF config

The architecture's own declaration of itself — what `transformers`/vLLM read
to know how to load it.

| JSON field | Feeds | Notes |
| --- | --- | --- |
| `architectures` | recognizing the model family | e.g. `Qwen3_5ForConditionalGeneration`. Cross-reference against [vLLM's supported models](https://docs.vllm.ai/en/latest/models/supported_models/) — an unlisted architecture will fail to load regardless of VRAM. |
| `model_type` | hybrid-attention detection | gpu flags known linear-attention families (`qwen3_5`/`qwen3_6`/`qwen3_8`, `jamba`, `mamba`, `bamba`, `deltanet`/`gdn` in the string, ...) — these need an **Ampere-or-newer** GPU for their Triton/FLA kernels. Confirmed the hard way: a Turing card (Quadro RTX 8000) failed to boot vLLM at all for a `qwen3_5` model; Ampere (RTX A6000) worked immediately. |
| `torch_dtype` | `runtime.dtype`, VRAM estimate (fallback) | Used only when the Hub API's `safetensors.parameters` is absent — the API's real on-disk dtype is preferred when both exist, since a repo can ship a different dtype than it trained in. |
| `quantization_config.quant_method` | VRAM estimate, `runtime.dtype: auto` | `fp8`/`awq`/`gptq`/`bitsandbytes_4bit` change the byte width per parameter; when present, set `runtime.dtype: auto` and let vLLM read the quantization from the checkpoint instead of forcing a dtype that fights it. |
| `max_position_embeddings` | `runtime.context_length` | The model's *native* window. `inspect-model` suggests `min(native, 32768)` — the native number is often a YaRN-extended marketing ceiling (e.g. 1,048,576), not a sane default: allocating KV cache for the full window is neither necessary nor cheap for typical agentic/tool-use turns. |
| `auto_map` | `model.trust_remote_code` | Its *presence* (regardless of value) is the actual signal `transformers` uses to know a repo ships custom modeling code. Absence doesn't guarantee you never need `trust_remote_code: true` — some very new architectures need it even with native library support during a transition period — but presence is a reliable "you definitely do." |
| `rope_scaling` | context length ceiling sanity-check | Confirms whether a large `max_position_embeddings` is native or extended (YaRN/linear/dynamic) — extended windows cost more KV cache per token than the base architecture would suggest. |

### Tool-call parser: a third, unwritten source

There is no machine-readable field anywhere that says "use `--tool-call-parser
X`" — this is `gpu4agents/core/tool_parsers.py`'s own hand-maintained table,
matched against vLLM's [tool calling
docs](https://docs.vllm.ai/en/latest/features/tool_calling/) and, sometimes,
empirical testing when the docs don't cover a model family yet (the Qwen3.5+
`qwen3_xml` entry was added after a live test showed `hermes` left the model's
tool call as inert text in `content` instead of a parsed `tool_calls` array).

Consequences:

- It only matches on **repo name substrings** (`"qwen3.5"`, `"llama-3"`, ...).
  A fine-tune with a custom name (`empero-ai/Qwythos-9B-v2`, based on Qwen3.5)
  won't be recognized even though the underlying format is known — set
  `runtime.tool_call_parser` explicitly once you know the base model.
- It can be wrong or incomplete for anything genuinely new. If `inspect-model`
  reports no parser guessed, **or an explicit one you set doesn't actually
  produce a `tool_calls` array**, the fix is the same: pick a candidate
  parser (start from the model's base-architecture family, e.g. Qwen3.8 → try
  what worked for Qwen3.5), restart the server with `--enable-auto-tool-choice
  --tool-call-parser <candidate>`, and send one real tool-calling request —
  a working parser puts the call in `message.tool_calls`, not `message.content`.
  See [`docs/ADDING_A_PROVIDER.md`](ADDING_A_PROVIDER.md) for how the fingerprint
  interacts with runtime-only changes like this if you're doing it through
  `gpu setup` rather than by hand.
- When you do confirm one empirically, add it to `_TOOL_PARSER_HINTS` in
  `gpu4agents/core/tool_parsers.py` (a one-line PR) so `tool_calling: auto`
  picks it up for everyone next time, the same way the Qwen3.5+ finding did.

## Don't write a new jambu.yaml per model - use the catalog

Every model you add this way is tempting to just copy the whole file: same
provider, same lifecycle, same budget, only `model:`/`compute.gpu:` actually
differ. Do that a handful of times and a repo ends up with
`jambu.qwythos.yaml`, `jambu.model-b.yaml`, `jambu.model-c.yaml`... duplicated
again in every sub-project directory that wants one - a pile of near-identical
files where changing a lifecycle default means editing all of them.

The fix: split "what varies by model" (the catalog) from "what varies by
project" (the active `jambu.yaml`), and reference the former by name.

```yaml
# jambu.models.yaml - one entry per model, shared by every project in the
# repo (found by walking up from a project's jambu.yaml, same as jambu.yaml
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
# jambu.yaml - just this project's policy + which model
gpu_runtime:
  version: 1
  model_profile: qwythos-9b   # the only thing that changes to switch models
  lifecycle: {idle_timeout: 15m, max_lifetime: 6h}
  budget: {max_hourly_cost_usd: 1.50}
```

The project's own fields always win over the profile's - a profile is a set
of defaults, not a lock, so a project can still override just one leaf
(`compute.gpu.min_vram_gb: 80`, say) without redeclaring the whole model.

`gpu inspect-model <repo> --add-profile <name>` writes straight into
the catalog (creating it if it doesn't exist) instead of printing a snippet
to paste into a new file - the research and the config land in the same
command. `gpu profiles` lists what's in it.

This is the same base+overlay pattern as docker-compose's `extends`, Helm
`values.yaml`, or Kustomize overlays - one place owns "what", each consumer
only says "which one, and what's different here."

**When a full standalone file still makes sense:** a genuinely one-off
experiment nobody else will reuse, or a config for a completely different
project that will never share this repo's catalog. Most "I need to try model
X" situations are the catalog case, not this one.

## Dynamic / un-modeled vLLM flags

`jambu.yaml`'s schema deliberately models the fields that affect *lifecycle,
cost, and provider portability* — it does not, and cannot, keep up with every
flag vLLM ships (vLLM adds flags faster than any fixed contract could track).
For everything else, `runtime:` has two escape hatches, applied in this order
(later wins on conflict, matching how vLLM's own argparse takes the last
occurrence of a repeated flag):

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

Setting a flag gpu already manages through another field (`--model`,
`--port`, `--enable-auto-tool-choice`, ...) via `vllm_args` still works — it
wins, since it's applied after those — but `gpu validate` warns about it,
because the config field that's supposed to control that behavior (e.g.
`runtime.tool_calling`) will silently stop reflecting what's actually running.
