# Serve any model behind an OpenAI-compatible API (instructions for the agent)

You are a coding agent and the work needs a model that is not a chat model, or a chat model on your own terms: an image generator, a voice, a transcriber, an embedding model, a detector, an open model with its own policy. GPU for Agents (`gpu`) rents the GPU, serves the model there behind the OpenAI API shapes, and stops the machine once nobody calls it. This page is the contract. Follow it exactly.

## 1. Choosing the model and its server

```bash
gpu init --model stabilityai/sd-turbo      # reads the model card, writes gpu.yaml
gpu offers                                 # prices; ask the user before renting
gpu setup                                  # rents, deploys, waits until the model answers
eval "$(gpu env)"                          # OPENAI_BASE_URL, OPENAI_API_KEY, JAMBU_MODEL_ID
```

`gpu init` reads the model's Hugging Face card (`pipeline_tag`, `library_name`) and writes one of three engines:

| The model | `runtime.engine` | `model.task` | Routes |
|---|---|---|---|
| chat, vision-language (`text-generation`, `image-text-to-text`) | `vllm` | chat | `/v1/chat/completions`, `/v1/completions`, `/v1/responses` |
| embeddings (`feature-extraction`, `sentence-similarity`) | `hf` | `embeddings` | `/v1/embeddings` |
| images (`text-to-image`, diffusers) | `hf` | `image` | `/v1/images/generations` |
| transcription (`automatic-speech-recognition`) | `hf` | `transcription` | `/v1/audio/transcriptions`, `/v1/audio/translations` |
| voice (`text-to-speech`, `text-to-audio`) | `hf` | `speech` | `/v1/audio/speech` |
| any other pipeline (detection, classification, translation, depth, ...) | `hf` | `pipeline` | `/v1/pipeline` |
| a model the above cannot load | `custom` | (yours) | (yours, section 4) |

Flags win over the card: `--engine hf --task speech`, `--min-vram-gb 24`. A chat model vLLM cannot run can go on `hf` with `--task chat`. `GET /v1/models` on the endpoint lists the model, its task and its routes.

`engine: hf` runs gpu's own server (`gpu4agents/serve/hf/server.py`) on `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime`. It installs `transformers`, `diffusers`, `sentence-transformers` or `soundfile` as the task needs, then loads the model. When the model card asks for more packages, list them, pinned:

```yaml
gpu_runtime:
  model:
    id: org/model
    task: speech
  runtime:
    engine: hf
    packages: ["phonemizer==3.3.0"]
```

## 2. Call it with any OpenAI client

The base URL and key come from `gpu env` (or `gpu_connection` over MCP). The key is per machine.

```python
import base64
from openai import OpenAI
client = OpenAI()                      # reads OPENAI_BASE_URL and OPENAI_API_KEY
model = "stabilityai/sd-turbo"         # JAMBU_MODEL_ID

# image
img = client.images.generate(model=model, prompt="a red fox, watercolor", size="512x512",
                             extra_body={"num_inference_steps": 2, "guidance_scale": 0.0, "seed": 7})
open("fox.png", "wb").write(base64.b64decode(img.data[0].b64_json))

# voice (WAV; mp3, flac and opus when the machine's libsndfile can write them)
client.audio.speech.create(model=model, voice="default", input="Hello.", response_format="wav").write_to_file("hi.wav")

# transcription (wav, flac, ogg, mp3); response_format json, text, srt, vtt, verbose_json
text = client.audio.transcriptions.create(model=model, file=open("hi.wav", "rb")).text

# embeddings (float or base64; the SDK asks for base64 by default)
vectors = client.embeddings.create(model=model, input=["first", "second"]).data
```

Any other pipeline task takes JSON, with images and audio as base64 data URLs, and returns the pipeline's output as JSON (images inside it come back as base64 PNG):

```bash
curl -s "$OPENAI_BASE_URL/pipeline" -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{"inputs": "data:image/jpeg;base64,...", "parameters": {"threshold": 0.9}}'
```

Image extras go in the body (`extra_body` in the SDK): `num_inference_steps`, `guidance_scale`, `negative_prompt`, `seed`, `output_format` (png, jpeg, webp). `response_format: "url"` returns a data URL, since nothing is hosted.

## 3. Lifecycle

- Requests are activity. The server counts them on `/metrics` and the watchdog reads that, so the machine stays up while clients call it and shuts down by itself after `lifecycle.idle_timeout` (15 min) once they stop. `max_lifetime` and the session budget still apply.
- Changing the model keeps the machine. Edit `model.id` or `model.task` (or your server's code) and run `gpu setup`: it stops the running server and starts the new one on the same GPU. `gpu redeploy` restarts it with nothing changed. A bigger `min_vram_gb` does need a new machine (`gpu setup --recreate`).
- A machine serves one model, and the GPU takes one request at a time; the rest queue. For concurrent chat traffic, use vLLM.
- `gpu logs` shows the server's log (installs, download, load, tracebacks). A server that dies while loading ends `gpu setup` with its last lines.
- When the machine is stopped, it keeps its disk, weights included: `gpu setup` restarts it and redeploys in about a minute. On Vast.ai an idle machine is destroyed instead, because a stopped one still bills its disk, so the next `gpu setup` rents a new one and downloads again (`lifecycle.keep_disk: true` stops it instead).

## 4. Your own server (`runtime.engine: custom`)

When no built-in server loads the model (a library of its own, a pipeline of several models, video, a custom API), write the server. gpu runs it on the GPU and treats it exactly like the built-in one.

```bash
gpu init --engine custom --min-vram-gb 24    # writes gpu.yaml and server/ (server.py, gpujob.yaml)
```

`server/gpujob.yaml` says how to start it:

```yaml
run: python3 server.py
setup: pip install -r requirements.txt     # pinned; torch already comes with the image
env: {}
```

The server must:

1. Listen on `0.0.0.0:$PORT`.
2. Answer `GET /health` with 503 until the model is loaded, then 200. `gpu setup` waits for the 200.
3. Require `Authorization: Bearer $JAMBU_API_KEY` on every route except `/health` and `/metrics`.
4. Serve `GET /metrics` in Prometheus text with `jambu_requests_running` (a gauge) and `jambu_requests_total` (a counter). Without them, requests do not count as activity and the machine stops after `idle_timeout` even while in use.
5. Use the OpenAI request and response shapes where one exists, so OpenAI clients work unchanged. Invent a route only for what OpenAI has no shape for.
6. Exit non-zero when loading fails, after printing why. Never print secrets.

The template in `server/server.py` already does 1 to 4; write `load()` and the handlers. It also gets `JAMBU_MODEL_ID` (the config's `model.id`, if any), `HF_HOME` (a cache on the machine's disk) and `HF_TOKEN` when the project has one. The same exclusions as jobs apply to the upload (`.git`, virtualenvs, `.env` files, ...), and the 2 GB limit: download weights in `load()`, do not ship them.

## 5. Rules

1. Renting costs money: show `gpu offers` and get the user's yes before `gpu setup`.
2. Size `--min-vram-gb` from the weights: bf16 needs about 2 GB per billion parameters, plus room for activations (a 12B image model wants 32 GB or more; `gpu init` sizes it from the card when it can).
3. Use models for what their licenses and the user's purpose allow. Open models with other policies are for legitimate work the user owns: research, evaluation, red-teaming their own systems, creative work.
4. When the work is done, tell the user what ran, the endpoint, the cost (`gpu status`), and offer `gpu stop` or `gpu destroy`.
