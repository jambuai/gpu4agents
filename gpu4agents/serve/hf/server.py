"""OpenAI-compatible server for any Hugging Face model (`runtime.engine: hf`).

gpu uploads this directory to the rented machine and runs it as a service
job: ``install.sh`` installs the libraries the task needs, then this file loads
the model and answers on ``$PORT``. Standalone on purpose: it imports nothing
from gpu4agents, only the standard library until a backend loads its model.

One process serves one model for one task:

    chat           POST /v1/chat/completions   transformers (vLLM covers most chat models;
                                               this is for the ones it cannot run)
    embeddings     POST /v1/embeddings         sentence-transformers, else mean pooling
    image          POST /v1/images/generations diffusers
    transcription  POST /v1/audio/transcriptions, /v1/audio/translations
    speech         POST /v1/audio/speech       transformers text-to-speech
    pipeline       POST /v1/pipeline           any other transformers pipeline task

Every task also serves GET /health (503 while the model loads), GET /metrics
(the watchdog counts requests as activity from it) and GET /v1/models.
Everything but /health and /metrics needs ``Authorization: Bearer $JAMBU_API_KEY``.

Environment: JAMBU_MODEL_ID, JAMBU_TASK (auto: from the model's pipeline_tag),
JAMBU_SERVED_NAME, JAMBU_REVISION, JAMBU_DTYPE, JAMBU_TRUST_REMOTE_CODE,
JAMBU_API_KEY, PORT.
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import struct
import sys
import threading
import time
import traceback
import uuid
import wave
from email.parser import BytesParser
from email.policy import default as email_policy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MAX_BODY_BYTES = 200 * 1024**2

TASKS = ("chat", "embeddings", "image", "transcription", "speech", "pipeline")

# Hugging Face pipeline_tag -> task. Anything else with a tag is `pipeline`.
PIPELINE_TAG_TASK = {
    "text-generation": "chat",
    "image-text-to-text": "chat",
    "feature-extraction": "embeddings",
    "sentence-similarity": "embeddings",
    "text-to-image": "image",
    "automatic-speech-recognition": "transcription",
    "text-to-speech": "speech",
    "text-to-audio": "speech",
}

ROUTES = {
    "chat": ["/v1/chat/completions"],
    "embeddings": ["/v1/embeddings"],
    "image": ["/v1/images/generations"],
    "transcription": ["/v1/audio/transcriptions", "/v1/audio/translations"],
    "speech": ["/v1/audio/speech"],
    "pipeline": ["/v1/pipeline"],
}


class BadRequest(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


# --------------------------------------------------------------------------
# settings
# --------------------------------------------------------------------------


def env_flag(name):
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


class Settings:
    def __init__(self, environ=None):
        env = os.environ if environ is None else environ
        self.model = env.get("JAMBU_MODEL_ID", "").strip()
        self.task = (env.get("JAMBU_TASK") or "auto").strip().lower()
        self.served_name = env.get("JAMBU_SERVED_NAME", "").strip() or self.model
        self.revision = env.get("JAMBU_REVISION", "").strip() or None
        self.dtype = (env.get("JAMBU_DTYPE") or "auto").strip().lower()
        self.trust_remote_code = env.get("JAMBU_TRUST_REMOTE_CODE", "").lower() in ("1", "true")
        self.api_key = env.get("JAMBU_API_KEY", "").strip()
        self.port = int(env.get("PORT") or 8000)
        self.pipeline_tag = None


def resolve_task(settings):
    """The task to serve: JAMBU_TASK, else the model card's pipeline_tag."""
    if settings.task != "auto":
        if settings.task not in TASKS:
            raise SystemExit("JAMBU_TASK must be one of %s or auto" % ", ".join(TASKS))
        return settings.task
    from huggingface_hub import model_info

    info = model_info(settings.model, revision=settings.revision)
    settings.pipeline_tag = getattr(info, "pipeline_tag", None)
    library = getattr(info, "library_name", None)
    if settings.pipeline_tag in PIPELINE_TAG_TASK:
        return PIPELINE_TAG_TASK[settings.pipeline_tag]
    if library == "diffusers":
        return "image"
    if library == "sentence-transformers":
        return "embeddings"
    if settings.pipeline_tag:
        return "pipeline"
    raise SystemExit(
        "%s has no pipeline_tag on Hugging Face; set model.task in jambu.yaml" % settings.model
    )


# --------------------------------------------------------------------------
# helpers shared by the backends
# --------------------------------------------------------------------------


def torch_dtype(name):
    import torch

    if name in ("auto", ""):
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16 if torch.cuda.is_available() else torch.float32
    table = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
    }
    if name not in table:
        raise SystemExit("unsupported JAMBU_DTYPE %r" % name)
    return table[name]


def device():
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def message_text(content):
    """OpenAI message content (a string or a list of parts) -> its text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for part in content:
        if isinstance(part, dict) and part.get("type") in ("text", "input_text"):
            parts.append(str(part.get("text") or ""))
    return "\n".join(parts)


def decode_data_url(value):
    """``data:<mime>;base64,<payload>`` -> (mime, bytes); anything else -> None."""
    if not isinstance(value, str) or not value.startswith("data:") or "," not in value:
        return None
    head, payload = value[5:].split(",", 1)
    mime = head.split(";", 1)[0]
    if ";base64" not in head:
        raise BadRequest("only base64 data URLs are supported")
    return mime, base64.b64decode(payload)


def wav_bytes(samples, rate):
    """Float samples in [-1, 1] (a flat list, or numpy, mono) -> a 16-bit PCM WAV."""
    out = io.BytesIO()
    with wave.open(out, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(rate))
        handle.writeframes(pcm16(samples))
    return out.getvalue()


def pcm16(samples):
    try:
        import numpy as np

        array = np.clip(np.asarray(samples, dtype="float32").reshape(-1), -1.0, 1.0)
        return (array * 32767.0).astype("<i2").tobytes()
    except ImportError:
        values = [max(-1.0, min(1.0, float(s))) for s in samples]
        return struct.pack("<%dh" % len(values), *(int(v * 32767.0) for v in values))


def read_audio(data, target_rate):
    """Audio file bytes -> (mono float32 numpy array at ``target_rate``, seconds)."""
    import numpy as np

    try:
        import soundfile

        audio, rate = soundfile.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception as exc:  # noqa: BLE001 - libsndfile names the format it cannot read
        raise BadRequest(
            "cannot decode the audio (%s); send wav, flac, ogg or mp3" % exc
        ) from exc
    audio = audio.mean(axis=1)
    seconds = len(audio) / float(rate) if rate else 0.0
    if rate != target_rate and len(audio):
        try:
            import torch
            import torchaudio.functional as taf

            audio = taf.resample(torch.from_numpy(audio), rate, target_rate).numpy()
        except Exception:  # noqa: BLE001 - torchaudio is optional
            count = int(round(len(audio) * target_rate / float(rate)))
            positions = np.linspace(0, len(audio) - 1, num=count)
            audio = np.interp(positions, np.arange(len(audio)), audio).astype("float32")
    return audio, seconds


def jsonable(value):
    """Pipeline output -> JSON: numpy and tensors to lists, images to base64 PNG."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "save") and hasattr(value, "size"):  # a PIL image
        buffer = io.BytesIO()
        value.save(buffer, format="PNG")
        return {"image_png_b64": base64.b64encode(buffer.getvalue()).decode()}
    if hasattr(value, "detach"):  # a torch tensor
        value = value.detach().cpu().numpy()
    if hasattr(value, "tolist"):  # numpy array or scalar
        return jsonable(value.tolist())
    return str(value)


def timestamp(seconds, sep):
    millis = int(round((seconds or 0.0) * 1000))
    hours, millis = divmod(millis, 3600000)
    minutes, millis = divmod(millis, 60000)
    secs, millis = divmod(millis, 1000)
    return "%02d:%02d:%02d%s%03d" % (hours, minutes, secs, sep, millis)


def subtitles(segments, fmt):
    lines = ["WEBVTT", ""] if fmt == "vtt" else []
    sep = "." if fmt == "vtt" else ","
    for index, segment in enumerate(segments, 1):
        if fmt == "srt":
            lines.append(str(index))
        lines.append("%s --> %s" % (timestamp(segment["start"], sep), timestamp(segment["end"], sep)))
        lines.append(segment["text"].strip())
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# backends: one per task. load() runs once, in the loader thread.
# --------------------------------------------------------------------------


class Backend:
    task = ""

    def __init__(self, settings):
        self.settings = settings

    def load(self):
        raise NotImplementedError

    def common(self):
        kwargs = {"trust_remote_code": self.settings.trust_remote_code}
        if self.settings.revision:
            kwargs["revision"] = self.settings.revision
        return kwargs


class ChatBackend(Backend):
    task = "chat"

    def load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        s = self.settings
        self.tokenizer = AutoTokenizer.from_pretrained(s.model, **self.common())
        self.model = AutoModelForCausalLM.from_pretrained(
            s.model, torch_dtype=torch_dtype(s.dtype), device_map="auto", **self.common()
        )

    def chat(self, body):
        import torch

        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise BadRequest("messages is required")
        conversation = [
            {"role": str(m.get("role") or "user"), "content": message_text(m.get("content"))}
            for m in messages
            if isinstance(m, dict)
        ]
        if getattr(self.tokenizer, "chat_template", None):
            prompt = self.tokenizer.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )
        else:
            prompt = "\n".join("%s: %s" % (m["role"], m["content"]) for m in conversation)
            prompt += "\nassistant:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        temperature = float(body.get("temperature", 1.0) or 0.0)
        max_tokens = int(body.get("max_completion_tokens") or body.get("max_tokens") or 512)
        generate = {"max_new_tokens": max_tokens, "do_sample": temperature > 0}
        if temperature > 0:
            generate["temperature"] = temperature
            if body.get("top_p") is not None:
                generate["top_p"] = float(body["top_p"])
        if self.tokenizer.pad_token_id is None:
            generate["pad_token_id"] = self.tokenizer.eos_token_id
        with torch.inference_mode():
            output = self.model.generate(**inputs, **generate)
        prompt_tokens = int(inputs["input_ids"].shape[1])
        completion_ids = output[0][prompt_tokens:]
        text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        stop = body.get("stop")
        for marker in [stop] if isinstance(stop, str) else (stop or []):
            if marker and marker in text:
                text = text.split(marker, 1)[0]
        finish = "length" if len(completion_ids) >= max_tokens else "stop"
        return text, finish, prompt_tokens, int(len(completion_ids))


class EmbeddingsBackend(Backend):
    task = "embeddings"

    def load(self):
        s = self.settings
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            SentenceTransformer = None
        if SentenceTransformer is not None:
            kwargs = {"device": device(), "trust_remote_code": s.trust_remote_code}
            if s.revision:
                kwargs["revision"] = s.revision
            self.st = SentenceTransformer(s.model, **kwargs)
            self.tokenizer = self.st.tokenizer
            return
        from transformers import AutoModel, AutoTokenizer

        self.st = None
        self.tokenizer = AutoTokenizer.from_pretrained(s.model, **self.common())
        self.model = AutoModel.from_pretrained(
            s.model, torch_dtype=torch_dtype(s.dtype), **self.common()
        ).to(device())

    def embed(self, texts):
        import torch

        if self.st is not None:
            vectors = self.st.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
            return [v.astype("float32") for v in vectors]
        batch = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(
            self.model.device
        )
        with torch.inference_mode():
            hidden = self.model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        pooled = torch.nn.functional.normalize(pooled.float(), dim=-1)
        return [row for row in pooled.cpu().numpy()]

    def count_tokens(self, texts):
        try:
            return sum(len(ids) for ids in self.tokenizer(texts)["input_ids"])
        except Exception:  # noqa: BLE001 - usage is informative only
            return sum(len(t.split()) for t in texts)


class ImageBackend(Backend):
    task = "image"

    def load(self):
        from diffusers import AutoPipelineForText2Image, DiffusionPipeline

        s = self.settings
        kwargs = {"torch_dtype": torch_dtype(s.dtype)}
        if s.revision:
            kwargs["revision"] = s.revision
        if s.trust_remote_code:
            kwargs["trust_remote_code"] = True
        try:
            self.pipe = AutoPipelineForText2Image.from_pretrained(s.model, **kwargs)
        except Exception:  # noqa: BLE001 - not every pipeline is in the auto table
            self.pipe = DiffusionPipeline.from_pretrained(s.model, **kwargs)
        if env_flag("JAMBU_CPU_OFFLOAD"):
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(device())
        self.pipe.set_progress_bar_config(disable=True)

    def generate(self, body):
        import torch

        prompt = body.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise BadRequest("prompt is required")
        n = max(1, min(int(body.get("n") or 1), 8))
        call = {"prompt": prompt, "num_images_per_prompt": n}
        size = str(body.get("size") or "auto")
        if size != "auto":
            try:
                width, height = (int(x) for x in size.lower().split("x"))
            except ValueError:
                raise BadRequest("size must look like 1024x1024") from None
            call["width"], call["height"] = width - width % 8, height - height % 8
        for key in ("num_inference_steps", "guidance_scale", "negative_prompt"):
            if body.get(key) is not None:
                call[key] = body[key]
        if body.get("seed") is not None:
            call["generator"] = torch.Generator(device=device()).manual_seed(int(body["seed"]))
        with torch.inference_mode():
            images = self.pipe(**call).images
        fmt = str(body.get("output_format") or "png").lower()
        fmt = {"jpg": "jpeg"}.get(fmt, fmt)
        if fmt not in ("png", "jpeg", "webp"):
            raise BadRequest("output_format must be png, jpeg or webp")
        encoded = []
        for image in images:
            buffer = io.BytesIO()
            image.save(buffer, format=fmt.upper())
            encoded.append((fmt, base64.b64encode(buffer.getvalue()).decode()))
        return encoded


class TranscriptionBackend(Backend):
    task = "transcription"

    def load(self):
        from transformers import pipeline

        s = self.settings
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=s.model,
            torch_dtype=torch_dtype(s.dtype),
            device=device(),
            **self.common(),
        )
        extractor = getattr(self.pipe, "feature_extractor", None)
        self.rate = int(getattr(extractor, "sampling_rate", 16000) or 16000)
        self.is_whisper = "whisper" in type(self.pipe.model).__name__.lower()

    def transcribe(self, data, language=None, translate=False, prompt=None):
        audio, seconds = read_audio(data, self.rate)
        kwargs = {"return_timestamps": True}
        if seconds > 30:
            kwargs["chunk_length_s"] = 30
        generate = {}
        if self.is_whisper:
            if language:
                generate["language"] = language
            generate["task"] = "translate" if translate else "transcribe"
        elif translate:
            raise BadRequest("this model cannot translate; use /v1/audio/transcriptions")
        if generate:
            kwargs["generate_kwargs"] = generate
        try:
            result = self.pipe({"raw": audio, "sampling_rate": self.rate}, **kwargs)
        except ValueError:
            # Models without timestamp support (CTC without an offset table).
            kwargs.pop("return_timestamps", None)
            result = self.pipe({"raw": audio, "sampling_rate": self.rate}, **kwargs)
        segments = []
        for index, chunk in enumerate(result.get("chunks") or []):
            start, end = (list(chunk.get("timestamp") or ()) + [None, None])[:2]
            segments.append(
                {
                    "id": index,
                    "start": float(start or 0.0),
                    "end": float(end if end is not None else seconds),
                    "text": chunk.get("text") or "",
                }
            )
        return (result.get("text") or "").strip(), segments, seconds


class SpeechBackend(Backend):
    task = "speech"

    def load(self):
        from transformers import pipeline

        self.pipe = pipeline(
            "text-to-speech", model=self.settings.model, device=device(), **self.common()
        )

    def speak(self, text):
        import numpy as np

        result = self.pipe(text)
        audio = np.asarray(result["audio"], dtype="float32")
        if audio.ndim > 1:
            audio = audio.reshape(audio.shape[0], -1)[0] if audio.shape[0] <= 2 else audio.reshape(-1)
        return audio, int(result["sampling_rate"])


class PipelineBackend(Backend):
    task = "pipeline"

    def load(self):
        from transformers import pipeline

        s = self.settings
        self.pipe = pipeline(
            task=s.pipeline_tag or None, model=s.model, device=device(), **self.common()
        )

    def run(self, body):
        if "inputs" not in body:
            raise BadRequest('send {"inputs": ..., "parameters": {...}}')
        inputs = self._decode(body["inputs"])
        parameters = body.get("parameters") or {}
        if not isinstance(parameters, dict):
            raise BadRequest("parameters must be an object")
        return jsonable(self.pipe(inputs, **parameters))

    def _decode(self, value):
        if isinstance(value, list):
            return [self._decode(v) for v in value]
        if isinstance(value, dict):
            return {k: self._decode(v) for k, v in value.items()}
        decoded = decode_data_url(value)
        if decoded is None:
            return value
        mime, data = decoded
        if mime.startswith("image/"):
            from PIL import Image

            return Image.open(io.BytesIO(data)).convert("RGB")
        if mime.startswith("audio/"):
            rate = int(getattr(getattr(self.pipe, "feature_extractor", None), "sampling_rate", 16000))
            audio, _ = read_audio(data, rate)
            return {"raw": audio, "sampling_rate": rate}
        return data


BACKENDS = {
    cls.task: cls
    for cls in (
        ChatBackend,
        EmbeddingsBackend,
        ImageBackend,
        TranscriptionBackend,
        SpeechBackend,
        PipelineBackend,
    )
}


# --------------------------------------------------------------------------
# the HTTP server
# --------------------------------------------------------------------------


class Service:
    """The model, its readiness, the request counters and one GPU at a time."""

    def __init__(self, settings, task, backend):
        self.settings = settings
        self.task = task
        self.backend = backend
        self.status = "loading"
        self.error = None
        self.loaded_at = None
        self.gpu = threading.Lock()
        self.counters = threading.Lock()
        self.running = 0
        self.total = 0
        self.errors = 0
        self.seconds = 0.0

    def load(self):
        started = time.time()
        try:
            self.backend.load()
        except Exception as exc:  # noqa: BLE001 - reported, then the process exits
            self.status, self.error = "error", "%s: %s" % (type(exc).__name__, exc)
            raise
        self.status, self.loaded_at = "ready", time.time()
        print(
            "jambu-serve: %s ready (%s) in %.0fs on port %d"
            % (self.settings.model, self.task, time.time() - started, self.settings.port),
            flush=True,
        )

    def metrics(self):
        with self.counters:
            values = (self.running, self.total, self.errors, self.seconds)
        return (
            "# TYPE jambu_requests_running gauge\njambu_requests_running %d\n"
            "# TYPE jambu_requests_total counter\njambu_requests_total %d\n"
            "# TYPE jambu_request_errors_total counter\njambu_request_errors_total %d\n"
            "# TYPE jambu_request_seconds_total counter\njambu_request_seconds_total %.3f\n"
            "# TYPE jambu_model_ready gauge\njambu_model_ready %d\n"
            % (values + (1 if self.status == "ready" else 0,))
        )


class Handler(BaseHTTPRequestHandler):
    server_version = "jambu-serve/1"
    protocol_version = "HTTP/1.1"
    service = None  # set by make_server

    def log_message(self, fmt, *args):  # noqa: A003 - one line per request instead
        return

    # -- plumbing ---------------------------------------------------------

    def _send(self, status, body, content_type="application/json", headers=None):
        if not isinstance(body, bytes):
            body = json.dumps(body).encode() if content_type == "application/json" else str(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status, message, kind="invalid_request_error"):
        self._send(status, {"error": {"message": message, "type": kind, "code": status}})

    def _authorized(self):
        key = self.service.settings.api_key
        if not key:
            return True
        header = self.headers.get("Authorization") or ""
        return header == "Bearer " + key

    def _body(self):
        """The whole body, read before any reply so a kept-alive connection stays in sync."""
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = -1
        if length < 0 or length > MAX_BODY_BYTES:
            self.close_connection = True
            return None
        return self.rfile.read(length) if length > 0 else b""

    def _json(self, raw):
        try:
            body = json.loads(raw or b"{}")
        except ValueError:
            raise BadRequest("the body is not valid JSON") from None
        if not isinstance(body, dict):
            raise BadRequest("the body must be a JSON object")
        return body

    def _form(self, raw):
        """multipart/form-data -> ({field: text}, {field: (filename, bytes)})."""
        ctype = self.headers.get("Content-Type") or ""
        if not ctype.startswith("multipart/form-data"):
            raise BadRequest("send multipart/form-data with a file field")
        message = BytesParser(policy=email_policy).parsebytes(
            b"Content-Type: " + ctype.encode() + b"\r\nMIME-Version: 1.0\r\n\r\n" + raw
        )
        fields, files = {}, {}
        for part in message.iter_parts():
            name = part.get_param("name", header="content-disposition")
            if not name:
                continue
            filename = part.get_filename()
            payload = part.get_payload(decode=True) or b""
            if filename is not None:
                files[name] = (filename, payload)
            else:
                fields[name] = payload.decode("utf-8", "replace")
        return fields, files

    # -- routes -----------------------------------------------------------

    def do_GET(self):  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        service = self.service
        if path == "/health":
            if service.status == "ready":
                self._send(200, {"status": "ok", "task": service.task})
            else:
                self._send(503, {"status": service.status, "error": service.error})
            return
        if path == "/metrics":
            self._send(200, service.metrics(), "text/plain; version=0.0.4")
            return
        if not self._authorized():
            self._error(401, "missing or wrong API key", "authentication_error")
            return
        if path == "/v1/models":
            self._send(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": service.settings.served_name,
                            "object": "model",
                            "created": int(service.loaded_at or time.time()),
                            "owned_by": "gpu4agents",
                            "root": service.settings.model,
                            "task": service.task,
                            "routes": ROUTES[service.task],
                        }
                    ],
                },
            )
            return
        self._error(404, "no route GET %s" % path)

    def do_POST(self):  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/")
        service = self.service
        raw = self._body()
        if raw is None:
            self._error(413, "request body over %d MB" % (MAX_BODY_BYTES // 1024**2))
            return
        if not self._authorized():
            self._error(401, "missing or wrong API key", "authentication_error")
            return
        if path not in ROUTES[service.task]:
            known = [route for routes in ROUTES.values() for route in routes]
            if path in known:
                self._error(
                    404,
                    "%s serves %s (%s), not %s"
                    % (service.settings.served_name, service.task, ", ".join(ROUTES[service.task]), path),
                )
            else:
                self._error(404, "no route POST %s" % path)
            return
        if service.status != "ready":
            self._error(503, "the model is still loading", "server_error")
            return
        started = time.time()
        with service.counters:
            service.running += 1
            service.total += 1
        try:
            with service.gpu:
                getattr(self, "_" + service.task)(path, raw)
        except BadRequest as exc:
            with service.counters:
                service.errors += 1
            self._error(exc.status, str(exc))
        except Exception as exc:  # noqa: BLE001 - the client gets the error, the log the trace
            with service.counters:
                service.errors += 1
            traceback.print_exc()
            self._error(500, "%s: %s" % (type(exc).__name__, exc), "server_error")
        finally:
            with service.counters:
                service.running -= 1
                service.seconds += time.time() - started

    # -- one handler per task -----------------------------------------------

    def _chat(self, path, raw):
        body = self._json(raw)
        text, finish, prompt_tokens, completion_tokens = self.service.backend.chat(body)
        created = int(time.time())
        ident = "chatcmpl-" + uuid.uuid4().hex[:24]
        model = self.service.settings.served_name
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        if body.get("stream"):
            # The whole answer in one chunk: clients that stream still work.
            chunks = [
                {"choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}]},
                {"choices": [{"index": 0, "delta": {}, "finish_reason": finish}], "usage": usage},
            ]
            payload = b"".join(
                b"data: "
                + json.dumps(dict(c, id=ident, object="chat.completion.chunk", created=created, model=model)).encode()
                + b"\n\n"
                for c in chunks
            )
            self._send(200, payload + b"data: [DONE]\n\n", "text/event-stream")
            return
        self._send(
            200,
            {
                "id": ident,
                "object": "chat.completion",
                "created": created,
                "model": model,
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}
                ],
                "usage": usage,
            },
        )

    def _embeddings(self, path, raw):
        body = self._json(raw)
        inputs = body.get("input")
        if isinstance(inputs, str):
            inputs = [inputs]
        if not isinstance(inputs, list) or not inputs or not all(isinstance(x, str) for x in inputs):
            raise BadRequest("input must be a string or a list of strings")
        backend = self.service.backend
        vectors = backend.embed(inputs)
        dims = body.get("dimensions")
        fmt = body.get("encoding_format") or "float"
        data = []
        for index, vector in enumerate(vectors):
            if dims:
                vector = vector[: int(dims)]
            if fmt == "base64":
                values = [float(x) for x in vector]
                embedding = base64.b64encode(struct.pack("<%df" % len(values), *values)).decode()
            else:
                embedding = [float(x) for x in vector]
            data.append({"object": "embedding", "index": index, "embedding": embedding})
        tokens = backend.count_tokens(inputs)
        self._send(
            200,
            {
                "object": "list",
                "data": data,
                "model": self.service.settings.served_name,
                "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
            },
        )

    def _image(self, path, raw):
        body = self._json(raw)
        images = self.service.backend.generate(body)
        as_url = body.get("response_format") == "url"
        data = []
        for fmt, encoded in images:
            if as_url:
                data.append({"url": "data:image/%s;base64,%s" % (fmt, encoded)})
            else:
                data.append({"b64_json": encoded})
        self._send(200, {"created": int(time.time()), "data": data})

    def _transcription(self, path, raw):
        fields, files = self._form(raw)
        if "file" not in files:
            raise BadRequest("the file field is required")
        _, data = files["file"]
        translate = path.endswith("/translations")
        text, segments, seconds = self.service.backend.transcribe(
            data, language=fields.get("language") or None, translate=translate, prompt=fields.get("prompt")
        )
        fmt = fields.get("response_format") or "json"
        if fmt == "text":
            self._send(200, text + "\n", "text/plain; charset=utf-8")
        elif fmt in ("srt", "vtt"):
            if not segments:
                segments = [{"id": 0, "start": 0.0, "end": seconds, "text": text}]
            self._send(200, subtitles(segments, fmt), "text/plain; charset=utf-8")
        elif fmt == "verbose_json":
            self._send(
                200,
                {
                    "task": "translate" if translate else "transcribe",
                    "language": fields.get("language") or None,
                    "duration": round(seconds, 3),
                    "text": text,
                    "segments": segments,
                },
            )
        else:
            self._send(200, {"text": text})

    def _speech(self, path, raw):
        body = self._json(raw)
        text = body.get("input")
        if not isinstance(text, str) or not text.strip():
            raise BadRequest("input is required")
        audio, rate = self.service.backend.speak(text)
        fmt = str(body.get("response_format") or "wav").lower()
        headers = {"X-Sample-Rate": str(rate)}
        if fmt == "pcm":
            self._send(200, pcm16(audio), "audio/pcm", headers)
            return
        if fmt in ("mp3", "flac", "opus", "aac"):
            encoded = self._soundfile(audio, rate, fmt)
            if encoded is not None:
                self._send(200, encoded[1], encoded[0], headers)
                return
            headers["X-Jambu-Format"] = "wav (%s is not available on this machine)" % fmt
        self._send(200, wav_bytes(audio, rate), "audio/wav", headers)

    @staticmethod
    def _soundfile(audio, rate, fmt):
        try:
            import soundfile

            buffer = io.BytesIO()
            if fmt == "mp3":
                soundfile.write(buffer, audio, rate, format="MP3")
                return "audio/mpeg", buffer.getvalue()
            if fmt == "flac":
                soundfile.write(buffer, audio, rate, format="FLAC")
                return "audio/flac", buffer.getvalue()
            if fmt == "opus":
                soundfile.write(buffer, audio, rate, format="OGG", subtype="OPUS")
                return "audio/ogg", buffer.getvalue()
        except Exception:  # noqa: BLE001 - fall back to WAV
            return None
        return None

    def _pipeline(self, path, raw):
        body = self._json(raw)
        result = self.service.backend.run(body)
        self._send(
            200,
            {
                "object": "pipeline.result",
                "model": self.service.settings.served_name,
                "task": self.service.settings.pipeline_tag,
                "output": result,
            },
        )


def make_server(service, host="0.0.0.0"):
    handler = type("BoundHandler", (Handler,), {"service": service})
    server = ThreadingHTTPServer((host, service.settings.port), handler)
    server.daemon_threads = True
    return server


def main():
    settings = Settings()
    if not settings.model:
        raise SystemExit("JAMBU_MODEL_ID is not set")
    task = resolve_task(settings)
    if task == "pipeline" and not settings.pipeline_tag:
        from huggingface_hub import model_info

        settings.pipeline_tag = getattr(model_info(settings.model, revision=settings.revision), "pipeline_tag", None)
    service = Service(settings, task, BACKENDS[task](settings))
    server = make_server(service)
    print(
        "jambu-serve: loading %s as %s (%s), listening on :%d"
        % (settings.model, task, ", ".join(ROUTES[task]), settings.port),
        flush=True,
    )

    def _load():
        try:
            service.load()
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            print("jambu-serve: the model failed to load", file=sys.stderr, flush=True)
            os._exit(1)

    threading.Thread(target=_load, daemon=True).start()
    server.serve_forever()


if __name__ == "__main__":
    main()
