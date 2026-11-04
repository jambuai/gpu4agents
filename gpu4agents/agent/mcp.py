"""`gpu mcp`: a local Model Context Protocol server over stdio.

The client (Claude Code, Claude Desktop, Cursor, ...) starts it as a child
process and talks newline-delimited JSON-RPC on stdin/stdout. Nothing is
hosted. Every tool runs the `gpu` CLI with ``--json`` in the project
directory, so an agent gets exactly the contract an application gets
(docs/INTEGRATION.md): same output, same exit codes, same file lock.

``gpu_setup`` is the one tool that would outlast a client's tool timeout. It
starts setup as a background job (see jobs.py) and waits up to
``wait_seconds``; calling it again re-attaches to the running job instead of
starting another one.

stdout carries the protocol and nothing else: everything a tool prints goes
through a subprocess pipe, and ``sys.stdout`` is pointed at stderr while the
server runs.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

from .. import __version__
from ..core.config import CONFIG_FILENAMES, find_config
from .jobs import Job

PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
SERVER_NAME = "gpu"

DEFAULT_SETUP_WAIT_S = 45
MAX_SETUP_WAIT_S = 300
DEFAULT_RUN_TIMEOUT_S = 600
CLI_TIMEOUT_S = 900
OUTPUT_TAIL_CHARS = 12_000

EXIT_MEANINGS = {
    1: ("unexpected error", "read the error; retry later"),
    2: ("config error", "fix gpu.yaml; gpu_setup(recreate=true) replaces a machine built from an old config"),
    3: ("no instance or no endpoint", "call gpu_setup"),
    4: ("provider API error", "retry with backoff"),
    5: ("provider authentication failed, or no key", "the user runs `gpu login` in their own terminal"),
    6: ("no capacity matches the config", "retry later, or relax compute.gpu / the budget in gpu.yaml"),
    7: ("provisioning failed", "retry; the failed machine was already destroyed"),
    8: ("provider timed out", "retry with backoff"),
    9: ("the provider cannot do this", "change the config"),
    10: ("over budget.max_hourly_cost_usd", "ask the user before raising the budget in gpu.yaml"),
    11: ("the model server did not start", "read stderr (the server's log tail); the machine was destroyed"),
    12: ("another gpu command holds this project's lock", "usually a setup: follow it with gpu_setup, or cancel it with gpu_stop"),
}

INSTRUCTIONS = """\
gpu rents a cloud GPU (Vast.ai built in; other providers through an adapter),
serves a model on it behind an OpenAI-compatible API, and stops it by itself
when nobody uses it. Any Hugging Face model: chat models on vLLM; image,
speech, transcription, embeddings and other pipeline models on gpu's own
server (gpu_init picks from the model card; the result says which routes).

Rules:
- Renting costs the user money. Before the first gpu_setup in a conversation,
  tell the user the model, the price cap (gpu_offers shows prices) and ask.
- gpu_setup takes 5-15 minutes the first time. It returns status "running"
  while it works: call gpu_setup again to keep waiting. Never start a new
  project or recreate because it is slow.
- A stopped machine restarts in about a minute with the same key, but
  base_url may change: always take it from the latest gpu_setup/gpu_connection.
- When the work is done, offer gpu_stop (keeps the disk, cheap) or
  gpu_destroy (nothing billed). gpu_destroy needs confirm=true and the user's
  explicit yes. Finish by checking gpu_ls for anything still running.
- Never print or ask for the provider's API key; the user runs `gpu login`
  in a terminal. The endpoint api_key is per machine.
- To run your own code on the GPU (training, image or audio models, anything
  that needs CUDA), write it in a directory following
  https://gpu4agents.jambu.ai/jobs.md, then gpu_setup, gpu_job_run,
  gpu_job_status until it ends, and gpu_job_pull for the outputs.
- To serve a model the built-in servers cannot load, write your own server
  following https://gpu4agents.jambu.ai/serving.md (gpu_init with
  engine="custom" writes a starting one), then gpu_setup.
"""


class ToolError(Exception):
    """A tool failed in a way the agent can act on (reported with isError)."""

    def __init__(self, payload: dict[str, Any]) -> None:
        super().__init__(payload.get("error", "error"))
        self.payload = payload


@dataclass
class CliResult:
    returncode: int
    stdout: str
    stderr: str
    data: Any = None

    def error(self) -> dict[str, Any]:
        lines = [line for line in self.stderr.splitlines() if line.strip()]
        message = next(
            (line.split("error:", 1)[1].strip() for line in reversed(lines) if "error:" in line),
            lines[-1] if lines else f"gpu exited with code {self.returncode}",
        )
        meaning, hint = EXIT_MEANINGS.get(self.returncode, ("error", "read the error"))
        payload: dict[str, Any] = {
            "error": message,
            "exit_code": self.returncode,
            "meaning": meaning,
            "next": hint,
        }
        if len(lines) > 1:
            payload["stderr_tail"] = lines[-30:]
        if isinstance(self.data, dict):
            payload["output"] = self.data
        return payload


def _tail_text(text: str, limit: int = OUTPUT_TAIL_CHARS) -> str:
    return text if len(text) <= limit else "...[truncated]...\n" + text[-limit:]


def _schema(properties: dict[str, Any], required: Optional[list[str]] = None) -> dict[str, Any]:
    props = {
        "project_dir": {
            "type": "string",
            "description": "Directory holding gpu.yaml (or below it). Defaults to the server's project.",
        },
        **properties,
    }
    schema: dict[str, Any] = {"type": "object", "properties": props, "additionalProperties": False}
    if required:
        schema["required"] = required
    return schema


@dataclass
class Tool:
    name: str
    title: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], Any]
    annotations: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "inputSchema": self.input_schema,
            "annotations": {"title": self.title, **self.annotations},
        }


class Server:
    def __init__(self, project: Optional[Path] = None, python: Optional[str] = None) -> None:
        env_project = os.environ.get("GPU_PROJECT_DIR")
        self.project = Path(project or env_project or os.getcwd()).expanduser()
        self.python = python or sys.executable
        self.jobs: dict[Path, Job] = {}
        self.tools = {tool.name: tool for tool in self._tools()}

    # -- plumbing ------------------------------------------------------

    def _start_dir(self, args: dict[str, Any]) -> Path:
        raw = args.get("project_dir") or self.project
        return Path(raw).expanduser().resolve()

    def _project_dir(self, args: dict[str, Any]) -> Path:
        start = self._start_dir(args)
        config = find_config(start)
        if config is None:
            raise ToolError(
                {
                    "error": f"no gpu.yaml in {start} or any parent",
                    "exit_code": 2,
                    "next": "call gpu_init with a model in the project directory, or pass project_dir",
                }
            )
        return config.parent

    def _argv(self, args: list[str], json_output: bool = True) -> list[str]:
        return [self.python, "-m", "gpu4agents", *(["--json"] if json_output else []), *args]

    def cli(self, args: list[str], cwd: Path, timeout: float = CLI_TIMEOUT_S) -> CliResult:
        try:
            proc = subprocess.run(
                self._argv(args),
                cwd=str(cwd),
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise ToolError(
                {"error": f"gpu {args[0]} did not finish within {int(timeout)}s", "exit_code": 8}
            ) from exc
        result = CliResult(proc.returncode, proc.stdout, proc.stderr)
        try:
            result.data = json.loads(proc.stdout) if proc.stdout.strip() else None
        except ValueError:
            result.data = None
        return result

    def _ok(self, result: CliResult) -> Any:
        if result.returncode != 0:
            raise ToolError(result.error())
        return result.data

    def _job(self, project_dir: Path) -> Job:
        if project_dir not in self.jobs:
            self.jobs[project_dir] = Job(project_dir)
        return self.jobs[project_dir]

    # -- tools ---------------------------------------------------------

    def tool_init(self, args: dict[str, Any]) -> Any:
        directory = self._start_dir(args)
        directory.mkdir(parents=True, exist_ok=True)
        argv = ["init", "--model", str(args["model"])]
        for key in ("engine", "task"):
            if args.get(key):
                argv += [f"--{key}", str(args[key])]
        if args.get("force"):
            argv.append("--force")
        self._ok(self.cli(argv, directory))
        config = directory / CONFIG_FILENAMES[0]
        text = config.read_text()
        if args.get("max_hourly_cost_usd") is not None:
            text = re.sub(
                r"(max_hourly_cost_usd:\s*)[0-9.]+", rf"\g<1>{float(args['max_hourly_cost_usd']):.2f}", text
            )
        if args.get("min_vram_gb") is not None:
            text = re.sub(r"(min_vram_gb:\s*)[0-9]+", rf"\g<1>{int(args['min_vram_gb'])}", text)
        config.write_text(text)
        written = (yaml.safe_load(text) or {}).get("gpu_runtime") or {}
        engine = (written.get("runtime") or {}).get("engine")
        result = {
            "config_path": str(config),
            "model": args["model"],
            "engine": engine,
            "task": (written.get("model") or {}).get("task") or ("chat" if engine == "vllm" else None),
            "next": "gpu_validate, then gpu_offers to see prices",
        }
        if engine == "custom":
            service_dir = (written.get("runtime") or {}).get("service_dir") or "server"
            result["server_dir"] = str(directory / service_dir)
            result["next"] = (
                f"write the model into {service_dir}/server.py following "
                "https://gpu4agents.jambu.ai/serving.md, then gpu_validate and gpu_offers"
            )
        return result

    def tool_validate(self, args: dict[str, Any]) -> Any:
        return self._ok(self.cli(["validate"], self._project_dir(args)))

    def tool_offers(self, args: dict[str, Any]) -> Any:
        limit = int(args.get("limit") or 5)
        return self._ok(self.cli(["offers", "--limit", str(limit)], self._project_dir(args)))

    def tool_setup(self, args: dict[str, Any]) -> Any:
        project_dir = self._project_dir(args)
        job = self._job(project_dir)
        wait = min(float(args.get("wait_seconds", DEFAULT_SETUP_WAIT_S)), MAX_SETUP_WAIT_S)
        started = False
        if not job.running():
            argv = ["setup"] + (["--recreate"] if args.get("recreate") else [])
            job.start(self._argv(argv), project_dir)
            started = True
        job.wait(wait)
        report = job.report()
        report["started_now"] = started
        status = report["status"]
        if status == "running":
            report["next"] = (
                "still provisioning (a first setup takes 5-15 min); call gpu_setup again to keep "
                "waiting, or gpu_status / gpu_logs(source='onstart') to watch"
            )
            return report
        if status == "succeeded":
            connection = report.pop("result", None) or {}
            connection.pop("watchdog", None)
            return {
                "status": "ready",
                "elapsed_s": report.get("elapsed_s"),
                **connection,
                "next": "use base_url, api_key and model with any OpenAI client, or gpu_run",
            }
        code = int(report.get("exit_code") or 1)
        stderr = "\n".join(report.get("stderr_tail") or [])
        error = CliResult(code, "", stderr).error()
        error["setup"] = {k: report.get(k) for k in ("status", "elapsed_s", "pid")}
        if status == "lost":
            error["error"] = "the setup process died without reporting; call gpu_status"
        raise ToolError(error)

    def tool_status(self, args: dict[str, Any]) -> Any:
        project_dir = self._project_dir(args)
        payload = self._ok(self.cli(["status"], project_dir))
        job = self._job(project_dir).report()
        if job["status"] != "none":
            job.pop("result", None)
            payload["setup_job"] = job
        return payload

    def tool_connection(self, args: dict[str, Any]) -> Any:
        return self._ok(self.cli(["env"], self._project_dir(args)))

    def tool_run(self, args: dict[str, Any]) -> Any:
        project_dir = self._project_dir(args)
        command = [str(part) for part in args.get("command") or []]
        if not command:
            raise ToolError({"error": "command is empty", "exit_code": 2})
        timeout = float(args.get("timeout_seconds") or DEFAULT_RUN_TIMEOUT_S)
        argv = self._argv(["run", "--label", str(args.get("label") or "mcp"), "--", *command], json_output=False)
        kwargs: dict[str, Any] = {}
        if os.name != "nt":
            kwargs["start_new_session"] = True
        proc = subprocess.Popen(
            argv,
            cwd=str(project_dir),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            **kwargs,
        )
        timed_out = False
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            # SIGINT first: gpu run ends the workload record cleanly on Ctrl-C.
            if os.name != "nt":
                os.killpg(proc.pid, signal.SIGINT)
            else:
                proc.terminate()
            try:
                stdout, stderr = proc.communicate(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()
        result = {
            "exit_code": proc.returncode,
            "timed_out": timed_out,
            "stdout": _tail_text(stdout),
            "stderr": _tail_text(stderr),
        }
        if timed_out:
            result["next"] = "the command was interrupted; raise timeout_seconds or run it from a shell with `gpu run`"
        return result

    def tool_logs(self, args: dict[str, Any]) -> Any:
        argv = ["logs", "--lines", str(int(args.get("lines") or 100))]
        if args.get("source"):
            argv += ["--source", str(args["source"])]
        return self._ok(self.cli(argv, self._project_dir(args)))

    def tool_lock(self, args: dict[str, Any]) -> Any:
        argv = ["lock"]
        if args.get("duration"):
            argv += ["--for", str(args["duration"])]
        if args.get("reason"):
            argv += ["--reason", str(args["reason"])]
        return self._ok(self.cli(argv, self._project_dir(args)))

    def tool_unlock(self, args: dict[str, Any]) -> Any:
        return self._ok(self.cli(["unlock"], self._project_dir(args)))

    def _cancel_setup(self, project_dir: Path) -> Optional[dict[str, Any]]:
        """Interrupt a running setup (it destroys the machine it was booting)."""
        job = self._job(project_dir)
        if not job.running():
            return None
        ended = job.cancel()
        report = job.report()
        if not ended:
            raise ToolError(
                {
                    "error": "the running setup did not stop within 2 minutes",
                    "exit_code": 12,
                    "next": "call gpu_status, then try again",
                    "setup": report,
                }
            )
        return {"status": "cancelled", "elapsed_s": report.get("elapsed_s")}

    def tool_stop(self, args: dict[str, Any]) -> Any:
        project_dir = self._project_dir(args)
        cancelled = self._cancel_setup(project_dir)
        result = self._ok(self.cli(["stop", "--reason", str(args.get("reason") or "agent")], project_dir))
        if cancelled:
            result["setup_cancelled"] = cancelled
        return result

    def tool_destroy(self, args: dict[str, Any]) -> Any:
        if args.get("confirm") is not True:
            raise ToolError(
                {
                    "error": "gpu_destroy needs confirm=true",
                    "exit_code": 2,
                    "next": "ask the user to confirm deleting the machine, then call again with confirm=true",
                }
            )
        project_dir = self._project_dir(args)
        cancelled = self._cancel_setup(project_dir)
        argv = ["destroy", "--yes", "--reason", str(args.get("reason") or "agent")]
        result = self._ok(self.cli(argv, project_dir))
        if cancelled:
            result["setup_cancelled"] = cancelled
        return result

    def tool_job_run(self, args: dict[str, Any]) -> Any:
        project_dir = self._project_dir(args)
        argv = ["job", "run", "--detach", "--no-start"]
        if args.get("dir"):
            argv += ["--dir", str(Path(str(args["dir"])).expanduser())]
        for key in ("setup", "timeout", "label"):
            if args.get(key):
                argv += [f"--{key}", str(args[key])]
        for key, value in (args.get("env") or {}).items():
            argv += ["--env", f"{key}={value}"]
        if args.get("command"):
            argv += ["--", str(args["command"])]
        result = self.cli(argv, project_dir)
        if result.returncode == 3:
            error = result.error()
            error["next"] = "start the machine first with gpu_setup (it costs money: ask the user), then call gpu_job_run again"
            raise ToolError(error)
        data = self._ok(result)
        data["next"] = "call gpu_job_status(job_id) to follow it; gpu_job_pull(job_id) once it ends"
        return data

    def tool_job_status(self, args: dict[str, Any]) -> Any:
        project_dir = self._project_dir(args)
        job_id = args.get("job_id")
        if not job_id:
            return self._ok(self.cli(["job", "ls"], project_dir))
        info = self._ok(self.cli(["job", "status", str(job_id)], project_dir))
        logs = self.cli(["job", "logs", str(job_id)], project_dir)
        if logs.returncode == 0 and isinstance(logs.data, dict):
            info["log_tail"] = _tail_text(logs.data.get("data") or "", int(args.get("log_chars") or OUTPUT_TAIL_CHARS))
        if info.get("state") == "running":
            info["next"] = "still running; call gpu_job_status again later (the machine stays up while it runs)"
        else:
            info["next"] = "gpu_job_pull(job_id) downloads the outputs; then offer gpu_stop"
        return info

    def tool_job_pull(self, args: dict[str, Any]) -> Any:
        argv = ["job", "pull", str(args["job_id"])]
        if args.get("dest"):
            argv += ["--out", str(Path(str(args["dest"])).expanduser())]
        return self._ok(self.cli(argv, self._project_dir(args)))

    def tool_job_cancel(self, args: dict[str, Any]) -> Any:
        return self._ok(self.cli(["job", "cancel", str(args["job_id"])], self._project_dir(args)))

    def tool_ls(self, args: dict[str, Any]) -> Any:
        start = self._start_dir(args)
        config = find_config(start)
        cwd = config.parent if config else (start if start.is_dir() else Path.home())
        return self._ok(self.cli(["ls"], cwd))

    def _tools(self) -> list[Tool]:
        read_only = {"readOnlyHint": True, "openWorldHint": True}
        return [
            Tool(
                "gpu_init",
                "Create gpu.yaml",
                "Write a starter gpu.yaml for a HuggingFace model in project_dir. Use once per "
                "project, before anything else. Any model: the model card picks the server (chat on "
                "vLLM; image, speech, transcription, embeddings or other pipelines on gpu's hf "
                "server) and the result says which. Defaults: 24 GB VRAM (or sized from the model), "
                "$1.00/h cap, stop after 15 min idle.",
                _schema(
                    {
                        "model": {"type": "string", "description": "HuggingFace model id, e.g. Qwen/Qwen2.5-7B-Instruct or openai/whisper-large-v3"},
                        "engine": {
                            "type": "string",
                            "enum": ["auto", "vllm", "hf", "custom", "none"],
                            "description": "auto (default) reads the model card. custom: a server you write (serving.md).",
                        },
                        "task": {
                            "type": "string",
                            "enum": ["chat", "embeddings", "image", "transcription", "speech", "pipeline"],
                            "description": "With engine hf: override the task the model card gives.",
                        },
                        "max_hourly_cost_usd": {"type": "number", "description": "Price cap per hour."},
                        "min_vram_gb": {"type": "integer", "description": "Minimum GPU memory."},
                        "force": {"type": "boolean", "description": "Overwrite an existing gpu.yaml."},
                    },
                    ["model"],
                ),
                self.tool_init,
                {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False},
            ),
            Tool(
                "gpu_validate",
                "Validate config",
                "Check gpu.yaml and the provider credentials without renting anything.",
                _schema({}),
                self.tool_validate,
                read_only,
            ),
            Tool(
                "gpu_offers",
                "Show GPU offers",
                "Cheapest machines that match gpu.yaml, with their hourly price. Nothing is rented.",
                _schema({"limit": {"type": "integer", "minimum": 1, "maximum": 50}}),
                self.tool_offers,
                read_only,
            ),
            Tool(
                "gpu_setup",
                "Start the GPU",
                "Make the model endpoint ready: rent a machine if there is none, restart it if stopped, "
                "wait until the model answers. COSTS MONEY: confirm with the user first. Returns "
                "status 'running' while it works (first time 5-15 min): call again to keep waiting; "
                "it never starts a second machine. On 'ready' returns base_url, api_key, model.",
                _schema(
                    {
                        "wait_seconds": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": MAX_SETUP_WAIT_S,
                            "description": f"How long to block before reporting progress (default {DEFAULT_SETUP_WAIT_S}).",
                        },
                        "recreate": {
                            "type": "boolean",
                            "description": "Replace a machine built from an older gpu.yaml (exit code 2 tells you when).",
                        },
                    }
                ),
                self.tool_setup,
                {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
            ),
            Tool(
                "gpu_status",
                "GPU status",
                "State, endpoint, cost so far, idle time and what the watchdog will do next, plus "
                "the progress of a background gpu_setup.",
                _schema({}),
                self.tool_status,
                read_only,
            ),
            Tool(
                "gpu_connection",
                "Endpoint details",
                "base_url, api_key and model for an OpenAI-compatible client, without starting anything.",
                _schema({}),
                self.tool_connection,
                read_only,
            ),
            Tool(
                "gpu_run",
                "Run a command against the GPU",
                "Run a local command with OPENAI_BASE_URL, OPENAI_API_KEY and JAMBU_MODEL_ID set, "
                "keeping the GPU alive while it runs (starts it first if needed, which costs money). "
                "Returns exit code and output tails.",
                _schema(
                    {
                        "command": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                            "description": "argv, e.g. [\"python\", \"eval.py\", \"--n\", \"50\"]",
                        },
                        "timeout_seconds": {"type": "number", "minimum": 1},
                        "label": {"type": "string"},
                    },
                    ["command"],
                ),
                self.tool_run,
                {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
            ),
            Tool(
                "gpu_job_run",
                "Run code on the GPU",
                "Upload a directory of code to the running GPU machine and start a shell command "
                "there (training, image/audio/video models, anything needing CUDA). Returns job_id "
                "at once; the job keeps the machine up while it runs. Files it writes to "
                "$JAMBU_OUTPUT_DIR come back with gpu_job_pull. Needs a machine started with "
                "gpu_setup. Contract: https://gpu4agents.jambu.ai/jobs.md",
                _schema(
                    {
                        "dir": {"type": "string", "description": "Directory to upload (default: project_dir). gpujob.yaml there may declare run/setup/env/timeout."},
                        "command": {"type": "string", "description": "Shell command, e.g. \"python train.py --epochs 3\". Default: run: in gpujob.yaml."},
                        "setup": {"type": "string", "description": "Shell command run first, e.g. \"pip install -r requirements.txt\"."},
                        "env": {"type": "object", "additionalProperties": {"type": "string"}},
                        "timeout": {"type": "string", "description": "Kill the job after this long, e.g. 2h."},
                        "label": {"type": "string"},
                    }
                ),
                self.tool_job_run,
                {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
            ),
            Tool(
                "gpu_job_status",
                "Job status and log",
                "State (running, succeeded, failed, timeout, cancelled), exit code, run time, cost, "
                "output files and the tail of the log of one job. Without job_id, lists this project's jobs.",
                _schema(
                    {
                        "job_id": {"type": "string"},
                        "log_chars": {"type": "integer", "minimum": 100, "maximum": 100000},
                    }
                ),
                self.tool_job_status,
                read_only,
            ),
            Tool(
                "gpu_job_pull",
                "Download job outputs",
                "Download what a job wrote to $JAMBU_OUTPUT_DIR into gpu-outputs/<job_id>/ (or dest). "
                "Works while the job still runs. Do it before gpu_destroy: the files live on the machine's disk.",
                _schema({"job_id": {"type": "string"}, "dest": {"type": "string"}}, ["job_id"]),
                self.tool_job_pull,
                {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
            ),
            Tool(
                "gpu_job_cancel",
                "Cancel a job",
                "Stop a running job (TERM, then KILL after 10 s). Outputs written so far stay.",
                _schema({"job_id": {"type": "string"}}, ["job_id"]),
                self.tool_job_cancel,
                {"readOnlyHint": False, "destructiveHint": True, "idempotentHint": True, "openWorldHint": True},
            ),
            Tool(
                "gpu_logs",
                "GPU logs",
                "Last lines of a log on the machine: runtime (vLLM), onstart (boot), watchdog or provider.",
                _schema(
                    {
                        "source": {"type": "string", "enum": ["runtime", "onstart", "watchdog", "provider"]},
                        "lines": {"type": "integer", "minimum": 1, "maximum": 2000},
                    }
                ),
                self.tool_logs,
                read_only,
            ),
            Tool(
                "gpu_lock",
                "Keep the GPU up",
                "Keep the machine running through quiet periods (a long job, a demo). Requests "
                "already count as activity; lock only for silences longer than the idle timeout.",
                _schema(
                    {
                        "duration": {"type": "string", "description": "e.g. 2h; default lifecycle.lock.default_ttl"},
                        "reason": {"type": "string"},
                    }
                ),
                self.tool_lock,
                {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
            ),
            Tool(
                "gpu_unlock",
                "Release the lock",
                "Return to the normal idle and lifetime rules.",
                _schema({}),
                self.tool_unlock,
                {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
            ),
            Tool(
                "gpu_stop",
                "Stop the GPU",
                "Stop the machine now. Compute billing ends; the disk and model cache are kept, so "
                "the next gpu_setup restarts it in about a minute. Cancels a setup in progress "
                "(a machine still booting is destroyed).",
                _schema({"reason": {"type": "string"}}),
                self.tool_stop,
                {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
            ),
            Tool(
                "gpu_destroy",
                "Delete the GPU",
                "Delete the machine and its disk: nothing is billed afterwards, and the next "
                "gpu_setup rents and downloads from scratch. Cancels a setup in progress. Requires "
                "confirm=true after the user agrees.",
                _schema(
                    {"confirm": {"type": "boolean"}, "reason": {"type": "string"}},
                    ["confirm"],
                ),
                self.tool_destroy,
                {"readOnlyHint": False, "destructiveHint": True, "idempotentHint": True, "openWorldHint": True},
            ),
            Tool(
                "gpu_ls",
                "List all GPUs",
                "Every machine gpu created in the provider account, from any project, with state and "
                "price. Use it to make sure nothing is left billing.",
                _schema({}),
                self.tool_ls,
                read_only,
            ),
        ]

    # -- JSON-RPC ------------------------------------------------------

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        tool = self.tools.get(name)
        if tool is None:
            raise KeyError(name)
        try:
            payload = tool.handler(arguments or {})
            is_error = False
        except ToolError as exc:
            payload, is_error = exc.payload, True
        except Exception as exc:  # a tool bug must not kill the server
            payload, is_error = {"error": f"{type(exc).__name__}: {exc}", "exit_code": 1}, True
        text = json.dumps(payload, indent=2, default=str)
        return {"content": [{"type": "text", "text": text}], "isError": is_error}

    def handle(self, message: dict[str, Any]) -> Optional[dict[str, Any]]:
        """One request in, one response out (None for notifications)."""
        method = message.get("method")
        msg_id = message.get("id")
        if msg_id is None:
            return None  # notifications/initialized, cancelled, ...
        params = message.get("params") or {}
        try:
            if method == "initialize":
                requested = params.get("protocolVersion")
                version = requested if requested in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0]
                result: Any = {
                    "protocolVersion": version,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": SERVER_NAME, "title": "gpu (GPU for Agents)", "version": __version__},
                    "instructions": INSTRUCTIONS,
                }
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": [tool.describe() for tool in self.tools.values()]}
            elif method == "tools/call":
                try:
                    result = self.call_tool(params.get("name", ""), params.get("arguments") or {})
                except KeyError:
                    return _error(msg_id, -32602, f"unknown tool: {params.get('name')}")
            else:
                return _error(msg_id, -32601, f"method not found: {method}")
        except Exception as exc:
            return _error(msg_id, -32603, f"{type(exc).__name__}: {exc}")
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    def serve(self, stdin=None, stdout=None) -> None:
        stdin = stdin or sys.stdin
        out = stdout or sys.stdout
        original = sys.stdout
        sys.stdout = sys.stderr  # a stray print must never corrupt the protocol
        try:
            for line in stdin:
                line = line.strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except ValueError:
                    response: Optional[dict[str, Any]] = _error(None, -32700, "parse error")
                else:
                    if isinstance(message, list):  # batches (2025-03-26)
                        replies = [r for r in (self.handle(m) for m in message) if r is not None]
                        response = replies or None  # type: ignore[assignment]
                    elif isinstance(message, dict):
                        response = self.handle(message)
                    else:
                        response = _error(None, -32600, "invalid request")
                if response is not None:
                    out.write(json.dumps(response) + "\n")
                    out.flush()
        finally:
            sys.stdout = original


def _error(msg_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}
