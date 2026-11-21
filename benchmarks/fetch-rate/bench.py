"""Weight-fetch benchmark: the fetcher against the downloads the servers do today.

Cases, each on an empty directory, deleted afterwards:
  gguf/fetcher-16      26.6 GB GGUF, our fetcher, 16 connections
  gguf/fetcher-32      the same, 32 connections
  gguf/single-stream   one connection, what llama.cpp's -hf does (capped at 180 s)
  st/fetcher-16        55.6 GB of safetensors, our fetcher
  st/hf-default        huggingface_hub.snapshot_download as vLLM calls it
  st/hf-xet-high       the same with HF_XET_HIGH_PERFORMANCE=1

Results go to $JAMBU_OUTPUT_DIR/results.json, one line per case as it ends.
"""

import json
import os
import shutil
import subprocess
import sys
import time

import fetch_ext

OUT = os.environ.get("JAMBU_OUTPUT_DIR", "out")
WORK = "/workspace/bench"
ST = "xtools-at/Qwensploit-V2-27B-CyberSecurity-Qwen3.8"
GGUF = "peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-GGUF"
GGUF_QUANT = "Q5_K_XL"
results = []


def _hub_python():
    """The interpreter vLLM runs under (the image's PATH python is a shim)."""
    for exe in ("/usr/bin/python3", "python3", sys.executable):
        if subprocess.run([exe, "-c", "import huggingface_hub"], capture_output=True).returncode == 0:
            return exe
    return sys.executable


HUB_PY = _hub_python()


def record(case, **fields):
    row = dict(case=case, **fields)
    results.append(row)
    print(json.dumps(row), flush=True)
    with open(os.path.join(OUT, "results.json"), "w") as handle:
        json.dump(results, handle, indent=1)


def fresh(name):
    path = os.path.join(WORK, name)
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path)
    return path


def fetcher(case, repo, engine, quant, workers):
    dest = fresh(case.replace("/", "_"))
    found = fetch_ext.fetch_plan(repo, "main", engine, quant)
    started = time.time()
    fetch_ext.fetch_download(repo, "main", found["files"], dest, workers=workers)
    seconds = time.time() - started
    record(case, bytes=found["bytes_total"], seconds=round(seconds, 1),
           mb_per_s=round(found["bytes_total"] / 1e6 / seconds, 1), workers=workers)
    shutil.rmtree(dest, ignore_errors=True)
    return found


def single_stream(case, repo, path, cap_s=180):
    response, _ = fetch_ext._fetch_open(fetch_ext.fetch_resolve_url(repo, "main", path))
    started = time.time()
    got = 0
    with response:
        while time.time() - started < cap_s:
            block = response.read(1 << 20)
            if not block:
                break
            got += len(block)
    seconds = time.time() - started
    record(case, bytes=got, seconds=round(seconds, 1), mb_per_s=round(got / 1e6 / seconds, 1),
           capped=seconds >= cap_s)


def hf_snapshot(case, repo, high_performance):
    dest = fresh(case.replace("/", "_"))
    env = dict(os.environ, HF_HOME=os.path.join(dest, "hf"))
    env.pop("HF_XET_HIGH_PERFORMANCE", None)
    if high_performance:
        env["HF_XET_HIGH_PERFORMANCE"] = "1"
    code = (
        "import huggingface_hub as h;"
        f"h.snapshot_download({repo!r}, allow_patterns=['*.safetensors','*.json','*.txt','*.model'])"
    )
    started = time.time()
    done = subprocess.run([HUB_PY, "-c", code], env=env)
    seconds = time.time() - started
    size = sum(
        os.path.getsize(os.path.join(root, f))
        for root, _, files in os.walk(os.path.join(dest, "hf"))
        for f in files
        if not os.path.islink(os.path.join(root, f))
    )
    record(case, bytes=size, seconds=round(seconds, 1), mb_per_s=round(size / 1e6 / seconds, 1),
           exit_code=done.returncode)
    shutil.rmtree(dest, ignore_errors=True)


def host():
    versions = subprocess.run(
        [HUB_PY, "-c",
         "import importlib.metadata as m\n"
         "for p in ('huggingface_hub','hf_xet','hf_transfer'):\n"
         "    try: print(p, m.version(p))\n"
         "    except Exception: print(p, 'absent')"],
        capture_output=True, text=True,
    ).stdout.split("\n")
    record("host", python=HUB_PY, cpus=os.cpu_count(), free_gb=round(shutil.disk_usage("/workspace").free / 1e9),
           packages=[v for v in versions if v])


def main():
    os.makedirs(OUT, exist_ok=True)
    host()
    only = set(filter(None, os.environ.get("BENCH_CASES", "").split(",")))

    def wanted(case):
        return not only or case in only

    if wanted("gguf/fetcher-16"):
        fetcher("gguf/fetcher-16", GGUF, "llama", GGUF_QUANT, 16)
    if wanted("gguf/fetcher-32"):
        fetcher("gguf/fetcher-32", GGUF, "llama", GGUF_QUANT, 32)
    if wanted("gguf/fetcher-64"):
        fetcher("gguf/fetcher-64", GGUF, "llama", GGUF_QUANT, 64)
    if wanted("gguf/single-stream"):
        found = fetch_ext.fetch_plan(GGUF, "main", "llama", GGUF_QUANT)
        single_stream("gguf/single-stream", GGUF, found["files"][0][0])
    if wanted("st/fetcher-16"):
        fetcher("st/fetcher-16", ST, "vllm", None, 16)
    if wanted("st/fetcher-32"):
        fetcher("st/fetcher-32", ST, "vllm", None, 32)
    if wanted("st/fetcher-64"):
        fetcher("st/fetcher-64", ST, "vllm", None, 64)
    if wanted("st/hf-default"):
        hf_snapshot("st/hf-default", ST, False)
    if wanted("st/hf-xet-high"):
        hf_snapshot("st/hf-xet-high", ST, True)
    shutil.rmtree(WORK, ignore_errors=True)


if __name__ == "__main__":
    main()
