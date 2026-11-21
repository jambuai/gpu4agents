"""Weight fetcher: an extension the CLI uploads to the watchdog (``POST /extend``).

vLLM and llama.cpp download a model's weights themselves when they start. Both
are slow at it on a rented machine: on Vast we measured 75-176 MB/s for vLLM
(upper bound, load included) and 20-67 MB/s for llama.cpp's ``-hf``, on hosts
that advertise several Gbps. This fetcher downloads the same files first, with
many HTTP range requests in parallel, and hands the server a local path.

How it meets the model server:

- The boot script creates the marker ``/workspace/.jambu-fetch`` (empty) before
  it starts the launcher. The launcher waits while the marker exists: up to
  120 s for this extension to arrive, and for as long as the marker holds
  "fetching", which this extension writes when it starts.
- When the files are complete this extension writes the local model reference
  to ``<marker>.done`` and removes the marker. The launcher passes that path to
  the server (vLLM ``--model``, llama.cpp ``-m``).
- When it cannot fetch (a repo layout it does not handle, no space, a network
  that keeps failing) it removes the marker without writing ``.done``, and the
  server downloads its own weights as before. Nothing here can stop a model
  that would have started without it.

The plan is saved next to the marker, so a watchdog that respawns (or a machine
that restarts from a stop) resumes or confirms the fetch on load, without the
CLI.

Routes (behind X-Jambu-Token):
    POST /fetch  -> {repo, revision, engine, quant}: start (idempotent)
    GET  /fetch  -> status: state, bytes_done, bytes_total, rate_mbps, ...

Standalone (the benchmark, or a quick check on any machine):
    python3 fetch_ext.py REPO DEST [--engine vllm|llama] [--quant Q]
        [--revision R] [--workers N] [--chunk-mb M]

Standard library only, like the rest of the watchdog.
"""

import json
import os
import queue
import re
import shutil
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

# Loaded with exec() into the watchdog's globals. Imported or run on its own
# (tests, the benchmark), it brings its own minimal versions of those names.
if "log_event" not in globals():  # pragma: no cover - only outside the watchdog
    try:
        from gpu4agents.watchdog.agent_body import EXT_ROUTES, log_event  # noqa: F401
    except ImportError:
        EXT_ROUTES = {}

        def log_event(event, **fields):
            print(json.dumps(dict(fields, event=event)), flush=True)


FETCH_MARKER = os.environ.get("JAMBU_FETCH_MARKER", "/workspace/.jambu-fetch")
FETCH_ROOT = os.environ.get("JAMBU_FETCH_ROOT", "/workspace/models")
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")

# 32 connections: 446 MB/s on an 8 Gbps host, where 64 gave 366 and 16 less.
FETCH_WORKERS = 32
FETCH_CHUNK = 64 << 20
FETCH_RETRIES = 6
FETCH_RATE_WINDOW_S = 120.0
# Files that are not weights but are big are almost never needed to serve
# (sample images, datasets); a tokenizer is tens of MB.
FETCH_SIDE_FILE_MAX = 512 << 20
FETCH_DISK_MARGIN = 2 << 30

_FETCH_WEIGHT_SUFFIXES = (
    ".safetensors", ".bin", ".pt", ".pth", ".gguf", ".onnx", ".onnx_data",
    ".msgpack", ".h5", ".ckpt", ".ot", ".tflite", ".mlmodel", ".npz",
)
_FETCH_LLAMA_DEFAULT = "Q4_K_M"
# One part of a split GGUF: <name>-00001-of-00003.gguf
_FETCH_SPLIT = re.compile(r"^(.*)-\d{5}-of-\d{5}$")

_FETCH_LOCK = threading.Lock()
_FETCH = {"state": "idle"}


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------


class _FetchNoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_FETCH_OPENER = urllib.request.build_opener(_FetchNoRedirect)


def _fetch_token():
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""


def _fetch_open(url, token="", byte_range=None, timeout=60.0):
    """GET ``url``, following redirects by hand.

    The token goes only to the Hugging Face host: the CDN it redirects to holds
    a signed URL and must not see a second credential. Returns (response,
    final_url).
    """
    hf_host = urllib.parse.urlsplit(HF_ENDPOINT).netloc
    for _ in range(8):
        headers = {"User-Agent": "gpu4agents-fetch/1"}
        if token and urllib.parse.urlsplit(url).netloc == hf_host:
            headers["Authorization"] = "Bearer " + token
        if byte_range is not None:
            headers["Range"] = "bytes=%d-%d" % byte_range
        request = urllib.request.Request(url, headers=headers)
        try:
            return _FETCH_OPENER.open(request, timeout=timeout), url
        except urllib.error.HTTPError as exc:
            if exc.code in (301, 302, 303, 307, 308) and exc.headers.get("Location"):
                url = urllib.parse.urljoin(url, exc.headers["Location"])
                exc.close()
                continue
            raise
    raise OSError("too many redirects for %s" % url)


def _fetch_json(url, token="", timeout=60.0):
    response, _ = _fetch_open(url, token, timeout=timeout)
    with response:
        data = json.loads(response.read().decode())
        link = response.headers.get("Link") or ""
    return data, link


def _fetch_next_link(link):
    for part in link.split(","):
        if 'rel="next"' in part:
            return part.split(";")[0].strip().strip("<>")
    return None


def fetch_list_files(repo, revision="main", token="", timeout=60.0):
    """[(path, size)] of every file in the repo at ``revision``."""
    url = "%s/api/models/%s/tree/%s?recursive=true" % (
        HF_ENDPOINT,
        repo,
        urllib.parse.quote(revision, safe=""),
    )
    files = []
    while url:
        data, link = _fetch_json(url, token, timeout)
        for entry in data:
            if entry.get("type") == "file":
                size = (entry.get("lfs") or {}).get("size") or entry.get("size") or 0
                files.append((entry["path"], int(size)))
        url = _fetch_next_link(link)
    return files


def fetch_resolve_url(repo, revision, path):
    return "%s/%s/resolve/%s/%s" % (
        HF_ENDPOINT,
        repo,
        urllib.parse.quote(revision, safe=""),
        urllib.parse.quote(path),
    )


# --------------------------------------------------------------------------
# what to fetch
# --------------------------------------------------------------------------


def _fetch_is_weight(path):
    return path.lower().endswith(_FETCH_WEIGHT_SUFFIXES)


def _fetch_side_files(files):
    return [p for p, size in files if not _fetch_is_weight(p) and size <= FETCH_SIDE_FILE_MAX]


def fetch_select(files, engine, quant=None, read_json=None):
    """(paths, local_ref) the server needs, or (None, reason) to leave it to the server.

    ``local_ref`` is relative to the download directory: "" for a vLLM model
    directory, the first GGUF file for llama.cpp. ``read_json(path)`` reads a
    small JSON file of the repo (the safetensors index).
    """
    sizes = dict(files)
    if engine == "vllm":
        root = [p for p in sizes if "/" not in p]
        for suffix, index in (
            (".safetensors", "model.safetensors.index.json"),
            (".bin", "pytorch_model.bin.index.json"),
        ):
            weights = sorted(p for p in root if p.endswith(suffix))
            if not weights:
                continue
            if index in sizes and read_json is not None:
                shards = set((read_json(index).get("weight_map") or {}).values())
                missing = shards - set(sizes)
                if missing:
                    return None, "the index names shards that are not in the repo: %s" % sorted(missing)[:3]
                weights = sorted(shards)
            return sorted(set(_fetch_side_files(files)) | set(weights)), ""
        return None, "no .safetensors or .bin weights at the repo root"
    if engine == "llama":
        ggufs = [p for p in sizes if p.lower().endswith(".gguf")]
        # llama.cpp's -hf also loads a vision projector when the repo has one;
        # with -m it needs --mmproj, so the reference carries both.
        mmproj = [p for p in ggufs if "mmproj" in p.lower()]
        if len(mmproj) > 1:
            return None, "%d mmproj files: left to llama.cpp's own download" % len(mmproj)
        ggufs = [p for p in ggufs if p not in mmproj]
        tag = (quant or _FETCH_LLAMA_DEFAULT).lower()
        groups = {}
        for path in ggufs:
            stem = path[:-5]
            split = _FETCH_SPLIT.match(stem)
            groups.setdefault(split.group(1) if split else stem, []).append(path)
        matches = {k: v for k, v in groups.items() if tag in k.lower()}
        exact = {k: v for k, v in matches.items() if k.lower().endswith(tag)}
        chosen = exact if len(exact) == 1 else matches
        if len(chosen) != 1:
            return None, "%d GGUF files match quant %r" % (len(chosen), quant or _FETCH_LLAMA_DEFAULT)
        parts = sorted(next(iter(chosen.values())))
        if any(c.isspace() for p in parts + mmproj for c in p):
            # The launcher splits the reference on spaces.
            return None, "a file name with a space: left to llama.cpp's own download"
        if mmproj:
            return parts + mmproj, "%s --mmproj {dir}/%s" % (parts[0], mmproj[0])
        return parts, parts[0]
    return None, "engine %r is not fetched ahead" % engine


def fetch_plan(repo, revision, engine, quant=None, token="", timeout=60.0):
    """{files: [[path, size]], local_ref, bytes_total} or {skip: reason}."""
    files = fetch_list_files(repo, revision, token, timeout)

    def read_json(path):
        response, _ = _fetch_open(fetch_resolve_url(repo, revision, path), token, timeout=timeout)
        with response:
            return json.loads(response.read().decode())

    paths, ref = fetch_select(files, engine, quant, read_json)
    if paths is None:
        return {"skip": ref}
    sizes = dict(files)
    chosen = [[p, sizes[p]] for p in paths]
    return {
        "files": chosen,
        "local_ref": ref,
        "bytes_total": sum(size for _, size in chosen),
    }


def fetch_model_dir(repo):
    return os.path.join(FETCH_ROOT, repo.replace("/", "--"))


# --------------------------------------------------------------------------
# download
# --------------------------------------------------------------------------


class FetchProgress:
    def __init__(self, total):
        self.total = total
        self.done = 0
        self.lock = threading.Lock()
        self.samples = [(time.time(), 0)]

    def add(self, count):
        with self.lock:
            self.done += count
            now = time.time()
            if now - self.samples[-1][0] >= 1.0:
                self.samples.append((now, self.done))
                cutoff = now - FETCH_RATE_WINDOW_S - 5
                while len(self.samples) > 2 and self.samples[1][0] < cutoff:
                    self.samples.pop(0)

    def rate(self, window=FETCH_RATE_WINDOW_S):
        """Bytes/s over the last ``window`` seconds (or since the start)."""
        with self.lock:
            now = time.time()
            done = self.done
            start_t, start_b = self.samples[0]
            for t, b in self.samples:
                if now - t <= window:
                    break
                start_t, start_b = t, b
        span = now - start_t
        return (done - start_b) / span if span > 0 else 0.0


def _fetch_chunks_path(target):
    return target + ".jambu-chunks"


def _fetch_done_chunks(target):
    try:
        with open(_fetch_chunks_path(target)) as handle:
            return {int(line) for line in handle if line.strip()}
    except OSError:
        return set()


def fetch_download(repo, revision, files, dest, token="", workers=FETCH_WORKERS, chunk=FETCH_CHUNK,
             progress=None, stop=None):
    """Fetch ``files`` ([[path, size]]) into ``dest``; resumable, parallel by range.

    A file is written as ``<name>.part`` and renamed when every chunk is in;
    finished chunks are listed in ``<name>.part.jambu-chunks``, so a restart
    fetches only what is missing.
    """
    progress = progress or FetchProgress(sum(size for _, size in files))
    work = queue.Queue()
    handles = {}
    remaining = {}
    urls = {}
    lock = threading.Lock()
    failure = []

    for path, size in files:
        final = os.path.join(dest, path)
        if os.path.exists(final) and os.path.getsize(final) == size:
            progress.add(size)
            continue
        os.makedirs(os.path.dirname(final) or dest, exist_ok=True)
        part = final + ".part"
        fd = os.open(part, os.O_RDWR | os.O_CREAT, 0o644)
        os.ftruncate(fd, size)
        handles[path] = (fd, part, final)
        done = _fetch_done_chunks(part)
        count = max(1, (size + chunk - 1) // chunk)
        pending = [i for i in range(count) if i not in done]
        progress.add(sum(min(chunk, size - i * chunk) for i in done if i < count))
        remaining[path] = len(pending)
        urls[path] = fetch_resolve_url(repo, revision, path)
        for i in pending:
            work.put((path, size, i))
        if not pending:
            remaining[path] = 0

    def finish(path):
        fd, part, final = handles[path]
        os.fsync(fd)
        os.close(fd)
        os.replace(part, final)
        try:
            os.remove(_fetch_chunks_path(part))
        except OSError:
            pass

    for path in [p for p, n in remaining.items() if n == 0]:
        finish(path)

    def fetch_chunk(path, size, index):
        fd, part, _ = handles[path]
        start = index * chunk
        end = min(size, start + chunk) - 1
        if size == 0:
            return
        for attempt in range(FETCH_RETRIES):
            got = 0
            try:
                response, final_url = _fetch_open(urls[path], token, (start, end))
                with response:
                    if response.status == 200 and (start, end) != (0, size - 1):
                        raise OSError("the server ignored the byte range")
                    offset = start
                    while True:
                        block = response.read(1 << 20)
                        if not block:
                            break
                        os.pwrite(fd, block, offset)
                        offset += len(block)
                        got += len(block)
                        progress.add(len(block))
                        if stop is not None and stop.is_set():
                            raise OSError("stopped")
                if got != end - start + 1:
                    raise OSError("short read: %d of %d bytes" % (got, end - start + 1))
                if final_url != urls[path]:
                    urls[path] = final_url  # the signed CDN URL: skip the resolve hop
                return
            except (OSError, urllib.error.URLError) as exc:
                progress.add(-got)
                if stop is not None and stop.is_set():
                    raise
                if isinstance(exc, urllib.error.HTTPError) and exc.code in (401, 403, 404, 410):
                    urls[path] = fetch_resolve_url(repo, revision, path)  # the signed URL expired
                    if exc.code in (401, 404) and attempt >= 1:
                        raise
                if attempt == FETCH_RETRIES - 1:
                    raise
                time.sleep(min(30, 2 ** (attempt + 1)))

    def worker():
        while not failure:
            try:
                path, size, index = work.get_nowait()
            except queue.Empty:
                return
            try:
                fetch_chunk(path, size, index)
            except Exception as exc:  # noqa: BLE001 - reported once, ends the fetch
                failure.append("%s: %s" % (path, exc))
                return
            with lock:
                with open(_fetch_chunks_path(handles[path][1]), "a") as handle:
                    handle.write("%d\n" % index)
                remaining[path] -= 1
                if remaining[path] == 0:
                    finish(path)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(max(1, workers))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failure:
        for path, (fd, _, _) in handles.items():
            if remaining.get(path):
                try:
                    os.close(fd)
                except OSError:
                    pass
        raise OSError(failure[0])
    return progress


# --------------------------------------------------------------------------
# the watchdog side: markers, status, routes
# --------------------------------------------------------------------------


def _fetch_write(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        handle.write(text)
    os.replace(tmp, path)


def _fetch_remove(path):
    try:
        os.remove(path)
    except OSError:
        pass


def _fetch_key(request):
    return json.dumps(
        {k: request.get(k) for k in ("repo", "revision", "engine", "quant")}, sort_keys=True
    )


def fetch_status():
    with _FETCH_LOCK:
        out = {k: v for k, v in _FETCH.items() if not k.startswith("_")}
        progress = _FETCH.get("_progress")
    if progress is not None:
        out["bytes_done"] = progress.done
        out["rate_mbps"] = round(progress.rate() * 8 / 1e6, 1)
    return out


def _fetch_release(local_ref=None):
    """Let the launcher go: with a local path, or (None) to its own download."""
    if local_ref is not None:
        _fetch_write(FETCH_MARKER + ".done", local_ref + "\n")
    _fetch_remove(FETCH_MARKER)


def _run_fetch(request):
    token = _fetch_token()
    repo = request["repo"]
    revision = request.get("revision") or "main"
    started = time.time()
    try:
        found = fetch_plan(repo, revision, request.get("engine"), request.get("quant"), token)
        if "skip" in found:
            _fetch_release(None)
            with _FETCH_LOCK:
                _FETCH.update(state="skipped", reason=found["skip"])
            log_event("fetch_skipped", repo=repo, reason=found["skip"])
            return
        dest = fetch_model_dir(repo)
        os.makedirs(dest, exist_ok=True)
        free = shutil.disk_usage(dest).free
        need = found["bytes_total"] + FETCH_DISK_MARGIN
        if free < need:
            raise OSError("the disk has %.1f GB free; the weights need %.1f GB" % (free / 1e9, need / 1e9))
        progress = FetchProgress(found["bytes_total"])
        with _FETCH_LOCK:
            _FETCH.update(state="fetching", bytes_total=found["bytes_total"], files=len(found["files"]),
                          started_at=started, _progress=progress)
        log_event("fetch_started", repo=repo, files=len(found["files"]), bytes_total=found["bytes_total"])
        fetch_download(repo, revision, found["files"], dest, token, progress=progress)
        seconds = round(time.time() - started, 1)
        ref = found["local_ref"].replace("{dir}", dest)
        local = os.path.join(dest, ref) if ref else dest
        _fetch_write(FETCH_MARKER + ".plan.json", json.dumps(dict(request, done=True, local_ref=local)))
        _fetch_release(local)
        with _FETCH_LOCK:
            _FETCH.update(state="done", finished_at=time.time(), seconds=seconds, local_ref=local)
        log_event("fetch_done", repo=repo, bytes_total=found["bytes_total"], seconds=seconds,
                  mb_per_s=round(found["bytes_total"] / 1e6 / max(seconds, 0.001), 1))
    except Exception as exc:  # noqa: BLE001 - the server's own download is the fallback
        _fetch_release(None)
        with _FETCH_LOCK:
            _FETCH.update(state="failed", error=str(exc)[:300])
        log_event("fetch_failed", repo=repo, error=str(exc)[:300])


def fetch_start(request):
    request = {k: request.get(k) for k in ("repo", "revision", "engine", "quant")}
    if not request.get("repo") or request.get("engine") not in ("vllm", "llama"):
        return 400, {"error": "fetch needs repo and engine (vllm or llama)"}
    key = _fetch_key(request)
    with _FETCH_LOCK:
        if _FETCH.get("key") == key and _FETCH.get("state") in ("planning", "fetching", "done"):
            pass
        elif not os.path.exists(FETCH_MARKER):
            # The launcher is not waiting (it gave up, or this boot script made
            # no marker): fetching now would only download the weights twice.
            _FETCH.clear()
            _FETCH.update(state="skipped", key=key, repo=request["repo"],
                          reason="the model server is not waiting for a fetch")
        else:
            _FETCH.clear()
            _FETCH.update(state="planning", key=key, repo=request["repo"])
            _fetch_write(FETCH_MARKER + ".plan.json", json.dumps(request))
            _fetch_write(FETCH_MARKER, "fetching\n")
            threading.Thread(target=_run_fetch, args=(request,), daemon=True).start()
    return 200, fetch_status()


def fetch_resume():
    """On load: finish what a previous run of this watchdog started, or confirm it."""
    try:
        with open(FETCH_MARKER + ".plan.json") as handle:
            saved = json.load(handle)
    except (OSError, ValueError):
        return
    if saved.get("done"):
        local = saved.get("local_ref") or ""
        if local and os.path.exists(local):
            with _FETCH_LOCK:
                _FETCH.update(state="done", key=_fetch_key(saved), repo=saved.get("repo"), local_ref=local)
            if os.path.exists(FETCH_MARKER):
                _fetch_release(local)
            return
    if os.path.exists(FETCH_MARKER):
        fetch_start(saved)


def fetch_route(request, method, path, body):
    if path != "/fetch":
        return False
    if method == "GET":
        request._reply(200, fetch_status())
    elif method == "POST":
        request._reply(*fetch_start(body or {}))
    else:
        request._reply(405, {"error": "GET or POST"})
    return True


if "_STATE" in globals() and "load_extension" in globals():
    # Inside the watchdog (exec'd into its globals).
    fetch_resume()
    EXT_ROUTES["fetch"] = fetch_route


def _fetch_main(argv):  # pragma: no cover - exercised by the benchmark on a GPU host
    import argparse

    parser = argparse.ArgumentParser(description="Fetch a Hugging Face model's serving files in parallel.")
    parser.add_argument("repo")
    parser.add_argument("dest")
    parser.add_argument("--engine", default="vllm", choices=("vllm", "llama"))
    parser.add_argument("--quant")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--workers", type=int, default=FETCH_WORKERS)
    parser.add_argument("--chunk-mb", type=int, default=FETCH_CHUNK >> 20)
    args = parser.parse_args(argv)
    token = _fetch_token()
    found = fetch_plan(args.repo, args.revision, args.engine, args.quant, token)
    if "skip" in found:
        print(json.dumps({"skip": found["skip"]}))
        return 2
    started = time.time()
    progress = FetchProgress(found["bytes_total"])
    stop = threading.Event()

    def report():
        while not stop.wait(10):
            print(json.dumps({"elapsed_s": round(time.time() - started), "bytes_done": progress.done,
                              "rate_mb_s_10s": round(progress.rate(10) / 1e6, 1)}), flush=True)

    threading.Thread(target=report, daemon=True).start()
    fetch_download(args.repo, args.revision, found["files"], args.dest, token, workers=args.workers,
             chunk=args.chunk_mb << 20, progress=progress)
    stop.set()
    seconds = time.time() - started
    print(json.dumps({"repo": args.repo, "files": len(found["files"]), "bytes": found["bytes_total"],
                      "seconds": round(seconds, 1), "mb_per_s": round(found["bytes_total"] / 1e6 / seconds, 1),
                      "workers": args.workers, "chunk_mb": args.chunk_mb, "local_ref": found["local_ref"]}))
    return 0


if __name__ == "__main__" and "_STATE" not in globals():  # pragma: no cover
    import sys

    sys.exit(_fetch_main(sys.argv[1:]))
