# Run your own code on the GPU (instructions for the agent)

You are a coding agent and the work needs a GPU: training or fine-tuning, an image, audio or video model, a batch over a dataset, a kernel benchmark, anything that wants CUDA. GPU for Agents (`gpu`) rents the machine, runs your code on it as a **job**, brings back the files it writes, and stops the machine once nothing runs. This page is the contract a job follows. Follow it exactly.

## 1. The machine

The job runs on the project's machine, described by `gpu.yaml`. For a machine that only runs jobs:

```bash
gpu init --jobs --min-vram-gb 24     # runtime.engine: none, image pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
```

Size `--min-vram-gb` for the job (a 7B model in bf16 needs about 16 GB to run and much more to fully fine-tune; LoRA on 7B fits in 24 GB). To use a different image, set `compute.image`. It needs `bash` and `python3`.

A project that serves a model with vLLM can run jobs too. The job then reaches the model over loopback through `OPENAI_BASE_URL`, `OPENAI_API_KEY` and `JAMBU_MODEL_ID`, which are already set, and vLLM holds most of the GPU memory.

## 2. The job directory

Put everything the job needs in one directory (the project directory by default):

```
my-job/
  gpujob.yaml        # what to run (optional; flags override it)
  train.py           # your code
  requirements.txt   # installed by setup:
```

`gpujob.yaml`:

```yaml
run: python train.py --epochs 3          # required here or as `gpu job run -- CMD`
setup: pip install -r requirements.txt   # runs first; if it fails, run does not
env:                                     # extra environment variables
  EPOCHS: "3"
timeout: 2h                              # killed after this long (exit 124)
exclude: [data/raw, "*.ckpt"]            # left out of the upload
label: lora-qwen-7b
```

These never upload: `.git`, virtualenvs, `__pycache__`, `node_modules`, `.gpu`, `.env` and `.env.*`, and `gpu-outputs/`. The upload limit is 2 GB. Have the job download large data itself (Hugging Face, a URL) instead of shipping it.

## 3. What the job can count on

- It runs in its own copy of the directory, with that copy as the working directory.
- `JAMBU_OUTPUT_DIR`: write every result here (files, checkpoints, images, metrics). Nothing else comes back.
- `JAMBU_JOB_ID` and `JAMBU_JOB_DIR` are set.
- `HF_HOME` points to a cache on the machine's disk, so a model downloaded once is reused by the next job on that machine. `HF_TOKEN` is set when the project has one (`gpu login` or `.env`).
- stdout and stderr go into one log. Python output is unbuffered.
- The exit code is the job's result: 0 means succeeded, anything else means failed.
- The provider credential the machine uses to stop itself is removed from the environment.

## 4. Rules for the code you write

1. Write results only to `$JAMBU_OUTPUT_DIR`. Write checkpoints there as you go, so a pull mid-run or a cut-off job still leaves something.
2. Print progress lines (step, loss, items done). The log is the only way anyone sees the job running.
3. Exit non-zero on failure. Do not catch every exception and exit 0.
4. Never print secrets, and never write them into outputs.
5. Install dependencies in `setup:`, pinned (`torch` comes with the default image; do not reinstall it).
6. Keep the job under `lifecycle.max_lifetime` (6 h by default) and the session budget: those stop the machine even while a job runs. Set `timeout:` below them.

## 5. Running it

With the user's yes (renting costs money; show `gpu offers` first):

```bash
gpu job run                      # pack, upload, run, stream the log, download the outputs
gpu job run -- python eval.py    # the command, without gpujob.yaml
gpu job run --detach             # return once it starts
gpu job status                   # state, exit code, time, cost, output files (latest job)
gpu job logs -f                  # follow the log
gpu job pull                     # outputs into gpu-outputs/<job id>/ (works mid-run)
gpu job cancel                   # TERM, then KILL after 10 s
gpu job ls
```

Through MCP: `gpu_setup` until ready, `gpu_job_run`, then `gpu_job_status(job_id)` until the state is no longer `running`, then `gpu_job_pull(job_id)`.

States: `running`, `succeeded`, `failed`, `timeout`, `cancelled`, `lost` (the machine restarted under it). `gpu job run` exits with the job's code, 124 on timeout and 130 on cancel.

## 6. Afterwards

The machine stops by itself after `idle_timeout` (15 min) once no job runs. A stopped machine keeps its disk, so `gpu setup` restarts it and `gpu job pull` still works. `gpu destroy` deletes the disk and every job file on it, so pull the outputs first. Tell the user what ran, how long it took, what it cost (`gpu job status` reports `cost_usd`) and where the outputs are.
