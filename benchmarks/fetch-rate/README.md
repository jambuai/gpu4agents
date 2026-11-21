# Weight-fetch benchmark

Measures how fast a rented Vast machine gets model weights from Hugging Face: with `gpu4agents/watchdog/fetch_ext.py`, and with the downloads vLLM and llama.cpp do on their own. It backs the numbers in [docs/engineering/fast-provisioning.md](../../docs/engineering/fast-provisioning.md).

```bash
cp ../../gpu4agents/watchdog/fetch_ext.py .   # the job imports it
gpu offers                                     # show the card and price, get a yes
gpu setup --offer <id>
gpu job run                                    # about 30 minutes
gpu destroy --confirm
```

The results land in `gpu-outputs/<job id>/results.json`.

## Results, 2026-11-21

Two Vast hosts, about $1.20 in all. The raw numbers are in `results-2026-11-21.json`.

| Download | A100 SXM4, 864 Mbps advertised | Quadro RTX 8000, 7988 Mbps advertised |
| --- | --- | --- |
| 27.5 GB GGUF, one connection (llama.cpp `-hf`) | 67 MB/s | 119 MB/s |
| 27.5 GB GGUF, fetcher, 32 connections | 117 MB/s | 446 MB/s |
| 55.6 GB safetensors, fetcher, 32 connections | 100 MB/s | 439 MB/s |
| 55.6 GB safetensors, `huggingface_hub` 2.2.0 + `hf_xet` 1.7.0 (vLLM) | 112 MB/s | 1065 MB/s |

On the slow host the link caps every method at about 110 MB/s. On the fast host, `hf_xet` beats the fetcher, and the fetcher beats a single connection by 3.7 times. That is why `runtime.fetch: auto` uses the fetcher for llama.cpp only.
