"""Queue smoke job: a CUDA matmul for about a minute, result written to $JAMBU_OUTPUT_DIR."""
import json, os, time
import torch

assert torch.cuda.is_available(), "no CUDA"
out = os.environ["JAMBU_OUTPUT_DIR"]
dev = torch.device("cuda")
a = torch.randn(4096, 4096, device=dev)
b = torch.randn(4096, 4096, device=dev)
start, n = time.time(), 0
while time.time() - start < float(os.environ.get("SMOKE_SECONDS", "60")):
    c = a @ b
    n += 1
torch.cuda.synchronize()
elapsed = time.time() - start
result = {
    "job": os.environ.get("JAMBU_JOB_ID"),
    "gpu": torch.cuda.get_device_name(0),
    "matmuls": n,
    "tflops": round(2 * 4096**3 * n / elapsed / 1e12, 2),
    "checksum": float(c[0, 0].item()),
    "previous_outputs": os.environ.get("JAMBU_PREVIOUS_OUTPUT_DIR"),
}
with open(os.path.join(out, "result.json"), "w") as f:
    json.dump(result, f)
print(json.dumps(result), flush=True)
