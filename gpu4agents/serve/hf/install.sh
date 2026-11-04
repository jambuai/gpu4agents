#!/bin/bash
# Installs what server.py needs for $JAMBU_TASK on top of the image's PyTorch.
# The torch build the image ships is pinned as a constraint, so pip never
# swaps it for one built against another CUDA.
set -eu

PY=$(command -v python3 || command -v python)
case "${JAMBU_TASK:-auto}" in
  chat) packages="transformers accelerate" ;;
  embeddings) packages="sentence-transformers transformers accelerate" ;;
  image) packages="diffusers transformers accelerate" ;;
  transcription) packages="transformers accelerate soundfile" ;;
  speech) packages="transformers accelerate soundfile" ;;
  *) packages="transformers accelerate diffusers sentence-transformers soundfile" ;;
esac
# brotli>=1.2: huggingface_hub downloads through httpx2, which calls brotli's
# output_buffer_limit; the PyTorch image's conda brotli 1.1 lacks it and every
# weight download died with "process() takes no keyword arguments".
packages="$packages safetensors sentencepiece protobuf huggingface_hub pillow brotli>=1.2 ${JAMBU_PACKAGES:-}"

constraints=$(mktemp)
"$PY" - > "$constraints" <<'EOF'
import importlib.metadata as metadata
for name in ("torch", "torchvision", "torchaudio"):
    try:
        print(f"{name}=={metadata.version(name)}")
    except metadata.PackageNotFoundError:
        pass
EOF
echo "jambu-serve: installing $packages"
# shellcheck disable=SC2086 - the package list is meant to split
PIP_ROOT_USER_ACTION=ignore "$PY" -m pip install --quiet --no-cache-dir \
  --disable-pip-version-check -c "$constraints" $packages
rm -f "$constraints"
