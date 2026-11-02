#!/bin/sh
# GPU for Agents installer: https://gpu4agents.jambu.ai
#
#   curl -fsSL https://gpu4agents.jambu.ai/install.sh | sh
#   curl -fsSL https://gpu4agents.jambu.ai/install.sh | sh -s -- --agent codex
#
# Installs the `gpu` command with uv, which downloads a suitable Python by
# itself, so nothing but curl is needed. Then gives the agent runtimes found on
# this machine (Claude Code, Codex, Gemini CLI, Cursor) the gpu4agents skill and
# the gpu MCP server. It never asks for or touches your GPU provider's key:
# run `gpu login` afterwards.
#
# Options:
#   --agent NAME   claude | codex | gemini | cursor | auto (default) | none
#   --no-modify-path  do not add ~/.local/bin to your shell profile
#
# Environment:
#   GPU4AGENTS_SPEC  what to install (default: gpu4agents>=0.7.0); a version pin or a wheel path
#   GPU4AGENTS_PYTHON  Python version uv uses for the tool (default: 3.12)

set -eu

AGENT="auto"
MODIFY_PATH=1
SPEC="${GPU4AGENTS_SPEC:-gpu4agents>=0.7.0}"
PYVER="${GPU4AGENTS_PYTHON:-3.12}"

while [ $# -gt 0 ]; do
  case "$1" in
    --agent) AGENT="${2:-}"; shift 2 ;;
    --agent=*) AGENT="${1#*=}"; shift ;;
    --no-modify-path) MODIFY_PATH=0; shift ;;
    -h|--help) sed -n '2,20p' "$0" 2>/dev/null || true; exit 0 ;;
    *) echo "gpu4agents: unknown option $1" >&2; exit 2 ;;
  esac
done

say() { printf '\033[1;32m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

case "$(uname -s)" in
  Linux|Darwin) ;;
  *) die "GPU for Agents installs on Linux and macOS (on Windows, run this inside WSL)." ;;
esac
command -v curl >/dev/null 2>&1 || die "curl is required"

BIN="${XDG_BIN_HOME:-$HOME/.local/bin}"
ORIG_PATH="$PATH"
export PATH="$BIN:$HOME/.cargo/bin:$PATH"

# 1. uv: a single binary that installs Python tools and Python itself.
if ! command -v uv >/dev/null 2>&1; then
  say "Installing uv (https://docs.astral.sh/uv/)"
  if [ "$MODIFY_PATH" = 1 ]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  else
    curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
  fi
  command -v uv >/dev/null 2>&1 || die "uv was installed but is not on PATH; open a new shell and run this again"
fi

# 2. The gpu command. A tool from the old jambu-gpu package owns the same
#    executable name, so it goes first.
if uv tool list 2>/dev/null | grep -q '^jambu-gpu '; then
  say "Removing the old jambu-gpu tool (renamed to gpu4agents)"
  uv tool uninstall jambu-gpu >/dev/null
fi
say "Installing gpu4agents (Python $PYVER via uv)"
uv tool install --upgrade --python "$PYVER" "$SPEC"

GPU="$BIN/gpu"
[ -x "$GPU" ] || GPU="$(command -v gpu || true)"
[ -n "$GPU" ] && [ -x "$GPU" ] || die "gpu was installed but not found in $BIN"
say "Installed $("$GPU" --version) at $GPU"

if [ "$MODIFY_PATH" = 1 ]; then
  uv tool update-shell >/dev/null 2>&1 || true
fi

# 3. The skill and the MCP server for the agent runtimes.
case "$AGENT" in
  none) ;;
  auto|claude|codex|gemini|cursor)
    say "Setting up agent runtime(s): $AGENT"
    "$GPU" agent install -t "$AGENT" --global || warn "gpu agent install failed; run it again later: gpu agent install -t $AGENT --global"
    ;;
  *) warn "unknown --agent $AGENT; skipped (use claude, codex, gemini, cursor or auto)" ;;
esac

# 4. What is left for the user.
echo
if "$GPU" login --check >/dev/null 2>&1; then
  say "Your GPU provider account is connected. Ask your agent for a GPU experiment."
else
  say "One step left: connect a GPU provider account."
  echo "    In a terminal, run:  gpu login"
  echo "    It asks for a Vast.ai key (built in). For another provider, ask your agent"
  echo "    to add it: https://gpu4agents.jambu.ai/providers.md"
fi
case ":$ORIG_PATH:" in
  *":$BIN:"*) ;;
  *) echo "    Open a new terminal (or run: export PATH=\"$BIN:\$PATH\") so \`gpu\` is on your PATH." ;;
esac
echo "    Then restart your agent so it loads the gpu4agents skill and tools."
