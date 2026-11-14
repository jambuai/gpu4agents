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
#   --keep-gpu-alias  leave a shell alias named gpu alone (see step 4)
#
# Environment:
#   GPU4AGENTS_SPEC  what to install (default: gpu4agents>=0.14.0); a version pin or a wheel path
#   GPU4AGENTS_PYTHON  Python version uv uses for the tool (default: 3.12)

set -eu

AGENT="auto"
MODIFY_PATH=1
FIX_ALIAS=1
SPEC="${GPU4AGENTS_SPEC:-gpu4agents>=0.14.0}"
PYVER="${GPU4AGENTS_PYTHON:-3.12}"

while [ $# -gt 0 ]; do
  case "$1" in
    --agent) AGENT="${2:-}"; shift 2 ;;
    --agent=*) AGENT="${1#*=}"; shift ;;
    --no-modify-path) MODIFY_PATH=0; shift ;;
    --keep-gpu-alias) FIX_ALIAS=0; shift ;;
    -h|--help) sed -n '2,21p' "$0" 2>/dev/null || true; exit 0 ;;
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

# 4. A shell alias named gpu. Oh My Zsh's git plugin, on by default, defines
#    gpu='git push upstream', and an alias wins over the command: typing
#    `gpu login`, in a terminal or in the agent's shell, would run git instead.
#    New shells get the alias removed, in a marked block at the end of the rc
#    file. The same command is also installed as `gpu4agents`, which no alias takes.
MARK="# >>> gpu4agents >>>"
gpu_kind() {  # how an interactive $1 shell resolves `gpu`: alias, function, command or none
  case "$1" in
    zsh) out=$(zsh -ic 'whence -w gpu' </dev/null 2>/dev/null | tail -n 1) ;;
    bash) out=$(bash -ic 'type -t gpu' </dev/null 2>/dev/null | tail -n 1) ;;
  esac
  case "$out" in
    *alias*) echo alias ;; *function*) echo function ;; *command*|*file*) echo command ;; *) echo none ;;
  esac
}
ALIAS_NOTE=""
for sh_name in zsh bash; do
  command -v "$sh_name" >/dev/null 2>&1 || continue
  rc="$HOME/.${sh_name}rc"
  [ -f "$rc" ] || continue
  kind=$(gpu_kind "$sh_name")
  [ "$kind" = alias ] || [ "$kind" = function ] || continue
  if [ "$kind" = function ] || [ "$FIX_ALIAS" = 0 ]; then
    warn "In $sh_name, gpu is a shell $kind of yours, so typing gpu does not run GPU for Agents."
    warn "Use the full name instead: gpu4agents (for example: gpu4agents login)."
    ALIAS_NOTE="gpu4agents"
    continue
  fi
  if ! grep -q "$MARK" "$rc" 2>/dev/null; then
    {
      echo ""
      echo "$MARK"
      echo "# gpu runs GPU for Agents. Your shell had gpu as an alias (Oh My Zsh's git"
      echo "# plugin: gpu = git push upstream). Delete these lines to get the alias back."
      echo "unalias gpu 2>/dev/null"
      echo "# <<< gpu4agents <<<"
    } >> "$rc"
  fi
  if [ "$(gpu_kind "$sh_name")" = alias ]; then
    warn "gpu is still an alias in $sh_name after editing $rc; use gpu4agents instead."
    ALIAS_NOTE="gpu4agents"
  else
    say "Your terminal had gpu as a shortcut for another command (Oh My Zsh sets it to git push upstream)."
    echo "    New terminals now run GPU for Agents when you type gpu. The other command still works typed in full."
    echo "    (Changed $rc; delete the gpu4agents block there to undo it.)"
    [ -n "$ALIAS_NOTE" ] || ALIAS_NOTE="new-shell"
  fi
done

# 5. What is left for the user.
CMD=gpu
[ "$ALIAS_NOTE" = gpu4agents ] && CMD=gpu4agents
echo
if "$GPU" login --check >/dev/null 2>&1; then
  say "Your GPU provider account is connected. Ask your agent for something that needs a GPU."
else
  say "One step left: connect a GPU provider account (about 2 minutes)."
  echo "    Open your terminal app and type:  $CMD login"
  echo "    It shows the providers you can use and walks you through it: Vast.ai is"
  echo "    ready to use, and your agent can add one you already have."
fi
case ":$ORIG_PATH:" in
  *":$BIN:"*) ;;
  *) echo "    Open a new terminal (or run: export PATH=\"$BIN:\$PATH\") so \`gpu\` is on your PATH." ;;
esac
echo "    Then restart your agent so it loads the gpu4agents skill and tools."
