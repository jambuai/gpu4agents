# Install GPU for Agents (instructions for the agent)

You are a coding agent (Claude Code, Codex, Gemini CLI, Cursor, or another) and the user asked you to set up GPU for Agents. It gives you a `gpu` command, a skill and MCP tools to rent a cloud GPU, serve a model on it and run experiments. A watchdog on the machine shuts it down when idle, so the user pays only while it works. Vast.ai is the built-in GPU provider, and any other provider can be added with an adapter.

Follow these steps in order. Run the commands yourself, and tell the user briefly what each step did.

## 1. Install the `gpu` command and give it to this agent

Pick the name of the runtime you are running in: `claude` (Claude Code), `codex` (Codex), `gemini` (Gemini CLI) or `cursor` (Cursor). If you are none of these, use `auto`.

```bash
curl -fsSL https://gpu4agents.jambu.ai/install.sh | sh -s -- --agent <runtime>
```

If that address does not resolve, use the copy in the repository:

```bash
curl -fsSL https://raw.githubusercontent.com/jambuai/gpu4agents/main/install.sh | sh -s -- --agent <runtime>
```

The script needs only `curl` (Linux or macOS; on Windows, run it inside WSL). It installs [uv](https://docs.astral.sh/uv/) if missing, uses it to install the `gpu4agents` package with a Python it downloads by itself, and then installs the skill and registers the `gpu` MCP server for your runtime. It does not ask for or read any key.

Check: `~/.local/bin/gpu --version` prints 0.12.0 or later.

For the rest of this session, call the command by its full path, `~/.local/bin/gpu`. Your shell may define `gpu` as an alias: Oh My Zsh's git plugin, on by default, makes `gpu` mean `git push upstream`, and an alias wins over the command, so plain `gpu setup` could run git instead. The installer removes that alias for new shells and says so; tell the user if it did. The same command is also installed as `gpu4agents`.

## 2. Connect a GPU provider account

Run `~/.local/bin/gpu login --check`.

- Exit code 0: an account is connected. Go to step 3.
- Exit code 5: no account is connected yet. Tell the user, in plain words and without technical terms, something close to this:

  > One last step, about two minutes: connect the account that pays for the GPUs. Open your terminal app (not this chat) and type `gpu login`. It shows the providers you can use and walks you through it: Vast.ai works right away, and if you already use another one (RunPod, Lambda, your company's servers), pick "Another provider" and I'll add it. Your key stays on your computer; I never see it.

  If the installer said it kept a `gpu` alias, tell them to type `gpu4agents login` instead.

  If the user picks another provider, write its adapter after this installation by following https://gpu4agents.jambu.ai/providers.md. The user then runs `gpu login --provider <name>`.

  Wait until the user says it is done, then run `~/.local/bin/gpu login --check` again. If the user wants to connect later, go on to step 3.

Never ask the user to paste a key into this conversation, never read `~/.gpu/credentials` or `.env` files, and never print a key.

## 3. Learn how to use it

Run `gpu agent skill` and read the output. It is the skill you just installed: the rules (ask before renting, wait out a slow setup, clean up), the commands, monitoring, adding a provider, and error codes. Follow it for the rest of this session.

The MCP tools (`gpu_setup`, `gpu_status`, ...) appear after the user restarts the agent. Until then, the CLI does everything: `gpu --json <command>`.

## 4. Tell the user it is ready

Summarize in a few lines:

- what was installed (the `gpu` command, the skill, the MCP server for this runtime);
- whether a provider account is connected, or what they still need to do (`gpu login` in their terminal, or an adapter for their provider), in plain words;
- that they should restart the agent to get the tools;
- an example request: "Compare Qwen 7B and Llama 8B on 50 prompts", and that you will show the price and ask before renting anything.

Do not rent a GPU as part of the installation.
