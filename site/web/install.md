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

Check: `~/.local/bin/gpu --version` prints 0.9.0 or later. Use `~/.local/bin/gpu` for the rest of this session if plain `gpu` is not found.

## 2. Connect a GPU provider account

Run `gpu login --check`.

- Exit code 0: an account is connected. Go to step 3.
- Exit code 5: no working key yet. Ask the user which GPU provider they want to use.
  - Vast.ai, or no preference: tell the user, in these words or close:

    > Create an API key at https://cloud.vast.ai/manage-keys/ and add a few dollars of credit. Then open your own terminal, run `gpu login`, and paste the key into the hidden prompt.

  - Another provider (RunPod, Lambda, their company's cluster, ...): `gpu` needs an adapter for it first. Tell the user you can write one, and do it after this installation by following https://gpu4agents.jambu.ai/providers.md. The user then runs `gpu login --provider <name>`.

  Wait until the user says it is done, then run `gpu login --check` again. If the user wants to connect later, go on to step 3.

Never ask the user to paste a key into this conversation, never read `~/.gpu/credentials` or `.env` files, and never print a key.

## 3. Learn how to use it

Run `gpu agent skill` and read the output. It is the skill you just installed: the rules (ask before renting, wait out a slow setup, clean up), the commands, monitoring, adding a provider, and error codes. Follow it for the rest of this session.

The MCP tools (`gpu_setup`, `gpu_status`, ...) appear after the user restarts the agent. Until then, the CLI does everything: `gpu --json <command>`.

## 4. Tell the user it is ready

Summarize in a few lines:

- what was installed (the `gpu` command, the skill, the MCP server for this runtime);
- whether a provider account is connected, or what they still need to do (`gpu login`, or an adapter for their provider);
- that they should restart the agent to get the tools;
- an example request: "Compare Qwen 7B and Llama 8B on 50 prompts", and that you will show the price and ask before renting anything.

Do not rent a GPU as part of the installation.
