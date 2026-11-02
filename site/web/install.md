# Install GPU for Agents (instructions for the agent)

You are a coding agent (Claude Code, Codex, Gemini CLI, Cursor, or another) and the user asked you to set up **GPU for Agents**: a `gpu` command, a skill and MCP tools that let you rent a cloud GPU, serve a model and run experiments, while a watchdog on the machine shuts it down when idle so the user pays only while it works.

Follow these steps in order. Run the commands yourself; tell the user briefly what each step did.

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

Check: `~/.local/bin/gpu --version` prints 0.6.0 or later. Use `~/.local/bin/gpu` for the rest of this session if plain `gpu` is not found.

## 2. Connect the user's GPU account

Run `gpu login --check`.

- Exit code 0: the account is connected. Go to step 3.
- Exit code 5: no working key yet. Tell the user, in these words or close:

  > Create an API key at https://cloud.vast.ai/manage-keys/ and add a few dollars of credit. Then open your own terminal, run `gpu login`, and paste the key into the hidden prompt.

  Wait until the user says it is done, then run `gpu login --check` again.

Never ask the user to paste the key into this conversation, never read `~/.gpu/credentials` or `.env` files, and never print the key.

## 3. Learn how to use it

Run `gpu agent skill` and read the output: it is the skill you just installed, with the rules (ask before renting, wait out a slow setup, clean up), the commands, monitoring and error codes. Follow it from now on in this session.

The MCP tools (`gpu_setup`, `gpu_status`, ...) appear after the user restarts the agent. Until then, the CLI does everything: `gpu --json <command>`.

## 4. Tell the user it is ready

Summarize in a few lines:

- what was installed (the `gpu` command, the skill, the MCP server for this runtime);
- whether the account is connected, or what they still need to do (`gpu login`);
- that they should restart the agent to get the tools;
- an example request: "Compare Qwen 7B and Llama 8B on 50 prompts", and that you will show the price and ask before renting anything.

Do not rent a GPU as part of the installation.
