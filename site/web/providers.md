# Add a GPU provider to GPU for Agents (instructions for the agent)

You are a coding agent, and the user wants GPU for Agents (`gpu`) to rent from a provider it does not ship with, such as RunPod, Lambda, Paperspace, a company cluster or their own servers. Vast.ai is built in. Any other provider is one Python file that implements the provider contract. The core, the watchdog that stops idle machines, the CLI and the MCP tools stay the same.

Follow these steps in the user's project directory. Do not rent anything until step 6, and ask first.

## 1. Make sure `gpu` is installed

`gpu --version` must print 0.7.0 or later. If it does not, follow https://gpu4agents.jambu.ai/install.md first.

## 2. Write the adapter skeleton

```bash
gpu providers new <name> --display-name "<Provider name>"
gpu init --provider <name> --model <org/model>   # skip if jambu.yaml exists
```

The first command writes `gpu_providers/<name>.py`. The second writes a `jambu.yaml` that selects it:

```yaml
gpu_runtime:
  provider:
    name: <name>
    plugin: ./gpu_providers/<name>.py:<Name>Provider
```

In an existing `jambu.yaml`, add those two lines under `gpu_runtime.provider` yourself.

## 3. Read the provider's API docs, then fill in the file

Every method that raises `NotImplementedError` needs the provider's API. Read the contract in `gpu4agents/providers/base.py` (run `python -c "import gpu4agents.providers.base as b; print(b.__file__)"` to find it) and the guide at https://github.com/jambuai/gpu4agents/blob/main/docs/ADDING_A_PROVIDER.md. The Vast.ai adapter in `gpu4agents/providers/vast/` is a full working example.

What each part must do:

| Part | What it does |
| --- | --- |
| `capabilities` | Declare only what the API supports. Set `stop_eliminates_billing=False` if a stopped machine still bills for disk. |
| `verify_credentials` | Call an account endpoint with the key the user typed into `gpu login`. Return only safe fields, such as `{"balance_usd": 12.5}`. Never return an email, a name or the key. |
| `find_offers` | List GPUs for rent that fit `spec.gpu.min_vram_gb`, `spec.gpu.count`, `spec.disk_gb` and `spec.max_hourly_cost_usd`, cheapest first. |
| `setup` | Rent one machine: image `spec.image`, environment `spec.env`, label `spec.label`, every port in `spec.ports` published, and `spec.onstart` run at boot. That script starts the model server and the watchdog. |
| `status`, `_to_instance` | Map the provider's states onto `InstanceState` and its published ports onto `endpoints["model"]` and `endpoints["watchdog"]`. |
| `stop`, `destroy` | Succeed when the machine is already stopped or gone. |
| `list_instances` | Return only machines whose label starts with `jambu-`. Never touch anything else in the account. |
| `WATCHDOG_ACTIONS` | Python, stdlib only, that runs on the machine itself: `provider_stop(ctx)` and `provider_destroy(ctx)` call the provider's API so an idle machine stops even when no laptop is on. |

Raise only the errors in `gpu4agents.core.errors`: `AuthenticationError`, `CapacityUnavailableError`, `ProvisioningError`, `ProviderTimeoutError`, `ProviderError`. The CLI turns them into the usual exit codes.

If the provider has no API for something (no stop, no price list), set that capability to `False` and leave the method raising. Tell the user which features that turns off.

## 4. Connect the account

Ask the user to run this in their own terminal and paste the key into the hidden prompt:

```bash
gpu login --provider <name>
```

Never ask for the key in the chat. `gpu login --check` then exits 0.

## 5. Check without renting

```bash
gpu --json validate
gpu --json offers
gpu providers
```

`offers` must list real GPUs with real prices. Fix the adapter until it does.

## 6. One real run, with the user's yes

Show the user the cheapest offer and ask before renting. Use a small model to keep it cheap (`Qwen/Qwen2.5-0.5B-Instruct` on a 16 to 24 GB GPU). Then:

```bash
gpu --json setup          # rent, boot, serve the model; can take 5 to 15 minutes
gpu --json status         # state ready, endpoint up, watchdog reachable
gpu --json destroy --yes  # with the user's yes
gpu --json ls             # nothing of ours left running
```

If `watchdog.reachable` is false in `status`, the watchdog port is not published: check `endpoints["watchdog"]` in `_to_instance`.

## 7. Tell the user what changed

List the file you wrote, the capabilities you turned off and why, and the result of the test run. To share the adapter with other projects, package it and declare it under the `gpu4agents.providers` entry point group: every `gpu` on that machine then finds it without `plugin:`.
