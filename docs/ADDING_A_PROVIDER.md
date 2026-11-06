# Adding a provider

Vast.ai, RunPod and Lambda are built in. Any other provider is a class that implements
the contract in `gpu4agents/providers/base.py`. Lifecycle, execution and CLI code
stay the same; if you find yourself editing `core/`, the contract is leaking.

An adapter reaches `gpu` in one of three ways:

| Where it lives | How `gpu` finds it | Use it for |
| --- | --- | --- |
| A file in the project | `provider.plugin: ./gpu_providers/acme.py:AcmeProvider` in gpu.yaml | one project, or an agent writing an adapter on the spot |
| An installed package | an entry point in the `gpu4agents.providers` group | sharing an adapter across projects or publishing it |
| This repository | `providers/<name>/` plus a line in `providers/registry.py` | a provider that ships with gpu4agents |

`gpu providers new acme` writes the skeleton for the first route, with every
method that needs the provider's API marked. `gpu init --provider acme` writes a
gpu.yaml that selects it. Agents can follow [PROVIDERS.md](../PROVIDERS.md),
which walks through the same steps.

An installed package declares its adapter like this:

```toml
# pyproject.toml of the package
[project.entry-points."gpu4agents.providers"]
acme = "acme_gpu.provider:AcmeProvider"
```

The rest of this guide applies to all three routes.

## 1. Implement the contract

```python
# gpu4agents/providers/acme/provider.py
from ...core.models import ProviderCapabilities
from ..base import GPUProvider, WatchdogSpec


class AcmeProvider(GPUProvider):
    name = "acme"
    display_name = "Acme GPU"
    required_credentials = ["ACME_API_KEY"]
    credentials_url = "https://console.acme.example/api-keys"
    capabilities = ProviderCapabilities(
        stop=True,
        destroy=True,
        ssh=True,
        public_ports=True,
        gpu_selection=True,
        pricing_query=True,
        remote_exec=True,
        logs=True,
        stop_eliminates_billing=True,
    )

    @classmethod
    def verify_credentials(cls, values): ...   # `gpu login` calls it before saving
    def validate(self, config): ...
    def setup(self, spec): ...
    def start(self, instance): ...
    def run(self, instance, workload): ...
    def status(self, instance): ...
    def stop(self, instance): ...
    def destroy(self, instance): ...
    def logs(self, instance): ...
```

`gpu login --provider acme` prompts for each name in `required_credentials`,
passes the values to `verify_credentials`, and saves them in
`~/.gpu/credentials` only if it returns. Raise `AuthenticationError` for a
rejected key, and return only fields that are safe to print, such as
`{"balance_usd": 12.5}`.

For an adapter that ships in this repository, mirror the layout of `vast/`, `runpod/` or `lambdacloud/`:

```
providers/acme/
├── provider.py   the contract implementation
├── client.py     raw HTTP; the ONLY place that knows the wire format
├── mapper.py     payloads <-> normalized models, state mapping
└── config.py     provider.options schema and constants
```

## 2. Declare capabilities honestly

The core branches on `provider.capabilities`, never on `provider.name`. Set
`stop_eliminates_billing=False` if a stopped instance still costs money; the policy
then knows it may need to escalate to destroy, and `lifecycle.stopped_ttl` applies.

Set `boot_progress=True` only if, while the machine boots, `status().message` shows the image
pull moving (a new line per layer, as Vast.ai does). The core then gives up on a pull that
stops moving for `health.pull_stall_timeout`; without it, only `health.boot_timeout` applies.

Set `list_instances=True` and implement `list_instances()` if the API can enumerate
the account. Return only instances whose label starts with `JAMBU_LABEL_PREFIX`
(`core/models.py`); `gpu ls` and `gpu sweep` are built on it and stay unavailable
otherwise.

## 3. Normalize state and errors

Map every provider status onto `InstanceState`
(`provisioning / starting / ready / busy / idle / stopped / failed / destroyed`),
and raise only the normalized errors from `core/errors.py`:
`AuthenticationError`, `CapacityUnavailableError`, `ProvisioningError`,
`ProviderTimeoutError`, `UnsupportedCapabilityError`. CLI behaviour must be
identical across providers.

Report a machine that will not boot as `failed`, so it does not sit in
`starting`: `setup` then destroys it at once and rents another machine
(`compute.host_attempts`). Set `Offer.host_id` and `Instance.host_id` to the
physical machine when the API exposes it, and leave out offers whose `host_id` is
in `ComputeSpec.exclude_hosts`, or `setup` may rent the same bad machine again.

## 4. Supply the watchdog actions

The lifecycle *policy* is the core's; the adapter contributes only the two calls
that shut the machine down from inside itself:

```python
WATCHDOG_ACTIONS = '''
def provider_stop(ctx):
    ...   # stdlib only: urllib, json, os

def provider_destroy(ctx):
    ...
'''

def watchdog_spec(self, spec_or_instance=None):
    return WatchdogSpec(
        actions_source=WATCHDOG_ACTIONS,
        env={"ACME_API_KEY": self.credentials.get("ACME_API_KEY") or ""},
        context={"provider": self.name},
    )
```

The instance id may not exist when the boot script is generated. Either resolve it
from the container's own environment (as the Vast adapter does with
`VAST_CONTAINERLABEL`), or rely on the CLI's `/identify` call. Support both if you
can, so the guard works even if the CLI never reconnects.

Return `None` only if the provider genuinely cannot self-terminate; users then have
to run `gpu guard` externally, and `validate` will warn.

## 5. Publish the ports

`ComputeSpec.ports` lists the container ports the core needs published. Map them to
`Instance.endpoints` under the names `model` and `watchdog`. Without a published
watchdog port the remote guard is unreachable.

## 6. Register it (built-in adapters only)

A plugin file or an installed package needs no registration. An adapter that
ships in this repository is added here:

```python
# providers/registry.py
def load_builtin_providers():
    from .vast.provider import VastProvider
    from .acme.provider import AcmeProvider

    registry.register(VastProvider)
    registry.register(AcmeProvider)
    return registry
```

## 7. Test it

Copy `tests/test_vast_provider.py` and swap the mocked payloads. `tests/test_engine.py`
drives the whole engine through `tests/fakes.py::FakeProvider` with no core
changes, and `test_the_core_never_branches_on_a_provider_name` fails the build if
`core/` learns a provider's name.

Then test against the real API without renting (`gpu validate`, `gpu offers`),
and once with a small model (`gpu setup`, `gpu status`, `gpu destroy --yes`).
`status` must show `watchdog.reachable: true`; if it does not, the watchdog port
is not published.
