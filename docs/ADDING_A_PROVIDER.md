# Adding a provider

Adding a provider means adding `gpu4agents/providers/<name>/` and one line in the
registry. **No lifecycle, execution or CLI code changes.** If you find yourself
editing `core/`, the contract is leaking.

## 1. Implement the contract

```python
# gpu4agents/providers/runpod/provider.py
from ...core.models import ProviderCapabilities
from ..base import GPUProvider, WatchdogSpec


class RunpodProvider(GPUProvider):
    name = "runpod"
    display_name = "RunPod"
    required_credentials = ["RUNPOD_API_KEY"]
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

    def validate(self, config): ...
    def setup(self, spec): ...
    def start(self, instance): ...
    def run(self, instance, workload): ...
    def status(self, instance): ...
    def stop(self, instance): ...
    def destroy(self, instance): ...
    def logs(self, instance): ...
```

Suggested layout, mirroring the Vast adapter:

```
providers/runpod/
├── provider.py   the contract implementation
├── client.py     raw HTTP; the ONLY place that knows the wire format
├── mapper.py     payloads <-> normalized models, state mapping
└── config.py     provider.options schema and constants
```

## 2. Declare capabilities honestly

The core never branches on `provider.name` — only on `provider.capabilities`. Set
`stop_eliminates_billing=False` if a stopped instance still costs money; the policy
then knows it may need to escalate to destroy, and `lifecycle.stopped_ttl` applies.

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

Report a machine that will not boot as `failed` rather than leaving it in
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
        env={"RUNPOD_API_KEY": self.credentials.get("RUNPOD_API_KEY") or ""},
        context={"provider": self.name},
    )
```

The instance id may not exist when the boot script is generated. Either resolve it
from the container's own environment (as the Vast adapter does with
`VAST_CONTAINERLABEL`), or rely on the CLI's `/identify` call — support both if you
can, so the guard works even if the CLI never reconnects.

Return `None` only if the provider genuinely cannot self-terminate; users then have
to run `gpu guard` externally, and `validate` will warn.

## 5. Publish the ports

`ComputeSpec.ports` lists the container ports the core needs published. Map them to
`Instance.endpoints` under the names `model` and `watchdog`. Without a published
watchdog port the remote guard is unreachable.

## 6. Register it

```python
# providers/registry.py
def load_builtin_providers():
    from .vast.provider import VastProvider
    from .runpod.provider import RunpodProvider

    registry.register(VastProvider)
    registry.register(RunpodProvider)
    return registry
```

## 7. Test it

Copy `tests/test_vast_provider.py` and swap the mocked payloads. Then check the
adapter really is swappable — `tests/test_engine.py` drives the whole engine
through `tests/fakes.py::FakeProvider` with zero core changes, and
`test_the_core_never_branches_on_a_provider_name` fails the build if `core/` learns
a provider's name.
