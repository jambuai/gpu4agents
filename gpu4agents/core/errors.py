"""Normalized error model (spec section 22).

Provider adapters MUST translate their native failures into these types so the
CLI behaves identically regardless of which provider is configured.
"""

from __future__ import annotations

from typing import Any


class JambuError(Exception):
    """Base class for every error raised by the CLI."""

    exit_code = 1


class ConfigError(JambuError):
    """jambu.yaml (or legacy config.yml) is missing, malformed or semantically invalid."""

    exit_code = 2


class StateError(JambuError):
    """Local runtime state is missing or inconsistent."""

    exit_code = 3


class LockBusy(StateError):
    """Another gpu command (usually a setup, for minutes) holds this project's lock."""

    exit_code = 12


class ProviderError(JambuError):
    """Base class for every provider-originated failure."""

    exit_code = 4


class AuthenticationError(ProviderError):
    exit_code = 5


class CapacityUnavailableError(ProviderError):
    exit_code = 6


class ProvisioningError(ProviderError):
    exit_code = 7


class ProviderTimeoutError(ProviderError):
    exit_code = 8


class UnsupportedCapabilityError(ProviderError):
    exit_code = 9


class BudgetExceededError(JambuError):
    exit_code = 10


class RuntimeStartupError(JambuError):
    exit_code = 11


class HostFailure(RuntimeStartupError):
    """The machine failed, not the config: another host may well succeed.

    Raised while an instance boots, before its container runs: the provider
    reports it failed (an image pull that never finished, a registry the host
    cannot reach), it vanished, or it never reached running within
    ``health.boot_timeout``. A model server that dies once the container runs
    is a plain RuntimeStartupError - the same config would fail anywhere.
    """

    def __init__(self, message: str, reason: str = "", instance: Any = None) -> None:
        super().__init__(message)
        # Short cause for one-line output, e.g. "image pull timed out after 15m".
        self.reason = reason or message
        # The models.Instance that failed, so its machine can be excluded.
        self.instance = instance


class WorkloadFailed(JambuError):
    """The user workload exited non-zero. Carries the original exit code."""

    def __init__(self, message: str, returncode: int) -> None:
        super().__init__(message)
        self.returncode = returncode
        self.exit_code = returncode or 1
