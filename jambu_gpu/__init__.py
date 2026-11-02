"""Compatibility alias: this package was renamed to gpu4agents in 0.5.0.

Kept so sweep timers installed by jambu-gpu (``python -m jambu_gpu sweep --apply``)
keep working after an upgrade. New code imports gpu4agents.
"""

from gpu4agents import __version__

__all__ = ["__version__"]
