"""gpu dashboard: a read-only, host-wide control-plane web UI.

A stdlib loopback HTTP server (sibling to ``gpu gateway``) that shows every
GPU, proxy and service this host can see and watches their reachability.
"""

from __future__ import annotations

from .collect import project_snapshots

__all__ = ["project_snapshots"]
