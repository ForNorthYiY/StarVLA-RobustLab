"""Route Python DNS lookups through a chosen systemd-resolved interface.

This is an opt-in workaround for hosts where a broken TUN interface owns the
global DNS route. Enable it only for one process by adding this directory to
PYTHONPATH and setting PHYSICAL_DNS_INTERFACE (for example, ``ens1f0``).
"""

from __future__ import annotations

import os
import re
import socket
import subprocess
from functools import lru_cache
from ipaddress import ip_address


_original_getaddrinfo = socket.getaddrinfo
_interface = os.environ.get("PHYSICAL_DNS_INTERFACE")
_ipv4_pattern = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


@lru_cache(maxsize=256)
def _resolve_on_interface(host: str) -> str | None:
    if not _interface or not isinstance(host, str):
        return None
    try:
        ip_address(host)
        return host
    except ValueError:
        pass
    result = subprocess.run(
        ["resolvectl", "query", "-i", _interface, "-4", host],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        return None
    match = _ipv4_pattern.search(result.stdout)
    return match.group(0) if match else None


def _getaddrinfo(host, port, *args, **kwargs):
    resolved = _resolve_on_interface(host)
    return _original_getaddrinfo(resolved or host, port, *args, **kwargs)


if _interface:
    socket.getaddrinfo = _getaddrinfo
