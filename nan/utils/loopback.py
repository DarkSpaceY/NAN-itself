"""
Loopback policy: the one rule that keeps the local chain local.

Two directions, one rule each:

    inbound   Host must be a loopback name and, when present, this
              port; a browser Origin is allowed only when it is
              loopback on this port. Native clients (curl, local ws
              tools) send no Origin and pass.
    outbound  every URL the core calls (service endpoints) or hands
              to the desktop (panel url, icon asset) is loopback.
"""

from __future__ import annotations

import ipaddress
from typing import Any
from urllib.parse import urlsplit

_LOOPBACK_NAMES = {"localhost", "::1"}


def is_loopback_host(host: str) -> bool:
    """True for localhost, ::1 and IP addresses in 127.0.0.0/8."""
    name = host.strip().lower().strip("[]")

    if name in _LOOPBACK_NAMES:
        return True

    try:
        return ipaddress.ip_address(name).is_loopback

    except ValueError:
        return False


def is_loopback_url(url: str) -> bool:
    """An absolute http(s) url whose host is loopback."""
    parts = urlsplit(url.strip())

    return (
        parts.scheme in {"http", "https"}
        and parts.hostname is not None
        and is_loopback_host(parts.hostname)
    )


def is_local_asset(value: str) -> bool:
    """An icon reference: a local path or a loopback url."""
    # "//evil.com/x" is a protocol-relative URL, not a path: it
    # resolves to the page's scheme and a foreign host.
    if value.startswith("//"):
        return False

    return value.startswith("/") or is_loopback_url(value)


def entry_allowed(headers: Any, port: int) -> bool:
    """
    The inbound rule: loopback Host on this port, and any Origin
    must be loopback on this port too.
    """
    host = urlsplit("//" + str(headers.get("host") or ""))

    if host.hostname is None or not is_loopback_host(host.hostname):
        return False

    # .port raises on a malformed authority ("127.0.0.1:abc").
    try:
        host_port = host.port
    except ValueError:
        return False

    if host_port is not None and host_port != port:
        return False

    origin = headers.get("origin")

    if origin is None:
        return True

    source = urlsplit(str(origin))

    try:
        source_port = source.port
    except ValueError:
        return False

    return (
        source.scheme in {"http", "https"}
        and source.hostname is not None
        and is_loopback_host(source.hostname)
        and source_port == port
    )
