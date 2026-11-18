"""
Listener startup policy: a configured port is a contract, not a
preference.

The core never falls back to another port. When a configured port is
taken, boot stops with a message naming it instead of letting a stale
instance keep answering -- or a second instance half-start.
"""

from __future__ import annotations

import errno
import socket


class PortUnavailableError(RuntimeError):
    """A configured listener port cannot be bound by this boot."""


def ensure_port_available(host: str, port: int, label: str) -> None:
    """
    Raise PortUnavailableError when ``host:port`` cannot be bound.

    The check runs before any component starts, so a conflict aborts
    cleanly instead of after agents and autostart services are already
    running. Uvicorn still does the real bind later; this only closes
    the window where a foreign process would answer in its place.
    """
    try:
        addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)

    except OSError as exc:
        raise PortUnavailableError(f"{label} address {host}:{port} is not usable: {exc}") from exc

    checked: set[tuple[int, object]] = set()

    for family, _, _, _, sockaddr in addresses:
        if (family, sockaddr) in checked:
            continue

        checked.add((family, sockaddr))

        with socket.socket(family, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            try:
                probe.bind(sockaddr)

            except OSError as exc:
                if exc.errno == errno.EADDRINUSE:
                    raise PortUnavailableError(
                        f"{label} port {host}:{port} is already in use; "
                        "stop the process that owns it (another NAN instance?) "
                        "or change the port in config/nan.yaml"
                    ) from exc

                raise PortUnavailableError(f"{label} cannot bind {host}:{port}: {exc}") from exc
