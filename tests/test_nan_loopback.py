"""The loopback policy: inbound entries and outbound destinations."""

from __future__ import annotations

from nan.utils.loopback import entry_allowed, is_local_asset, is_loopback_host, is_loopback_url


def test_loopback_hosts() -> None:
    for host in ("127.0.0.1", "127.1.2.3", "localhost", "LOCALHOST", "::1", "[::1]"):
        assert is_loopback_host(host), host

    for host in ("evil.com", "127.0.0.1.evil.com", "8.8.8.8", "0.0.0.0", ""):
        assert not is_loopback_host(host), host


def test_loopback_urls() -> None:
    for url in (
        "http://127.0.0.1:9002/ui",
        "http://localhost:9001/turn",
        "https://[::1]:9000/ui",
    ):
        assert is_loopback_url(url), url

    for url in (
        "https://evil.com/ui",
        "http://127.0.0.1.evil.com/ui",
        "file:///etc/passwd",
        "//127.0.0.1/ui",
        "not a url",
    ):
        assert not is_loopback_url(url), url


def test_local_asset() -> None:
    assert is_local_asset("/icons/record.svg")
    assert is_local_asset("http://127.0.0.1:9002/icon.svg")

    assert not is_local_asset("https://cdn.example.com/icon.svg")
    assert not is_local_asset("icon.svg")

    # protocol-relative urls are not paths: //evil.com/x resolves
    # against the page's scheme to a foreign host
    assert not is_local_asset("//evil.com/icon.svg")
    assert not is_local_asset("//127.0.0.1:9002/icon.svg")


def test_entry_guard() -> None:
    headers = {"host": "127.0.0.1:8765"}

    assert entry_allowed(headers, 8765)

    # native clients send no Origin
    assert entry_allowed({**headers, "origin": "http://127.0.0.1:8765"}, 8765)

    # DNS rebinding: a foreign Host on a loopback socket
    assert not entry_allowed({"host": "evil.com:8765"}, 8765)

    # hostile page: foreign Origin
    assert not entry_allowed({**headers, "origin": "http://evil.com"}, 8765)

    # a sandboxed panel has an opaque origin, which the browser
    # sends as "null"
    assert not entry_allowed({**headers, "origin": "null"}, 8765)

    # wrong port on either side
    assert not entry_allowed({"host": "127.0.0.1:9999"}, 8765)
    assert not entry_allowed({**headers, "origin": "http://127.0.0.1:9999"}, 8765)

    # a missing Host never passes
    assert not entry_allowed({}, 8765)

    # malformed ports are rejected, not raised (a ValueError in
    # the handler would be a 500 instead of a 403)
    assert not entry_allowed({"host": "127.0.0.1:notaport"}, 8765)
    assert not entry_allowed({**headers, "origin": "http://127.0.0.1:notaport"}, 8765)
