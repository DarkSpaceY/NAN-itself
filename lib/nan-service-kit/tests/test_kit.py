"""
The kit's own behaviour: where the defaults come from, and the calls a
link makes over a service's life.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from loguru import logger
from nan_service_kit import HubLink, ServiceConfig, Surface, setup_logging
from nan_service_kit.config import HUB_PORT, read_config
from nan_service_kit.testing import StubHub


def _write(tmp_path: Path, name: str, text: str) -> Path:
    home = tmp_path / name

    home.mkdir()

    path = home / "config.yaml"

    path.write_text(text, encoding="utf-8")

    return path


def test_defaults_fill_what_the_file_omits(tmp_path, monkeypatch):
    monkeypatch.setenv("NAN_INJECTION_PORT", "9999")

    config = read_config(ServiceConfig, _write(tmp_path, "weather", "port: 9001\n"))

    assert config.service_id == "weather"
    assert config.hub_url == "http://127.0.0.1:9999"
    assert config.mode == "inject"
    assert config.surface is None
    assert config.home == tmp_path / "weather"
    assert config.logs_dir == tmp_path / "weather" / "logs"


def test_hub_url_without_the_core_env(tmp_path, monkeypatch):
    monkeypatch.delenv("NAN_INJECTION_PORT", raising=False)

    config = read_config(ServiceConfig, _write(tmp_path, "weather", "port: 9001\n"))

    assert config.hub_url == f"http://127.0.0.1:{HUB_PORT}"


def test_the_file_wins(tmp_path):
    path = _write(
        tmp_path,
        "weather",
        "service_id: forecast\nport: 9001\nmode: archive\n"
        "hub_url: http://127.0.0.1:7000\nsurface:\n  title: Weather\n",
    )

    config = read_config(ServiceConfig, path)

    assert (config.service_id, config.mode, config.hub_url) == (
        "forecast",
        "archive",
        "http://127.0.0.1:7000",
    )
    assert isinstance(config.surface, Surface)
    assert config.surface.title == "Weather"


def test_link_registers_announces_and_unregisters(tmp_path):
    path = _write(tmp_path, "weather", "port: 9001\nsurface:\n  title: Weather\n")

    config = read_config(ServiceConfig, path)

    stub = StubHub()

    async def run() -> None:
        link = HubLink(config, transport=stub.transport())

        await link.register()

        await link.beat()

        await link.stop()

    asyncio.run(run())

    assert stub.posts[0] == (
        "/register",
        {
            "service_id": "weather",
            "endpoint": "http://127.0.0.1:9001/turn",
            "mode": "inject",
        },
    )

    assert stub.bodies("/surfaces") == [
        {
            "service_id": "weather",
            "url": "http://127.0.0.1:9001/",
            "title": "Weather",
        }
    ]

    assert stub.deletes == ["/surfaces/weather"]


def test_a_forgotten_service_registers_again(tmp_path):
    config = read_config(ServiceConfig, _write(tmp_path, "weather", "port: 9001\n"))

    stub = StubHub()

    stub.heartbeat = "unknown"

    async def run() -> None:
        link = HubLink(config, transport=stub.transport())

        await link.beat()

        await link.stop()

    asyncio.run(run())

    assert len(stub.bodies("/register")) == 1


def test_logging_lands_under_logs(tmp_path):
    config = read_config(ServiceConfig, _write(tmp_path, "weather", "port: 9001\n"))

    setup_logging(config)

    logger.info("hello")

    assert "hello" in (config.logs_dir / "weather.log").read_text(encoding="utf-8")
