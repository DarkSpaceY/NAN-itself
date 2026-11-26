#!/usr/bin/env python3
"""
Agent-facing CLI for the mcp gateway.

    mcp list [--full | --json]
    mcp schema <server> <tool> [--json]
    mcp call <server> <tool> ['<json-arguments>']

Success writes the tool result to stdout. Failure writes one JSON
line to stderr and exits non-zero:

    2  usage: bad arguments or malformed JSON
    3  unavailable: unknown server/tool, or the gateway is down
    4  tool failure: the tool raised, timed out, or returned an error

The gateway's port is read from the sibling config.yaml, so the CLI
and the service never disagree about the endpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent

EXIT_USAGE = 2
EXIT_UNAVAILABLE = 3
EXIT_TOOL = 4

_TIMEOUT = 10.0
_CALL_TIMEOUT = 180.0

# Loopback calls must never go through an HTTP proxy: a proxy would
# turn "gateway is down" into an unrelated 502.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class CliError(Exception):
    """A failure that already has a wire code and an exit status."""

    def __init__(self, code: str, message: str, status: int) -> None:
        super().__init__(message)

        self.code = code
        self.message = message
        self.status = status


class UsageError(Exception):
    """Bad command line: reported as invalid_arguments."""


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise UsageError(message)


def _port() -> int:
    try:
        text = (HERE / "config.yaml").read_text(encoding="utf-8")

    except OSError as exc:
        raise CliError(
            "service_unavailable", f"cannot read config.yaml: {exc}", EXIT_UNAVAILABLE
        ) from exc

    for line in text.splitlines():
        if line.startswith("port:"):
            try:
                return int(line.split(":", 1)[1].strip())

            except ValueError:
                break

    raise CliError("service_unavailable", "config.yaml has no usable port", EXIT_UNAVAILABLE)


def _request(
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
    *,
    timeout: float = _TIMEOUT,
) -> Any:
    data = None if body is None else json.dumps(body).encode("utf-8")

    request = urllib.request.Request(
        f"http://127.0.0.1:{_port()}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data is not None else {},
    )

    try:
        with _OPENER.open(request, timeout=timeout) as response:
            raw = response.read()

    except urllib.error.HTTPError as exc:
        raw = exc.read()

        try:
            payload = json.loads(raw)

        except (json.JSONDecodeError, UnicodeDecodeError):
            payload = None

        if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
            error = payload["error"]

            code = str(error.get("code") or "tool_error")

            raise CliError(code, str(error.get("message") or ""), _status_for(code)) from exc

        raise CliError(
            "service_unavailable", f"HTTP {exc.code} from the mcp gateway", EXIT_UNAVAILABLE
        ) from exc

    except TimeoutError as exc:
        raise CliError("timeout", f"gateway call timed out: {exc}", EXIT_TOOL) from exc

    except (urllib.error.URLError, OSError) as exc:
        raise CliError(
            "service_unavailable", f"cannot reach the mcp gateway: {exc}", EXIT_UNAVAILABLE
        ) from exc

    try:
        return json.loads(raw)

    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise CliError(
            "service_unavailable", f"gateway returned invalid JSON: {exc}", EXIT_UNAVAILABLE
        ) from exc


def _get(path: str) -> Any:
    return _request("GET", path)


def _post(path: str, body: dict[str, Any]) -> Any:
    return _request("POST", path, body, timeout=_CALL_TIMEOUT)


def _status_for(code: str) -> int:
    if code in ("unknown_server", "unknown_tool", "server_unavailable"):
        return EXIT_UNAVAILABLE

    if code == "invalid_arguments":
        return EXIT_USAGE

    return EXIT_TOOL


def _oneline(value: Any) -> str:
    return " ".join(str(value or "").split())


def _summary(value: Any, limit: int = 160) -> str:
    """The first paragraph, flattened and capped; details stay in the table."""
    text = _oneline(str(value or "").split("\n\n", 1)[0])

    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"

    return text


def _type_name(node: Any) -> str:
    if not isinstance(node, dict):
        return "any"

    if "type" in node:
        value = node["type"]

        if isinstance(value, list):
            return "|".join(_type_name({"type": item}) for item in value)

        return str(value)

    for key in ("anyOf", "oneOf"):
        options = node.get(key)

        if isinstance(options, list) and options:
            return "|".join(_type_name(option) for option in options)

    if "enum" in node:
        return "enum"

    return "any"


def _render_list(data: Any, full: bool) -> None:
    servers = data.get("servers") or []

    for entry in sorted(servers, key=lambda item: str(item.get("name", ""))):
        name = str(entry.get("name", ""))

        tools = entry.get("tools") or []

        if not full:
            names = [str(tool.get("name", "")) for tool in tools]

            print(f"{name}: {', '.join(names) if names else '(no tools)'}")

            continue

        print(f"{name}:")

        if not tools:
            print("  (no tools)")

        for tool in tools:
            description = _summary(tool.get("description", ""))

            suffix = f" — {description}" if description else ""

            print(f"  {tool.get('name', '')}{suffix}")

    starting = [str(name) for name in data.get("starting") or []]

    for name in sorted(starting):
        print(f"starting: {name}")

    unavailable = data.get("unavailable") or {}

    for name in sorted(unavailable):
        print(f"unavailable: {name} ({_oneline(unavailable[name])})")

    if not servers and not starting and not unavailable:
        print("no MCP servers configured")


def _render_schema(data: Any, raw: bool) -> None:
    if not data.get("ok"):
        error = data.get("error")

        if isinstance(error, dict):
            code = str(error.get("code") or "unknown_tool")

            raise CliError(code, str(error.get("message") or ""), _status_for(code))

        raise CliError(
            "unknown_tool", str(error or "schema unavailable"), EXIT_UNAVAILABLE
        )

    schema = data.get("schema") or {}

    if raw:
        print(json.dumps(schema, ensure_ascii=False, indent=2))

        return

    header = f"{data.get('server', '')}.{data.get('tool', '')}"

    description = _summary(data.get("description", ""))

    if description:
        header += f" — {description}"

    print(header)

    properties = schema.get("properties") or {}

    if not isinstance(properties, dict) or not properties:
        print("  (no arguments)")

        return

    required = set(schema.get("required") or [])

    rows: list[list[str]] = []

    for name, raw_property in properties.items():
        prop = raw_property if isinstance(raw_property, dict) else {}

        rows.append(
            [
                str(name),
                _type_name(prop),
                "yes" if name in required else "no",
                json.dumps(prop["default"], ensure_ascii=False) if "default" in prop else "-",
                _oneline(prop.get("description", "")) or "-",
            ]
        )

    headers = ["name", "type", "required", "default", "description"]

    widths = [len(header) for header in headers]

    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    print("  " + "  ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))

    for row in rows:
        print("  " + "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))


def _render_call(data: Any) -> None:
    if not data.get("ok"):
        error = data.get("error")

        if isinstance(error, dict):
            code = str(error.get("code") or "tool_error")

            raise CliError(code, str(error.get("message") or ""), _status_for(code))

        raise CliError("tool_error", str(error or "tool call failed"), EXIT_TOOL)

    result = data.get("result")

    if isinstance(result, str):
        if result:
            print(result)

        return

    print(json.dumps(result, ensure_ascii=False, indent=2))


def _parser() -> Parser:
    parser = Parser(
        prog="mcp",
        description="Call MCP tools from the shell.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exit codes:\n"
            "  0  success\n"
            "  2  usage error (bad arguments or malformed JSON)\n"
            "  3  unavailable (unknown server/tool, or the gateway is down)\n"
            "  4  tool failure (raised, timed out, or returned an error)\n\n"
            "Examples:\n"
            "  mcp list\n"
            "  mcp schema search web_search\n"
            "  mcp call search web_search '{\"query\":\"uv python\",\"max_results\":3}'\n"
        ),
    )

    sub = parser.add_subparsers(dest="command", required=True)

    list_parser = sub.add_parser(
        "list", help="Connected servers, ones still starting, and unavailable ones."
    )

    list_mode = list_parser.add_mutually_exclusive_group()

    list_mode.add_argument("--full", action="store_true", help="Include tool descriptions.")

    list_mode.add_argument("--json", action="store_true", help="Print the raw response.")

    schema_parser = sub.add_parser("schema", help="Show one tool's arguments.")

    schema_parser.add_argument("server")

    schema_parser.add_argument("tool")

    schema_parser.add_argument("--json", action="store_true", help="Print the raw JSON schema.")

    call_parser = sub.add_parser("call", help="Call a tool and print its result.")

    call_parser.add_argument("server")

    call_parser.add_argument("tool")

    call_parser.add_argument(
        "arguments", nargs="?", default="{}", help="JSON object of arguments (default: {})."
    )

    return parser


def _fail(code: str, message: str, status: int) -> int:
    print(json.dumps({"error": {"code": code, "message": message}}, ensure_ascii=False), file=sys.stderr)

    return status


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)

        if args.command == "list":
            data = _get("/tools")

            if args.json:
                print(json.dumps(data, ensure_ascii=False, indent=2))

            else:
                _render_list(data, args.full)

        elif args.command == "schema":
            path = f"/schema/{urllib.parse.quote(args.server, safe='')}/{urllib.parse.quote(args.tool, safe='')}"

            _render_schema(_get(path), args.json)

        elif args.command == "call":
            try:
                arguments = json.loads(args.arguments)

            except json.JSONDecodeError as exc:
                raise UsageError(f"arguments must be valid JSON: {exc}") from exc

            if not isinstance(arguments, dict):
                raise UsageError("arguments must be a JSON object")

            data = _post(
                "/call", {"server": args.server, "tool": args.tool, "arguments": arguments}
            )

            _render_call(data)

        return 0

    except UsageError as exc:
        return _fail("invalid_arguments", str(exc), EXIT_USAGE)

    except CliError as exc:
        return _fail(exc.code, exc.message, exc.status)


if __name__ == "__main__":
    sys.exit(main())
