"""segv: inspect the memory service over its own HTTP API.

The CLI is a pure client: it never touches the seen set (search is
exempt from deduplication by design). The default agent comes from
NAN_AGENT_ID, which the shell exports for the current agent.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Any

DEFAULT_URL = "http://127.0.0.1:9004"
AGENT_ENV = "NAN_AGENT_ID"
MODES = ("hybrid", "raw", "event", "bm25", "dense")


def _base(args: argparse.Namespace) -> str:
    return (args.url or os.environ.get("SEGV_URL") or DEFAULT_URL).rstrip("/")


def _agent(args: argparse.Namespace) -> str:
    agent = (args.agent or os.environ.get(AGENT_ENV) or "").strip()

    if not agent:
        raise SystemExit(f"segv: no agent: set {AGENT_ENV} or pass --agent")

    return agent


def _call(
    args: argparse.Namespace,
    method: str,
    path: str,
    *,
    body: dict[str, Any] | None = None,
) -> Any:
    url = _base(args) + path
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            return json.loads(response.read().decode())

    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")

        raise SystemExit(f"segv: {exc.code} {url}: {detail}") from exc

    except urllib.error.URLError as exc:
        raise SystemExit(f"segv: cannot reach {url}: {exc.reason}") from exc


def _get(args: argparse.Namespace, path: str, **params: Any) -> Any:
    query = urllib.parse.urlencode(
        {key: value for key, value in params.items() if value is not None}
    )

    return _call(args, "GET", f"{path}?{query}" if query else path)


def _clock(value: Any) -> str:
    try:
        stamp = float(value or 0.0)

    except (TypeError, ValueError):
        return "-"

    if stamp <= 0:
        return "-"

    return datetime.fromtimestamp(stamp).strftime("%m-%d %H:%M:%S")


def _clip(text: Any, width: int = 96) -> str:
    line = " ".join(str(text or "").split())

    return line if len(line) <= width else line[: width - 1] + "…"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]

    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    lines = ["  ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True))]

    for row in rows:
        lines.append("  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)))

    return "\n".join(lines)


def _emit(args: argparse.Namespace, payload: Any, render: Any) -> None:
    if getattr(args, "json", False):
        print(json.dumps(payload, ensure_ascii=False, indent=2))

    else:
        print(render())


# ----------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------


def cmd_search(args: argparse.Namespace) -> int:
    agent = _agent(args)
    query = " ".join(args.query).strip()

    if not query:
        raise SystemExit("segv: empty query")

    payload = _call(
        args,
        "POST",
        "/api/recall",
        body={
            "agent_id": agent,
            "query": query,
            "top_k": args.top_k,
            "pool": args.pool,
            "mode": args.mode,
            "rerank": not args.no_ce,
        },
    )
    hits = payload.get("hits") or []

    def render() -> str:
        if not hits:
            return "no hits"

        rows = []

        for hit in hits:
            text = hit.get("text") if args.full else hit.get("event")

            rows.append(
                [
                    f"{float(hit.get('score') or 0.0):.4f}",
                    f"{hit.get('seq')}:{hit.get('ordinal')}",
                    _clock(hit.get("started_at")),
                    _clip(text),
                ]
            )

        if getattr(args, "plain", False):
            return "\n".join("\t".join(row) for row in rows)

        return _table(["SCORE", "LOC", "TIME", "CUE"], rows)

    _emit(args, payload, render)

    return 0


def cmd_grep(args: argparse.Namespace) -> int:
    agent = _agent(args)
    pattern = " ".join(args.pattern).strip()

    if not pattern:
        raise SystemExit("segv: empty pattern")

    payload = _get(
        args,
        f"/api/agents/{urllib.parse.quote(agent, safe='')}/grep",
        pattern=pattern,
        limit=args.limit,
        before=args.before,
        timeout=args.scan_timeout,
        context=args.context,
    )
    hits = payload.get("hits") or []

    def render() -> str:
        if not hits:
            if payload.get("truncated"):
                return (
                    "no matches in the scanned window (budget reached; "
                    f"continue with --before {payload['next_before']})"
                )

            return "no matches"

        lines = [
            f"[{hit['seq']}] {_clock(hit.get('started_at'))}  {hit['snippet']}"
            for hit in hits
        ]

        if payload.get("truncated"):
            lines.append(
                f"(scan budget reached; continue with --before {payload['next_before']})"
            )

        return "\n".join(lines)

    _emit(args, payload, render)

    return 0


def cmd_state(args: argparse.Namespace) -> int:
    agent = _agent(args)
    payload = _get(args, "/api/cache/state", agent_id=agent)

    def render() -> str:
        if not payload.get("enabled", True):
            return "cache disabled"

        pointer = payload.get("pointer")
        location = f"{pointer['seq']}:{pointer['ordinal']}" if pointer else "-"
        lines = [
            f"agent: {payload.get('agent_id')}",
            "  epoch: {}  pointer: {}  version: {}  pending: {}  seen: {}".format(
                payload.get("last_epoch"),
                location,
                payload.get("version"),
                payload.get("pending"),
                payload.get("seen"),
            ),
        ]
        snapshot = payload.get("snapshot") or {}
        lines.append(f"goal: {snapshot.get('goal') or '-'}")

        sections = (
            ("unresolved_items", "text"),
            ("locked_decisions", "text"),
            ("forbidden_behaviors", "text"),
            ("entity_map", "name"),
        )

        for key, label in sections:
            entries = snapshot.get(key) or []
            lines.append(f"{key}:")

            if not entries:
                lines.append("  -")
                continue

            for entry in entries:
                value = entry.get(label) or ""
                extra = entry.get("meaning") if label == "name" else None

                if extra:
                    value = f"{value} — {extra}"

                lines.append(f"  [{entry.get('id')}] {value}")

        return "\n".join(lines)

    _emit(args, payload, render)

    return 0


def cmd_tail(args: argparse.Namespace) -> int:
    agent = _agent(args)
    payload = _get(args, "/api/cache/tail", agent_id=agent, limit=args.limit)
    cues = payload.get("cues") or []

    def render() -> str:
        if not cues:
            return "tail is empty"

        rows = [
            [
                f"{cue.get('seq')}:{cue.get('ordinal')}",
                _clock(cue.get("ts")),
                _clip(cue.get("text")),
            ]
            for cue in cues
        ]

        if getattr(args, "plain", False):
            return "\n".join("\t".join(row) for row in rows)

        return _table(["LOC", "TIME", "CUE"], rows)

    _emit(args, payload, render)

    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    payload = {
        "l0": _get(args, "/api/stats"),
        "fact": _get(args, "/api/fact/stats"),
        "cache": _get(args, "/api/cache/stats"),
    }

    def render() -> str:
        lines: list[str] = []

        for section, values in payload.items():
            lines.append(f"[{section}]")

            for key, value in values.items():
                lines.append(f"  {key}: {value}")

        return "\n".join(lines)

    _emit(args, payload, render)

    return 0


def cmd_turns(args: argparse.Namespace) -> int:
    agent = _agent(args)
    payload = _get(
        args,
        f"/api/agents/{urllib.parse.quote(agent, safe='')}/turns",
        limit=args.limit,
        before=args.before,
    )
    turns = payload.get("turns") or []

    def render() -> str:
        if not turns:
            return "no turns"

        rows = [
            [
                str(turn.get("seq")),
                _clock(turn.get("started_at")),
                str((turn.get("meta") or {}).get("origin") or "-"),
                str((turn.get("meta") or {}).get("epoch")),
                _clip((turn.get("raw") or {}).get("reply") or turn.get("summary")),
            ]
            for turn in turns
        ]

        if getattr(args, "plain", False):
            return "\n".join("\t".join(row) for row in rows)

        return _table(["SEQ", "TIME", "ORIGIN", "EPOCH", "REPLY"], rows)

    _emit(args, payload, render)

    return 0


def cmd_turn(args: argparse.Namespace) -> int:
    agent = _agent(args)
    payload = _get(
        args, f"/api/turns/{urllib.parse.quote(agent, safe='')}/{args.seq}"
    )

    print(json.dumps(payload, ensure_ascii=False, indent=2))

    return 0


# ----------------------------------------------------------------------
# Parser
# ----------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)

    common.add_argument("--url", default=None, help="service URL (or SEGV_URL)")
    common.add_argument("--agent", default=None, help=f"agent id (or {AGENT_ENV})")
    common.add_argument("--json", action="store_true", help="print the raw JSON")
    common.add_argument("--table", action="store_true", help="aligned table (default)")
    common.add_argument("--plain", action="store_true", help="one item per line")
    common.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout")
    common.add_argument("-h", "--help", "-help", action="help", help="show this help")

    parser = argparse.ArgumentParser(
        prog="segv",
        add_help=False,
        description="Inspect SEGV memory: search facts, snapshot, tail, turns.",
    )
    parser.add_argument("-h", "--help", "-help", action="help", help="show this help")

    sub = parser.add_subparsers(dest="command")

    search = sub.add_parser(
        "search", parents=[common], add_help=False, help="rank fact cues"
    )
    search.add_argument("query", nargs="+", help="query text")
    search.add_argument("--mode", choices=MODES, default="hybrid")
    search.add_argument("--top-k", type=int, default=10)
    search.add_argument("--pool", type=int, default=200)
    search.add_argument("--no-ce", action="store_true", help="skip the reranker")
    search.add_argument("--full", action="store_true", help="show the chunk text")
    search.set_defaults(handler=cmd_search)

    grep = sub.add_parser(
        "grep", parents=[common], add_help=False, help="scan raw turns, newest first"
    )
    grep.add_argument("pattern", nargs="+", help="substring to find (case-insensitive)")
    grep.add_argument("--limit", type=int, default=5)
    grep.add_argument("--before", type=int, default=None, help="scan turns older than SEQ")
    grep.add_argument(
        "--scan-timeout",
        type=float,
        default=3.0,
        help="wall-clock budget for the raw scan in seconds",
    )
    grep.add_argument(
        "--context", type=int, default=48, help="snippet chars either side of a match"
    )
    grep.set_defaults(handler=cmd_grep)

    state = sub.add_parser(
        "state", parents=[common], add_help=False, help="show the snapshot"
    )
    state.set_defaults(handler=cmd_state)

    tail = sub.add_parser(
        "tail", parents=[common], add_help=False, help="show uncompressed cues"
    )
    tail.add_argument("--limit", type=int, default=20)
    tail.set_defaults(handler=cmd_tail)

    stats = sub.add_parser(
        "stats", parents=[common], add_help=False, help="show service counters"
    )
    stats.set_defaults(handler=cmd_stats)

    turns = sub.add_parser(
        "turns", parents=[common], add_help=False, help="list recent turns"
    )
    turns.add_argument("--limit", type=int, default=20)
    turns.add_argument("--before", type=int, default=None)
    turns.set_defaults(handler=cmd_turns)

    turn = sub.add_parser(
        "turn", parents=[common], add_help=False, help="show one turn as JSON"
    )
    turn.add_argument("seq", type=int)
    turn.set_defaults(handler=cmd_turn)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()

    if not args or args[0] in ("help", "-help", "--help", "-h"):
        parser.print_help()

        return 0

    parsed = parser.parse_args(args)

    if parsed.command is None:
        parser.print_help()

        return 0

    return int(parsed.handler(parsed))


if __name__ == "__main__":
    raise SystemExit(main())
