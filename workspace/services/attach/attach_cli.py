"""
`attach`: send one file and one request to the attach service.

    attach notes/paper.txt "extract the method in three sentences"
    attach shots/plot.png "what does the y-axis measure?"

Text and images take the same form. Only for content that actually has
to be read: a small file, a known line range or a grep hit is cheaper
through the shell.

The file must sit inside the workspace. A text file holds at most
128,000 characters and an image at most 8 MB; anything larger is
refused with `too_large` rather than truncated. The answer is printed
as plain text (`--json` keeps the envelope: kind, path, model,
elapsed_s, plus chars for text or mime/bytes for an image).

Exit codes: 0 success, 2 usage error, 3 file too large, 4 service or
model error, 5 the file's capability is not configured.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any

DEFAULT_URL = "http://127.0.0.1:9006"
URL_ENV = "ATTACH_URL"


def _post(url: str, payload: dict[str, Any], timeout: float) -> tuple[int, dict[str, Any]]:
    """One JSON request. Local, so the machine's proxy is deliberately skipped."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    try:
        with opener.open(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode())

    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")

        try:
            return exc.code, json.loads(body)

        except ValueError:
            return exc.code, {"status": "error", "message": body[:300]}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv

    parser = argparse.ArgumentParser(
        prog="attach",
        description=(
            "Read one workspace file -- text or image -- through the attach "
            "service. Only for content that needs reading; a small file or a "
            "line range is cheaper with cat/sed."
        ),
    )

    parser.add_argument("path", help="File to read, relative to the workspace.")
    parser.add_argument("ask", nargs="+", help="What to extract or answer from it.")
    parser.add_argument("--url", help=f"Service URL (or {URL_ENV}).")
    parser.add_argument("--json", action="store_true", help="Print the raw envelope.")
    parser.add_argument(
        "--timeout",
        type=float,
        default=900.0,
        help="Client-side wait for the answer, in seconds.",
    )

    args = parser.parse_args(argv)

    url = (args.url or os.environ.get(URL_ENV) or DEFAULT_URL).rstrip("/")

    body = {"path": args.path, "ask": " ".join(args.ask).strip()}

    try:
        status, payload = _post(url + "/attach", body, args.timeout)

    except TimeoutError:
        print(f"attach: no answer within {args.timeout}s", file=sys.stderr)

        return 4

    except (urllib.error.URLError, OSError) as exc:
        reason = getattr(exc, "reason", exc)

        print(f"attach: cannot reach {url}: {reason}", file=sys.stderr)

        return 4

    if status == 200 and payload.get("status") == "ok":
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))

        else:
            print(payload.get("text") or "")

        return 0

    code = str(payload.get("code") or payload.get("status") or status)

    if code == "too_large":
        unit = payload.get("unit") or "characters"

        print(
            f"attach: too_large: {payload.get('size')} {unit} > "
            f"{payload.get('limit')} limit ({args.path})",
            file=sys.stderr,
        )

        return 3

    if code == "capability_unavailable":
        print(
            f"attach: capability_unavailable: set capabilities."
            f"{payload.get('capability')} in services/attach/config.yaml",
            file=sys.stderr,
        )

        return 5

    message = payload.get("message") or json.dumps(payload)

    print(f"attach: {code}: {message}", file=sys.stderr)

    return 4


if __name__ == "__main__":
    raise SystemExit(main())
