"""
CallToolResult construction and value serialization.
"""

from __future__ import annotations

import json
from typing import Any

import mcp.types as types


def text_result(
    text: str,
) -> types.CallToolResult:
    return types.CallToolResult(
        content=[
            types.TextContent(
                type="text",
                text=text,
            )
        ]
    )


def error_result(
    text: str,
) -> types.CallToolResult:
    return types.CallToolResult(
        isError=True,
        content=[
            types.TextContent(
                type="text",
                text=text,
            )
        ],
    )


def serialize_value(
    value: Any,
) -> str:
    if value is None:
        return "null"

    if isinstance(value, str):
        return value

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            default=str,
        )

    except Exception:
        return str(value)
