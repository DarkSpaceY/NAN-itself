"""The cue model, reached over the OpenAI chat-completions protocol.

One runtime, no vendor SDK: any server that speaks /v1/chat/completions can
be the source of cue sentences. Retries are for a shared local daemon that
may still be loading a model, not a second code path.
"""

from __future__ import annotations

import time
from typing import Any

import httpx
from loguru import logger

from .config import SlmConfig


class Slm:
    def __init__(self, config: SlmConfig) -> None:
        self.config = config
        self._http = httpx.Client(
            timeout=config.timeout,
            trust_env=False,
            headers={"Authorization": f"Bearer {config.api_key}"} if config.api_key else None,
        )

    def _payload(self, system: str, user: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": False,
            "seed": self.config.seed,
        }

        if self.config.reasoning_effort is not None:
            payload["reasoning_effort"] = self.config.reasoning_effort

        return payload

    @staticmethod
    def _content(reply: dict[str, Any]) -> str:
        choices = reply.get("choices") or []

        if not choices:
            return ""

        content = (choices[0].get("message") or {}).get("content")

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            return "".join(
                str(part.get("text") or "")
                for part in content
                if isinstance(part, dict)
            ).strip()

        return ""

    def chat(self, system: str, user: str) -> str:
        """Return the assistant text, or "" after the retry budget."""
        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        payload = self._payload(system, user)
        last: Exception | None = None

        for attempt in range(max(1, self.config.attempts)):
            try:
                response = self._http.post(url, json=payload)
                response.raise_for_status()
                content = self._content(response.json())

                if content:
                    return content

                last = RuntimeError("empty completion")

            except Exception as exc:
                last = exc

            if attempt + 1 < self.config.attempts:
                time.sleep(min(8.0, 2.0**attempt))

        logger.warning(
            "segv-fact cue model failed after {} attempts: {}: {}",
            self.config.attempts,
            type(last).__name__,
            last,
        )

        return ""

    def close(self) -> None:
        self._http.close()
