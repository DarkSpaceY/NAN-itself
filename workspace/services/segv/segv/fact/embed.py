"""Embeddings over the OpenAI embeddings protocol.

Vectors are normalized on arrival; fact quantizes to int8 at the storage
boundary. The request shape is the only contract -- no Ollama-specific
endpoint, no local model load in this process.
"""

from __future__ import annotations

import time

import httpx
import numpy as np
from loguru import logger

from .config import EmbedConfig


class Embedder:
    def __init__(self, config: EmbedConfig) -> None:
        self.config = config
        self._http = httpx.Client(
            timeout=config.timeout,
            trust_env=False,
            headers={"Authorization": f"Bearer {config.api_key}"} if config.api_key else None,
        )

    @staticmethod
    def _normalize(vector: list[float]) -> list[float]:
        norm = float(np.linalg.norm(vector))

        return [float(x) / norm for x in vector] if norm else [float(x) for x in vector]

    def _request(self, texts: list[str]) -> list[list[float]]:
        url = f"{self.config.base_url.rstrip('/')}/embeddings"
        payload = {"model": self.config.model, "input": texts}
        last: Exception | None = None

        for attempt in range(max(1, self.config.attempts)):
            try:
                response = self._http.post(url, json=payload)
                response.raise_for_status()
                data = response.json().get("data") or []

                if len(data) != len(texts):
                    raise RuntimeError(
                        f"embedding count mismatch: {len(data)} != {len(texts)}"
                    )

                if data and isinstance(data[0], dict) and "index" in data[0]:
                    data = sorted(data, key=lambda row: int(row.get("index") or 0))

                vectors = [row["embedding"] for row in data]

                for vector in vectors:
                    if len(vector) != self.config.dim:
                        raise RuntimeError(
                            f"embedding dim {len(vector)} != {self.config.dim}"
                        )

                return vectors

            except Exception as exc:
                last = exc

            if attempt + 1 < self.config.attempts:
                time.sleep(min(8.0, 2.0**attempt))

        raise RuntimeError(
            f"embedding request failed after {self.config.attempts} attempts: "
            f"{type(last).__name__}: {last}"
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        out: list[list[float]] = []
        batch_size = max(1, self.config.batch_size)

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            out.extend(self._normalize(vector) for vector in self._request(batch))

        if len(out) != len(texts):
            logger.warning("segv-fact embedder returned {} of {}", len(out), len(texts))

        return out

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def close(self) -> None:
        self._http.close()
