# @tool
"""
Digest provider: file hashing utilities, kept deliberately small
to show the full local-provider contract.

Annotated example of a local Python tool provider:
- `# @tool` header in the first 20 lines (before the docstring),
- exactly one LocalToolProvider subclass with a unique `id`,
- @tool methods fully annotated (schema derives from annotations),
- docstring with an Args: section for every parameter,
- no framework imports: LocalToolProvider / @tool / text_result /
  error_result are injected into the file namespace by the loader.
"""

from __future__ import annotations

import hashlib
from typing import Any


class DigestProvider(LocalToolProvider):

    id = "digest"

    @tool
    async def sha256(
        self,
        path: str,
        chunk_size: int = 65536,
    ) -> Any:
        """Compute the SHA-256 hex digest of a local file.

        Returns a JSON object with `path` and `sha256`.

        Args:
            path: file path (absolute, or relative to the repo root).
            chunk_size: read chunk size in bytes.
        """
        # Heavy/optional imports belong INSIDE the method body so
        # provider load and hot reload stay fast. (hashlib is stdlib,
        # imported at top here for readability.)
        from pathlib import Path

        from nan_itself.utils import paths

        resolved = Path(path)

        if not resolved.is_absolute():
            resolved = paths.repo_root() / resolved

        digest = hashlib.sha256()

        with resolved.open("rb") as handle:
            while chunk := handle.read(chunk_size):
                digest.update(chunk)

        return {"path": str(resolved), "sha256": digest.hexdigest()}

    @tool
    async def sha256_text(self, text: str) -> Any:
        """Compute the SHA-256 hex digest of a UTF-8 string.

        Args:
            text: the text to hash.
        """
        return {
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()
        }
