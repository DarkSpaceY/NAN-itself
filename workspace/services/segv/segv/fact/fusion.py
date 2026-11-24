"""Weighted reciprocal-rank fusion (SEGV v2, unchanged)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def fuse_rrf(
    rankings: Sequence[tuple[str, list[str]]],
    weights: Mapping[str, float],
    *,
    k: int = 2,
    seq: Mapping[str, int] | None = None,
    top_k: int = 20,
    order: Sequence[str] | None = None,
) -> list[tuple[str, float]]:
    """Fuse route rankings; `order` controls accumulation.

    A route may be listed before its ranking exists: it contributes
    nothing. `seq` only breaks exact score ties, newest first.
    """
    by_name = {name: ids for name, ids in rankings}
    names = list(order) if order else [name for name, _ in rankings]
    score: dict[str, float] = {}

    for name in names:
        weight = float(weights.get(name, 0.0))

        if weight == 0.0:
            continue

        seen: set[str] = set()

        for rank, doc_id in enumerate(by_name.get(name, [])):
            if doc_id in seen:
                continue

            seen.add(doc_id)

            score[doc_id] = score.get(doc_id, 0.0) + weight / (k + rank + 1)

    seq = seq or {}

    items = sorted(score.items(), key=lambda kv: (-kv[1], -int(seq.get(kv[0], 0))))

    return items[:top_k]
