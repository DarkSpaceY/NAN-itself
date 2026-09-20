"""
Shared escalating-retry schedule.

One form, two callers:

    - CoreAgent.run_forever(): failed turns retry with
      escalating delays; a successful turn resets the index
    - SourceTracker: failed source loads are retried after
      escalating delays while the file stays unchanged

The schedule is a tuple of delays; consecutive failures walk
it forward and clamp at the last entry.
"""

from __future__ import annotations


def next_backoff(
    values: tuple[float, ...],
    index: int,
) -> tuple[float, int]:
    """
    Return (delay, next_index) for the failure at `index`.

    The index is clamped to the last entry, so repeated
    failures keep waiting `values[-1]` forever.
    """
    if not values:
        return (0.0, 0)

    index = max(index, 0)

    delay = values[min(index, len(values) - 1)]

    next_index = min(
        index + 1,
        len(values) - 1,
    )

    return (delay, next_index)
