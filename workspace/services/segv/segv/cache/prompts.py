"""The two fixed strings an epoch opening carries.

The guide is sent once per epoch, next to the recent turns; the status
line reports whether the index is caught up. Long-term memory is not
pushed any more: the model pulls it with `segv search`.
"""

from __future__ import annotations

GUIDE = """<segv-guide>
Context was cleared. The blocks below are what survived it:
- <segv-recent>: the last archived turns, each with its seq, time and origin.
- <segv-pointer>: the newest archived turn seq.
- <segv-status>: index health; lag_turns > 0 means search may miss recent turns.
Your task and constraints are not stored in these blocks: they live in
AGENT.md, notes/ and the earliest turns. Pull anything else on demand:
  segv search "text" [--mode hybrid|lexical|vector|grep] [--rerank]
      [--seq-from N --seq-to N] [--since T --until T] [--limit K] [--full]
  segv turns --limit 20     list recent turns
  segv turn <seq>           print one turn in full
hybrid is the default and runs without the cross-encoder; pass --rerank
when ranking quality matters, and use grep for exact words or while the
index is rebuilding.
</segv-guide>"""


def status_block(index_ok: bool, lag: int, latest: int) -> str:
    state = "ok" if index_ok else "warming"

    return (
        f'<segv-status index="{state}" lag_turns="{lag}" '
        f'latest_seq="{latest}"/>'
    )
