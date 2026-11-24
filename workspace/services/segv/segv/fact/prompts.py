"""Frozen cue prompt, adapted from SEGV v2's v5 contract.

The unit changed from one message to one chunk: a chunk may pack several
rendered messages, so the current line carries the chunk's roles instead of a
single speaker. Everything else is the v5 contract — one plain sentence, only
current facts, next chunk is context only, language follows the input, no
quotes, fixed timestamp anchor.
"""

CUE_SYSTEM = (
    "You write one plain-text memory sentence. No JSON, no markdown. "
    "Write it in the same language as the current message."
)

CUE_PROMPT = (
    "Prev chunk ({p_spk}): {prev}\n"
    "Current chunk [{ts}] ({spk}):\n{text}\n"
    "Next chunk ({n_spk}; context only, not the current chunk): {nxt}\n"
    "One sentence about the current chunk, written in the same language as "
    "that chunk. Start it with the timestamp {ts}, then describe what the "
    "roles ({spk}) did. Only facts from the current chunk; the next chunk is "
    "context only and its nouns/facts must not appear. Full names; resolve "
    "pronouns and relative dates; max 35 words (about 40 characters for CJK); "
    "no quotes."
)
