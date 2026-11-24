"""Text preparation (SEGV v2 tokenizer and context documents)."""

from __future__ import annotations

import re

# Aligned with the frozen LoCoMo tokenizer: a small stopword set and the
# original [\w']+ token regex, so raw-BM25 can be compared directly.
STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "to",
    "of", "in", "on", "at", "and", "or", "but", "for", "with", "about",
    "what", "when", "where", "who", "how", "why", "did", "do", "does",
    "i", "you", "he", "she", "it", "we", "they", "my", "your",
    "his", "her", "their", "our", "me", "him", "them", "that", "this",
    "these", "those", "there", "here", "have", "has", "had", "will",
    "would", "can", "could", "should", "up", "out", "so", "if", "then",
}

# Minimal Chinese stopword set, mapped category-by-category from STOPWORDS so
# the two languages filter the same functional roles. Only 1-2 character
# entries: the CJK tokenizer emits unigrams and bigrams, so longer entries can
# never match a token.
CN_STOPWORDS = {
    # copula / structural particles  (be, did, do, have, has, had)
    "是", "的", "了", "着", "过", "吗", "呢", "吧", "啊",
    # prepositions  (to, of, in, on, at, for, with, about, up, out)
    "在", "上", "下", "到", "对", "从", "向", "与", "把", "被", "给", "为", "关于",
    # conjunctions  (and, or, but, so, if, then)
    "和", "或", "但", "而", "就", "也", "还", "如果", "那么", "因为", "所以",
    "但是",
    # pronouns / demonstratives  (i, you, he, ... this, that, these, there, here)
    "我", "你", "他", "她", "它", "这", "那", "哪", "谁", "几", "们",
    "我们", "你们", "他们", "我的", "你的", "他的", "她的",
    "这个", "那个", "这些", "那些", "这里", "那里",
    # question words  (what, when, where, who, how, why)
    "什么", "哪里", "怎么", "多少",
}

_TOKEN_RE = re.compile(r"[\w']+")
# CJK Unified Ideographs (+ Ext A, Compatibility Ideographs).
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")
SENTENCE_RE = re.compile(r"(?<=[.!?。！？])\s+")


def _cjk_tokens(run: str) -> list[str]:
    """CJK run -> unigrams plus adjacent bigrams (no cross-script tokens)."""
    tokens = list(run)

    tokens.extend(run[i : i + 2] for i in range(len(run) - 1))

    return tokens


def tokenize(text: str, mode: str = "stop") -> list[str]:
    """Tokenize mixed Chinese/English text.

    Non-CJK segments keep the frozen ``[\\w']+`` behaviour. CJK runs are
    emitted as character unigrams and adjacent bigrams, the strongest
    dictionary-free baseline for Chinese BM25.
    """
    if mode not in ("stop", "plain"):
        raise ValueError(f"unknown tokenizer mode: {mode}")

    tokens: list[str] = []
    pos = 0

    for match in _CJK_RE.finditer(text):
        tokens.extend(_TOKEN_RE.findall(text[pos : match.start()].lower()))
        tokens.extend(_cjk_tokens(match.group(0)))
        pos = match.end()

    tokens.extend(_TOKEN_RE.findall(text[pos:].lower()))

    if mode == "stop":
        return [t for t in tokens if t not in STOPWORDS and t not in CN_STOPWORDS]

    return tokens


def split_sentences(text: str, limit: int) -> list[str]:
    """Pack sentences into pieces of at most `limit`; hard-cut long ones."""
    text = text.strip()

    if not text:
        return []

    if len(text) <= limit:
        return [text]

    out: list[str] = []
    buffer = ""

    for sentence in SENTENCE_RE.split(text):
        sentence = sentence.strip()

        if not sentence:
            continue

        if len(sentence) > limit:
            if buffer:
                out.append(buffer)
                buffer = ""

            out.extend(sentence[i : i + limit] for i in range(0, len(sentence), limit))

        elif len(buffer) + len(sentence) + 1 > limit:
            out.append(buffer)

            buffer = sentence

        else:
            buffer = f"{buffer} {sentence}".strip() if buffer else sentence

    if buffer:
        out.append(buffer)

    return out


def raw_doc(
    ts: str,
    speaker: str,
    text: str,
    prev: str = "",
    nxt: str = "",
    ctx_chars: int = 240,
) -> str:
    """Raw-text embedding document: current text plus text-only neighbours."""
    head = f"[{ts}] {speaker}: " if ts else (f"{speaker}: " if speaker else "")

    parts = []

    if prev:
        parts.append("...context: " + prev[-ctx_chars:])

    parts.append(head + text)

    if nxt:
        parts.append("...context: " + nxt[:ctx_chars])

    return "\n".join(parts)


def event_doc(
    prev_event: str,
    event: str,
    next_event: str,
    ctx_chars: int = 240,
) -> str:
    """Event embedding document: the cue plus neighbouring cues, no prefix."""
    parts = []

    if prev_event:
        parts.append("...context: " + prev_event[-ctx_chars:])

    parts.append(event)

    if next_event:
        parts.append("...context: " + next_event[:ctx_chars])

    return "\n".join(parts)
