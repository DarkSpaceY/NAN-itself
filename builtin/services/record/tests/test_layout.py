"""The ledger has to stay readable inside the desktop's narrow pane."""

from __future__ import annotations

import re
from pathlib import Path

CSS = Path(__file__).resolve().parents[1] / "web" / "style.css"


def _track_min_px(track: str) -> int:
    track = track.strip().rstrip(",")
    if re.fullmatch(r"\d+px", track):
        return int(track[:-2])
    match = re.fullmatch(r"minmax\((\d+)px,\s*[^)]+\)", track)
    if match:
        return int(match.group(1))
    if re.fullmatch(r"minmax\(0,\s*[^)]+\)", track) or "fr" in track:
        return 0
    raise AssertionError(f"unparsed track {track}")


def _tracks(declaration: str) -> list[str]:
    return re.findall(r"minmax\([^)]+\)|\d+px", declaration)


def _track_max_px(track: str) -> int | None:
    """Fixed ceiling of a track. A flexible track has no ceiling."""
    track = track.strip().rstrip(",")
    if re.fullmatch(r"\d+px", track):
        return int(track[:-2])
    match = re.fullmatch(r"minmax\([^,]+,\s*(\d+(?:\.\d+)?)rem\)", track)
    if match:
        return round(float(match.group(1)) * 16)
    if "fr" in track:
        return None
    raise AssertionError(f"unparsed track {track}")


def _media_block(css: str, query: str) -> str:
    start = css.index(query)
    open_at = css.index("{", start)
    depth = 0
    for index in range(open_at, len(css)):
        if css[index] == "{":
            depth += 1
        elif css[index] == "}":
            depth -= 1
            if depth == 0:
                return css[open_at + 1 : index]
    raise AssertionError(f"unclosed {query}")


def test_narrow_pane_keeps_a_turn_column() -> None:
    css = CSS.read_text(encoding="utf-8")
    cols = re.search(r"--cols:\s*([^;]+);", css)
    assert cols is not None
    minimum = sum(_track_min_px(track) for track in _tracks(cols.group(1)))

    # A desktop pane is about 420px. Padding and the row inset take ~80,
    # so the fixed tracks have to leave a turn column.
    assert minimum <= 340

    cells = re.search(r"\.ledger-head > \*,\s*\.row > \* \{([^}]+)\}", css)
    assert cells is not None and "min-width: 0" in cells.group(1)

    span = re.search(r"\.range-span \{([^}]+)\}", css)
    assert span is not None and "white-space: nowrap" in span.group(1)

    narrow = _media_block(css, "@media (max-width: 720px)")
    assert "position: absolute" in narrow

    narrow_cols = re.search(r"--cols:\s*([^;]+);", narrow)
    assert narrow_cols is not None
    ceilings = [_track_max_px(track) for track in _tracks(narrow_cols.group(1))]
    fixed = sum(size for size in ceilings if size is not None)

    # The pane row is about 380px. Everything that cannot grow has to
    # leave the turn a readable slice.
    assert fixed <= 300
