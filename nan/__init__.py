"""
NAN: the formal core package (the validated design of
paper_search/nan_core, re-layered for production use).

One core process runs every agent (1 agent = 1 asyncio task):
turn / handle / verbs / injection hub / gateway. External
injection services are independent local processes that
register over HTTP and contribute text to every turn.

Design source of truth: paper_search/nan_core/SPEC.md.
Entry point: ``python -m nan.app`` (reads config/nan.yaml).
"""

from __future__ import annotations
