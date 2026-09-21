"""
Audio module unit tests (no hardware, no model weights).

Covers the post-STT contract:
    - PCM ring facts: monotonic seq, window trim, base64 snapshot,
      JSON-safe publication into DataSpace
    - utterance-event ring: speech events without content
      (transcription is the voice module's territory)
    - enrichment attach (speaker / utt_tag / emotion) via fakes
"""

from __future__ import annotations

import base64
import importlib.util
import json
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np

from nan_itself.modules.model import DataSpace
from nan_itself.utils.audio import Utterance


REPO = Path(__file__).resolve().parents[1]

FRAME_BYTES = 16000 * 30 // 1000 * 2  # 30 ms @ 16 kHz s16le


def _load_audio_module():
    """
    Load builtin/modules/audio.py the way the loader does
    (Module/Turn injected into the file namespace).
    """
    spec = importlib.util.spec_from_file_location(
        "audio_module_test",
        REPO / "builtin" / "modules" / "audio.py",
    )

    assert spec is not None and spec.loader is not None

    module = importlib.util.module_from_spec(spec)

    sys.modules[spec.name] = module

    from nan_itself.modules.model import Module, Turn

    module.Module = Module

    module.Turn = Turn

    spec.loader.exec_module(module)

    return module


def _fresh_instance():
    audio = _load_audio_module()

    instance = audio.AudioModule()

    instance.pcm_ring_window_s = 0.1  # 3200 bytes ≈ 3.3 frames

    instance._pcm_ring_budget = int(
        instance.pcm_ring_window_s * instance.sample_rate * 2
    )

    instance.data = DataSpace("audio")

    return instance


class _FakeMic:
    def __init__(
        self,
        frames: list[bytes],
        stop_event: threading.Event,
    ) -> None:
        self._frames = list(frames)

        self._stop_event = stop_event

        self.dropped = 0

        self.closed = False

    def start(self) -> None:
        pass

    def read(self, timeout: float | None = None):
        if not self._frames:
            self._stop_event.set()

            return None

        return self._frames.pop(0)

    def close(self) -> None:
        self.closed = True


class _SilentPipeline:
    """Stand-in for AudioPipeline: no events, neutral stats."""

    latest_stats: dict[str, Any] = {}

    ambient_kind = "quiet"

    class _Tracker:
        noise_floor = -60.0

    tracker = _Tracker()

    def process(self, pcm: bytes) -> list[dict[str, Any]]:
        return []

    def quiet_seconds(self, now: float) -> None:
        return None


class _FakeEmbedder:
    def embed(self, pcm: bytes):
        return np.ones(192, dtype=np.float32)


class _FakeMatcher:
    def assign(self, vector, voiced_ms: float):
        return "person-1", 0.87, False

    def known_count(self) -> int:
        return 1

    def labels(self) -> list[str]:
        return ["person-1"]

    def serialize(self) -> dict[str, Any]:
        return {}

    def restore(self, payload: Any) -> None:
        pass


class _FakeTagger:
    def tag(self, pcm: bytes) -> list[tuple[str, float]]:
        return [("Dog", 0.9), ("Speech", 0.8)]


class _FakeEmotion:
    def recognize(self, pcm: bytes) -> dict[str, Any]:
        return {"emotion": "happy", "prob": 0.9}


# ============================================================================
# PCM ring facts
# ============================================================================


def test_pcm_ring_seq_and_window_trim():
    instance = _fresh_instance()

    frames = [b"\x00" * FRAME_BYTES for _ in range(5)]

    instance.mic_factory = lambda: _FakeMic(
        frames, instance._stop_event
    )

    instance.pipeline = _SilentPipeline()

    instance._run_capture_session()

    # 5 frames = 4800 bytes > 3200 budget -> 3 newest frames kept.
    assert instance._pcm_seq == 5

    assert len(instance._pcm_ring) == 3

    assert instance._pcm_ring_bytes == 3 * FRAME_BYTES

    snapshot = instance._pcm_ring_snapshot()

    assert snapshot["seq"] == 5

    assert snapshot["sample_rate"] == 16000

    assert snapshot["frame_ms"] == 30

    assert len(snapshot["chunks"]) == 3

    # Chunk i carries global seq (seq - len(chunks) + i): the
    # first kept chunk is frame #2 (0-based).
    assert base64.b64decode(snapshot["chunks"][0]) == frames[2]

    assert base64.b64decode(snapshot["chunks"][-1]) == frames[4]


def test_publish_once_carries_json_safe_pcm_ring():
    instance = _fresh_instance()

    instance.mic_factory = lambda: _FakeMic(
        [b"\x00" * FRAME_BYTES for _ in range(2)],
        instance._stop_event,
    )

    instance.pipeline = _SilentPipeline()

    instance._run_capture_session()

    instance._publish_once()

    snapshot = instance.data.snapshot()

    ring = snapshot["pcm_ring"]

    assert ring["seq"] == 2

    assert len(ring["chunks"]) == 2

    # The whole published state survives a JSON round-trip
    # (Facade persists DataSpace to disk on stop).
    encoded = json.dumps(snapshot)

    decoded = json.loads(encoded)

    assert base64.b64decode(decoded["pcm_ring"]["chunks"][0]) == (
        b"\x00" * FRAME_BYTES
    )


def test_consumer_resync_arithmetic():
    """
    Document the consumer contract: chunk i maps to global seq
    (seq - len(chunks) + i); a consumer behind the window takes
    everything.
    """
    instance = _fresh_instance()

    instance.mic_factory = lambda: _FakeMic(
        [b"\x00" * FRAME_BYTES for _ in range(5)],
        instance._stop_event,
    )

    instance.pipeline = _SilentPipeline()

    instance._run_capture_session()

    snapshot = instance._pcm_ring_snapshot()

    seq = snapshot["seq"]

    first = seq - len(snapshot["chunks"])

    # Consumer that has consumed up to 3: new chunks start at
    # index max(0, 3 - first).
    consumer_seq = 3

    start = max(0, consumer_seq - first)

    assert [first + i for i in range(start, len(snapshot["chunks"]))] == [
        3,
        4,
    ]

    # Consumer that has consumed up to 0 (or was started late):
    # the window overtook it -> resync with everything.
    assert max(0, 0 - first) == 0


# ============================================================================
# Utterance-event ring (no transcript content)
# ============================================================================


def _utterance(voiced_ms: int) -> Utterance:
    return Utterance(
        pcm=b"\x00" * FRAME_BYTES,
        voiced_ms=voiced_ms,
        total_ms=voiced_ms + 300,
        pauses_ms=[],
        f0s=[],
    )


def test_analyze_utterance_records_event_without_text():
    instance = _fresh_instance()

    instance._analyze_utterance(_utterance(600))

    assert len(instance._heard) == 1

    event = instance._heard[0]

    assert event["voiced_ms"] == 600

    assert "text" not in event

    assert "confidence" not in event

    assert instance._stats["last_heard_ts"] == event["ts"]


def test_analyze_utterance_attaches_speaker_tag_emotion():
    instance = _fresh_instance()

    instance._embedder = _FakeEmbedder()

    instance.matcher = _FakeMatcher()

    instance._tagger = _FakeTagger()

    instance._emotion = _FakeEmotion()

    instance._analyze_utterance(_utterance(900))

    event = instance._heard[0]

    assert event["speaker"] == "person-1"

    assert event["utt_tag"] == "Dog 0.90"

    assert event["emotion"] == "happy"

    lines = instance._render_heard(list(instance._heard))

    assert len(lines) == 1

    # Clock prefix + "(speaker) <duration> [tag] [emo]" suffix.
    assert lines[0].endswith(
        " (person-1) 900ms [Dog 0.90] [emo:happy]"
    )


def test_render_heard_formats_short_voiced_ms():
    instance = _fresh_instance()

    instance._analyze_utterance(_utterance(600))

    lines = instance._render_heard(list(instance._heard))

    assert len(lines) == 1

    assert lines[0].startswith("- heard ")

    assert lines[0].endswith("600ms")


def test_serialize_state_drops_transcript_counters():
    instance = _fresh_instance()

    state = instance.serialize_state()

    assert set(state) == {
        "utterances_total",
        "transients_total",
        "noise_floor_seed",
    }

    # Restore accepts the same shape and validates counters.
    instance.restore_state(
        {
            "utterances_total": 3,
            "transients_total": 1,
            "noise_floor_seed": -55.0,
        }
    )

    assert instance.pipeline.tracker.noise_floor == -55.0
