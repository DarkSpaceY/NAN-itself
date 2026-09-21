"""
Voice module unit tests (no hardware, no model weights).

Covers:
    - ring consumer arithmetic (_consume_ring): fresh skip-history,
      sequential delivery, overtake resync, malformed seq
    - say channel: TaskPayload schema, depth-8 FIFO, on_target wake
    - fast path: transcribe -> fast_reply -> speak; suppressed when
      a say task is pending, when the transcript is empty, or when
      the fast path is disabled
    - say tasks: compose -> stream playback; compose failure is
      one-shot feedback, never a crash
    - barge-in: interruptible playback stops + FIFO drains + event;
      non-interruptible playback plays through with the gate flag
      down; fast-path chatter is preempted by a pending say task
    - provisioning: loud failure without the audio dependency,
      without a published ring, or with a bad sample rate
    - query() projection and serialize/restore validation
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from nan_itself.modules.model import DataSpace
from nan_itself.utils.audio import Utterance


REPO = Path(__file__).resolve().parents[1]


def _load_voice_module():
    """
    Load builtin/modules/voice.py the way the loader does.
    voice.py imports Module symbols explicitly from
    nan_itself.modules.action, so no namespace injection is
    needed beyond registering the spec.
    """
    spec = importlib.util.spec_from_file_location(
        "voice_module_test",
        REPO / "builtin" / "modules" / "voice.py",
    )

    assert spec is not None and spec.loader is not None

    module = importlib.util.module_from_spec(spec)

    sys.modules[spec.name] = module

    spec.loader.exec_module(module)

    return module


def _fresh_instance():
    voice = _load_voice_module()

    instance = voice.VoiceModule()

    instance.data = DataSpace("voice")

    instance.dependencies = {}

    return instance


def _utterance() -> Utterance:
    return Utterance(
        pcm=b"\x00" * 960,
        voiced_ms=600,
        total_ms=900,
        pauses_ms=[],
        f0s=[],
    )


class _FakeTranscriber:
    def __init__(self, text: str) -> None:
        self.text = text

    def transcribe(self, utt: Any) -> dict:
        return {"text": self.text}


class _FakeSLM:
    def __init__(
        self,
        reply: str | None = None,
        composed: dict[str, str] | None = None,
    ) -> None:
        self.reply = reply

        self.composed = composed

        self.fast_calls: list[str] = []

        self.compose_calls: list[dict[str, Any]] = []

    def fast_reply(self, user_text: str, now=None):
        self.fast_calls.append(user_text)

        return self.reply

    def compose(self, task):
        self.compose_calls.append(dict(task))

        return self.composed


class _FakeTTS:
    def __init__(self, chunks: int = 3) -> None:
        self.chunks = chunks

        self.calls: list[tuple[str, str]] = []

    @property
    def sample_rate(self) -> int:
        return 16000

    def speak(self, text: str, instruct: str = "", speed=None):
        self.calls.append((text, instruct))

        for _ in range(self.chunks):
            yield np.zeros(1600, dtype=np.float32)


class _FakeStream:
    """Records writes; optional per-write hook."""

    def __init__(self, on_write=None) -> None:
        self.writes = 0

        self._on_write = on_write

    def write(self, audio) -> None:
        self.writes += 1

        if self._on_write is not None:
            self._on_write(self.writes)

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass


class _FakeReader:
    def __init__(self, value: dict[str, Any]) -> None:
        self._value = value

    def snapshot(self) -> dict[str, Any]:
        return dict(self._value)


def _say_task(**overrides: Any) -> dict[str, Any]:
    task: dict[str, Any] = {
        "intent": "greet",
        "key_points": "向用户问好",
        "tone": "happy",
        "interruptible": True,
    }

    task.update(overrides)

    return task


# ============================================================================
# Ring consumer arithmetic
# ============================================================================


def test_consume_ring_fresh_skips_history():
    ring = {
        "seq": 5,
        "chunks": ["c0", "c1", "c2"],
        "sample_rate": 16000,
        "frame_ms": 30,
    }

    # A fresh consumer (seq None) skips history entirely and
    # positions itself at the publisher's current seq.
    chunks, seq = _fresh_instance()._consume_ring(ring, None)

    assert chunks == []

    assert seq == 5

    # Already caught up: nothing new.
    chunks, seq = _fresh_instance()._consume_ring(ring, 5)

    assert chunks == []

    assert seq == 5


def test_consume_ring_sequential_delivery():
    ring = {
        "seq": 5,
        "chunks": ["c0", "c1", "c2"],
    }

    # Chunk i carries global seq (seq - len + i) = 2, 3, 4.
    # A consumer at 3 has seen c0 (seq 2) -> gets c1, c2.
    chunks, seq = _fresh_instance()._consume_ring(ring, 3)

    assert chunks == ["c1", "c2"]

    assert seq == 5


def test_consume_ring_overtook_resyncs():
    ring = {
        "seq": 5,
        "chunks": ["c0", "c1", "c2"],
    }

    # The window starts at seq 2; a consumer at 0 was overtaken.
    chunks, seq = _fresh_instance()._consume_ring(ring, 0)

    assert chunks == []

    assert seq == 5


def test_consume_ring_malformed_seq():
    chunks, seq = _fresh_instance()._consume_ring(
        {"seq": "bogus", "chunks": ["c0"]},
        None,
    )

    assert chunks == []

    assert seq is None


# ============================================================================
# Say channel contract
# ============================================================================


def test_say_channel_schema_and_depth():
    voice = _load_voice_module()

    spec = voice.VoiceModule.channels["say"]

    assert spec.model is voice.TaskPayload

    assert spec.depth == 8

    payload, error = spec.validate(
        {"intent": "greet", "key_points": "hi"}
    )

    assert error is None

    assert payload == {
        "intent": "greet",
        "key_points": "hi",
        "tone": "",
        "interruptible": True,
    }

    _, error = spec.validate({"intent": "greet"})

    assert error is not None


def test_on_target_wakes_speak_thread():
    instance = _fresh_instance()

    assert not instance._say_event.is_set()

    result = instance.set_target("say", _say_task())

    assert result == "written"

    assert instance._say_event.is_set()

    assert instance.current_target("say") is not None


# ============================================================================
# Fast path
# ============================================================================


def test_fast_path_answers_trivial_turn():
    instance = _fresh_instance()

    instance.transcriber = _FakeTranscriber("你好")

    instance.slm = _FakeSLM(reply="你好呀")

    spoken: list[dict[str, Any]] = []

    instance._speak_text = lambda text, **kw: spoken.append(
        {"text": text, **kw}
    )

    instance._handle_utterance(_utterance())

    assert instance._stats["transcripts_total"] == 1

    assert instance._stats["last_transcript"] == "你好"

    assert instance.slm.fast_calls == ["你好"]

    assert instance._stats["fast_answers_total"] == 1

    assert spoken == [
        {
            "text": "你好呀",
            "instruct": "",
            "source": "fast",
            "interruptible": True,
        }
    ]


def test_fast_path_skipped_when_say_task_pending():
    instance = _fresh_instance()

    instance.set_target("say", _say_task())

    instance.transcriber = _FakeTranscriber("你好")

    instance.slm = _FakeSLM(reply="你好呀")

    spoken: list[Any] = []

    instance._speak_text = lambda *a, **kw: spoken.append(a)

    instance._handle_utterance(_utterance())

    # The transcript is still a fact; the fast path stays quiet.
    assert instance._stats["transcripts_total"] == 1

    assert instance.slm.fast_calls == []

    assert instance._stats["fast_answers_total"] == 0

    assert spoken == []


def test_fast_path_skipped_on_empty_transcript():
    instance = _fresh_instance()

    instance.transcriber = _FakeTranscriber("   ")

    instance.slm = _FakeSLM(reply="你好呀")

    instance._handle_utterance(_utterance())

    assert instance._stats["transcripts_total"] == 0

    assert instance.slm.fast_calls == []


def test_fast_path_disabled():
    instance = _fresh_instance()

    instance.fast_path_enabled = False

    instance.transcriber = _FakeTranscriber("你好")

    instance.slm = _FakeSLM(reply="你好呀")

    instance._handle_utterance(_utterance())

    assert instance._stats["transcripts_total"] == 1

    assert instance.slm.fast_calls == []


# ============================================================================
# Say task playback
# ============================================================================


def test_speak_once_composes_and_plays():
    instance = _fresh_instance()

    instance.set_target("say", _say_task())

    instance.slm = _FakeSLM(
        composed={
            "instruct": "用开心的语气说",
            "text": "哈喽，很高兴见到你",
        }
    )

    instance.tts = _FakeTTS(chunks=3)

    stream = _FakeStream()

    instance._open_stream = lambda: stream

    assert instance._speak_once() is True

    # The task was consumed up front (barge-in drains what is
    # left behind it).
    assert instance.current_target("say") is None

    assert instance.slm.compose_calls == [_say_task()]

    assert instance.tts.calls == [
        ("哈喽，很高兴见到你", "用开心的语气说")
    ]

    assert stream.writes == 3

    assert instance._stats["tasks_total"] == 1

    assert instance._current_state() == "listening"


def test_speak_once_empty_slot_returns_false():
    instance = _fresh_instance()

    assert instance._speak_once() is False


def test_illegal_instruct_is_loud_and_neutral():
    instance = _fresh_instance()

    instance.set_target("say", _say_task())

    instance.slm = _FakeSLM(
        composed={
            "instruct": "用外星人的语气说",
            "text": "哈喽",
        }
    )

    instance.tts = _FakeTTS(chunks=1)

    instance._open_stream = lambda: _FakeStream()

    assert instance._speak_once() is True

    # Neutral speech, never the hallucinated directive.
    assert instance.tts.calls == [("哈喽", "")]

    assert any(
        "not on the TTS whitelist" in event
        for event in instance._events
    )


def test_compose_failure_is_feedback_not_crash():
    instance = _fresh_instance()

    instance.set_target("say", _say_task())

    instance.slm = _FakeSLM(composed=None)

    assert instance._speak_once() is True

    assert instance._stats["compose_failures_total"] == 1

    assert instance._current_state() == "listening"

    assert "user interrupted" not in list(instance._events)

    assert any(
        "failed to compose" in event
        for event in instance._events
    )


# ============================================================================
# Barge-in
# ============================================================================


def test_barge_in_stops_playback_and_drains_fifo():
    instance = _fresh_instance()

    instance.set_target("say", _say_task())

    instance.slm = _FakeSLM(
        composed={"instruct": "用开心的语气说", "text": "哈喽"}
    )

    # During the first write the "listen thread" hears 0.5 s of
    # continuous speech and a second task lands on the FIFO.
    def on_write(writes: int) -> None:
        instance._barge_event.set()

        instance.set_target("say", _say_task(intent="followup"))

    instance.tts = _FakeTTS(chunks=5)

    instance._open_stream = lambda: _FakeStream(on_write)

    assert instance._speak_once() is True

    # Playback stopped after the first chunk.
    assert instance._current_state() == "listening"

    # The whole FIFO was drained (say tasks behind the current
    # one die with the barge-in).
    assert instance.current_target("say") is None

    assert instance._stats["bargeins_total"] == 1

    assert "user interrupted" in list(instance._events)


def test_non_interruptible_task_plays_through():
    instance = _fresh_instance()

    instance.set_target("say", _say_task(interruptible=False))

    instance.slm = _FakeSLM(
        composed={"instruct": "用认真的语气说", "text": "听我说"}
    )

    gate_during_write: list[Any] = []

    def on_write(writes: int) -> None:
        gate_during_write.append(
            instance._speaking_interruptible
        )

    instance.tts = _FakeTTS(chunks=3)

    instance._open_stream = lambda: _FakeStream(on_write)

    assert instance._speak_once() is True

    # All chunks played; the barge gate was down the whole time
    # (this is exactly the flag the listen thread consults).
    assert gate_during_write == [False, False, False]

    assert instance._stats["bargeins_total"] == 0

    assert "user interrupted" not in list(instance._events)


def test_fast_path_chatter_preempted_by_say_task():
    instance = _fresh_instance()

    # The agent fires a say task after the first chunk of
    # fast-path chatter: the chatter stops immediately.
    def on_write(writes: int) -> None:
        instance.set_target("say", _say_task())

    instance.tts = _FakeTTS(chunks=5)

    instance._open_stream = lambda: _FakeStream(on_write)

    instance._speak_text(
        "随便聊聊",
        instruct="",
        source="fast",
        interruptible=True,
    )

    # Chatter stopped, the preempting task survives (it is not
    # barge-in, so the FIFO is not drained).
    assert instance.current_target("say") is not None

    assert instance._stats["bargeins_total"] == 0

    assert "user interrupted" not in list(instance._events)


# ============================================================================
# Provisioning (loud failure, Facade DOWN + backoff)
# ============================================================================


def test_provision_requires_audio_dependency():
    instance = _fresh_instance()

    with pytest.raises(RuntimeError, match="audio module"):
        instance._provision()


def test_provision_requires_published_ring():
    instance = _fresh_instance()

    instance.dependencies = {"audio": _FakeReader({})}

    with pytest.raises(RuntimeError, match="pcm_ring"):
        instance._provision()


def test_provision_rejects_incompatible_rate():
    instance = _fresh_instance()

    instance.dependencies = {
        "audio": _FakeReader(
            {"pcm_ring": {"seq": 1, "sample_rate": 44100}}
        )
    }

    with pytest.raises(RuntimeError, match="webrtcvad"):
        instance._provision()


# ============================================================================
# query() projection + persistence
# ============================================================================


def test_query_projects_state_transcripts_and_channel():
    instance = _fresh_instance()

    instance._transcripts.append(
        {"ts": 1758400000.0, "text": "你好"}
    )

    instance._answered.append(
        {"ts": 1758400000.0, "text": "你好呀"}
    )

    rendered = asyncio.run(instance.query(None))

    assert rendered is not None

    assert "[Voice] state:" in rendered

    assert '- user said' in rendered

    assert '"你好"' in rendered

    assert '- answered myself' in rendered

    assert '"你好呀"' in rendered

    # The say channel registry is always surfaced so the model
    # does not re-send targets it already sent.
    assert "[channel] voice/say" in rendered


def test_serialize_restore_roundtrip():
    instance = _fresh_instance()

    instance.transcriber = _FakeTranscriber("你好")

    instance.slm = _FakeSLM(reply=None)

    instance._handle_utterance(_utterance())

    state = instance.serialize_state()

    assert state["transcripts_total"] == 1

    assert state["last_transcripts"] == [
        {
            "ts": instance._transcripts[0]["ts"],
            "text": "你好",
        }
    ]

    fresh = _fresh_instance()

    fresh.restore_state(state)

    assert fresh._stats["transcripts_total"] == 1

    assert list(fresh._transcripts) == state["last_transcripts"]


def test_restore_state_validates_shape():
    instance = _fresh_instance()

    with pytest.raises(TypeError):
        instance.restore_state(["not", "an", "object"])

    with pytest.raises(TypeError, match="must be an integer"):
        instance.restore_state({"transcripts_total": "1"})

    with pytest.raises(TypeError, match="must be objects"):
        instance.restore_state(
            {"last_transcripts": ["nope"]}
        )
