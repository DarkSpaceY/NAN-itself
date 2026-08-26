"""
Audio module unit contracts.

Hardware is never touched: frames are synthesized, the VAD gate
is replaced by deterministic doubles, and the transcriber is an
echo stub. Every thread body is driven through extracted methods.
"""

import json
import time

import pytest

from src.nan_itself.modules.builtin.audio import (
    AudioModule,
)
from src.nan_itself.utils.audio import (
    AudioPipeline,
    EchoTranscriber,
    FeatureTracker,
    UtteranceSegmenter,
    rms_dbfs,
)


FRAME_MS = 30

SAMPLE_RATE = 16000


def level_frame(dbfs: float) -> bytes:
    """
    One constant-amplitude 30ms frame at roughly the dBFS level.
    Constant DC reads as a pure loudness value under rms_dbfs().
    """
    amplitude = int(32767 * (10 ** (dbfs / 20.0)))

    if amplitude < 0:
        amplitude = 0

    sample = amplitude.to_bytes(2, "little", signed=True)

    return sample * (SAMPLE_RATE * FRAME_MS // 1000)


class AlwaysNonSpeechGate:
    """Deterministic VAD double: everything is non-speech."""

    mode = "stub"

    def classify(self, pcm, sample_rate, reference_dbfs,
                 rise_db=10.0) -> bool:
        return False


class LoudIsSpeechGate:
    """
    Deterministic VAD double for segmentation flows: frames above
    the level are speech, everything below is not.
    """

    mode = "stub"

    def __init__(self, threshold_dbfs: float = -30.0) -> None:
        self.threshold = threshold_dbfs

    def classify(self, pcm, sample_rate, reference_dbfs,
                 rise_db=10.0) -> bool:
        return rms_dbfs(pcm) > self.threshold


def make_pipeline() -> AudioPipeline:
    pipeline = AudioPipeline(
        preroll_frames=5,
        trailing_silence_frames=7,
        min_utterance_frames=4,
        max_utterance_frames=30,
    )

    pipeline.vad = AlwaysNonSpeechGate()

    return pipeline


def make_module(**overrides) -> AudioModule:
    module = AudioModule()

    module.hear_preview_cap = 60

    for key, value in overrides.items():
        setattr(module, key, value)

    return module


# ======================================================================
# Signal primitives
# ======================================================================


def test_rms_dbfs_scale():
    assert rms_dbfs(b"\x00\x00" * 480) == -120.0

    half = level_frame(-6.02)

    value = rms_dbfs(half)

    assert -7.0 < value < -5.0


# ======================================================================
# UtteranceSegmenter
# ======================================================================


def test_segmenter_includes_preroll_and_emits_after_trailing():
    segmenter = UtteranceSegmenter(
        preroll_frames=5,
        trailing_silence_frames=7,
        min_utterance_frames=4,
        max_utterance_frames=200,
    )

    silence = level_frame(-120.0)

    speech = level_frame(-25.0)

    for _ in range(12):
        assert segmenter.feed(silence, False) == []

    # Speech onset: the current frame plus the preroll tail enter
    # the buffer together; nothing is emitted yet.
    assert segmenter.feed(speech, True) == []

    for _ in range(9):
        assert segmenter.feed(speech, True) == []

    utterances = []

    for _ in range(8):
        utterances.extend(segmenter.feed(silence, False))

    assert len(utterances) == 1

    utterance = utterances[0]

    # Preroll (kept) + 1 trigger + 9 speech + trailing silence.
    assert utterance.voiced_ms == 10 * FRAME_MS

    assert utterance.total_ms >= 15 * FRAME_MS

    # State machine fully reset afterwards.
    assert segmenter.buffer == []

    assert segmenter.voiced_count == 0


def test_segmenter_drops_too_short_bursts():
    segmenter = UtteranceSegmenter(
        preroll_frames=3,
        trailing_silence_frames=5,
        min_utterance_frames=8,
        max_utterance_frames=400,
    )

    silence = level_frame(-120.0)

    speech = level_frame(-25.0)

    for _ in range(4):
        segmenter.feed(silence, False)

    assert segmenter.feed(speech, True) == []

    for _ in range(3):
        assert segmenter.feed(speech, True) == []

    out: list = []

    for _ in range(6):
        out.extend(segmenter.feed(silence, False))

    assert out == []


def test_segmenter_force_closes_at_max_length():
    segmenter = UtteranceSegmenter(
        preroll_frames=2,
        trailing_silence_frames=50,
        min_utterance_frames=4,
        max_utterance_frames=20,
    )

    speech = level_frame(-25.0)

    emitted: list = []

    for _ in range(40):
        emitted.extend(segmenter.feed(speech, True))

    assert len(emitted) >= 1

    for utterance in emitted:
        assert utterance.total_ms <= 20 * FRAME_MS + FRAME_MS


def test_pipeline_transient_debounce_and_cooldown():
    pipeline = make_pipeline()

    quiet = level_frame(-120.0)

    bang = level_frame(-40.0)

    t = [100.0]

    def step(pcm):
        t[0] += 0.03

        return pipeline.process(pcm, now=t[0])

    for _ in range(80):
        step(quiet)

    events = step(bang)

    assert any(e["type"] == "transient" for e in events)

    # Inside cooldown -> suppressed.
    events = step(bang)

    assert not any(
        e["type"] == "transient" for e in events
    )

    # Past cooldown -> fires again.
    t[0] += 2.0

    events = step(bang)

    assert any(e["type"] == "transient" for e in events)


def test_feature_tracker_floor_only_sinks_when_settled_quiet():
    tracker = FeatureTracker(floor_alpha=0.3, settle_frames=5)

    loud_speech = -20.0

    for _ in range(20):
        tracker.feed(loud_speech, True)

    assert tracker.noise_floor == -60.0

    for _ in range(20):
        tracker.feed(loud_speech, True)

    # Even sustained "speech-like" loud audio may lift the floor
    # only marginally when it towers above it.
    assert tracker.noise_floor <= -55.0

    floor_before = tracker.noise_floor

    for _ in range(20):
        tracker.feed(-50.0, False)

    assert tracker.noise_floor > floor_before


# ======================================================================
# Transcript bookkeeping + query projection
# ======================================================================


def test_accept_transcript_records_and_dedups():
    module = make_module(dedup_window_s=60.0)

    first = {
        "text": "嘿 NAN 帮我看下报错",
        "confidence": 0.93,
        "language": "zh",
    }

    assert module._accept_transcript(first) is True

    assert module._stats["transcripts_total"] == 1

    # Same normalized text inside window -> dropped.
    repeat = dict(first, confidence=0.91)

    assert module._accept_transcript(repeat) is False

    assert module._stats["dedup_skipped_total"] == 1

    assert module._stats["transcripts_total"] == 1

    assert len(module._heard) == 1


def test_query_renders_ambient_and_heard_newest_first():
    module = make_module()

    with module._state_lock:
        module._stats["available"] = True

        module._stats["speech_active"] = True

        module._stats["level_dbfs"] = -38.2

        module._stats["noise_floor_dbfs"] = -55.1

        now = time.time()

        module._heard.extend(
            [
                {"ts": now - 70, "text": "earlier words",
                 "confidence": 0.88, "language": "en"},
                {"ts": now - 10, "text": "latest words",
                 "confidence": 0.42, "language": "en"},
            ],
        )

    rendered = time_machine_query(module)

    assert "[Ambient]" in rendered

    assert "- hearing: speech active" in rendered

    assert "-38.2" in rendered and "-55.1" in rendered

    assert "[Heard]" in rendered

    heard_section = rendered.split("[Heard]")[1]

    latest_at = heard_section.index("latest words")

    earlier_at = heard_section.index("earlier words")

    assert latest_at < earlier_at

    assert "(conf 0.42)" in heard_section


def time_machine_query(module):
    import asyncio

    return asyncio.run(module.query(turn=None))


def test_query_render_limit_and_preview_cap():
    module = make_module(
        hear_render_limit=3,
        hear_preview_cap=10,
    )

    long_text = "x" * 200

    now = time.time()

    with module._state_lock:
        module._stats["available"] = True

        module._stats["quiet_s"] = 5.0

        for i in range(6):
            # Ascending chronological order (oldest first).
            module._heard.append(
                {"ts": now - (5 - i), "text": long_text,
                 "confidence": 0.9, "language": None},
            )

    rendered = time_machine_query(module)

    heard_section = rendered.split("[Heard]")[1]

    assert heard_section.count('"') // 2 == 3

    assert '"xxxxxxxxxx' in heard_section

    assert long_text not in heard_section


def test_query_returns_none_for_dead_empty_room():
    module = make_module()

    with module._state_lock:
        module._stats["available"] = True

        module._stats["quiet_s"] = 999.0

    assert time_machine_query(module) is None


def test_query_still_renders_heard_even_when_quiet_long():
    module = make_module()

    with module._state_lock:
        module._stats["available"] = True

        module._stats["quiet_s"] = 500.0

        module._heard.append(
            {"ts": time.time(), "text": "past sentence",
             "confidence": 0.77, "language": "zh"},
        )

    rendered = time_machine_query(module)

    assert "[Heard]" in rendered

    assert "past sentence" in rendered


def test_query_reports_unavailable_device():
    module = make_module()

    with module._state_lock:
        module._stats["available"] = False

        module._stats["reason"] = "no default input device"

    rendered = time_machine_query(module)

    assert rendered.startswith("[Audio]")

    assert "input unavailable: no default input device" in rendered


def test_on_turn_marks_are_recorded():
    module = make_module()

    import asyncio

    from src.nan_itself.modules.model import TurnRecord

    record = TurnRecord(
        agent_hash="h",
        parent_hash=None,
        depth=0,
        task=None,
        user_input="hi",
        world={},
        reply=None,
        error=None,
        started_at=12345.0,
        ended_at=12346.0,
    )

    asyncio.run(module.on_turn(record))

    assert list(module._turn_marks) == [12345.0]


# ======================================================================
# Persistence + timing bound
# ======================================================================


def test_serialize_restore_roundtrip_applies_seed():
    module = make_module()

    with module._state_lock:
        module._stats["utterances_total"] = 11

        module._stats["transcripts_total"] = 9

        module._stats["transients_total"] = 3

    module.pipeline.tracker.noise_floor = -47.3

    state = module.serialize_state()

    blob = json.dumps(state)

    revived = make_module()

    revived.restore_state(json.loads(blob))

    assert revived._stats["utterances_total"] == 0

    assert revived.pipeline.tracker.noise_floor == pytest.approx(
        -47.3,
    )


def test_restore_state_rejects_bad_payloads():
    module = make_module()

    with pytest.raises(TypeError):
        module.restore_state([1, 2])

    with pytest.raises(TypeError):
        module.restore_state({"noise_floor_seed": "loud"})

    with pytest.raises(TypeError):
        module.restore_state({"utterances_total": "many"})


def test_query_projection_is_fast_under_load():
    module = make_module()

    now = time.time()

    with module._state_lock:
        module._stats["available"] = True

        module._stats["quiet_s"] = 1.0

        for i in range(module.hear_history):
            module._heard.append(
                {"ts": now - i, "text": f"sentence number {i}",
                 "confidence": 0.8, "language": None},
            )

    started = time.perf_counter()

    for _ in range(200):
        assert time_machine_query(module) is not None

    elapsed = time.perf_counter() - started

    assert elapsed < 10.0  # 200 full renders; projection only


# ======================================================================
# End-to-end wiring without hardware
# ======================================================================


def test_capture_events_flow_into_transcript_ring():
    """
    Pipeline output feeds _enqueue_utterance and the transcript
    loop body (_accept_transcript via transcriber double).
    """
    module = make_module()

    module.transcriber = EchoTranscriber(text="heard phrase")

    pipeline = module.pipeline

    # -22 dBFS speech frames gate as speech; quiet frames do not.
    pipeline.vad = LoudIsSpeechGate(threshold_dbfs=-30.0)

    speech = level_frame(-22.0)

    silence = level_frame(-120.0)

    now = 500.0

    processed = 0

    def step(pcm):
        nonlocal processed

        processed += 1

        return pipeline.process(pcm, now=now + processed * 0.03)

    for _ in range(12):
        for event in step(silence):
            dispatch_event(module, event)

    for _ in range(14):
        for event in step(speech):
            dispatch_event(module, event)

    out = []

    # Module defaults: trailing_silence_frames=23.
    for _ in range(25):
        for event in step(silence):
            out.append(event)

            dispatch_event(module, event)

    with module._state_lock:
        queued = module._stats["utterances_total"]

    assert queued == 1

    while not module._utterance_queue.empty():
        utt = module._utterance_queue.get_nowait()

        accepted = module._accept_transcript(
            module.transcriber.transcribe(utt),
        )

        assert accepted

    with module._state_lock:
        assert module._stats["transcripts_total"] == 1

        text = module._heard[-1]["text"]

        # Simulate the capture session being live for rendering.
        module._stats["available"] = True

        module._stats["quiet_s"] = 2.0

    assert text.startswith("heard phrase")

    rendered = time_machine_query(module)

    assert "heard phrase" in rendered


def dispatch_event(module, event) -> None:
    if event["type"] == "transient":
        with module._state_lock:
            module._stats["transients_total"] += 1

            module._stats["last_transient_ts"] = time.time()

    elif event["type"] == "utterance":
        module._enqueue_utterance(event["utterance"])
