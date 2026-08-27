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
    AmbientClassifier,
    AudioPipeline,
    EchoTranscriber,
    FeatureTracker,
    PitchTracker,
    SpeakerMatcher,
    Utterance,
    UtteranceSegmenter,
    WhisperTranscriber,
    cosine_similarity,
    rms_dbfs,
    spectral_centroid_hz,
    spectral_flatness,
    zero_crossing_rate,
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
        # Short window: keeps feature means reactive enough for
        # step changes in these deterministic scenarios.
        ambient_window_frames=10,
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


def test_query_renders_inside_module_territory():
    """
    Territory contract: one module, one [Audio] header; every
    line (ambient, heard, diagnostics) lives inside it. The
    module must not mint global-looking sections in <module>.
    """
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

    assert rendered.startswith("[Audio]\n")

    assert "- hearing: speech active" in rendered

    assert "-38.2" in rendered and "-55.1" in rendered

    # Sub-content renders as territory lines, not sections.
    assert "[Ambient]" not in rendered

    assert "[Heard]" not in rendered

    assert rendered.index("latest words") < rendered.index(
        "earlier words",
    )

    assert "(conf 0.42)" in rendered


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

    assert rendered.startswith("[Audio]\n")

    assert rendered.count('"') // 2 == 3

    assert '"xxxxxxxxxx' in rendered

    assert long_text not in rendered


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

    assert rendered.startswith("[Audio]\n")

    assert "- heard " in rendered

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

    # 25 speech frames ~ 750ms voiced: clears the 600ms
    # minimum-utterance bar.
    for _ in range(25):
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

    module.embedder_factory = lambda: FakeEmbedder(
        [list(range(4))],
    )

    while not module._utterance_queue.empty():
        utt = module._utterance_queue.get_nowait()

        accepted = module._accept_transcript(
            module.transcriber.transcribe(utt),
        )

        assert accepted

        if utt.voiced_ms >= module.embed_min_voiced_ms:
            module._attribute_speaker(utt)

    with module._state_lock:
        assert module._stats["transcripts_total"] == 1

        text = module._heard[-1]["text"]

        assert module._heard[-1]["speaker"] == "voice-1"


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


# ======================================================================
# W2: spectral features + ambient classification
# ======================================================================


def noise_frame(seed: int, amplitude: int = 1500) -> bytes:
    import random

    rng = random.Random(seed)

    import array

    samples = array.array(
        "h",
        [
            rng.randint(-amplitude, amplitude)
            for _ in range(SAMPLE_RATE * FRAME_MS // 1000)
        ],
    )

    return samples.tobytes()


def test_zero_crossing_rate_tone_vs_noise():
    assert rms_dbfs(level_frame(-20.0)) > -30.0  # sanity

    # Constant frame: no sign changes at all.
    assert zero_crossing_rate(level_frame(-20.0)) == 0.0

    # Broadband noise: roughly half the samples flip sign.
    assert zero_crossing_rate(noise_frame(1)) > 0.3


def test_spectral_centroid_orders_tone_vs_noise():
    # Same content each frame: centroid for white noise sits far
    # above the near-DC constant frame.
    tone = spectral_centroid_hz(level_frame(-20.0))

    hiss = spectral_centroid_hz(noise_frame(2))

    assert hiss > tone


def test_spectral_flatness_tone_vs_noise():
    tone_flatness = spectral_flatness(level_frame(-20.0))

    noise_flatness = spectral_flatness(noise_frame(3))

    assert tone_flatness < 0.1

    assert noise_flatness > 0.3


def test_ambient_classifier_decision_matrix():
    classifier = AmbientClassifier()

    base = dict(
        dbfs=-30.0,
        noise_floor=-60.0,
        zcr=0.05,
        flatness=0.05,
    )

    assert classifier.classify(is_speech=True, **base) == "speech"

    assert classifier.classify(is_speech=False, **base) == "tonal"

    assert (
        classifier.classify(
            is_speech=False,
            dbfs=-59.0,
            noise_floor=-60.0,
            zcr=0.05,
            flatness=0.05,
        )
        == "quiet"
    )

    assert (
        classifier.classify(
            is_speech=False,
            dbfs=-30.0,
            noise_floor=-60.0,
            zcr=0.5,
            flatness=0.8,
        )
        == "noisy"
    )

    # Ambiguous mid-flatness broadband clatter -> noisy.
    assert (
        classifier.classify(
            is_speech=False,
            dbfs=-30.0,
            noise_floor=-60.0,
            zcr=0.3,
            flatness=0.28,
        )
        == "noisy"
    )


def test_pipeline_populates_ambient_features_and_kind():
    pipeline = make_pipeline()

    quiet = level_frame(-120.0)

    # Constant loud hum: non-speech under the stub gate, above
    # the quiet margin, spectrally flat-none -> tonal.
    hum = level_frame(-25.0)

    t = [0.0]

    def step(pcm):
        t[0] += 0.03

        return pipeline.process(pcm, now=t[0])

    for _ in range(35):
        step(quiet)

    assert pipeline.latest_stats["ambient_kind"] == "quiet"

    for _ in range(40):
        step(hum)

    stats = pipeline.latest_stats

    assert stats["ambient_kind"] == "tonal"

    assert stats["zcr"] == pytest.approx(0.0, abs=1e-6)

    assert stats["flatness"] < 0.1

    # Windowed means: centroid is a finite number.
    assert stats["centroid_hz"] >= 0.0

    # Broadband noise louder than the adapted floor flips the
    # ruling... but only transiently: the noise floor tracks
    # sustained ambience, so a steady hiss becomes the new
    # normal and the room reads quiet again. Both halves are
    # contractual.
    for i in range(6):
        step(noise_frame(10 + i, amplitude=6000))

    assert pipeline.latest_stats["ambient_kind"] == "noisy"

    for i in range(60):
        step(noise_frame(50 + i, amplitude=6000))

    assert pipeline.latest_stats["ambient_kind"] == "quiet"

    assert pipeline.tracker.noise_floor > -30.0


# ======================================================================
# W3: pitch contour, speech rate, word timestamps, hotwords
# ======================================================================


def tone_frame(hz: float, amplitude: int = 8000) -> bytes:
    import array
    import math

    count = SAMPLE_RATE * FRAME_MS // 1000

    samples = array.array(
        "h",
        [
            int(
                amplitude
                * math.sin(2 * math.pi * hz * i / SAMPLE_RATE)
            )
            for i in range(count)
        ],
    )

    return samples.tobytes()


def test_pitch_tracker_locks_onto_sine_and_rejects_noise():
    tracker = PitchTracker()

    stats = tracker.feed(tone_frame(200.0))

    assert stats["f0_hz"] == pytest.approx(200.0, abs=4.0)

    assert stats["pitch_strength"] > 0.5

    # Broadband noise is unvoiced: no contour growth, low f0.
    tracker_noise = PitchTracker()

    for i in range(5):
        stats = tracker_noise.feed(noise_frame(20 + i))

    assert list(tracker_noise.contour) == []

    assert stats["pitch_strength"] < 0.5

    # Digital silence has no energy at all.
    silent = PitchTracker().feed(level_frame(-120.0))

    assert silent["f0_hz"] == 0.0


def test_pitch_trend_detects_rising_and_steady():
    tracker = PitchTracker(contour_frames=64)

    for _ in range(12):
        tracker.feed(tone_frame(160.0))

    for _ in range(12):
        tracker.feed(tone_frame(250.0))

    assert tracker.trend() == "rising"

    tracker = PitchTracker(contour_frames=64)

    for _ in range(16):
        tracker.feed(tone_frame(200.0))

    assert tracker.trend() == "steady"

    # Too few voiced points -> no verdict.
    assert PitchTracker().trend() == ""


def test_transcriber_double_provides_word_timestamps():
    echo = EchoTranscriber(text="hello nan world")

    echo.calls = 0

    utterance = Utterance(
        pcm=b"",
        voiced_ms=900,
        total_ms=1200,
    )

    result = echo.transcribe(utterance)

    assert result["voiced_s"] == pytest.approx(0.9)

    words = result["words"]

    assert [w["w"] for w in words] == [
        "hello",
        "nan",
        "world",
        "#1",
    ]

    assert words[0]["start"] < words[0]["end"]

    assert words[2]["start"] > words[0]["start"]


def test_accept_transcript_computes_rate_and_hotword():
    module = make_module()

    module.hotwords = ("nan",)

    module.transcriber = EchoTranscriber(text="hello nan world")

    utterance = Utterance(
        pcm=b"",
        voiced_ms=900,
        total_ms=1200,
    )

    result = module.transcriber.transcribe(utterance)

    # "hello nan world #1" contains the configured hotword.
    assert module._accept_transcript(result) is True

    with module._state_lock:
        entry = module._heard[-1]

        rate = entry["rate"]

        assert entry["hotword"] == "nan"

        assert module._stats["hotwords_total"] == 1

    assert rate == pytest.approx(4 / 0.9, rel=0.05)

    assert module._stats["last_speech_rate"] == rate

    # A named utterance tags the hotword too.
    named = {
        "text": "嘿 nan 帮我看下这个报错",
        "confidence": 0.93,
        "language": "zh",
        "words": [],
        "voiced_s": 0.0,
    }

    assert module._accept_transcript(named) is True

    with module._state_lock:
        assert module._heard[-1]["hotword"] == "nan"

        assert module._stats["hotwords_total"] == 2

    assert module._stats["last_hotword"] == "nan"


def test_query_renders_voice_line_and_hotword_tags():
    module = make_module()

    module.hotwords = ("nan",)

    with module._state_lock:
        module._stats["available"] = True

        module._stats["speech_active"] = True

        module._stats["f0_hz"] = 198.4

        module._stats["pitch_trend"] = "rising"

        module._stats["quiet_s"] = 0.0

        module._heard.append(
            {"ts": time.time(), "text": "嘿 nan",
             "confidence": 0.9, "language": "zh",
             "rate": 2.5, "hotword": "nan"},
        )

    rendered = time_machine_query(module)

    assert "- voice: ~198 Hz (rising)" in rendered

    assert "[hot:nan]" in rendered

    # Territory preserved: everything under the one header.
    assert rendered.count("[Audio]") == 1

    assert "[Ambient]" not in rendered

    assert "[Heard]" not in rendered


def test_query_hides_voice_line_when_unvoiced():
    module = make_module()

    with module._state_lock:
        module._stats["available"] = True

        module._stats["speech_active"] = True

        module._stats["f0_hz"] = 0.0

        module._stats["pitch_trend"] = ""

    rendered = time_machine_query(module)

    assert "- hearing: speech active" in rendered

    assert "- voice:" not in rendered


def test_pipeline_merges_pitch_into_stats_during_speech():
    pipeline = AudioPipeline(
        preroll_frames=5,
        trailing_silence_frames=7,
        min_utterance_frames=4,
        max_utterance_frames=400,
        ambient_window_frames=10,
    )

    # Sine at -21 dBFS gates as speech and carries clean F0.
    pipeline.vad = LoudIsSpeechGate(threshold_dbfs=-25.0)

    voice = tone_frame(200.0, amplitude=4100)

    for _ in range(12):
        pipeline.process(voice)

    stats = pipeline.latest_stats

    assert stats["f0_hz"] == pytest.approx(200.0, abs=4.0)

    assert stats["pitch_strength"] > 0.5

    assert stats["pitch_trend"] == "steady"

    # Silence frames stop the trend verdict but keep stats sane.
    for _ in range(3):
        pipeline.process(level_frame(-120.0))

    assert pipeline.latest_stats["pitch_trend"] == ""


def test_hotwords_env_parsing():
    import os

    env_backup = os.environ.get("NAN_AUDIO_HOTWORDS")

    try:
        os.environ["NAN_AUDIO_HOTWORDS"] = "nan, 小娜 , NAN"

        module = AudioModule()

        assert module.hotwords == ("nan", "小娜", "nan")

    finally:
        if env_backup is None:
            os.environ.pop("NAN_AUDIO_HOTWORDS", None)

        else:
            os.environ["NAN_AUDIO_HOTWORDS"] = env_backup


# ======================================================================
# W4: speaker identity (auto-enrolling registry)
# ======================================================================


import numpy as np


class FakeEmbedder:
    """Deterministic embedding source for unit flows."""

    def __init__(self, vectors, fail: bool = False) -> None:
        self.vectors = [np.asarray(v, dtype=np.float32)
                        for v in vectors]

        self.calls = 0

        self.fail = fail

    def load(self) -> None:
        if self.fail:
            raise RuntimeError("no model binary")

    def embed(self, pcm: bytes):
        if self.fail:
            raise RuntimeError("no model binary")

        self.calls += 1

        return self.vectors[(self.calls - 1) % len(self.vectors)]


def test_cosine_similarity_basics():
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)

    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)

    assert cosine_similarity([0, 0], [1, 0]) == 0.0

    assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)


def test_speaker_matcher_auto_clusters_and_promotes():
    matcher = SpeakerMatcher(
        promote_min_utterances=3,
        promote_min_voiced_ms=3000.0,
    )

    v1 = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)

    label, score, promoted = matcher.assign(v1, voiced_ms=500)

    assert (label, promoted) == ("voice-1", False)

    assert score >= 0.0  # first sight: nothing to compare with

    # Near voice joins the same cluster; centroid blends.
    near = np.asarray([0.9, 0.1, 0.0], dtype=np.float32)

    label, score, promoted = matcher.assign(near, voiced_ms=500)

    assert label == "voice-1"

    assert score > 0.9

    assert not promoted

    # A distant voice opens a second cluster.
    stranger = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)

    label, _, promoted = matcher.assign(stranger, voiced_ms=100)

    assert label == "voice-2"

    assert not promoted

    # Third long utterance on voice-1 crosses both thresholds.
    label, _, promoted = matcher.assign(v1, voiced_ms=2500)

    assert label == "voice-1"

    assert promoted is True

    assert matcher.known_count() == 1

    centroid = matcher.clusters["voice-1"]["centroid"]

    # ([1,0,0] + [0.9,0.1,0]) / 2 blended again with [1,0,0].
    assert centroid[1] == pytest.approx(0.1 / 3, abs=1e-6)

    assert matcher.clusters["voice-1"]["count"] == 3


def test_matcher_nearest_must_exceed_threshold():
    matcher = SpeakerMatcher(threshold=0.9)

    matcher.assign([1.0, 0.0], voiced_ms=100)

    # 45 degrees: cosine 0.707 < 0.9 -> must NOT join voice-1.
    label, score, _ = matcher.assign([0.7071, 0.7071])

    assert label == "voice-2"

    assert score == pytest.approx(0.7071, abs=1e-3)


def test_matcher_only_promoted_voices_survive_serialization():
    matcher = SpeakerMatcher(
        promote_min_utterances=2,
        promote_min_voiced_ms=2000.0,
    )

    matcher.assign([1.0, 0.0], voiced_ms=1000)

    matcher.assign([1.0, 0.0], voiced_ms=1500)  # promotes voice-1

    matcher.assign([0.0, 1.0], voiced_ms=9000)  # voice-2, below bar

    payload = matcher.serialize()

    assert list(payload["voices"]) == ["voice-1"]

    revived = SpeakerMatcher()

    revived.restore(payload)

    assert revived.known_count() == 1

    assert "voice-1" in revived.clusters

    assert "voice-2" not in revived.clusters

    # ID numbering never reuses dead session ids: next fresh
    # voice continues past every id ever minted.
    label, _, _ = revived.assign([0.0, 1.0], voiced_ms=1)

    assert label == "voice-3"

    # Same voice after restore still matches its old cluster.
    label, score, _ = revived.assign([1.0, 0.0], voiced_ms=100)

    assert label == "voice-1"

    assert score > 0.99

    # No promotion replay: already persisted.
    assert revived.known_count() == 1


def test_matcher_restore_rejects_bad_payloads():
    matcher = SpeakerMatcher()

    with pytest.raises(TypeError):
        matcher.restore([1, 2])

    with pytest.raises(TypeError):
        matcher.restore({"voices": [], "next_id": 1})

    with pytest.raises(TypeError):
        matcher.restore({"voices": {}, "next_id": 0})


def test_attribute_speaker_tags_heard_and_stats():
    module = make_module()

    module.embedder_factory = lambda: FakeEmbedder(
        [[1.0, 0.0]],
    )

    utt = Utterance(pcm=b"", voiced_ms=600, total_ms=700)

    assert module._accept_transcript(
        {"text": "it is me", "confidence": 0.9,
         "language": "en"},
    )

    module._attribute_speaker(utt)

    with module._state_lock:
        assert module._heard[-1]["speaker"] == "voice-1"

        assert module._stats["last_speaker"] == "voice-1"

        assert module._stats["last_speaker_score"] == (
            pytest.approx(0.0, abs=1e-6)  # first sight
        )

        assert module._stats["voices_session"] == 1

        assert module._stats["voices_known"] == 0

        assert module._stats["speaker_backend"] == "FakeEmbedder"


def test_attribute_speaker_degrades_on_backend_failure():
    module = make_module()

    module.embedder_factory = lambda: FakeEmbedder(
        [], fail=True,
    )

    utt = Utterance(pcm=b"", voiced_ms=600, total_ms=700)

    assert module._accept_transcript(
        {"text": "who said that", "confidence": 0.9,
         "language": "en"},
    )

    module._attribute_speaker(utt)

    with module._state_lock:
        assert "speaker" not in module._heard[-1]

        assert module._stats["last_speaker"] is None

        assert module._stats["speaker_backend"].startswith(
            "unavailable:",
        )


def test_registry_write_through_on_promotion(tmp_path):
    module = make_module()

    module.registry_path = tmp_path / "voices.json"

    # The matcher snapshot these thresholds at construction.
    module.matcher.promote_min_utterances = 3

    module.matcher.promote_min_voiced_ms = 3000.0

    module.embedder_factory = lambda: FakeEmbedder(
        [[1.0, 0.0]],
    )

    utt = Utterance(pcm=b"", voiced_ms=2000, total_ms=2000)

    for text in ("still me", "still me here", "me again"):
        assert module._accept_transcript(
            {"text": text, "confidence": 0.9,
             "language": "en"},
        )

        module._attribute_speaker(utt)

    assert module.registry_path.is_file()

    payload = json.loads(
        module.registry_path.read_text(),
    )

    assert list(payload["voices"]) == ["voice-1"]

    assert payload["voices"]["voice-1"]["count"] == 3

    # A fresh module boots with the learned voice.
    revived = make_module()

    revived.registry_path = module.registry_path

    revived._load_registry()

    assert revived.matcher.known_count() == 1

    with revived._state_lock:
        assert revived._stats["voices_known"] == 1


def test_registry_corruption_is_quarantined(tmp_path):
    module = make_module()

    module.registry_path = tmp_path / "voices.json"

    module.registry_path.write_bytes(b"not json at all {{")

    module._load_registry()

    quarantined = list(tmp_path.glob("*.corrupt-*"))

    assert len(quarantined) == 1

    assert module.matcher.clusters == {}

    with module._state_lock:
        assert module._stats["voices_known"] == 0


def test_registry_throttled_saving(tmp_path):
    module = make_module()

    module.registry_path = tmp_path / "voices.json"

    module.registry_save_throttle_s = 999.0

    module.embedder_factory = lambda: FakeEmbedder(
        [[1.0, 0.0]],
    )

    utt = Utterance(pcm=b"", voiced_ms=600, total_ms=600)

    assert module._accept_transcript(
        {"text": "one", "confidence": 0.9, "language": "en"},
    )

    module._attribute_speaker(utt)

    assert not module.registry_path.exists()  # dirty, throttled

    module._maybe_save_registry(promoted=False)  # still inside window

    assert not module.registry_path.exists()

    module._registry_last_save = 0.0  # force window open

    module._maybe_save_registry(promoted=False)

    assert module.registry_path.exists()


def test_render_heard_includes_speaker_tag():
    module = make_module()

    with module._state_lock:
        module._stats["available"] = True

        module._stats["quiet_s"] = 1.0

        module._heard.append(
            {"ts": time.time(), "text": "where is my coffee",
             "confidence": 0.91, "language": "en",
             "rate": 2.0, "hotword": None,
             "speaker": "voice-3"},
        )

    rendered = time_machine_query(module)

    assert "- heard" in rendered

    assert "(voice-3)" in rendered

    assert "where is my coffee" in rendered

    assert rendered.count("[Audio]") == 1


# ======================================================================
# Anti-hallucination gate (L7)
# ======================================================================


def test_whisper_hallucination_gate_matrix():
    """
    "Hello 佳佳" class of bugs: whisper inventing words for claps
    and noise. The gate drops a segment only when BOTH signals
    say non-speech; confident audio always survives.
    """
    gate = WhisperTranscriber()  # lazy: no model load here

    # Non-speech + low confidence -> the "佳佳" case.
    assert gate.segment_passes(0.90, 0.30) is False

    # High no_speech but confident -> real speech, keep.
    assert gate.segment_passes(0.90, 0.80) is True

    # Low no_speech but weak confidence -> keep ( borderline ).
    assert gate.segment_passes(0.20, 0.30) is True

    # Boundary is strict-greater on both sides -> keep.
    assert gate.segment_passes(0.60, 0.50) is True

    # Just past both boundaries -> drop.
    assert gate.segment_passes(0.61, 0.49) is False
