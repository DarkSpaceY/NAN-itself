# @module

"""
Voice: the first bidirectional (reactive) builtin Module.

Perceives speech, decides, and acts -- speaking -- through the
say channel downlink (docs/design/voice.md). The audio module
owns the microphone and publishes a rolling PCM ring; voice is a
downstream consumer (requires = ["audio"]) and owns everything
language: VAD endpointing, STT, the semantic/surface split
(main-agent task -> 0.6B SLM utterance -> CosyVoice2 TTS), the
SLM fast path for trivial turns, and barge-in.

Threads outside the event loop:

    listen    audio ring -> barge-in VAD (while speaking) /
              AudioPipeline endpointing -> utterance queue
    analyze   utterance -> STT -> transcript fact -> fast path
    speak     say channel FIFO -> SLM compose -> TTS stream

Playback and the fast path share one play lock: speech never
overlaps itself. A say task always preempts fast-path chatter
(checked before and during fast-path playback). Barge-in
(>= 0.5 s continuous human speech while an interruptible task
plays) stops playback, drains the say FIFO, emits an event and
returns to listening.

v1 honest scope (per spec): headphones assumed -- self-hearing
during playback is expected noise, AEC is v2. No preemptive
generation: an agent turn starts only after STT confirms the
end of an utterance. STT is gated during playback: speech while
speaking triggers barge-in only; it is transcribed from the
next endpoint after playback stops.

Provisioning (whisper, SLM, TTS) happens in start(); failures
raise -- the Facade marks the module DOWN with the error and
retries with backoff, so missing weights revive loudly when
they land (core principle 7). A missing audio dependency or a
ring that has not been published yet is the same kind of loud,
retrying failure.
"""

from __future__ import annotations

import array
import base64
import json
import math
import queue
import sys
import threading
import time
import asyncio
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Mapping

import numpy as np
import sounddevice as sd
import torch
import webrtcvad
import yaml
from faster_whisper import WhisperModel
from loguru import logger
from pydantic import BaseModel

from nan_itself.modules.action import (
    ActionSurface,
    ChannelSpec,
)
from nan_itself.utils import paths as _paths


# ======================================================================
# Inlined audio DSP toolkit (formerly nan_itself.utils.audio).
# utils/ is core-architecture only now: the voice module owns
# its whole per-frame pipeline (constants, feature helpers,
# VAD, utterance segmentation, the pipeline, whisper STT).
# ======================================================================

SAMPLE_RATE = 16000

BYTES_PER_SAMPLE = 2


def frame_bytes(frame_ms: int) -> int:
    return SAMPLE_RATE * frame_ms // 1000 * BYTES_PER_SAMPLE


def rms_dbfs(pcm: bytes | memoryview) -> float:
    """
    Convert one PCM16 chunk into dBFS. Digital silence -> -120.
    """
    count = len(pcm) // BYTES_PER_SAMPLE

    if count == 0:
        return -120.0

    samples = array.array("h")
    samples.frombytes(bytes(pcm))

    acc = 0.0

    for s in samples:
        acc += float(s) * float(s)

    rms = math.sqrt(acc / count)

    if rms <= 0:
        return -120.0

    dbfs = 20.0 * math.log10(rms / 32768.0)

    return max(-120.0, min(0.0, dbfs))


def zero_crossing_rate(pcm: bytes | memoryview) -> float:
    """
    Fraction of adjacent-sample sign changes in one chunk.
    Pure tone -> low; hiss/clatter -> high.
    """
    samples = array.array("h")

    samples.frombytes(bytes(pcm))

    count = len(samples)

    if count < 2:
        return 0.0

    crossings = 0

    previous = samples[0]

    for sample in samples[1:]:
        if (sample >= 0) != (previous >= 0):
            crossings += 1

        previous = sample

    return crossings / (count - 1)


def _spectrum(pcm: bytes | memoryview):
    samples = np.frombuffer(
        bytes(pcm),
        dtype=np.int16,
    ).astype(np.float32) / 32768.0

    magnitude = np.abs(np.fft.rfft(samples))

    freqs = np.fft.rfftfreq(
        len(samples),
        d=1.0 / SAMPLE_RATE,
    )

    return freqs, magnitude


def spectral_centroid_hz(pcm: bytes | memoryview) -> float:
    """
    Magnitude-weighted mean frequency: bright vs muffled sound.
    """
    freqs, magnitude = _spectrum(pcm)

    total = magnitude.sum()

    if total <= 0:
        return 0.0

    return float((freqs * magnitude).sum() / total)


def spectral_flatness(pcm: bytes | memoryview) -> float:
    """
    Geometric/arithmetic mean ratio of the spectrum: 0 = tonal,
    1 = white noise. Guarded for near-silent frames.
    """
    _, magnitude = _spectrum(pcm)

    floor = 1e-10

    magnitude = magnitude + floor

    geometric = float(np.exp(np.log(magnitude).mean()))

    arithmetic = float(magnitude.mean())

    if arithmetic <= 0:
        return 0.0

    return min(1.0, geometric / arithmetic)


def spectral_bandwidth(pcm: bytes | memoryview) -> float:
    """
    Std-dev of the spectrum around its centroid: spread of
    energy across frequencies (Hz).
    """
    freqs, magnitude = _spectrum(pcm)

    total = magnitude.sum()

    if total <= 0:
        return 0.0

    centroid = float((freqs * magnitude).sum() / total)

    return float(
        np.sqrt(
            (magnitude * (freqs - centroid) ** 2).sum() / total
        )
    )


def spectral_rolloff(
    pcm: bytes | memoryview,
    pct: float = 0.85,
) -> float:
    """
    Frequency below which `pct` of the spectral energy lies.
    Telephone-band audio rolls off near 3.4 kHz; full-band
    content reaches much higher.
    """
    freqs, magnitude = _spectrum(pcm)

    total = magnitude.sum()

    if total <= 0:
        return 0.0

    cumulative = np.cumsum(magnitude)

    index = int(np.searchsorted(cumulative, pct * total))

    index = min(index, len(freqs) - 1)

    return float(freqs[index])


def peak_stats(pcm: bytes | memoryview) -> tuple[float, bool]:
    """
    (peak_dbfs, clipped) for one chunk. Clipped means samples
    touch >= 99% of full scale -- distortion happened.
    """
    samples = array.array("h")

    samples.frombytes(bytes(pcm))

    if not len(samples):
        return -120.0, False

    peak = max(abs(s) for s in samples)

    dbfs = (
        20.0 * math.log10(peak / 32768.0) if peak else -120.0
    )

    return max(-120.0, min(0.0, dbfs)), peak >= 32767 * 0.99


class PitchTracker:
    """
    Autocorrelation F0 over 30ms frames (75-500 Hz by default).

    Wiener-Khinchin: zero-padded |FFT|^2 -> IFFT gives the
    autocorrelation without circular wraparound. The peak lag in
    the voiced range is the period; peak/energy is confidence.

    The tracker also keeps a voiced-pitch contour and reduces it
    to a three-way trend (rising/falling/steady) for the future
    L8 emotion layer.
    """

    def __init__(
        self,
        min_hz: float = 75.0,
        max_hz: float = 500.0,
        strength_min: float = 0.5,
        contour_frames: int = 33,
    ) -> None:
        self.min_lag = max(2, int(SAMPLE_RATE / max_hz))

        self.max_lag = int(SAMPLE_RATE / min_hz)

        self.strength_min = strength_min

        self.contour: deque[float] = deque(
            maxlen=max(2, contour_frames),
        )

        self.f0 = 0.0

        self.strength = 0.0

    def feed(self, pcm: bytes) -> dict[str, float]:
        x = np.frombuffer(
            bytes(pcm),
            dtype=np.int16,
        ).astype(np.float32) / 32768.0

        x = x - x.mean()

        energy = float((x * x).sum())

        if energy <= 0:
            self.f0 = 0.0

            self.strength = 0.0

            return {
                "f0_hz": 0.0,
                "pitch_strength": 0.0,
            }

        n = 1 << (2 * len(x) - 1).bit_length()

        spec = np.fft.rfft(x, n)

        ac = np.fft.irfft(
            spec * np.conj(spec),
            n,
        )[: len(x)]

        segment = ac[self.min_lag : self.max_lag + 1]

        offset = int(segment.argmax())

        lag = self.min_lag + offset

        strength = (
            float(segment[offset] / ac[0]) if ac[0] > 0 else 0.0
        )

        if strength >= self.strength_min:
            self.f0 = SAMPLE_RATE / lag

            self.strength = strength

            self.contour.append(self.f0)

        else:
            self.f0 = 0.0

            self.strength = strength

        return {
            "f0_hz": round(self.f0, 1),
            "pitch_strength": round(self.strength, 3),
        }

    def trend(
        self,
        delta_ratio: float = 0.08,
        min_points: int = 8,
    ) -> str:
        """
        Compare mean pitch of the two contour halves. "" means
        not enough voiced data yet.
        """
        points = list(self.contour)

        if len(points) < min_points:
            return ""

        half = len(points) // 2

        head = sum(points[:half]) / half

        tail = sum(points[half:]) / (len(points) - half)

        if tail > head * (1.0 + delta_ratio):
            return "rising"

        if tail < head * (1.0 - delta_ratio):
            return "falling"

        return "steady"


class AmbientClassifier:
    """
    Heuristic speech/tonal/noisy/quiet ruling over windowed
    frame features. Thresholds are attributes, not constants:
    this is W2's coarse triage, not a claim about acoustics.

    Priority: speech (VAD verdict) > quiet (energy) > noisy
    (flat spectrum) > tonal (low ZCR + low flatness).
    """

    def __init__(
        self,
        quiet_margin_db: float = 4.0,
        noisy_flatness_min: float = 0.35,
        tonal_zcr_max: float = 0.12,
        tonal_flatness_max: float = 0.20,
    ) -> None:
        self.quiet_margin_db = quiet_margin_db

        self.noisy_flatness_min = noisy_flatness_min

        self.tonal_zcr_max = tonal_zcr_max

        self.tonal_flatness_max = tonal_flatness_max

    def classify(
        self,
        *,
        is_speech: bool,
        dbfs: float,
        noise_floor: float,
        zcr: float,
        flatness: float,
    ) -> str:
        if is_speech:
            return "speech"

        if dbfs < noise_floor + self.quiet_margin_db:
            return "quiet"

        if flatness >= self.noisy_flatness_min:
            return "noisy"

        if (
            zcr <= self.tonal_zcr_max
            and flatness <= self.tonal_flatness_max
        ):
            return "tonal"

        return "noisy" if zcr > 0.25 else "tonal"


class FeatureTracker:
    """
    Rolling level statistics with silence-anchored noise floor.

    The noise floor only sinks during sustained non-speech spans;
    speech can raise it slowly but never during a voiced burst.
    """

    def __init__(
        self,
        floor_alpha: float = 0.05,
        settle_frames: int = 50,
    ) -> None:
        self.noise_floor = -60.0

        self.current = -120.0

        self.quiet_frames = settle_frames

        self.settle_frames = settle_frames

        self.floor_alpha = floor_alpha

        self.speech_recent = False

    def feed(
        self,
        dbfs: float,
        is_speech: bool,
    ) -> dict[str, float]:
        self.current = dbfs

        if is_speech:
            self.quiet_frames = 0

            # Speech may pull the floor up very slowly only if it
            # sits well above it (persistent ambience, not voice).
            if dbfs > self.noise_floor + 12.0:
                pass

            else:
                blended = (
                    self.noise_floor * (1.0 - self.floor_alpha / 4)
                    + dbfs * (self.floor_alpha / 4)
                )

                self.noise_floor = min(
                    max(blended, -95.0), 0.0
                )

        else:
            self.quiet_frames += 1

            if self.quiet_frames >= self.settle_frames:
                blended = (
                    self.noise_floor * (1.0 - self.floor_alpha)
                    + dbfs * self.floor_alpha
                )

                self.noise_floor = min(
                    max(blended, -95.0), 0.0
                )

        self.speech_recent = is_speech

        return {
            "level_dbfs": round(self.current, 1),
            "noise_floor_dbfs": round(self.noise_floor, 1),
        }


class VadGate:
    """
    Thin webrtcvad wrapper over the frame contract.
    """

    def __init__(
        self,
        aggressiveness: int = 2,
    ) -> None:
        self.engine = webrtcvad.Vad(
            int(aggressiveness),
        )

    @property
    def mode(self) -> str:
        return "webrtcvad"

    def classify(
        self,
        pcm: bytes,
        sample_rate: int,
        reference_dbfs: float,
        rise_db: float = 10.0,
    ) -> bool:
        return self.engine.is_speech(
            pcm,
            sample_rate,
        )


@dataclass
class Utterance:
    pcm: bytes

    voiced_ms: int

    total_ms: int

    transient_events: list[dict] | None = None

    # Interior silence gaps (ms each), trailing closer excluded.
    pauses_ms: list[int] | None = None

    # Voiced F0 samples observed while this utterance was open.
    f0s: list[float] | None = None


class UtteranceSegmenter:
    """
    Slices the continuous frame stream into utterances.

    Feed one (pcm, is_speech) pair per call; returns zero or more
    completed utterances. State machine:

        IDLE      --(speech)--> SPEAKING   (pre-roll included)
        SPEAKING  --trailing silence--> emit
        SPEAKING  --max length--> force emit
    """

    def __init__(
        self,
        preroll_frames: int = 10,
        trailing_silence_frames: int = 23,
        # 600ms: below this whisper is mostly guessing, and the
        # guesses are exactly the hallucinations we filter later.
        min_utterance_frames: int = 20,
        max_utterance_frames: int = 833,
    ) -> None:
        # Noise-armor threshold: a segment whose voiced share
        # (excluding the closing trailing silence) is below this
        # is a clap/click/hum, not speech -- never transcribe it.
        self.min_voiced_ratio = 0.5

        # 30ms frames => 300ms preroll, ~700ms tail,
        # 600ms minimum, ~25s maximum.
        self.preroll: deque[tuple[bytes, bool]] = deque(
            maxlen=max(1, preroll_frames),
        )

        self.trailing_limit = trailing_silence_frames

        self.min_voiced = min_utterance_frames

        self.max_total = max_utterance_frames

        self.buffer: list[tuple[bytes, bool]] = []

        self.voiced_count = 0

        self.silence_run = 0

        self.pauses: list[int] = []

    def feed(
        self,
        pcm: bytes,
        is_speech: bool,
    ) -> list[Utterance]:
        out: list[Utterance] = []

        if not self.buffer:
            self.preroll.append((pcm, is_speech))

            if is_speech:
                self._begin()

            return out

        self.buffer.append((pcm, is_speech))

        if is_speech:
            if self.silence_run >= 2:  # >=60ms counts as a pause
                self.pauses.append(self.silence_run * 30)

            self.voiced_count += 1

            self.silence_run = 0

        else:
            self.silence_run += 1

        total = len(self.buffer)

        closed = False

        reason = ""

        if self.silence_run >= self.trailing_limit:
            closed = True

            reason = "trailing"

        elif total >= self.max_total:
            closed = True

            reason = "maxlen"

        if closed:
            out.extend(self._emit(reason=reason))

        return out

    def flush(self) -> list[Utterance]:
        """Force-close whatever is open (shutdown path)."""
        return self._emit(reason="flush")

    def _begin(self) -> None:
        self.buffer = list(self.preroll)

        self.preroll.clear()

        self.voiced_count = sum(
            1 for _, speech in self.buffer if speech
        )

        self.silence_run = 0

    def _emit(
        self,
        reason: str,
    ) -> list[Utterance]:
        buffered = self.buffer

        voiced = self.voiced_count

        pauses = self.pauses

        trailing = self.silence_run

        self.buffer = []

        self.voiced_count = 0

        self.silence_run = 0

        self.pauses = []

        if voiced < self.min_voiced:
            return []

        # Claps ride in on a long trailing silence; real speech
        # does not. Ratio excludes the closing run.
        speech_frames = len(buffered) - trailing

        if (
            speech_frames > 0
            and voiced / speech_frames < self.min_voiced_ratio
        ):
            return []

        pcm = b"".join(chunk for chunk, _ in buffered)

        return [
            Utterance(
                pcm=pcm,
                voiced_ms=voiced * 30,
                total_ms=len(buffered) * 30,
                pauses_ms=pauses,
            )
        ]


class AudioPipeline:
    """
    One object owns the full per-frame brain work.

    process() returns typed events:

        {"type": "utterance", "utterance": Utterance}
        {"type": "transient", "dbfs": float}

    Statistics live here too so query-side rendering never
    recomputes anything.
    """

    def __init__(
        self,
        vad_aggressiveness: int = 2,
        preroll_frames: int = 10,
        trailing_silence_frames: int = 23,
        min_utterance_frames: int = 8,
        max_utterance_frames: int = 833,
        transient_rise_db: float = 18.0,
        transient_cooldown_s: float = 1.0,
        ambient_window_frames: int = 33,
        pitch_min_hz: float = 75.0,
        pitch_max_hz: float = 500.0,
    ) -> None:
        self.tracker = FeatureTracker()

        self.vad = VadGate(vad_aggressiveness)

        self.classifier = AmbientClassifier()

        self.pitch = PitchTracker(
            min_hz=pitch_min_hz,
            max_hz=pitch_max_hz,
        )

        # ~1s of 30ms frames: stable enough for classification.
        self.feature_window: deque[tuple[float, float, float]] = (
            deque(maxlen=max(1, ambient_window_frames))
        )

        self.ambient_kind = "quiet"

        self.segmenter = UtteranceSegmenter(
            preroll_frames=preroll_frames,
            trailing_silence_frames=trailing_silence_frames,
            min_utterance_frames=min_utterance_frames,
            max_utterance_frames=max_utterance_frames,
        )

        self.transient_rise_db = transient_rise_db

        self.transient_cooldown_s = transient_cooldown_s

        self._last_transient_at = 0.0

        self.clip_total = 0

        self._last_clip_at = 0.0

        self._current_f0s: list[float] = []

        self.speech_active = False

        self.last_sound_monotonic: float | None = None

    def process(
        self,
        pcm: bytes,
        now: float | None = None,
    ) -> list[dict]:
        now = now if now is not None else time.monotonic()

        events: list[dict] = []

        dbfs = rms_dbfs(pcm)

        is_speech = self.vad.classify(
            pcm,
            SAMPLE_RATE,
            self.tracker.noise_floor,
        )

        stats = self.tracker.feed(dbfs, is_speech)

        self.speech_active = is_speech

        if dbfs > self.tracker.noise_floor or is_speech:
            self.last_sound_monotonic = now

        peak_dbfs, clipped = peak_stats(pcm)

        if clipped and now - self._last_clip_at >= 1.0:
            self._last_clip_at = now

            self.clip_total += 1

        # Ambient features: windowed means feed the classifier.
        # Six fields: zcr, centroid, flatness, bandwidth,
        # rolloff, dbfs (the last doubles as dynamic range).
        self.feature_window.append(
            (
                zero_crossing_rate(pcm),
                spectral_centroid_hz(pcm),
                spectral_flatness(pcm),
                spectral_bandwidth(pcm),
                spectral_rolloff(pcm),
                dbfs,
            ),
        )

        window = list(self.feature_window)

        n = len(window)

        zcr = sum(f[0] for f in window) / n

        centroid = sum(f[1] for f in window) / n

        flatness = sum(f[2] for f in window) / n

        bandwidth = sum(f[3] for f in window) / n

        rolloff = sum(f[4] for f in window) / n

        levels = sorted(f[5] for f in window)

        dynamic_range = levels[int(0.95 * (n - 1))] - levels[
            int(0.05 * (n - 1))
        ]

        self.ambient_kind = self.classifier.classify(
            is_speech=is_speech,
            dbfs=dbfs,
            noise_floor=self.tracker.noise_floor,
            zcr=zcr,
            flatness=flatness,
        )

        pitch_stats = self.pitch.feed(pcm)

        if is_speech and pitch_stats["f0_hz"] > 0:
            self._current_f0s.append(pitch_stats["f0_hz"])

            pitch_trend = self.pitch.trend()

        else:
            pitch_trend = ""

        if (
            not is_speech
            and dbfs >= self.tracker.noise_floor + self.transient_rise_db
            and now - self._last_transient_at
            >= self.transient_cooldown_s
        ):
            self._last_transient_at = now

            events.append(
                {"type": "transient", "dbfs": round(dbfs, 1)}
            )

        for utt in self.segmenter.feed(pcm, is_speech):
            utt.f0s = self._current_f0s

            self._current_f0s = []

            events.append(
                {"type": "utterance", "utterance": utt}
            )

        stats["mode"] = self.vad.mode

        stats["speech_active"] = self.speech_active

        stats["zcr"] = round(zcr, 3)

        stats["centroid_hz"] = round(centroid, 1)

        stats["flatness"] = round(flatness, 3)

        stats["bandwidth_hz"] = round(bandwidth, 1)

        stats["rolloff_hz"] = round(rolloff, 1)

        stats["peak_dbfs"] = round(peak_dbfs, 1)

        stats["clip_total"] = self.clip_total

        stats["dynamic_range_db"] = round(dynamic_range, 1)

        stats["snr_db"] = (
            round(dbfs - self.tracker.noise_floor, 1)
            if is_speech
            else None
        )

        stats["ambient_kind"] = self.ambient_kind

        stats.update(pitch_stats)

        stats["pitch_trend"] = pitch_trend

        self.latest_stats = stats

        return events

    def quiet_seconds(self, now: float | None = None) -> float | None:
        now = now if now is not None else time.monotonic()

        if self.last_sound_monotonic is None:
            return None

        return max(0.0, now - self.last_sound_monotonic)


class WhisperTranscriber:
    """
    Lazy faster-whisper adapter. model files stay under the repo
    models/whisper cache; import happens on first use so units
    without the package still collect cleanly.
    """

    def __init__(
        self,
        model_size: str = "base",
        models_dir: str | None = None,
        language: str | None = None,
        cpu_threads: int = 4,
        no_speech_max: float = 0.6,
        halluc_conf_max: float = 0.5,
        translate_enabled: bool = False,
    ) -> None:
        self.model_size = model_size

        self.models_dir = models_dir

        self.language = language

        self.cpu_threads = cpu_threads

        # Anti-hallucination gate: whisper invents words when fed
        # non-speech audio (claps, clicks, silence). A segment
        # that BOTH looks like non-speech to the model AND reads
        # as low-confidence is almost certainly invented.
        self.no_speech_max = no_speech_max

        self.halluc_conf_max = halluc_conf_max

        # Off by default: the second whisper pass doubles cost
        # for non-English utterances.
        self.translate_enabled = translate_enabled

        self._model = None

    def segment_passes(
        self,
        no_speech_prob: float,
        confidence: float,
    ) -> bool:
        """
        Pure gate so the hallucination filter is unit-testable
        without loading the model.
        """
        if (
            no_speech_prob > self.no_speech_max
            and confidence < self.halluc_conf_max
        ):
            return False

        return True

    def load(self) -> None:
        if self._model is not None:
            return

        kwargs: dict = {
            "device": "cpu",
            "compute_type": "int8",
            "cpu_threads": self.cpu_threads,
        }

        if self.models_dir:
            kwargs["download_root"] = str(self.models_dir)

        self._model = WhisperModel(
            self.model_size,
            **kwargs,
        )

    def transcribe(self, utt: Utterance) -> dict:
        self.load()

        audio = np.frombuffer(
            utt.pcm,
            dtype=np.int16,
        ).astype(np.float32) / 32768.0

        segments, info = self._model.transcribe(
            audio,
            language=self.language,
            beam_size=1,
            vad_filter=False,
            word_timestamps=True,
        )

        parts: list[str] = []

        confs: list[float] = []

        words: list[dict] = []

        for seg in segments:
            no_speech = float(
                getattr(seg, "no_speech_prob", 0.0) or 0.0
            )

            avg_logprob = seg.avg_logprob

            confidence = (
                math.exp(avg_logprob)
                if avg_logprob is not None
                else 0.5
            )

            if not self.segment_passes(no_speech, confidence):
                logger.info(
                    "whisper segment dropped as likely "
                    "hallucination (no_speech={:.2f}, conf={:.2f})",
                    no_speech,
                    confidence,
                )

                continue

            parts.append(seg.text.strip())

            confs.append(confidence)

            for word in getattr(seg, "words", None) or []:
                words.append(
                    {
                        "w": (word.word or "").strip(),
                        "start": float(word.start or 0.0),
                        "end": float(word.end or 0.0),
                        "p": round(
                            math.exp(word.probability)
                            if word.probability is not None
                            else 0.5,
                            3,
                        ),
                    },
                )

        text = " ".join(p for p in parts if p).strip()

        confidence = (
            round(sum(confs) / len(confs), 3)
            if confs
            else 0.5
        )

        result = {
            "text": text,
            "confidence": confidence,
            "language": getattr(info, "language", None),
            "words": words,
            "voiced_s": round(utt.voiced_ms / 1000.0, 3),
        }

        language = result.get("language")

        if (
            self.translate_enabled
            and text
            and language
            and language != "en"
        ):
            translated_segments, _ = self._model.transcribe(
                audio,
                language=language,
                task="translate",
                beam_size=1,
            )

            result["translation"] = " ".join(
                seg.text.strip()
                for seg in translated_segments
                if seg.text.strip()
            )

        return result


# ======================================================================
# Inlined CosyVoice2 TTS adapter (formerly nan_itself.utils.tts).
# ======================================================================

"""
CosyVoice2 TTS adapter (the voice module's speech surface).

Backed by an upstream FunAudioLLM/CosyVoice checkout (cloned to
models/voice/cosyvoice/ by convention, overridable via env at
the module layer) injected into sys.path here -- the repo itself
stays free of third-party code.

Contract:
    load()      heavy provisioning; raises loudly so the Facade
                can mark the module DOWN and retry with backoff
                (core principle 7). Missing checkout, missing
                weights or a missing reference wav are all loud.
    speak()     sentence-streamed: the upstream text normalizer
                splits long text and the generator yields float32
                mono chunks as they are synthesized.

The instruct directive is a whitelist-controlled control surface:
the SLM's choice is validated here, and anything hallucinated is
dropped before it can reach the synthesis model.
"""


NEUTRAL_INSTRUCT = "用自然的语气说"

# The SLM's whitelist: instruct must be one of these, verbatim.
ALLOWED_INSTRUCTS: tuple[str, ...] = (
    NEUTRAL_INSTRUCT,
    "用开心的语气说",
    "用兴奋的语气说",
    "用温柔的语气说",
    "用认真的语气说",
    "用抱歉的语气说",
    "用平静的语气说",
    "用快速的语气说",
)


def sanitize_instruct(instruct: str) -> str:
    """
    Whitelist filter for the SLM's instruct choice.

    Returns the directive verbatim when legal, "" otherwise --
    callers substitute NEUTRAL_INSTRUCT. Hallucinated control
    text must never reach the synthesis model.
    """
    text = (instruct or "").strip()

    return text if text in ALLOWED_INSTRUCTS else ""


class CosyVoiceTTS:
    """
    Lazy CosyVoice2 adapter around an upstream checkout.

    Provisioning failures raise out of load(): the caller (the
    voice module's start()) propagates them, the Facade marks
    the module DOWN with the error and retries with backoff, and
    the module revives once the checkout and weights land.
    """

    def __init__(
        self,
        checkout_dir: Path,
        model_dir: Path,
        prompt_wav: Path,
        speed: float = 1.0,
    ) -> None:
        self.checkout_dir = Path(checkout_dir)

        self.model_dir = Path(model_dir)

        self.prompt_wav = Path(prompt_wav)

        self.speed = float(speed)

        self._model: Any = None

    # ------------------------------------------------------------------
    # Provisioning
    # ------------------------------------------------------------------

    def _inject_sys_path(self) -> None:
        """
        The upstream package imports Matcha-TTS from
        third_party/, so both roots go on sys.path.
        """
        candidates = (
            self.checkout_dir,
            self.checkout_dir / "third_party" / "Matcha-TTS",
        )

        for path in candidates:
            resolved = str(path.resolve())

            if path.is_dir() and resolved not in sys.path:
                sys.path.insert(0, resolved)

    def load(self) -> None:
        if self._model is not None:
            return

        if not self.checkout_dir.is_dir():
            raise RuntimeError(
                "CosyVoice checkout missing at "
                f"{self.checkout_dir}; clone FunAudioLLM/CosyVoice "
                "there (git clone https://github.com/FunAudioLLM/"
                "CosyVoice.git) and install its requirements"
            )

        if not (self.model_dir / "cosyvoice2.yaml").is_file():
            raise RuntimeError(
                "CosyVoice2 weights missing at "
                f"{self.model_dir} (no cosyvoice2.yaml); download "
                "them with: modelscope download --model "
                f"iic/CosyVoice2-0.5B --local_dir {self.model_dir}"
            )

        if not self.prompt_wav.is_file():
            raise RuntimeError(
                "reference wav missing at "
                f"{self.prompt_wav}; drop a >=5s 16kHz mono clip "
                "of the desired voice there"
            )

        self._inject_sys_path()

        try:
            from cosyvoice.cli.cosyvoice import CosyVoice2

        except ImportError as exc:
            raise RuntimeError(
                "cosyvoice runtime not importable "
                f"({exc}); install the checkout's requirements "
                "(pynini via conda-forge on macOS arm64, "
                "WeTextProcessing fallback elsewhere)"
            ) from exc

        self._model = CosyVoice2(
            str(self.model_dir),
            load_jit=False,
            load_trt=False,
            load_vllm=False,
            fp16=False,
        )

        logger.info(
            "cosyvoice tts ready (sr={}, checkout={})",
            self._model.sample_rate,
            self.checkout_dir,
        )

    @property
    def sample_rate(self) -> int:
        self.load()

        return int(self._model.sample_rate)

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    def speak(
        self,
        text: str,
        instruct: str = "",
        speed: float | None = None,
    ) -> Iterator[np.ndarray]:
        """
        Yield float32 mono speech chunks (native sample rate)
        as they are synthesized. Empty instruct resolves to the
        neutral directive; illegal instructs are dropped.
        """
        self.load()

        text = (text or "").strip()

        if not text:
            return

        directive = sanitize_instruct(instruct) or NEUTRAL_INSTRUCT

        effective = self.speed if speed is None else float(speed)

        outputs = self._model.inference_instruct2(
            text,
            directive,
            str(self.prompt_wav),
            stream=True,
            speed=effective,
        )

        for output in outputs:
            chunk = (
                output.get("tts_speech")
                if isinstance(output, dict)
                else None
            )

            if chunk is None:
                continue

            audio = (
                chunk.detach().cpu().float().numpy().reshape(-1)
            )

            if audio.size:
                yield audio


# ======================================================================
# Inlined Qwen3 small-dialogue adapter (formerly
# nan_itself.utils.dialogue). ALLOWED_INSTRUCTS is the local
# whitelist defined in the TTS section above.
# ======================================================================

"""
Qwen3-0.6B dialogue adapter (the voice module's surface composer).

Semantic / surface split: the main agent sends a semantic task
(intent + key points + tone); this small local SLM composes the
actual spoken utterance -- wording plus one whitelisted CosyVoice
directive. It also answers trivial turns autonomously (the fast
path) under a fixed, versioned allowlist prompt; anything
non-trivial is escalated to the main agent.

Non-thinking mode (enable_thinking=False) for latency; Qwen3's
recommended non-thinking sampling (temperature 0.7 / top_p 0.8 /
top_k 20). Output must be a single JSON object; parsing is
defensive -- a malformed reply yields None and the module falls
back to escalation instead of speaking garbage.
"""


DEFAULT_REPO_ID = "Qwen/Qwen3-0.6B"

# Versioned fast-path policy: bump when the allowlist prompt
# changes meaningfully.
FAST_PATH_PROMPT_VERSION = 1

_COMPOSE_SYSTEM = """\
你是语音助手的话术引擎。主智能体交给你一个任务，你要把它组织成\
一句自然的中文口语，交给 TTS 直接朗读。

规则：
1. 只输出一个 JSON 对象，不要输出任何其他内容：
   {{"instruct": "...", "text": "..."}}
2. instruct 必须从下面这些指令中逐字选择一个，不得改写、不得自创：
{allowlist}
3. text 是要朗读的中文口语：像说话，不要书面语，简洁自然，
   可以使用这些标记增加表现力：[laughter] [breath] <strong>；
   不要使用任何其他标记或 <|...|> 形式的特殊符号。
4. 只组织表达，不回答任务之外的问题，不解释。"""

_COMPOSE_USER = """\
任务：
意图：{intent}
要点：{key_points}
语气偏好：{tone}"""

_FAST_PATH_SYSTEM = f"""\
你是本机语音助手的快速应答器。用户刚对你说了一句话。只有当这句\
话属于下面几类琐碎话轮时，你才可以直接回答；否则必须升级给主智\
能体处理：

1. 打招呼、告别、感谢（你好 / 早上好 / 晚安 / 谢谢 / 再见）
2. 问时间、日期、星期（当前时间会提供给你）
3. 简单确认与应答（好的 / 嗯 / 可以 / 不用了）

硬性规则：
- 你没有观点，不做承诺，不执行任何动作，不谈论需要记忆或个性的\
内容。
- 只输出一个 JSON 对象，不要输出任何其他内容：
  可以直接回答时：{{"answer": "一句自然的中文口语"}}
  不该回答时：{{"escalate": true}}

[fast-path policy v{FAST_PATH_PROMPT_VERSION}]"""

_FAST_PATH_USER = """\
当前时间：{now}
用户说：{user_text}"""


def _allowlist_block() -> str:
    return "\n".join(
        f"   - {instruct}" for instruct in ALLOWED_INSTRUCTS
    )


class SmallDialogue:
    """
    Lazy Qwen3-0.6B adapter. Weights live under
    models/voice/slm/<snapshot>; a missing directory downloads
    from the Hub during provisioning (honors HF_ENDPOINT and
    proxy env vars) -- a failing download raises out of load(),
    so the Facade marks the module DOWN and retries.
    """

    def __init__(
        self,
        model_path: Path,
        repo_id: str | None = None,
        max_new_tokens: int = 160,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
    ) -> None:
        self.model_path = Path(model_path)

        self.repo_id = repo_id or DEFAULT_REPO_ID

        self.max_new_tokens = max_new_tokens

        self.temperature = temperature

        self.top_p = top_p

        self.top_k = top_k

        self._tokenizer: Any = None

        self._model: Any = None

    # ------------------------------------------------------------------
    # Provisioning
    # ------------------------------------------------------------------

    def _weights_present(self) -> bool:
        return self.model_path.is_dir() and any(
            self.model_path.iterdir()
        )

    def _download(self) -> None:
        logger.info(
            "slm weights missing at {}; downloading {} "
            "(~1.5 GB, honors HF_ENDPOINT and proxy envs)",
            self.model_path,
            self.repo_id,
        )

        import huggingface_hub

        huggingface_hub.snapshot_download(
            repo_id=self.repo_id,
            local_dir=self.model_path,
        )

    def load(self) -> None:
        if self._model is not None:
            return

        if not self._weights_present():
            self._download()

        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
        )

        if torch.cuda.is_available():
            device = "cuda"

            dtype = torch.float16

        elif torch.backends.mps.is_available():
            device = "mps"

            dtype = torch.float16

        else:
            device = "cpu"

            dtype = torch.float32

        path = str(self.model_path)

        self._tokenizer = AutoTokenizer.from_pretrained(path)

        self._model = (
            AutoModelForCausalLM.from_pretrained(
                path,
                dtype=dtype,
            )
            .to(device)
            .eval()
        )

        logger.info(
            "dialogue slm loaded from {} on {}",
            path,
            device,
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _generate(
        self,
        system: str,
        user: str,
        max_new_tokens: int,
    ) -> str:
        self.load()

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        prompt = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
        ).to(self._model.device)

        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
            )

        text = self._tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        return self._strip_think(text)

    @staticmethod
    def _strip_think(text: str) -> str:
        """
        Defensive: non-thinking mode should never leak a think
        block, but a leaked one must not reach the JSON parser.
        """
        if "<think>" in text:
            return text.split("</think>", 1)[-1].strip()

        return text.strip()

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any] | None:
        start = text.find("{")

        end = text.rfind("}")

        if start < 0 or end <= start:
            return None

        try:
            payload = json.loads(text[start : end + 1])

        except Exception:
            return None

        return payload if isinstance(payload, dict) else None

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    def compose(
        self,
        task: Mapping[str, Any],
    ) -> dict[str, str] | None:
        """
        Task -> surface utterance {"instruct", "text"}.

        The SLM's instruct is returned verbatim (stripped) -- the
        voice module validates it against the TTS whitelist and
        reports illegal choices to the model. None means the SLM
        failed to produce a usable utterance.
        """
        raw = self._generate(
            _COMPOSE_SYSTEM.format(
                allowlist=_allowlist_block(),
            ),
            _COMPOSE_USER.format(
                intent=task.get("intent") or "narrate",
                key_points=task.get("key_points") or "",
                tone=task.get("tone") or "自然",
            ),
            self.max_new_tokens,
        )

        payload = self._extract_json(raw)

        if payload is None:
            return None

        text = str(payload.get("text") or "").strip()

        if not text:
            return None

        return {
            "instruct": str(payload.get("instruct") or "").strip(),
            "text": text,
        }

    def fast_reply(
        self,
        user_text: str,
        now: str | None = None,
    ) -> str | None:
        """
        Trivial-turn fast path. Returns the spoken reply, or
        None when the turn must escalate to the main agent.
        """
        raw = self._generate(
            _FAST_PATH_SYSTEM,
            _FAST_PATH_USER.format(
                now=now or time.strftime("%Y-%m-%d %H:%M"),
                user_text=(user_text or "").strip(),
            ),
            self.max_new_tokens,
        )

        payload = self._extract_json(raw)

        if payload is None:
            return None

        if payload.get("escalate"):
            return None

        answer = str(payload.get("answer") or "").strip()

        return answer or None


# ======================================================================
# Speaker diarization adapter (pyannote embedding + local registry)
# ======================================================================


def cosine_similarity(a, b) -> float:  # noqa: ANN001 - mirrors toolkit
    """Cosine between two 1-D vectors; 0.0 on any mismatch."""
    a = np.asarray(a, dtype=np.float64).ravel()

    b = np.asarray(b, dtype=np.float64).ravel()

    denom = float(np.linalg.norm(a) * np.linalg.norm(b))

    if denom <= 0.0:
        return 0.0

    return float(
        np.clip(np.dot(a, b) / denom, -1.0, 1.0)
    )


class VoiceSpeakerRegistry:
    """
    JSON-native speaker registry for per-utterance diarization.

    Mirrors the vision FaceMatcher stance: unmatched embeddings
    auto-enroll as person-N; the registry persists at
    data/databases/audio/voice_speakers.json and survives hot
    reloads. Pure projection: counters live in the module.
    """

    def __init__(
        self,
        path: Path,
        threshold: float = 0.75,
    ) -> None:
        self.path = path

        self.threshold = threshold

        self.people: dict[str, list[list[float]]] = {}

        self.matched_total = 0

    def load(self) -> None:
        if not self.path.is_file():
            self.people = {}

            return

        try:
            data = json.loads(
                self.path.read_text(encoding="utf-8")
            )

        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid speaker registry JSON: {self.path}"
            ) from exc

        if not isinstance(data, dict):
            raise ValueError(
                f"Invalid speaker registry JSON: {self.path}"
            )

        people: dict[str, list[list[float]]] = {}

        for name, vectors in data.items():
            if not isinstance(vectors, list):
                raise ValueError(
                    f"Registry entry {name!r} must be a list"
                )

            people[str(name)] = [
                [float(x) for x in vector]
                for vector in vectors
            ]

        self.people = people

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self.path.write_text(
            json.dumps(self.people, ensure_ascii=False),
            encoding="utf-8",
        )

    def match_or_enroll(self, embedding) -> str:  # noqa: ANN001
        """
        Return the best person-N at/above the threshold, or
        enroll a new person-N for this embedding. Persisted on
        every enroll (single-speaker households write rarely).
        """
        best_name: str | None = None

        best_score = 0.0

        for name, vectors in self.people.items():
            score = max(
                (
                    cosine_similarity(stored, embedding)
                    for stored in vectors
                ),
                default=0.0,
            )

            if score > best_score:
                best_name = name

                best_score = score

        if (
            best_name is not None
            and best_score >= self.threshold
        ):
            self.matched_total += 1

            return best_name

        index = len(self.people) + 1

        name = f"person-{index}"

        self.people[name] = [
            [float(x) for x in np.ravel(embedding)]
        ]

        self.save()

        return name


class PyannoteEmbedder:
    """
    Lazy pyannote speaker-embedding adapter.

    Weights live under models/voice/diarization/; a missing
    snapshot downloads from the Hub during provisioning (the
    gated pyannote models require accepting their terms on
    huggingface.co and a token in config/modules/voice.yaml).
    A failing download raises out of load(): DOWN + backoff,
    never a silent limbo.
    """

    def __init__(
        self,
        cache_dir: Path,
        repo_id: str = "pyannote/embedding",
        token: str = "",
    ) -> None:
        self.cache_dir = cache_dir

        self.repo_id = repo_id

        self.token = token or None

        self._inference: Any = None

    def load(self) -> None:
        import huggingface_hub

        try:
            path = huggingface_hub.snapshot_download(
                self.repo_id,
                cache_dir=str(self.cache_dir),
                token=self.token,
                local_files_only=True,
            )

        except Exception:
            if not self.token:
                raise RuntimeError(
                    "pyannote weights are not cached and no HF "
                    "token is configured; accept the model terms "
                    "on huggingface.co and set "
                    "diarization_token in config/modules/voice.yaml"
                ) from None

            path = huggingface_hub.snapshot_download(
                self.repo_id,
                cache_dir=str(self.cache_dir),
                token=self.token,
            )

        from pyannote.audio import Inference, Model

        model = Model.from_pretrained(path)

        self._inference = Inference(
            model,
            window="whole",
        )

    def embed(self, pcm: bytes) -> np.ndarray:
        """
        One 16 kHz mono PCM16 utterance -> speaker embedding.
        """
        import soundfile as sf

        if self._inference is None:
            raise RuntimeError("diarization embedder not loaded")

        import tempfile

        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)

        try:
            sf.write(
                str(temp_path),
                np.frombuffer(pcm, dtype=np.int16),
                SAMPLE_RATE,
                subtype="PCM_16",
            )

            result = self._inference(str(temp_path))

            return np.asarray(result).ravel()

        finally:
            temp_path.unlink(missing_ok=True)


class VoiceConfig(BaseModel):
    """
    Module-private config: config/modules/voice.yaml over these
    defaults. Path-valued fields are repo-relative strings
    (_resolve_path); whisper weights are shared with the audio
    module.
    """

    stt_model: str = "base"

    stt_language: str | None = None

    fast_path_enabled: bool = True

    tts_speed: float = 1.0

    whisper_models_dir: str = "models/whisper"

    slm_path: str = "models/voice/slm/qwen3-0.6b"

    slm_repo_id: str | None = None

    tts_checkout_dir: str = "models/voice/cosyvoice"

    tts_model_dir: str = "models/voice/tts/cosyvoice2-0.5b"

    tts_reference_wav: str = "models/voice/tts/reference.wav"

    diarization_repo: str = "pyannote/embedding"

    diarization_token: str = ""

    diarization_threshold: float = 0.75

    diarization_models_dir: str = "models/voice/diarization"

    voices_registry: str = (
        "data/databases/audio/voice_speakers.json"
    )


def _resolve_path(value: str) -> Path:
    """
    Absolute (and ~/) passes through; relative resolves against
    the repository root.
    """
    resolved = Path(value).expanduser()

    return (
        resolved
        if resolved.is_absolute()
        else _paths.repo_root() / resolved
    )


def _load_config() -> VoiceConfig:
    """
    Module-private config: config/modules/voice.yaml over the
    VoiceConfig defaults. Missing or empty file = pure defaults;
    anything unparsable is a loud construction failure.
    """
    path = (
        _paths.repo_root()
        / "config"
        / "modules"
        / "voice.yaml"
    )

    if not path.is_file():
        return VoiceConfig()

    data = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )

    if data is None:
        return VoiceConfig()

    if not isinstance(data, dict):
        raise ValueError(
            f"Invalid module config YAML: {path}"
        )

    return VoiceConfig(**data)


class TaskPayload(BaseModel):
    """
    Semantic task written by the main agent to the say channel.

    The SLM turns this into the actual spoken utterance; the
    agent never writes surface text.
    """

    intent: str

    key_points: str

    tone: str = ""

    interruptible: bool = True


class VoiceModule(ActionSurface):
    id = "voice"

    requires: ClassVar[tuple[str, ...]] = ("audio",)

    channels: ClassVar[Mapping[str, ChannelSpec]] = {
        "say": ChannelSpec(
            TaskPayload,
            depth=8,
            description=(
                "Speak a semantic task "
                "{intent, key_points, tone, interruptible}; "
                "depth-N FIFO, barge-in drains the queue"
            ),
        ),
    }

    # ------------------------------------------------------------------
    # Configuration (instance attributes; tests may override)
    # ------------------------------------------------------------------

    sample_rate: int = 16000

    frame_ms: int = 30

    vad_aggressiveness: int = 2

    # ~0.5s endpointing (spec) instead of audio's conversational
    # ~0.7s: voice replies feel snappier.
    trailing_silence_frames: int = 17

    preroll_frames: int = 10

    min_utterance_frames: int = 20

    max_utterance_frames: int = 833

    barge_in_min_speech_s: float = 0.5

    stt_model: str = "base"

    stt_language: str | None = None

    tts_speed: float = 1.0

    say_poll_s: float = 0.5

    publish_interval: float = 2.0

    fast_path_enabled: bool = True

    transcript_history: int = 8

    def __init__(self) -> None:
        # ActionSurface.__init__ creates the channel slots and
        # the one-shot event deque; skipping it would crash the
        # first set_target/current_target call.
        super().__init__()

        cfg = _load_config()

        self.stt_model = cfg.stt_model

        self.stt_language = cfg.stt_language

        self.fast_path_enabled = cfg.fast_path_enabled

        self.tts_speed = cfg.tts_speed

        # Reuse the audio module's whisper weights.
        self.whisper_models_dir = _resolve_path(
            cfg.whisper_models_dir
        )

        self.slm_path = _resolve_path(cfg.slm_path)

        self.slm_repo_id = cfg.slm_repo_id

        self.tts_checkout_dir = _resolve_path(
            cfg.tts_checkout_dir
        )

        self.tts_model_dir = _resolve_path(cfg.tts_model_dir)

        self.tts_reference_wav = _resolve_path(
            cfg.tts_reference_wav
        )

        self.diarization_repo = cfg.diarization_repo

        self.diarization_token = cfg.diarization_token

        self.diarization_threshold = (
            cfg.diarization_threshold
        )

        self.diarization_models_dir = _resolve_path(
            cfg.diarization_models_dir
        )

        self.voices_registry_path = _resolve_path(
            cfg.voices_registry
        )

        # Backends (provisioned in start(), swappable in tests).
        self.transcriber: Any = None

        self.slm: Any = None

        self.tts: Any = None

        self.embedder: Any = None

        self.registry: Any = None

        self._state_lock = threading.Lock()

        self._state: str = "idle"

        # Whether the utterance currently playing may be
        # interrupted (fast-path replies always are; tasks carry
        # their own flag).
        self._speaking_interruptible = True

        self._transcripts: deque[dict[str, Any]] = deque(
            maxlen=self.transcript_history,
        )

        self._answered: deque[dict[str, Any]] = deque(
            maxlen=self.transcript_history,
        )

        self._stats: dict[str, Any] = {
            "transcripts_total": 0,
            "fast_answers_total": 0,
            "tasks_total": 0,
            "compose_failures_total": 0,
            "bargeins_total": 0,
            "stt_errors_total": 0,
            "last_transcript": None,
            "last_answered": None,
            "load_error": None,
        }

        self._stop_event = threading.Event()

        # Wake the speak thread when a task lands on the slot.
        self._say_event = threading.Event()

        # Barge-in signal: listen thread sets, speak thread obeys.
        self._barge_event = threading.Event()

        # Serializes playback between the fast path (analyze
        # thread) and say tasks (speak thread).
        self._play_lock = threading.Lock()

        self._utterance_queue: queue.Queue[Any] = queue.Queue(
            maxsize=32
        )

        self._pipeline: Any = None

        self._threads: list[threading.Thread] = []

        self._publish_task: Any = None

    # ==================================================================
    # Lifecycle
    # ==================================================================

    async def start(self) -> None:
        logger.info("voice module starting")

        self._provision()

        for target, name in (
            (self._listen_loop, "voice-listen"),
            (self._analyze_loop, "voice-analyze"),
            (self._speak_loop, "voice-speak"),
        ):
            thread = threading.Thread(
                target=target,
                name=name,
                daemon=True,
            )

            thread.start()

            self._threads.append(thread)

        self._publish_task = asyncio.create_task(
            self._publish_ticker(),
        )

        # Park forever: a returning start() would spawn duplicate
        # thread triples (same stance as the audio module).
        try:
            await asyncio.Event().wait()

        finally:
            self._stop_event.set()

    async def stop(self) -> None:
        self._stop_event.set()

        for thread in self._threads:
            thread.join(timeout=3.0)

        if self._publish_task is not None:
            self._publish_task.cancel()

        logger.info("voice module stopped")

    def _provision(self) -> None:
        """
        Load every backend before the loops start. Failures raise
        out of start(): DOWN + backoff, never a silent limbo.
        """
        reader = self.dependencies.get("audio")

        if reader is None:
            raise RuntimeError(
                "voice requires the audio module (pcm_ring facts); "
                "audio is not loaded"
            )

        ring = reader.snapshot().get("pcm_ring")

        if not isinstance(ring, dict) or "seq" not in ring:
            raise RuntimeError(
                "audio has not published pcm_ring yet; voice "
                "retries with backoff until it does"
            )

        rate = int(ring.get("sample_rate") or 0)

        if rate not in (8000, 16000, 32000, 48000):
            raise RuntimeError(
                f"audio ring sample rate {rate} is not a "
                "webrtcvad-compatible rate"
            )

        self.sample_rate = rate

        self.transcriber = WhisperTranscriber(
            model_size=self.stt_model,
            models_dir=str(self.whisper_models_dir),
            language=self.stt_language,
        )

        self.transcriber.load()

        self.slm = SmallDialogue(
            model_path=self.slm_path,
            repo_id=self.slm_repo_id,
        )

        self.slm.load()

        self.tts = CosyVoiceTTS(
            checkout_dir=self.tts_checkout_dir,
            model_dir=self.tts_model_dir,
            prompt_wav=self.tts_reference_wav,
            speed=self.tts_speed,
        )

        self.tts.load()

        self.embedder = PyannoteEmbedder(
            cache_dir=self.diarization_models_dir,
            repo_id=self.diarization_repo,
            token=self.diarization_token,
        )

        self.embedder.load()

        self.registry = VoiceSpeakerRegistry(
            path=self.voices_registry_path,
            threshold=self.diarization_threshold,
        )

        self.registry.load()

        with self._state_lock:
            self._state = "listening"

        logger.info(
            "voice backends ready (stt+slm+tts+diarization)"
        )

    # ==================================================================
    # Threads
    # ==================================================================

    def _listen_loop(self) -> None:
        """
        Consume the audio ring. While speaking: bare VAD feeds
        the barge-in counter. Otherwise frames feed the utterance
        pipeline (endpointing). The pipeline is rebuilt after
        every speaking -> listening transition: audio that rode
        under our own playback is echo, not conversation.
        """
        consumer_seq: int | None = None

        barge_frames = 0

        barge_need = max(
            1,
            int(
                self.barge_in_min_speech_s
                * 1000
                // self.frame_ms
            ),
        )

        was_speaking = False

        while not self._stop_event.is_set():
            ring = self._read_ring()

            if ring is None:
                consumer_seq = None

                self._stop_event.wait(0.5)

                continue

            chunks, consumer_seq = self._consume_ring(
                ring,
                consumer_seq,
            )

            if consumer_seq is None:
                # Malformed ring (unparsable seq): wait instead
                # of spinning on the same snapshot.
                self._stop_event.wait(0.5)

                continue

            speaking = self._current_state() == "speaking"

            if was_speaking and not speaking:
                # Speaking just ended (finished or barge-in):
                # start endpointing from a clean slate.
                self._pipeline = None

                barge_frames = 0

            was_speaking = speaking

            for encoded in chunks:
                pcm = base64.b64decode(encoded)

                if speaking:
                    interruptible = self._speaking_interruptible

                    if interruptible and self._vad.classify(
                        pcm,
                        self.sample_rate,
                        -120.0,
                    ):
                        barge_frames += 1

                        if barge_frames >= barge_need:
                            self._trigger_barge_in()

                            barge_frames = 0

                            speaking = False

                    else:
                        barge_frames = 0

                else:
                    barge_frames = 0

                    self._feed_pipeline(pcm)

            self._stop_event.wait(self.frame_ms / 1000.0)

    @property
    def _vad(self) -> VadGate:
        """
        Bare webrtcvad gate for barge-in; created lazily so
        tests without the wheel still construct the module.
        """
        gate = getattr(self, "_vad_gate", None)

        if gate is None:
            gate = VadGate(self.vad_aggressiveness)

            self._vad_gate = gate

        return gate

    def _read_ring(self) -> dict[str, Any] | None:
        reader = self.dependencies.get("audio")

        if reader is None:
            return None

        ring = reader.snapshot().get("pcm_ring")

        return ring if isinstance(ring, dict) else None

    @staticmethod
    def _consume_ring(
        ring: dict[str, Any],
        consumer_seq: int | None,
    ) -> tuple[list[str], int | None]:
        """
        Slice the ring into chunks this consumer has not seen.

        Chunks are keyed by arithmetic on the publisher's
        monotonic seq (chunk i is global frame seq - len + i).
        A fresh consumer -- or one the window overtook -- skips
        history and starts from now.
        """
        chunks = ring.get("chunks") or []

        try:
            seq = int(ring.get("seq") or 0)

        except (TypeError, ValueError):
            return [], None

        first = seq - len(chunks)

        if consumer_seq is None or first > consumer_seq:
            return [], seq

        start = consumer_seq - first

        return chunks[start:], seq

    def _feed_pipeline(self, pcm: bytes) -> None:
        if self._pipeline is None:
            self._pipeline = AudioPipeline(
                vad_aggressiveness=self.vad_aggressiveness,
                preroll_frames=self.preroll_frames,
                trailing_silence_frames=(
                    self.trailing_silence_frames
                ),
                min_utterance_frames=self.min_utterance_frames,
                max_utterance_frames=self.max_utterance_frames,
            )

        for event in self._pipeline.process(pcm):
            if event["type"] != "utterance":
                continue

            try:
                self._utterance_queue.put_nowait(
                    event["utterance"]
                )

            except queue.Full:
                logger.warning(
                    "voice utterance queue full; dropped one"
                )

    def _trigger_barge_in(self) -> None:
        logger.info("barge-in detected")

        self._barge_event.set()

    def _analyze_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                utt = self._utterance_queue.get(
                    timeout=self.say_poll_s,
                )

            except queue.Empty:

                continue

            try:
                self._handle_utterance(utt)

            except Exception as exc:
                logger.exception(
                    "voice utterance handling failed: {}", exc
                )

                with self._state_lock:
                    self._stats["load_error"] = str(exc)

    def _handle_utterance(self, utt: Any) -> None:
        """
        One endpointed utterance: transcribe, publish the fact,
        then try the fast path. Non-trivial turns are escalated
        by doing nothing further -- the transcript is already a
        fact the main agent reads through query().
        """
        result = self.transcriber.transcribe(utt)

        text = (result.get("text") or "").strip()

        if not text:
            return

        now_ts = time.time()

        # Diarization: who said this? A broken embedder must not
        # kill the conversational loop -- log and continue
        # unlabeled (the transcript itself still stands).
        speaker = ""

        if self.embedder is not None:
            try:
                vector = self.embedder.embed(utt.pcm)

                speaker = self.registry.match_or_enroll(vector)

            except Exception as exc:
                logger.warning(
                    "voice diarization failed: {}", exc
                )

        with self._state_lock:
            self._transcripts.append(
                {
                    "ts": now_ts,
                    "text": text,
                    "speaker": speaker,
                }
            )

            self._stats["transcripts_total"] += 1

            self._stats["last_transcript"] = text

        logger.info(
            "voice heard ({}): {}",
            speaker or "unknown",
            text,
        )

        if not self.fast_path_enabled:
            return

        # A pending say task always preempts fast-path chatter.
        if self.current_target("say") is not None:
            return

        reply = self.slm.fast_reply(text)

        if reply is None:
            return

        with self._state_lock:
            self._answered.append(
                {"ts": time.time(), "text": reply}
            )

            self._stats["fast_answers_total"] += 1

            self._stats["last_answered"] = reply

        logger.info("voice answered myself: {}", reply)

        self._speak_text(
            reply,
            instruct="",
            source="fast",
            interruptible=True,
        )

    def _speak_loop(self) -> None:
        while not self._stop_event.is_set():
            if not self._speak_once():
                self._say_event.wait(self.say_poll_s)

                self._say_event.clear()

    def _speak_once(self) -> bool:
        """
        Consume one say task (or return False when the slot is
        empty): compose, then play. Barge-in drains the FIFO.
        """
        task = self.current_target("say")

        if task is None:
            return False

        # Consume immediately: barge-in drains whatever is
        # left behind this task.
        self.clear_target("say")

        with self._state_lock:
            self._stats["tasks_total"] += 1

        composed = self._compose(task)

        if composed is None:
            return True

        self._speak_text(
            composed["text"],
            instruct=composed["instruct"],
            source="task",
            interruptible=bool(
                task.get("interruptible", True)
            ),
        )

        return True

    def _compose(self, task: Mapping[str, Any]) -> dict[str, str] | None:
        """
        Semantic task -> surface utterance. Composition failures
        are one-shot feedback for the model, never crashes.
        """
        with self._state_lock:
            self._state = "thinking"

        try:
            composed = self.slm.compose(task)

        except Exception as exc:
            logger.warning("voice compose failed: {}", exc)

            composed = None

        if composed is None:
            with self._state_lock:
                self._stats["compose_failures_total"] += 1

                self._state = "listening"

            self.emit_event(
                "voice: SLM failed to compose the say task; "
                "skipped (rewrite the task and resend)"
            )

            return None

        # Loud whitelist enforcement (no silent substitution):
        # an illegal instruct becomes neutral speech, but the
        # model is told exactly what the SLM hallucinated.
        raw_instruct = composed.get("instruct", "")

        instruct = sanitize_instruct(raw_instruct)

        if raw_instruct and not instruct:
            self.emit_event(
                "voice: SLM instruct not on the TTS whitelist, "
                f"falling back to neutral: {raw_instruct!r}"
            )

            composed["instruct"] = ""

        return composed

    def _speak_text(
        self,
        text: str,
        instruct: str = "",
        source: str = "task",
        interruptible: bool = True,
    ) -> None:
        """
        Shared playback path (fast path and say tasks). Holds
        the play lock so speech never overlaps itself; obeys
        barge-in for interruptible speech; a say task preempts
        fast-path playback mid-stream.
        """
        with self._play_lock:
            self._barge_event.clear()

            with self._state_lock:
                self._speaking_interruptible = interruptible

                self._state = "speaking"

            stream: Any = None

            try:
                for audio in self.tts.speak(
                    text,
                    instruct=instruct,
                ):
                    if self._barge_event.is_set():
                        break

                    if (
                        source == "fast"
                        and self.current_target("say")
                        is not None
                    ):
                        break

                    if stream is None:
                        stream = self._open_stream()

                    stream.write(audio.reshape(-1, 1))

            except Exception as exc:
                logger.warning("voice playback failed: {}", exc)

                self.emit_event(
                    f"voice: playback failed ({exc})"
                )

            finally:
                if stream is not None:
                    try:
                        stream.stop()

                        stream.close()

                    except Exception:

                        pass

            interrupted = self._barge_event.is_set()

            with self._state_lock:
                self._speaking_interruptible = True

                self._state = "listening"

        if interrupted:
            # Drain the FIFO and surface the interruption once.
            while self.current_target("say") is not None:
                self.clear_target("say")

            with self._state_lock:
                self._stats["bargeins_total"] += 1

            self.emit_event("user interrupted")

    def _open_stream(self) -> Any:
        stream = sd.OutputStream(
            samplerate=self.tts.sample_rate,
            channels=1,
            dtype="float32",
        )

        stream.start()

        return stream

    def on_target(
        self,
        channel: str,
        payload: Any,
    ) -> bool:
        # Accept every well-formed payload and wake the speak
        # thread immediately.
        self._say_event.set()

        return True

    def _current_state(self) -> str:
        with self._state_lock:
            return self._state

    # ==================================================================
    # Publishing
    # ==================================================================

    async def _publish_ticker(self) -> None:
        while True:
            await asyncio.sleep(self.publish_interval)

            with self._state_lock:
                payload = {
                    key: value
                    for key, value in self._stats.items()
                }

                payload["state"] = self._state

            self.data.publish(payload)

    # ==================================================================
    # Module contract
    # ==================================================================

    async def query(self, turn: Any) -> str | None:
        with self._state_lock:
            state = self._state

            stats = dict(self._stats)

            transcripts = list(self._transcripts)

            answered = list(self._answered)

        lines = [f"[Voice] state: {state}"]

        if self.dependencies.get("audio") is None:
            lines.append("- audio ring unavailable")

        for item in reversed(transcripts):
            clock = datetime.fromtimestamp(item["ts"]).strftime(
                "%H:%M",
            )

            speaker = item.get("speaker") or ""

            who = f" ({speaker})" if speaker else ""

            lines.append(
                f'- user said{who} {clock}: "{item["text"]}"'
            )

        for item in reversed(answered):
            clock = datetime.fromtimestamp(item["ts"]).strftime(
                "%H:%M",
            )

            lines.append(
                f'- answered myself {clock}: "{item["text"]}"'
            )

        if stats.get("load_error"):
            lines.append(f"- error: {stats['load_error']}")

        section = self.render_action_section(turn)

        if section:
            lines.append(section)

        if len(lines) == 1 and state == "idle":
            return None

        return "\n".join(lines)

    # ==================================================================
    # Persistence (counters + the last transcript ring only;
    # channel residue is never persisted)
    # ==================================================================

    def serialize_state(self) -> dict[str, Any]:
        with self._state_lock:
            return {
                "transcripts_total": self._stats[
                    "transcripts_total"
                ],
                "fast_answers_total": self._stats[
                    "fast_answers_total"
                ],
                "tasks_total": self._stats["tasks_total"],
                "compose_failures_total": self._stats[
                    "compose_failures_total"
                ],
                "bargeins_total": self._stats["bargeins_total"],
                "last_transcripts": list(self._transcripts),
            }

    def restore_state(self, state: Any) -> None:
        if not isinstance(state, dict):
            raise TypeError("voice private state must be an object")

        total_keys = (
            "transcripts_total",
            "fast_answers_total",
            "tasks_total",
            "compose_failures_total",
            "bargeins_total",
        )

        for key in total_keys:
            value = state.get(key, 0)

            if not isinstance(value, int):
                raise TypeError(f"{key} must be an integer")

            self._stats[key] = value

        for item in state.get("last_transcripts") or []:
            if not isinstance(item, dict):
                raise TypeError(
                    "last_transcripts entries must be objects"
                )

            self._transcripts.append(
                {
                    "ts": float(item["ts"]),
                    "text": str(item["text"]),
                    "speaker": str(item.get("speaker") or ""),
                }
            )
