# @module

"""
Audio: hearing as an autonomous builtin Module.

This module owns the microphone. It publishes everything it
hears as facts and never interprets speech content -- STT
lives downstream (the voice module) which consumes the
rolling PCM ring published here.

Layers:

    L1  energy      RMS + noise floor bookkeeping
    L4  events      VAD slicing, silence spans, transient bangs
    L5  identity    speaker match, ambient/utterance tags, emotion

Two daemon threads outside the event loop:

    capture     mic frames -> AudioPipeline -> utterance queue
    analyze     utterance queue -> tagging/speaker/emotion -> ring

query() is a pure projection of already-computed rings/stats;
it never touches DSP or LLM work. Missing weights or no input
device raise out of start() -- the Facade marks the module DOWN
and retries with backoff. A device lost mid-session keeps the
capture loop projecting the real error as available:false while
it reopens every retry_interval seconds.
"""

from __future__ import annotations

import array
import base64
import json
import math
import multiprocessing as mp
import os
import queue
import threading
import time
import asyncio
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import sounddevice as sd
import webrtcvad
import yaml
from faster_whisper import WhisperModel
from loguru import logger
from pydantic import BaseModel

# Guarded per project convention: sherpa_onnx's native library can
# fail to load on some Linux runners (libonnxruntime.so). Speaker
# embedding / audio tagging features report unavailable when it is
# missing; pure DSP helpers keep working.
try:
    import sherpa_onnx
except Exception:  # pragma: no cover - depends on runner native libs
    sherpa_onnx = None

from nan_itself.utils import paths as _paths


# ======================================================================
# Inlined audio DSP toolkit (formerly backend/nan_itself/utils/audio.py).
# utils/ is core-architecture only now: the audio module owns its whole
# hearing stack -- constants, feature helpers, pitch/ambient/level
# tracking, VAD gate, utterance segmentation, the pipeline, the mic
# source, the speaker/tagger/emotion backends and the whisper worker.
# Frame contract: mono int16 PCM at 16 kHz; one frame == frame_ms.
# Kept verbatim, free of async and Module-framework code.
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


class SoundDeviceMicSource:
    """
    Default microphone source built on sounddevice.

    The PortAudio callback only enqueues raw bytes; everything
    heavy happens on the consumer thread. Overflow drops NEW
    frames (oldest-first ordering preserved) and counts losses.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        blocksize: int | None = None,
        device: int | str | None = None,
        queue_size: int = 200,
    ) -> None:
        self.sample_rate = sample_rate

        # 30ms blocks by default: matches the VAD frame contract.
        self.blocksize = blocksize or (
            sample_rate * 30 // 1000
        )

        self.device = device

        self.dropped = 0

        self._queue: queue.Queue[bytes] = queue.Queue(
            maxsize=queue_size,
        )

        self._stream = None

    def start(self) -> None:
        def _callback(data, frames, t, status):
            payload = bytes(data)

            try:
                self._queue.put_nowait(payload)

            except queue.Full:

                self.dropped += 1

        self._stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=self.blocksize,
            channels=1,
            dtype="int16",
            device=self.device,
            callback=_callback,
        )

        self._stream.start()

    def read(self, timeout: float = 0.5) -> bytes | None:
        try:
            return self._queue.get(timeout=timeout)

        except queue.Empty:

            return None

    def close(self) -> None:
        stream = self._stream

        self._stream = None

        if stream is not None:
            try:
                stream.stop()

                stream.close()

            except Exception:

                pass


class EchoTranscriber:
    """
    Test double: pretends to transcribe by echoing metadata.
    Fabricates word timestamps from the text so word-level
    contracts can be exercised without whisper. Never touches
    faster-whisper.
    """

    WORD_SPAN_S = 0.2

    def __init__(self, text: str = "echo",
                 translate_enabled: bool = False) -> None:
        self.text = text

        self.calls = 0

        self.translate_enabled = translate_enabled

    def load(self) -> None:
        pass

    def transcribe(self, utt: Utterance) -> dict:
        self.calls += 1

        full_text = f"{self.text} #{self.calls}"

        tokens = full_text.split()

        words = [
            {
                "w": token,
                "start": round(i * self.WORD_SPAN_S, 2),
                "end": round(
                    (i + 0.75) * self.WORD_SPAN_S, 2
                ),
                "p": 0.9,
            }
            for i, token in enumerate(tokens)
        ]

        result = {
            "text": full_text,
            "confidence": 0.9,
            "language": "zh",
            "words": words,
            "voiced_s": round(utt.voiced_ms / 1000.0, 3),
        }

        if self.translate_enabled:
            result["translation"] = f"EN: {full_text}"

        return result


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
# W4: speaker identity (embedding registry + streaming assignment)
# ======================================================================


def cosine_similarity(a, b) -> float:
    """
    Plain cosine over 1-D float vectors; zero vectors -> 0.0.
    """
    va = np.asarray(a, dtype=np.float32)

    vb = np.asarray(b, dtype=np.float32)

    denom = float(
        np.linalg.norm(va) * np.linalg.norm(vb)
    )

    if denom <= 0:
        return 0.0

    return float(np.dot(va, vb) / denom)


class SpeakerMatcher:
    """
    Auto-enrolling voice registry for a permanently running ear.

    No agent, no tool, no manual file ever enrolls a voice:
    clusters open on first sight (voice-N, numbering monotonic
    across restarts), absorb nearest matches with a running-mean
    centroid, and PROMOTE into the persistent registry once they
    carry enough evidence (utterance count + accumulated voiced
    duration). Only promoted voices survive a reboot; candidates
    stay session-scoped. Serialization is JSON-native so the
    registry file stays human-inspectable.
    """

    def __init__(
        self,
        threshold: float = 0.62,
        promote_min_utterances: int = 5,
        promote_min_voiced_ms: float = 20000.0,
    ) -> None:
        self.threshold = threshold

        self.promote_min_utterances = promote_min_utterances

        self.promote_min_voiced_ms = promote_min_voiced_ms

        self.clusters: dict[str, dict] = {}

        self.persisted: set[str] = set()

        self.next_id = 1

    def labels(self) -> list[str]:
        return list(self.clusters)

    def known_count(self) -> int:
        """Persistent (promoted/restored) voice count."""
        return len(self.persisted & set(self.clusters))

    def assign(
        self,
        vec,
        voiced_ms: float = 0.0,
    ) -> tuple[str, float, bool]:
        """
        Route one utterance embedding to a voice.

        Returns (label, best_similarity, promoted_now). The
        score is the nearest similarity even when it falls
        below the threshold (a new cluster opens then).
        """
        now = time.time()

        best_label: str | None = None

        best_score = -1.0

        for label, cluster in self.clusters.items():
            score = cosine_similarity(
                vec,
                cluster["centroid"],
            )

            if score > best_score:
                best_label, best_score = label, score

        promoted_now = False

        if (
            best_label is not None
            and best_score >= self.threshold
        ):
            cluster = self.clusters[best_label]

            count = cluster["count"]

            blended = (
                cluster["centroid"] * count
                + np.asarray(vec, dtype=np.float32)
            )

            cluster["centroid"] = (
                blended / (count + 1)
            ).astype(np.float32)

            cluster["count"] = count + 1

            cluster["voiced_ms"] += voiced_ms

            cluster["last_seen"] = now

        else:
            best_label = f"voice-{self.next_id}"

            self.next_id += 1

            self.clusters[best_label] = {
                "centroid": np.asarray(
                    vec,
                    dtype=np.float32,
                ).copy(),
                "count": 1,
                "voiced_ms": float(voiced_ms),
                "created_at": now,
                "last_seen": now,
            }

            best_score = max(best_score, 0.0)

        cluster = self.clusters[best_label]

        if (
            best_label not in self.persisted
            and cluster["count"] >= self.promote_min_utterances
            and cluster["voiced_ms"]
            >= self.promote_min_voiced_ms
        ):
            self.persisted.add(best_label)

            promoted_now = True

        return best_label, best_score, promoted_now

    # ------------------------------------------------------------------
    # Registry persistence (JSON-native, human-inspectable)
    # ------------------------------------------------------------------

    def serialize(self) -> dict:
        voices: dict[str, dict] = {}

        for label in sorted(self.persisted):
            cluster = self.clusters.get(label)

            if cluster is None:
                continue

            voices[label] = {
                "centroid": [
                    round(float(x), 5)
                    for x in cluster["centroid"]
                ],
                "count": cluster["count"],
                "voiced_ms": round(
                    cluster["voiced_ms"], 1
                ),
                "created_at": cluster["created_at"],
                "last_seen": cluster["last_seen"],
            }

        return {
            "next_id": self.next_id,
            "voices": voices,
        }

    def restore(self, data: Any) -> None:
        if not isinstance(data, dict):
            raise TypeError("voice registry must be an object")

        voices = data.get("voices", {})

        next_id = data.get("next_id", 1)

        if not isinstance(voices, dict):
            raise TypeError("voices must be an object")

        if not isinstance(next_id, int) or next_id < 1:
            raise TypeError("next_id must be a positive int")

        self.clusters.clear()

        self.persisted.clear()

        self.next_id = next_id

        for label, payload in voices.items():
            centroid = payload.get("centroid")

            if (
                not isinstance(centroid, list)
                or not centroid
            ):
                continue

            self.clusters[label] = {
                "centroid": np.asarray(
                    centroid,
                    dtype=np.float32,
                ),
                "count": int(payload.get("count", 1)),
                "voiced_ms": float(
                    payload.get("voiced_ms", 0.0)
                ),
                "created_at": float(
                    payload.get("created_at", 0.0)
                ),
                "last_seen": float(
                    payload.get("last_seen", 0.0)
                ),
            }

            self.persisted.add(label)

        used = [
            int(label.split("-")[1])
            for label in self.clusters
            if label.startswith("voice-")
            and label.split("-")[1].isdigit()
        ]

        if used:
            self.next_id = max(self.next_id, max(used) + 1)


class SherpaSpeakerEmbedder:
    """
    sherpa-onnx speaker-embedding adapter (CAM++ / ECAPA-class
    ONNX models). Lazy import + lazy load; nothing here touches
    torch. Model binary lives under models/speaker/.
    """

    def __init__(
        self,
        model_path: str | Path,
    ) -> None:
        self.model_path = str(model_path)

        self._extractor: Any = None

    @property
    def loaded(self) -> bool:
        return self._extractor is not None

    def load(self) -> None:
        if self._extractor is not None:
            return

        if sherpa_onnx is None:
            raise RuntimeError(
                "sherpa_onnx is not importable on this machine; "
                "speaker embedding is unavailable"
            )

        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=self.model_path,
        )

        self._extractor = (
            sherpa_onnx.SpeakerEmbeddingExtractor(config)
        )

    @property
    def dim(self) -> int:
        self.load()

        return int(self._extractor.dim)

    def embed(self, pcm: bytes) -> Any | None:
        """
        int16 mono PCM -> embedding vector. None when the chunk
        carries no usable audio.
        """
        self.load()

        samples = np.frombuffer(
            bytes(pcm),
            dtype=np.int16,
        ).astype(np.float32) / 32768.0

        if samples.size == 0:
            return None

        stream = self._extractor.create_stream()

        stream.accept_waveform(SAMPLE_RATE, samples)

        stream.input_finished()

        vector = self._extractor.compute(stream)

        if not vector:
            return None

        return np.asarray(vector, dtype=np.float32)


# ======================================================================
# W5/W6: audio tagging (AudioSet) + speech emotion (emotion2vec)
# ======================================================================


class SherpaAudioTagger:
    """
    CED audio tagging via sherpa-onnx: PCM -> top-k AudioSet
    events with probabilities. Inference is ~20ms per 10s clip
    (int8 tiny), cheap enough to run inside the capture thread.
    """

    def __init__(
        self,
        model_path: str | Path,
        labels_path: str | Path,
        top_k: int = 5,
    ) -> None:
        self.model_path = str(model_path)

        self.labels_path = str(labels_path)

        self.top_k = top_k

        self._tagger: Any = None

    @property
    def loaded(self) -> bool:
        return self._tagger is not None

    def load(self) -> None:
        if self._tagger is not None:
            return

        if sherpa_onnx is None:
            raise RuntimeError(
                "sherpa_onnx is not importable on this machine; "
                "audio tagging is unavailable"
            )

        config = sherpa_onnx.AudioTaggingConfig(
            model=sherpa_onnx.AudioTaggingModelConfig(
                ced=self.model_path,
                num_threads=2,
            ),
            labels=self.labels_path,
            top_k=self.top_k,
        )

        self._tagger = sherpa_onnx.AudioTagging(config)

    def tag(self, pcm: bytes) -> list[tuple[str, float]]:
        self.load()

        samples = np.frombuffer(
            bytes(pcm),
            dtype=np.int16,
        ).astype(np.float32) / 32768.0

        if samples.size == 0:
            return []

        stream = self._tagger.create_stream()

        stream.accept_waveform(SAMPLE_RATE, samples)

        events = self._tagger.compute(stream)

        return [
            (event.name, float(event.prob))
            for event in events
        ]


class OnnxEmotionRecognizer:
    """
    emotion2vec ONNX (raw-waveform frontend baked in):

        waveform -> [1, T, 768] embedding -> mean-pool
                 -> linear head (W, B from head JSON) -> softmax

    Trained on zh+en emotional speech; 9 classes by default.
    """

    def __init__(
        self,
        model_path: str | Path,
        head_path: str | Path,
    ) -> None:
        self.model_path = str(model_path)

        self.head_path = str(head_path)

        self._session: Any = None

        self.head: dict[str, Any] = {}

    @property
    def loaded(self) -> bool:
        return self._session is not None

    def load(self) -> None:
        if self._session is not None:
            return

        self._session = ort.InferenceSession(
            self.model_path,
            providers=["CPUExecutionProvider"],
        )

        head = json.load(
            open(self.head_path, encoding="utf-8"),
        )

        self.head = {
            "labels": list(head["labels"]),
            "weight": np.asarray(
                head["weight"], dtype=np.float32,
            ),
            "bias": np.asarray(
                head["bias"], dtype=np.float32,
            ),
        }

    def recognize(self, pcm: bytes) -> dict | None:
        self.load()

        samples = np.frombuffer(
            bytes(pcm),
            dtype=np.int16,
        ).astype(np.float32) / 32768.0

        if samples.size == 0:
            return None

        input_name = self._session.get_inputs()[0].name

        feats = self._session.run(
            None,
            {input_name: samples.reshape(1, -1)},
        )[0]

        pooled = feats[0].mean(axis=0)

        logits = (
            self.head["weight"] @ pooled + self.head["bias"]
        )

        exp = np.exp(logits - logits.max())

        probs = exp / exp.sum()

        order = np.argsort(-probs)

        labels = self.head["labels"]

        return {
            "emotion": labels[int(order[0])],
            "prob": round(float(probs[order[0]]), 3),
            "probs": [
                (labels[int(i)], round(float(probs[i]), 3))
                for i in order[:3]
            ],
        }


# ======================================================================
# 🟡 sweep: cheap spectral/temporal extras (pure numpy)
# ======================================================================


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


def lpc_formants(
    pcm: bytes | memoryview,
    sample_rate: int = SAMPLE_RATE,
    order: int | None = None,
    max_frames: int = 24,
) -> list[float]:
    """
    Median F1/F2/F3 via LPC root analysis over the chunk.

    Autocorrelation method + Levinson-Durbin, per 40ms frame
    (50% hop); roots inside the unit circle with positive
    imaginary part are candidate formants. Pure numpy; runs
    per-utterance, never per-frame.
    """
    x = np.frombuffer(
        bytes(pcm),
        dtype=np.int16,
    ).astype(np.float32) / 32768.0

    if order is None:
        order = 2 + sample_rate // 1000

    frame = int(0.04 * sample_rate)

    hop = frame // 2

    if len(x) < frame or frame <= order + 2:
        return []

    formant_samples: list[list[float]] = []

    for start in range(0, len(x) - frame + 1, hop):
        chunk = x[start : start + frame]

        chunk = chunk - chunk.mean()

        if float((chunk * chunk).sum()) <= 1e-8:
            continue

        # Biased autocorrelation.
        ac = np.correlate(chunk, chunk, "full")[
            len(chunk) - 1 : len(chunk) - 1 + order + 1
        ]

        if ac[0] <= 0:
            continue

        ac = ac / ac[0]

        # Levinson-Durbin.
        a = np.zeros(order, dtype=np.float64)

        error = ac[0]

        for k in range(1, order + 1):
            reflection = ac[k] - np.dot(
                a[: k - 1][::-1],
                ac[1:k],
            )

            reflection /= error

            a[: k - 1] = (
                a[: k - 1] - reflection * a[: k - 1][::-1]
            )

            a[k - 1] = reflection

            error *= 1.0 - reflection * reflection

            if error <= 0:
                break

        if error <= 0:
            continue

        roots = np.roots(np.r_[1.0, -a])

        roots = roots[np.abs(roots) < 0.999]

        angles = np.angle(roots)

        freqs = np.abs(angles) * sample_rate / (2 * np.pi)

        freqs = np.sort(freqs)

        freqs = freqs[
            (freqs >= 150) & (freqs <= 4500)
        ]

        if len(freqs) >= 3:
            formant_samples.append(
                [float(f) for f in freqs[:3]]
            )

        if len(formant_samples) >= max_frames:
            break

    if not formant_samples:
        return []

    stacked = np.asarray(formant_samples)

    return [
        round(float(np.median(stacked[:, i])), 1)
        for i in range(3)
    ]


def estimate_bpm(
    pcm: bytes | memoryview,
    min_bpm: float = 60.0,
    max_bpm: float = 180.0,
) -> float | None:
    """
    Crude tempo from onset-envelope autocorrelation at 100Hz.
    Returns None when no periodicity stands out -- silence and
    plain speech yield None, steady music yields a number.
    """
    x = np.frombuffer(
        bytes(pcm),
        dtype=np.int16,
    ).astype(np.float32) / 32768.0

    if len(x) < SAMPLE_RATE:  # <1s: no tempo claim
        return None

    envelope = np.abs(x)

    kernel = np.ones(SAMPLE_RATE // 100) / (SAMPLE_RATE // 100)

    envelope = np.convolve(envelope, kernel, "same")

    envelope = envelope[:: 160]  # ~100Hz track

    envelope = envelope - envelope.mean()

    if float((envelope * envelope).sum()) <= 1e-9:
        return None

    ac = np.correlate(envelope, envelope, "full")[len(envelope) - 1 :]

    if ac[0] <= 0:
        return None

    ac = ac / ac[0]

    lo = int(60.0 / max_bpm * 100)

    hi = min(int(60.0 / min_bpm * 100), len(ac) - 1)

    if hi - lo < 5:
        return None

    segment = ac[lo:hi]

    peak = int(segment.argmax())

    strength = float(segment[peak])

    if strength < 0.25:
        return None

    return round(60.0 * 100.0 / (lo + peak), 1)


def pitch_register(f0s: list[float]) -> str | None:
    """
    Neutral pitch-register readout from voiced F0 samples.
    Deliberately NOT a gender/age guess -- that needs a real
    model and this layer refuses to fake one.
    """
    if len(f0s) < 10:
        return None

    median = float(np.median(np.asarray(f0s)))

    if median < 140.0:
        register = "low"

    elif median > 195.0:
        register = "high"

    else:
        register = "mid"

    return f"{register} ({median:.0f}Hz)"


# ======================================================================
# Whisper subprocess isolation (pathological-input armor)
# ======================================================================


def _whisper_worker_main(
    jobs: "queue.Queue",
    results: "queue.Queue",
    model_size: str,
    models_dir: str | None,
    language: str | None,
    translate_enabled: bool,
) -> None:
    """
    Subprocess body: owns the whisper model. A pathological
    segment (noise loops, memory balloon) degrades THIS process
    only; the supervisor restarts it on timeout.
    """
    kwargs: dict = {"device": "cpu", "compute_type": "int8"}

    if models_dir:
        kwargs["download_root"] = models_dir

    model = WhisperModel(model_size, **kwargs)

    results.put({"event": "ready"})

    while True:
        job = jobs.get()

        if job is None:
            return

        pcm, voiced_ms, translate = job

        try:
            audio = np.frombuffer(
                pcm,
                dtype=np.int16,
            ).astype(np.float32) / 32768.0

            segments, info = model.transcribe(
                audio,
                language=language,
                beam_size=1,
                word_timestamps=True,
            )

            parts: list[str] = []

            confs: list[float] = []

            words: list[dict] = []

            for seg in segments:
                no_speech = float(
                    getattr(seg, "no_speech_prob", 0.0) or 0.0
                )

                logprob = seg.avg_logprob

                confidence = (
                    math.exp(logprob)
                    if logprob is not None
                    else 0.5
                )

                # The anti-hallucination gate runs IN the worker:
                # a hanging segment is exactly what we never
                # want to shuttle across the boundary.
                if not (
                    no_speech > 0.6 and confidence < 0.5
                ):
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
                "voiced_s": round(voiced_ms / 1000.0, 3),
            }

            if translate and text:
                translated, _ = model.transcribe(
                    audio,
                    language=result["language"],
                    task="translate",
                    beam_size=1,
                )

                result["translation"] = " ".join(
                    seg.text.strip()
                    for seg in translated
                    if seg.text.strip()
                )

            results.put(result)

        except Exception as exc:
            results.put({"error": str(exc)})


class WhisperWorkerProxy:
    """
    Main-process handle to the whisper worker subprocess.

    transcribe() waits at most result_timeout seconds; on
    timeout the worker is presumed wedged (pathological input)
    and restarted. The agent process never hangs.
    """

    def __init__(
        self,
        model_size: str = "base",
        models_dir: str | None = None,
        language: str | None = None,
        translate_enabled: bool = False,
        result_timeout: float = 30.0,
        ready_timeout: float = 120.0,
    ) -> None:
        self.model_size = model_size

        self.models_dir = models_dir

        self.language = language

        self.translate_enabled = translate_enabled

        self.result_timeout = result_timeout

        self.ready_timeout = ready_timeout

        self.restarts = 0

        self.timeouts = 0

        self._jobs = None

        self._results = None

        self._process = None

    def _ensure(self) -> None:
        if self._process is not None and self._process.is_alive():
            return

        self._jobs = mp.Queue()

        self._results = mp.Queue()

        self._process = mp.Process(
            target=_whisper_worker_main,
            args=(
                self._jobs,
                self._results,
                self.model_size,
                self.models_dir,
                self.language,
                self.translate_enabled,
            ),
            daemon=True,
            name="whisper-worker",
        )

        self._process.start()

        # Wait out the model load in the worker; the ready marker
        # also proves the queue wiring works both directions.
        try:
            marker = self._results.get(timeout=self.ready_timeout)

        except Exception:
            self._restart()

            raise RuntimeError("whisper worker never became ready")

        if not (isinstance(marker, dict) and "event" in marker):
            # Not a marker? Put it back for the real consumer.
            self._results.put(marker)

    def transcribe(self, utt: Utterance) -> dict:
        self._ensure()

        try:
            self._jobs.put(
                (utt.pcm, utt.voiced_ms, self.translate_enabled),
            )

        except Exception as exc:
            self._restart()

            return {"error": str(exc), "text": ""}

        try:
            result = self._results.get(
                timeout=self.result_timeout,
            )

        except Exception:
            self.timeouts += 1

            self._restart()

            return {
                "error": "whisper worker timeout",
                "text": "",
            }

        return result

    def _restart(self) -> None:
        self.restarts += 1

        if self._process is not None:
            try:
                self._process.terminate()

                self._process.join(timeout=2)

            except Exception:

                pass

        self._process = None


class AudioConfig(BaseModel):
    """
    Module-private config: config/modules/audio.yaml over these
    defaults. Path-valued fields are repo-relative strings
    (_resolve_path); `device` is the mic index/name.
    """

    device: int | str | None = None

    speaker_model: str = (
        "models/speaker/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"
    )

    voices_registry: str = "data/databases/audio/voices.json"

    tagger_model: str = "models/audio_tag/model.int8.onnx"

    tagger_labels: str = (
        "models/audio_tag/class_labels_indices.csv"
    )

    emotion_model: str = (
        "models/emotion/emotion2vec_plus_base.onnx"
    )

    emotion_head: str = "models/emotion/emotion2vec_head.json"


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


def _load_config() -> AudioConfig:
    """
    Module-private config: config/modules/audio.yaml over the
    AudioConfig defaults. Missing or empty file = pure defaults;
    anything unparsable is a loud construction failure.
    """
    path = (
        _paths.repo_root()
        / "config"
        / "modules"
        / "audio.yaml"
    )

    if not path.is_file():
        return AudioConfig()

    data = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )

    if data is None:
        return AudioConfig()

    if not isinstance(data, dict):
        raise ValueError(
            f"Invalid module config YAML: {path}"
        )

    return AudioConfig(**data)


class AudioModule(Module):
    id = "audio"

    # ------------------------------------------------------------------
    # Configuration (instance attributes; tests may override)
    # ------------------------------------------------------------------

    frame_ms: int = 30

    sample_rate: int = 16000

    vad_aggressiveness: int = 2

    preroll_frames: int = 10

    trailing_silence_frames: int = 23

    min_utterance_frames: int = 20

    max_utterance_frames: int = 833

    transient_rise_db: float = 18.0

    ambient_window_frames: int = 33

    hear_history: int = 8

    hear_render_limit: int = 5

    quiet_report_after_s: float = 120.0

    retry_interval: float = 30.0

    read_timeout: float = 0.5

    publish_interval: float = 2.0

    # Rolling raw-PCM window published as facts for downstream
    # consumers (voice). The ambient tagger tags this same ring.
    pcm_ring_window_s: float = 10.0

    speaker_threshold: float = 0.62

    embed_min_voiced_ms: int = 400

    promote_min_utterances: int = 5

    promote_min_voiced_ms: float = 20000.0

    registry_save_throttle_s: float = 60.0

    tagger_top_k: int = 5

    ambient_tag_interval_s: float = 10.0

    utt_tag_min_prob: float = 0.4

    emotion_min_voiced_ms: int = 800

    def __init__(self) -> None:
        self.sample_rate = 16000

        cfg = _load_config()

        self.device: int | str | None = cfg.device

        self.speaker_model_path = _resolve_path(
            cfg.speaker_model
        )

        self.registry_path = _resolve_path(
            cfg.voices_registry
        )

        self.matcher = SpeakerMatcher(
            threshold=self.speaker_threshold,
            promote_min_utterances=self.promote_min_utterances,
            promote_min_voiced_ms=self.promote_min_voiced_ms,
        )

        self._registry_dirty = False

        # Start inside the throttle window: ordinary learning
        # saves at most once per window; promotions always write.
        self._registry_last_save = time.time()

        # Injectable like mic_factory; tests swap in fakes.
        self.embedder_factory = (
            lambda: SherpaSpeakerEmbedder(
                self.speaker_model_path,
            )
        )

        self._embedder: Any = None

        self.tagger_model_path = _resolve_path(
            cfg.tagger_model
        )

        self.tagger_labels_path = _resolve_path(
            cfg.tagger_labels
        )

        self.tagger_factory = lambda: SherpaAudioTagger(
            self.tagger_model_path,
            self.tagger_labels_path,
            top_k=self.tagger_top_k,
        )

        self._tagger: Any = None

        self.emotion_model_path = _resolve_path(
            cfg.emotion_model
        )

        self.emotion_head_path = _resolve_path(
            cfg.emotion_head
        )

        self.emotion_factory = lambda: OnnxEmotionRecognizer(
            self.emotion_model_path,
            self.emotion_head_path,
        )

        self._emotion: Any = None

        # Rolling raw-PCM window published as facts. The capture
        # thread appends; the publish ticker snapshots it from
        # the event loop, so all access is guarded by the lock.
        self._pcm_ring: deque[bytes] = deque()

        self._pcm_ring_bytes = 0

        self._pcm_ring_budget = int(
            self.pcm_ring_window_s
            * self.sample_rate
            * 2
        )

        # Monotonic counter of every chunk ever appended. Chunk
        # i in the ring corresponds to seq - len(ring) + i, so
        # consumers resync by arithmetic, not by handshake.
        self._pcm_seq = 0

        self._last_ambient_tag = 0.0

        self.pipeline = AudioPipeline(
            vad_aggressiveness=self.vad_aggressiveness,
            preroll_frames=self.preroll_frames,
            trailing_silence_frames=self.trailing_silence_frames,
            min_utterance_frames=self.min_utterance_frames,
            max_utterance_frames=self.max_utterance_frames,
            transient_rise_db=self.transient_rise_db,
            ambient_window_frames=self.ambient_window_frames,
        )

        self.mic_factory = lambda: SoundDeviceMicSource(
            sample_rate=self.sample_rate,
            blocksize=self.sample_rate * self.frame_ms // 1000,
            device=self.device,
        )

        self._state_lock = threading.Lock()

        self._heard: deque[dict[str, Any]] = deque(
            maxlen=self.hear_history,
        )

        self._turn_marks: deque[float] = deque(maxlen=20)

        self._stats: dict[str, Any] = {
            "available": False,
            "reason": "not started yet",
            "level_dbfs": -120.0,
            "noise_floor_dbfs": -60.0,
            "speech_active": False,
            "quiet_s": None,
            "vad_mode": "unknown",
            "last_heard_ts": None,
            "last_transient_ts": None,
            "utterances_total": 0,
            "transients_total": 0,
            "dropped_frames_total": 0,
            "dropped_utterances_total": 0,
            "load_error": None,
            "f0_hz": 0.0,
            "pitch_strength": 0.0,
            "pitch_trend": "",
            "speaker_backend": "not loaded",
            "last_speaker": None,
            "last_speaker_score": None,
            "voices_known": 0,
            "voices_session": 0,
            "tagger_backend": "not loaded",
            "ambient_tags": [],
            "last_utt_tags": [],
            "emotion_backend": "not loaded",
            "last_emotion": None,
            "last_emotion_prob": None,
            "emotions_total": 0,
            "bpm": None,
            "last_formants": [],
            "last_pauses": None,
            "last_register": None,
        }

        self._stop_event = threading.Event()

        self._utterance_queue: queue.Queue[Utterance] = (
            queue.Queue(maxsize=32)
        )

        self._mic: Any = None

        self._threads: list[threading.Thread] = []

        self._publish_task: Any = None

    # ==================================================================
    # Lifecycle
    # ==================================================================

    async def start(self) -> None:
        logger.info("audio module starting")

        self._load_registry()

        self._provision_backends()

        # A missing input device is a provisioning failure too:
        # open the mic here so absence crashes start() and the
        # Facade retries with backoff until hardware shows up.
        self._open_mic()

        for target, name in (
            (self._capture_loop, "audio-capture"),
            (self._analyze_loop, "audio-analyze"),
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

        # Park forever: a returning start() tells the Facade this
        # is a state-only module, and it will re-run start() every
        # few seconds -- spawning duplicate thread pairs each time.
        try:
            await asyncio.Event().wait()

        finally:
            # Cancelled by the Facade on shutdown: make sure the
            # daemon threads see the stop flag too.
            self._stop_event.set()

    async def stop(self) -> None:
        self._stop_event.set()

        for thread in self._threads:
            thread.join(timeout=3.0)

        if self._publish_task is not None:
            self._publish_task.cancel()

        self._close_mic()

        logger.info("audio module stopped")

    # ==================================================================
    # Backend provisioning
    # ==================================================================

    def _provision_backends(self) -> None:
        """
        Load every model-backed backend before the loops start.

        First-run weight downloads (tagging, speaker,
        emotion models) happen here, in the service lifetime
        phase -- never inside the tick loops. A failing backend
        raises out of start(): the Facade marks the module DOWN
        with the error and retries with backoff, so missing
        weights come up loudly failed and revive once they land.
        """
        for attr, stats_key, factory, label in (
            (
                "_embedder",
                "speaker_backend",
                self.embedder_factory,
                "speaker embedding",
            ),
            (
                "_tagger",
                "tagger_backend",
                self.tagger_factory,
                "audio tagger",
            ),
            (
                "_emotion",
                "emotion_backend",
                self.emotion_factory,
                "emotion",
            ),
        ):
            backend = factory()

            backend.load()

            setattr(self, attr, backend)

            with self._state_lock:
                self._stats[stats_key] = type(
                    backend
                ).__name__

            logger.info("{} backend ready: {}", label, type(backend).__name__)

    # ==================================================================
    # Threads
    # ==================================================================

    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._run_capture_session()

            except Exception as exc:  # hardware or portaudio errors
                with self._state_lock:
                    self._stats["available"] = False

                    self._stats["reason"] = str(exc)

                logger.warning(
                    "audio capture session failed: {}", exc
                )

            self._close_mic()

            self._stop_event.wait(self.retry_interval)

    def _run_capture_session(self) -> None:
        mic = self._open_mic()

        with self._state_lock:
            self._stats["available"] = True

            self._stats["reason"] = ""

        while not self._stop_event.is_set():
            pcm = mic.read(timeout=self.read_timeout)

            if pcm is None:
                continue

            if len(pcm) != self.sample_rate * self.frame_ms // 1000 * 2:
                # Partial device blocks are ignored until aligned.
                continue

            dropped_before = getattr(mic, "dropped", 0)

            events = self.pipeline.process(pcm)

            now = time.monotonic()

            with self._state_lock:
                self._pcm_ring.append(pcm)

                self._pcm_ring_bytes += len(pcm)

                self._pcm_seq += 1

                while self._pcm_ring_bytes > self._pcm_ring_budget:
                    dropped = self._pcm_ring.popleft()

                    self._pcm_ring_bytes -= len(dropped)

            if (
                now - self._last_ambient_tag
                >= self.ambient_tag_interval_s
            ):
                self._last_ambient_tag = now

                self._tag_ambient()

            with self._state_lock:
                stats = self.pipeline.latest_stats.copy()

                self._stats.update(stats)

                quiet = self.pipeline.quiet_seconds(now)

                self._stats["quiet_s"] = (
                    round(quiet, 1)
                    if quiet is not None
                    else None
                )

                self._stats["dropped_frames_total"] += (
                    mic.dropped - dropped_before
                )

            for event in events:
                if event["type"] == "transient":
                    with self._state_lock:
                        self._stats["transients_total"] += 1

                        self._stats["last_transient_ts"] = (
                            time.time()
                        )

                elif event["type"] == "utterance":
                    self._enqueue_utterance(event["utterance"])

    def _open_mic(self) -> Any:
        if self._mic is None:
            mic = self.mic_factory()

            mic.start()

            self._mic = mic

        return self._mic

    def _close_mic(self) -> None:
        if self._mic is not None:
            try:
                self._mic.close()

            except Exception:

                pass

            self._mic = None

    def _enqueue_utterance(self, utt: Utterance) -> None:
        try:
            self._utterance_queue.put_nowait(utt)

            with self._state_lock:
                self._stats["utterances_total"] += 1

        except queue.Full:

            with self._state_lock:
                self._stats["dropped_utterances_total"] += 1

    def _analyze_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                utt = self._utterance_queue.get(
                    timeout=self.read_timeout,
                )

            except queue.Empty:

                continue

            try:
                self._analyze_utterance(utt)

            except Exception as exc:
                logger.exception(
                    "audio utterance analysis failed: {}", exc
                )

                with self._state_lock:
                    self._stats["load_error"] = str(exc)

    def _analyze_utterance(self, utt: Utterance) -> None:
        """
        One finished utterance: record the event as a fact (no
        content -- transcription belongs downstream), then enrich
        the event with whatever the cheap DSP backends attribute.
        """
        with self._state_lock:
            self._heard.append(
                {
                    "ts": time.time(),
                    "voiced_ms": utt.voiced_ms,
                },
            )

            self._stats["last_heard_ts"] = (
                self._heard[-1]["ts"]
            )

        self._tag_utterance(utt)

        # W4: attribute the utterance to a voice when it is
        # long enough to carry a stable embedding.
        if utt.voiced_ms >= self.embed_min_voiced_ms:
            self._attribute_speaker(utt)

        # W6: paralinguistic emotion needs more audio than
        # speaker identity to mean anything.
        if utt.voiced_ms >= self.emotion_min_voiced_ms:
            self._recognize_emotion(utt)

        self._utterance_extras(utt)

    # ------------------------------------------------------------------
    # Speaker identity (W4)
    # ------------------------------------------------------------------

    def _get_embedder(self) -> Any | None:
        """
        Provisioned in start(); a failure raises there, so any
        running session has a loaded embedder (tagging runs on
        an independent tagger).
        """
        return self._embedder

    def _load_registry(self) -> None:
        """
        Restore the auto-enrolled voice registry at boot.

        A corrupt registry is renamed .corrupt-<ts> for forensics
        and the module starts with an empty one -- same stance as
        the memory module's storage files: never truncate, never
        die, wait for a human.
        """
        path = self.registry_path

        if not path.is_file():
            return

        try:
            payload = json.loads(
                path.read_text(encoding="utf-8"),
            )

            self.matcher.restore(payload)

        except Exception as exc:
            stamp = time.strftime("%Y%m%d-%H%M%S")

            corrupt = path.with_name(
                f"{path.name}.corrupt-{stamp}",
            )

            try:
                path.rename(corrupt)

            except OSError:

                pass

            logger.error(
                "voice registry corrupt ({}); quarantined "
                "as {}; starting empty",
                exc,
                corrupt.name,
            )

            self.matcher = SpeakerMatcher(
                threshold=self.speaker_threshold,
                promote_min_utterances=(
                    self.promote_min_utterances
                ),
                promote_min_voiced_ms=(
                    self.promote_min_voiced_ms
                ),
            )

        with self._state_lock:
            self._sync_voice_stats()

    def _save_registry(self) -> None:
        """
        Atomic write-through of the auto-enrolled registry.
        """
        self.registry_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = json.dumps(
            self.matcher.serialize(),
            ensure_ascii=False,
            indent=1,
        )

        tmp = self.registry_path.with_suffix(".json.tmp")

        tmp.write_text(payload, encoding="utf-8")

        os.replace(tmp, self.registry_path)

        self._registry_dirty = False

        self._registry_last_save = time.time()

    def _maybe_save_registry(self, promoted: bool) -> None:
        """
        Promotion forces a save; ordinary learning is throttled.
        """
        if promoted:
            self._save_registry()

            return

        if not self._registry_dirty:
            return

        if (
            time.time() - self._registry_last_save
            >= self.registry_save_throttle_s
        ):
            self._save_registry()

    def _sync_voice_stats(self) -> None:
        self._stats["voices_known"] = (
            self.matcher.known_count()
        )

        self._stats["voices_session"] = len(
            self.matcher.labels()
        )

    def _attribute_speaker(self, utt: Utterance) -> None:
        embedder = self._get_embedder()

        if embedder is None:
            return

        try:
            vector = embedder.embed(utt.pcm)

        except Exception as exc:
            logger.warning("speaker embedding failed: {}", exc)

            return

        if vector is None:
            return

        label, score, promoted = self.matcher.assign(
            vector,
            voiced_ms=utt.voiced_ms,
        )

        self._registry_dirty = True

        with self._state_lock:
            if self._heard:
                self._heard[-1]["speaker"] = label

            self._stats["last_speaker"] = label

            self._stats["last_speaker_score"] = (
                round(score, 3) if score is not None else None
            )

            self._sync_voice_stats()

        if promoted:
            logger.info(
                "voice promoted to persistent registry: {}",
                label,
            )

        self._maybe_save_registry(promoted)

    # ------------------------------------------------------------------
    # W5: audio tagging (ambient window + per-utterance)
    # ------------------------------------------------------------------

    def _get_tagger(self) -> Any | None:
        return self._tagger

    def _tag_ambient(self) -> None:
        """
        Tag the rolling PCM window. CED-tiny int8 infers ~20ms
        per 10s clip, so this runs inline on the capture thread.
        """
        if self._pcm_ring_bytes < self.sample_rate:  # <1s audio
            return

        tagger = self._get_tagger()

        if tagger is None:
            return

        with self._state_lock:
            window = b"".join(self._pcm_ring)

        try:
            events = tagger.tag(window)

        except Exception as exc:
            logger.warning("ambient tagging failed: {}", exc)

            return

        with self._state_lock:
            self._stats["ambient_tags"] = [
                [name, round(prob, 2)]
                for name, prob in events[:2]
            ]

        musicish = any(
            "music" in name.lower() or "singing" in name.lower()
            for name, _ in events
        ) or self.pipeline.ambient_kind == "tonal"

        if musicish:
            try:
                bpm = estimate_bpm(window)

            except Exception:
                bpm = None

            with self._state_lock:
                self._stats["bpm"] = bpm

    def _tag_utterance(self, utt: Utterance) -> None:
        tagger = self._get_tagger()

        if tagger is None:
            return

        try:
            events = tagger.tag(utt.pcm)

        except Exception as exc:
            logger.warning("utterance tagging failed: {}", exc)

            return

        # "Speech"/"Male speech"/"Silence" are redundant here --
        # the interesting part is what rides ON the speech.
        filtered = [
            (name, prob)
            for name, prob in events
            if "speech" not in name.lower()
            and "silence" not in name.lower()
        ]

        with self._state_lock:
            self._stats["last_utt_tags"] = [
                [name, round(prob, 2)]
                for name, prob in filtered[:2]
            ]

            if self._heard:
                for name, prob in filtered:
                    if prob >= self.utt_tag_min_prob:
                        self._heard[-1]["utt_tag"] = (
                            f"{name} {prob:.2f}"
                        )

                        break

    # ------------------------------------------------------------------
    # W6: paralinguistic emotion (emotion2vec)
    # ------------------------------------------------------------------

    def _get_emotion(self) -> Any | None:
        return self._emotion

    def _recognize_emotion(self, utt: Utterance) -> None:
        recognizer = self._get_emotion()

        if recognizer is None:
            return

        try:
            result = recognizer.recognize(utt.pcm)

        except Exception as exc:
            logger.warning("emotion recognition failed: {}", exc)

            return

        if not result:
            return

        with self._state_lock:
            if self._heard:
                self._heard[-1]["emotion"] = result["emotion"]

            self._stats["last_emotion"] = result["emotion"]

            self._stats["last_emotion_prob"] = result["prob"]

            self._stats["emotions_total"] = (
                self._stats.get("emotions_total", 0) + 1
            )

    def _utterance_extras(self, utt: Utterance) -> None:
        """
        🟡-sweep extras: pause structure, LPC formants, pitch
        register. All cheap numpy, all DataSpace/heard-only.
        """
        pauses = utt.pauses_ms or []

        formants = (
            lpc_formants(utt.pcm)
            if utt.voiced_ms >= 800
            else []
        )

        register = pitch_register(utt.f0s or [])

        with self._state_lock:
            if self._heard:
                self._heard[-1]["pauses"] = len(pauses)

                if formants:
                    self._heard[-1]["formants"] = formants

                if register:
                    self._heard[-1]["register"] = register

            self._stats["last_pauses"] = len(pauses)

            if formants:
                self._stats["last_formants"] = formants

            if register:
                self._stats["last_register"] = register

    # ==================================================================
    # Publishing
    # ==================================================================

    def _pcm_ring_snapshot(self) -> dict[str, Any]:
        """
        JSON-safe rolling PCM window for downstream consumers.

        `seq` counts every chunk ever appended; the chunk at
        index i corresponds to global sequence number
        seq - len(chunks) + i, so a consumer that has consumed
        up to S resyncs by arithmetic (S < seq - len(chunks)
        means the window overtook it: take everything). Chunks
        are base64 strings because DataSpace persists to JSON.
        """
        with self._state_lock:
            chunks = list(self._pcm_ring)

            seq = self._pcm_seq

        return {
            "seq": seq,
            "sample_rate": self.sample_rate,
            "frame_ms": self.frame_ms,
            "chunks": [
                base64.b64encode(chunk).decode("ascii")
                for chunk in chunks
            ],
        }

    async def _publish_ticker(self) -> None:
        while True:
            await asyncio.sleep(self.publish_interval)

            self._publish_once()

    def _publish_once(self) -> None:
        with self._state_lock:
            payload = {
                key: value
                for key, value in self._stats.items()
            }

        payload["pcm_ring"] = self._pcm_ring_snapshot()

        self.data.publish(payload)

    # ==================================================================
    # Module contract
    # ==================================================================

    async def on_turn(self, record: Turn) -> None:
        with self._state_lock:
            self._turn_marks.append(record.started_at)

    async def query(self, turn: Turn) -> str | None:
        with self._state_lock:
            stats = dict(self._stats)

            heard = list(self._heard)

        # One module, one territory: everything lives under the
        # module's own header; no module may mint global-looking
        # sections inside <module>.
        lines = ["[Audio]"]

        if not stats.get("available"):
            reason = stats.get("reason") or "no input"

            lines.append(f"- input unavailable: {reason}")

            return "\n".join(lines)

        quiet = stats.get("quiet_s")

        if stats.get("speech_active"):
            lines.append("- hearing: speech active")

            f0 = stats.get("f0_hz") or 0.0

            if f0 > 0:
                voice_line = f"- voice: ~{int(f0)} Hz"

                trend = stats.get("pitch_trend") or ""

                if trend:
                    voice_line += f" ({trend})"

                lines.append(voice_line)

        elif quiet is not None:
            lines.append(
                f"- hearing: quiet {self._fmt_span(quiet)}"
            )
        else:
            lines.append("- hearing: no signal yet")

        ambient_kind = stats.get("ambient_kind")

        if (
            ambient_kind
            and not stats.get("speech_active")
            and ambient_kind != "quiet"
        ):
            ambient_line = f"- ambient: {ambient_kind}"

            tags = stats.get("ambient_tags") or []

            if tags:
                ambient_line += (
                    " ("
                    + ", ".join(
                        f"{name} {prob:.2f}"
                        for name, prob in tags
                    )
                    + ")"
                )

            lines.append(ambient_line)

        lines.append(
            f"- level: {stats.get('level_dbfs')} dBFS "
            f"(noise floor {stats.get('noise_floor_dbfs')})"
        )

        transient_ts = stats.get("last_transient_ts")

        if (
            transient_ts
            and time.time() - transient_ts < 60.0
        ):
            lines.append(
                "- sharp sound detected at "
                f"{datetime.fromtimestamp(transient_ts).strftime('%H:%M')}"
            )

        heard_lines = self._render_heard(heard)

        lines.extend(heard_lines)

        if (
            not heard_lines
            and not stats.get("speech_active")
            and (quiet is None or quiet >= self.quiet_report_after_s)
            and stats.get("utterances_total", 0) == 0
        ):
            # Nothing has ever been heard, the room is dead and
            # no speech is in flight: stay silent instead of
            # spamming empty ambience.
            return None

        return "\n".join(lines)

    def _render_heard(
        self,
        heard: list[dict[str, Any]],
    ) -> list[str]:
        """
        Speech events without content: when something was heard,
        who it belonged to, and what rode on the voice. The words
        themselves are the voice module's territory.
        """
        out: list[str] = []

        for item in reversed(heard):
            clock = datetime.fromtimestamp(item["ts"]).strftime(
                "%H:%M",
            )

            tags = ""

            speaker = item.get("speaker")

            if speaker:
                tags += f" ({speaker})"

            voiced_ms = item.get("voiced_ms", 0)

            if voiced_ms >= 1000:
                tags += f" {voiced_ms / 1000:.1f}s"

            else:
                tags += f" {voiced_ms}ms"

            utt_tag = item.get("utt_tag")

            if utt_tag:
                tags += f" [{utt_tag}]"

            emotion = item.get("emotion")

            if emotion:
                tags += f" [emo:{emotion}]"

            out.append(f"- heard {clock}{tags}")

            if len(out) >= self.hear_render_limit:
                break

        return out

    def _fmt_span(self, seconds: float) -> str:
        if seconds < 90:
            return f"{int(seconds)}s"

        minutes = seconds / 60

        if minutes < 90:
            return f"{int(minutes)}m"

        hours = minutes / 60

        return f"{hours:.1f}h"

    # ==================================================================
    # Persistence (counters plus the noise-floor seed; senses reset
    # on reboot except the restored noise floor)
    # ==================================================================

    def serialize_state(self) -> dict[str, Any]:
        with self._state_lock:
            return {
                "utterances_total": self._stats[
                    "utterances_total"
                ],
                "transients_total": self._stats[
                    "transients_total"
                ],
                "noise_floor_seed": self.pipeline.tracker.noise_floor,
            }

    def restore_state(self, state: Any) -> None:
        if not isinstance(state, dict):
            raise TypeError("audio private state must be an object")

        seed = state.get("noise_floor_seed")

        if seed is not None:
            if not isinstance(seed, (int, float)):
                raise TypeError("noise_floor_seed must be numeric")

            self.pipeline.tracker.noise_floor = float(seed)

        total_keys = (
            "utterances_total",
            "transients_total",
        )

        for key in total_keys:
            value = state.get(key, 0)

            if not isinstance(value, int):
                raise TypeError(f"{key} must be an integer")
