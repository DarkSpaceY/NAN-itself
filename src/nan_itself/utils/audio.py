# @builtin

"""
Audio DSP toolkit for the hearing module.

Everything here is deliberately free of async and of the Module
framework: pure functions, pure state machines, and thin hardware
adapters behind small interfaces.

Blocks (see docs/audio-design.md §2):

    FeatureTracker        RMS / noise floor / quiet span bookkeeping
    VadGate               webrtcvad wrapper w/ energy-gate fallback
    UtteranceSegmenter    pre-roll + trailing-silence utterance slicer
    AudioPipeline         composes the three above over 30ms frames
    WhisperTranscriber    lazy faster-whisper adapter
    SoundDeviceMicSource  default microphone source (sounddevice)

Frame contract: mono int16 PCM at 16 kHz; one frame == frame_ms.
"""

from __future__ import annotations

import array
import math
import queue
import time
from collections import deque
from dataclasses import dataclass
from typing import Iterator


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
    import array

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
    import numpy as np

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
    import numpy as np

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
        import numpy as np

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
    webrtcvad when available; energy gate fallback otherwise.

    The energy gate compares against a supplied reference (noise
    floor + margin), so it degrades gracefully instead of dying.
    """

    def __init__(
        self,
        aggressiveness: int = 2,
    ) -> None:
        self.engine = None

        try:
            import webrtcvad

            self.engine = webrtcvad.Vad(
                int(aggressiveness),
            )

        except Exception:

            self.engine = None

    @property
    def mode(self) -> str:
        return "webrtcvad" if self.engine else "energy"

    def classify(
        self,
        pcm: bytes,
        sample_rate: int,
        reference_dbfs: float,
        rise_db: float = 10.0,
    ) -> bool:
        if self.engine is not None:
            try:
                return self.engine.is_speech(
                    pcm,
                    sample_rate,
                )

            except Exception:

                # Malformed frame sizes fall back to energy.
                pass

        return rms_dbfs(pcm) > reference_dbfs + rise_db


@dataclass
class Utterance:
    pcm: bytes

    voiced_ms: int

    total_ms: int

    transient_events: list[dict] | None = None


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
        min_utterance_frames: int = 8,
        max_utterance_frames: int = 833,
    ) -> None:
        # 30ms frames => 300ms preroll, ~700ms tail,
        # 250ms minimum, ~25s maximum.
        self.preroll: deque[tuple[bytes, bool]] = deque(
            maxlen=max(1, preroll_frames),
        )

        self.trailing_limit = trailing_silence_frames

        self.min_voiced = min_utterance_frames

        self.max_total = max_utterance_frames

        self.buffer: list[tuple[bytes, bool]] = []

        self.voiced_count = 0

        self.silence_run = 0

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

        self.buffer = []

        self.voiced_count = 0

        self.silence_run = 0

        if voiced < self.min_voiced:
            return []

        pcm = b"".join(chunk for chunk, _ in buffered)

        return [
            Utterance(
                pcm=pcm,
                voiced_ms=voiced * 30,
                total_ms=len(buffered) * 30,
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

        # W2 ambient features: windowed means feed the classifier.
        self.feature_window.append(
            (
                zero_crossing_rate(pcm),
                spectral_centroid_hz(pcm),
                spectral_flatness(pcm),
            ),
        )

        zcr = sum(f[0] for f in self.feature_window) / len(
            self.feature_window,
        )

        centroid = sum(f[1] for f in self.feature_window) / len(
            self.feature_window,
        )

        flatness = sum(f[2] for f in self.feature_window) / len(
            self.feature_window,
        )

        self.ambient_kind = self.classifier.classify(
            is_speech=is_speech,
            dbfs=dbfs,
            noise_floor=self.tracker.noise_floor,
            zcr=zcr,
            flatness=flatness,
        )

        pitch_stats = self.pitch.feed(pcm)

        if is_speech and pitch_stats["f0_hz"] > 0:
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
            events.append(
                {"type": "utterance", "utterance": utt}
            )

        stats["mode"] = self.vad.mode

        stats["speech_active"] = self.speech_active

        stats["zcr"] = round(zcr, 3)

        stats["centroid_hz"] = round(centroid, 1)

        stats["flatness"] = round(flatness, 3)

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
        import sounddevice as sd

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

    def __init__(self, text: str = "echo") -> None:
        self.text = text

        self.calls = 0

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

        return {
            "text": full_text,
            "confidence": 0.9,
            "language": "zh",
            "words": words,
            "voiced_s": round(utt.voiced_ms / 1000.0, 3),
        }


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
    ) -> None:
        self.model_size = model_size

        self.models_dir = models_dir

        self.language = language

        self.cpu_threads = cpu_threads

        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return

        from faster_whisper import WhisperModel

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
        import numpy as np

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
            parts.append(seg.text.strip())

            if seg.avg_logprob is not None:
                confs.append(math.exp(seg.avg_logprob))

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

        return {
            "text": text,
            "confidence": confidence,
            "language": getattr(info, "language", None),
            "words": words,
            "voiced_s": round(utt.voiced_ms / 1000.0, 3),
        }


# ======================================================================
# W4: speaker identity (embedding registry + streaming assignment)
# ======================================================================


def cosine_similarity(a, b) -> float:
    """
    Plain cosine over 1-D float vectors; zero vectors -> 0.0.
    """
    import numpy as np

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
        import numpy as np

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
        import numpy as np

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

        import sherpa_onnx

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
        import numpy as np

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
