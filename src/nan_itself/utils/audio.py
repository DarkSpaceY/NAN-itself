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
    ) -> None:
        self.tracker = FeatureTracker()

        self.vad = VadGate(vad_aggressiveness)

        self.classifier = AmbientClassifier()

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
    Never touches faster-whisper.
    """

    def __init__(self, text: str = "echo") -> None:
        self.text = text

        self.calls = 0

    def load(self) -> None:
        pass

    def transcribe(self, utt: Utterance) -> dict:
        self.calls += 1

        return {
            "text": f"{self.text} #{self.calls}",
            "confidence": 0.9,
            "language": "zh",
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
        import io

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
        )

        parts: list[str] = []

        confs: list[float] = []

        for seg in segments:
            parts.append(seg.text.strip())

            if seg.avg_logprob is not None:
                confs.append(math.exp(seg.avg_logprob))

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
        }
