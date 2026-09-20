"""
Audio DSP toolkit for the hearing module.

Everything here is deliberately free of async and of the Module
framework: pure functions, pure state machines, and thin hardware
adapters behind small interfaces.

Blocks (see docs/audio-design.md §2):

    FeatureTracker        RMS / noise floor / quiet span bookkeeping
    VadGate               webrtcvad wrapper
    UtteranceSegmenter    pre-roll + trailing-silence utterance slicer
    AudioPipeline         composes the three above over 30ms frames
    WhisperTranscriber    lazy faster-whisper adapter
    SoundDeviceMicSource  default microphone source (sounddevice)

Frame contract: mono int16 PCM at 16 kHz; one frame == frame_ms.
"""

from __future__ import annotations

import array
import json
import math
import multiprocessing as mp
import queue
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort
import sherpa_onnx
import sounddevice as sd
import webrtcvad
from faster_whisper import WhisperModel
from loguru import logger


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
