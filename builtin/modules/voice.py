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

import base64
import os
import queue
import threading
import time
import asyncio
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Mapping

import numpy as np
import sounddevice as sd
from loguru import logger
from pydantic import BaseModel

from nan_itself.modules.action import (
    ActionSurface,
    ChannelSpec,
)
from nan_itself.utils import paths as _paths
from nan_itself.utils.audio import (
    AudioPipeline,
    VadGate,
    WhisperTranscriber,
)
from nan_itself.utils.dialogue import (
    SmallDialogue,
    default_slm_path,
)
from nan_itself.utils.tts import (
    CosyVoiceTTS,
    default_tts_paths,
)


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

        self.stt_model = os.getenv(
            "NAN_VOICE_STT_MODEL",
            "base",
        )

        self.stt_language = (
            os.getenv("NAN_VOICE_STT_LANGUAGE") or None
        )

        self.fast_path_enabled = (
            os.getenv("NAN_VOICE_FAST_PATH", "1") == "1"
        )

        speed = os.getenv("NAN_VOICE_TTS_SPEED")

        if speed:
            self.tts_speed = float(speed)

        # Reuse the audio module's whisper weights.
        models_dir = os.getenv("NAN_VOICE_STT_MODELS_DIR")

        self.whisper_models_dir = (
            Path(models_dir)
            if models_dir
            else _paths.repo_root() / "models" / "whisper"
        )

        slm_dir = os.getenv("NAN_VOICE_SLM_DIR")

        self.slm_path = (
            Path(slm_dir)
            if slm_dir
            else default_slm_path()
        )

        self.slm_repo_id = (
            os.getenv("NAN_VOICE_SLM_REPO") or None
        )

        checkout, tts_dir, reference = default_tts_paths()

        voice_checkout = os.getenv("NAN_VOICE_COSYVOICE_DIR")

        self.tts_checkout_dir = (
            Path(voice_checkout)
            if voice_checkout
            else checkout
        )

        voice_tts = os.getenv("NAN_VOICE_TTS_DIR")

        self.tts_model_dir = (
            Path(voice_tts) if voice_tts else tts_dir
        )

        voice_reference = os.getenv("NAN_VOICE_TTS_REFERENCE")

        self.tts_reference_wav = (
            Path(voice_reference)
            if voice_reference
            else reference
        )

        # Backends (provisioned in start(), swappable in tests).
        self.transcriber: Any = None

        self.slm: Any = None

        self.tts: Any = None

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

        with self._state_lock:
            self._state = "listening"

        logger.info("voice backends ready (stt+slm+tts)")

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

        with self._state_lock:
            self._transcripts.append(
                {"ts": now_ts, "text": text}
            )

            self._stats["transcripts_total"] += 1

            self._stats["last_transcript"] = text

        logger.info("voice heard: {}", text)

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

            lines.append(f'- user said {clock}: "{item["text"]}"')

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
                {"ts": float(item["ts"]), "text": str(item["text"])}
            )
