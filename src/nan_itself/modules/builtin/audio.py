# @builtin

"""
Audio: hearing as an autonomous builtin Module.

MVP layers (docs/audio-design.md):

    L1  energy      RMS + noise floor bookkeeping
    L4  events      VAD slicing, silence spans, transient bangs
    L7  content     faster-whisper transcription

Two daemon threads outside the event loop:

    capture     mic frames -> AudioPipeline -> utterance queue
    transcript  utterance queue -> WhisperTranscriber -> ring

query() is a pure projection of already-computed rings/stats;
it never touches DSP or LLM work. Hardware failure degrades to
available:false and retried every retry_interval seconds; the
agent process never dies because a microphone is missing.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
import asyncio
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from src.nan_itself.modules.model import (
    Module,
    ModuleTurn,
    TurnRecord,
)

from src.nan_itself.utils.audio import (
    AudioPipeline,
    SherpaSpeakerEmbedder,
    SoundDeviceMicSource,
    SpeakerMatcher,
    Utterance,
    WhisperTranscriber,
)


def _repo_root() -> Path:
    # .../src/nan_itself/modules/builtin/audio.py -> repo root
    return Path(__file__).resolve().parents[4]


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

    hear_preview_cap: int = 200

    dedup_window_s: float = 60.0

    quiet_report_after_s: float = 120.0

    retry_interval: float = 30.0

    read_timeout: float = 0.5

    publish_interval: float = 2.0

    whisper_model: str = "base"

    whisper_language: str | None = None

    hotwords: tuple[str, ...] = ()

    speaker_threshold: float = 0.62

    embed_min_voiced_ms: int = 400

    promote_min_utterances: int = 5

    promote_min_voiced_ms: float = 20000.0

    registry_save_throttle_s: float = 60.0

    def __init__(self) -> None:
        self.sample_rate = int(
            os.getenv("NAN_AUDIO_SAMPLE_RATE", "16000"),
        )

        self.device: int | str | None = (
            os.getenv("NAN_AUDIO_DEVICE") or None
        )

        self.whisper_model = os.getenv(
            "NAN_AUDIO_WHISPER_MODEL",
            "base",
        )

        hotword_env = os.getenv("NAN_AUDIO_HOTWORDS", "")

        self.hotwords = tuple(
            self._normalize_text(part)
            for part in hotword_env.split(",")
            if part.strip()
        )

        speaker_model = os.getenv("NAN_AUDIO_SPEAKER_MODEL")

        self.speaker_model_path = (
            Path(speaker_model)
            if speaker_model
            else _repo_root()
            / "models"
            / "speaker"
            / "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"
        )

        registry = os.getenv("NAN_AUDIO_VOICES_REGISTRY")

        self.registry_path = (
            Path(registry)
            if registry
            else _repo_root() / "data" / "audio" / "voices.json"
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

        self._embedder_failed = False

        models_dir = os.getenv("NAN_AUDIO_MODELS_DIR")

        self.models_dir = (
            Path(models_dir)
            if models_dir
            else _repo_root() / "models" / "whisper"
        )

        self.pipeline = AudioPipeline(
            vad_aggressiveness=self.vad_aggressiveness,
            preroll_frames=self.preroll_frames,
            trailing_silence_frames=self.trailing_silence_frames,
            min_utterance_frames=self.min_utterance_frames,
            max_utterance_frames=self.max_utterance_frames,
            transient_rise_db=self.transient_rise_db,
            ambient_window_frames=self.ambient_window_frames,
        )

        self.transcriber = WhisperTranscriber(
            model_size=self.whisper_model,
            models_dir=str(self.models_dir),
            language=self.whisper_language,
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
            "transcripts_total": 0,
            "transients_total": 0,
            "dedup_skipped_total": 0,
            "dropped_frames_total": 0,
            "dropped_utterances_total": 0,
            "load_error": None,
            "f0_hz": 0.0,
            "pitch_strength": 0.0,
            "pitch_trend": "",
            "last_speech_rate": None,
            "last_hotword": None,
            "hotwords_total": 0,
            "speaker_backend": "not loaded",
            "last_speaker": None,
            "last_speaker_score": None,
            "voices_known": 0,
            "voices_session": 0,
        }

        self._last_normalized_text: str = ""

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
        logger.info(
            "audio module starting (model={}, dir={})",
            self.whisper_model,
            self.models_dir,
        )

        self._load_registry()

        for target, name in (
            (self._capture_loop, "audio-capture"),
            (self._transcribe_loop, "audio-transcribe"),
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

    async def stop(self) -> None:
        self._stop_event.set()

        for thread in self._threads:
            thread.join(timeout=3.0)

        if self._publish_task is not None:
            self._publish_task.cancel()

        self._close_mic()

        logger.info("audio module stopped")

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

    def _transcribe_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                utt = self._utterance_queue.get(
                    timeout=self.read_timeout,
                )

            except queue.Empty:

                continue

            try:
                result = self.transcriber.transcribe(utt)

            except Exception as exc:
                logger.exception(
                    "audio transcription failed: {}", exc
                )

                with self._state_lock:
                    self._stats["load_error"] = str(exc)

                continue

            if not self._accept_transcript(result):
                continue

            # W4: attribute the utterance to a voice when it is
            # long enough to carry a stable embedding.
            if utt.voiced_ms >= self.embed_min_voiced_ms:
                self._attribute_speaker(utt)

    # ------------------------------------------------------------------
    # Speaker identity (W4)
    # ------------------------------------------------------------------

    def _get_embedder(self) -> Any | None:
        """
        Lazy singleton; a failing backend disables tagging for
        the session instead of poisoning every utterance.
        """
        if self._embedder is not None:
            return self._embedder

        if self._embedder_failed:
            return None

        try:
            embedder = self.embedder_factory()

            embedder.load()

        except Exception as exc:
            self._embedder_failed = True

            with self._state_lock:
                self._stats["speaker_backend"] = (
                    f"unavailable: {exc}"
                )

            logger.warning(
                "speaker embedding backend unavailable: {}",
                exc,
            )

            return None

        self._embedder = embedder

        with self._state_lock:
            self._stats["speaker_backend"] = type(
                embedder
            ).__name__

        return embedder

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
    # Transcript bookkeeping (extracted for direct unit driving)
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_text(text: str) -> str:
        return "".join(text.split()).lower()

    def _accept_transcript(self, result: dict[str, Any]) -> bool:
        """Record one transcription; False when deduplicated/empty."""
        text = (result.get("text") or "").strip()

        if not text:
            return False

        normalized = self._normalize_text(text)

        now_ts = time.time()

        with self._state_lock:
            if (
                normalized == self._last_normalized_text
                and self._heard
                and now_ts - self._heard[-1]["ts"]
                <= self.dedup_window_s
            ):
                self._stats["dedup_skipped_total"] += 1

                self._last_normalized_text = normalized

                return False

            self._last_normalized_text = normalized

            words = result.get("words") or []

            voiced_s = float(result.get("voiced_s") or 0.0)

            rate = (
                round(
                    len(words) / max(voiced_s, 0.3),
                    2,
                )
                if words and voiced_s > 0
                else None
            )

            hotword = next(
                (
                    hw
                    for hw in self.hotwords
                    if hw in normalized
                ),
                None,
            )

            self._heard.append(
                {
                    "ts": now_ts,
                    "text": text,
                    "confidence": round(
                        float(result.get("confidence", 0.0)),
                        2,
                    ),
                    "language": result.get("language"),
                    "rate": rate,
                    "hotword": hotword,
                },
            )

            self._stats["transcripts_total"] += 1

            self._stats["last_heard_ts"] = now_ts

            if rate is not None:
                self._stats["last_speech_rate"] = rate

            if hotword:
                self._stats["last_hotword"] = hotword

                self._stats["hotwords_total"] = (
                    self._stats.get("hotwords_total", 0) + 1
                )

        return True

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

            self.data.publish(payload)

    # ==================================================================
    # Module contract
    # ==================================================================

    async def on_turn(self, record: TurnRecord) -> None:
        with self._state_lock:
            self._turn_marks.append(record.started_at)

    async def query(self, turn: ModuleTurn) -> str | None:
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
            lines.append(f"- ambient: {ambient_kind}")

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
            and stats.get("transcripts_total", 0) == 0
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
        out: list[str] = []

        for item in reversed(heard):
            preview = item["text"][: self.hear_preview_cap]

            clock = datetime.fromtimestamp(item["ts"]).strftime(
                "%H:%M",
            )

            confidence = item.get("confidence", 0.0)

            tags = ""

            speaker = item.get("speaker")

            if speaker:
                tags += f" ({speaker})"

            if confidence < 0.6:
                tags += f" (conf {confidence})"

            hotword = item.get("hotword")

            if hotword:
                tags += f" [hot:{hotword}]"

            out.append(
                f'- heard {clock}{tags} "{preview}"'
            )

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
    # Persistence (counters only; senses reset on reboot)
    # ==================================================================

    def serialize_state(self) -> dict[str, Any]:
        with self._state_lock:
            return {
                "utterances_total": self._stats[
                    "utterances_total"
                ],
                "transcripts_total": self._stats[
                    "transcripts_total"
                ],
                "transients_total": self._stats[
                    "transients_total"
                ],
                "hotwords_total": self._stats.get(
                    "hotwords_total", 0
                ),
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
            "transcripts_total",
            "transients_total",
            "hotwords_total",
        )

        for key in total_keys:
            value = state.get(key, 0)

            if not isinstance(value, int):
                raise TypeError(f"{key} must be an integer")
