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

import base64
import json
import os
import queue
import threading
import time
import asyncio
from collections import deque
from datetime import datetime
from typing import Any

from loguru import logger
from pydantic import BaseModel

from nan_itself.utils.module_config import (
    load_module_config,
    resolve_path,
)
from nan_itself.utils.audio import (
    AudioPipeline,
    estimate_bpm,
    lpc_formants,
    OnnxEmotionRecognizer,
    pitch_register,
    SherpaAudioTagger,
    SherpaSpeakerEmbedder,
    SoundDeviceMicSource,
    SpeakerMatcher,
    Utterance,
)


class AudioConfig(BaseModel):
    """
    Module-private config: config/modules/audio.yaml over these
    defaults. Path-valued fields are repo-relative strings
    (resolve_path); `device` is the mic index/name.
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

        cfg = load_module_config("audio", AudioConfig)

        self.device: int | str | None = cfg.device

        self.speaker_model_path = resolve_path(
            cfg.speaker_model
        )

        self.registry_path = resolve_path(cfg.voices_registry)

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

        self.tagger_model_path = resolve_path(cfg.tagger_model)

        self.tagger_labels_path = resolve_path(
            cfg.tagger_labels
        )

        self.tagger_factory = lambda: SherpaAudioTagger(
            self.tagger_model_path,
            self.tagger_labels_path,
            top_k=self.tagger_top_k,
        )

        self._tagger: Any = None

        self.emotion_model_path = resolve_path(
            cfg.emotion_model
        )

        self.emotion_head_path = resolve_path(
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
