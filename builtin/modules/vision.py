# @module

"""
Vision: seeing as an autonomous builtin Module.

Layers implemented (docs/vision-design.md):

    L1  photometry    brightness/contrast/clipping/sharpness/...
    L2  motion        frame-diff energy, quiet spans, cuts, light
    L3  structure     optical flow, motion blob, saliency
    L4  events        glance slicing (the visual utterance)
    L5  objects       faces via face_recognition; YOLO if a model
                      has been dropped into models/vision/object
    L6  semantics     person registry, QR, OCR (easyocr)
    L7  caption       VLM caption of the current glance
                      (SmolVLM2-500M-Video-Instruct)

Two daemon threads outside the event loop:

    capture     camera frames -> VisionPipeline -> glance queue
    inference   glance queue -> face/QR/OCR/YOLO/caption -> ring

query() is a pure projection of already-computed rings/stats; it
never touches models or OpenCV. Missing weights or no camera
raise out of start() -- the Facade marks the module DOWN and
retries with backoff. A camera lost mid-session keeps the
capture loop projecting the real error as available:false while
it reopens every retry_interval seconds.
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
from typing import Any

from loguru import logger
from pydantic import BaseModel

from nan_itself.modules.model import Module, Turn

from nan_itself.utils import paths as _paths

from nan_itself.utils.module_config import (
    load_module_config,
    resolve_path,
)

from nan_itself.utils.vision import (
    CameraSource,
    FaceAnalyzer,
    FaceMatcher,
    OcrReader,
    QrScanner,
    VisionPipeline,
    VlmCaptioner,
    YoloOnnxDetector,
)


class VisionConfig(BaseModel):
    """
    Module-private config: config/modules/vision.yaml over these
    defaults. Path-valued fields are repo-relative strings
    (resolve_path); `device` is the camera index/name.
    """

    device: int | str | None = None

    faces_enabled: bool = True

    ocr_enabled: bool = True

    vlm_enabled: bool = True

    faces_registry: str = "data/databases/vision/faces.json"

    ocr_models_dir: str = "models/easyocr"

    vlm_dir: str = (
        "models/vision/vlm/SmolVLM2-500M-Video-Instruct"
    )


class VisionModule(Module):
    id = "vision"

    # ------------------------------------------------------------------
    # Configuration (instance attributes; tests may override)
    # ------------------------------------------------------------------

    frame_width: int = 640

    frame_height: int = 480

    target_fps: float = 15.0

    small_width: int = 160

    motion_ratio_min: float = 0.02

    motion_energy_min: float = 2.0

    burst_ratio: float = 0.08

    preroll_frames: int = 8

    trailing_quiet_frames: int = 20

    min_glance_frames: int = 6

    max_glance_frames: int = 450

    see_history: int = 8

    see_render_limit: int = 5

    see_preview_cap: int = 160

    quiet_report_after_s: float = 300.0

    retry_interval: float = 30.0

    read_timeout: float = 0.5

    publish_interval: float = 2.0

    faces_enabled: bool = True

    face_distance_threshold: float = 0.5

    face_max_width: int = 640

    promote_min_sightings: int = 5

    promote_min_seen_ms: float = 30000.0

    registry_save_throttle_s: float = 60.0

    qr_enabled: bool = True

    ocr_enabled: bool = True

    ocr_interval_s: float = 20.0

    ocr_min_glance_ms: int = 1000

    ocr_dedup_window_s: float = 120.0

    objects_enabled: bool = True

    objects_interval_s: float = 10.0

    vlm_max_tokens: int = 48

    vlm_interval_s: float = 30.0

    def __init__(self) -> None:
        cfg = load_module_config("vision", VisionConfig)

        self.device: int | str | None = cfg.device

        self.faces_enabled = cfg.faces_enabled

        self.ocr_enabled = cfg.ocr_enabled

        self.registry_path = resolve_path(cfg.faces_registry)

        self.matcher = FaceMatcher(
            threshold=self.face_distance_threshold,
            promote_min_sightings=(
                self.promote_min_sightings
            ),
            promote_min_seen_ms=self.promote_min_seen_ms,
        )

        self._registry_dirty = False

        # Start inside the throttle window: ordinary learning
        # saves at most once per window; promotions always write.
        self._registry_last_save = time.time()

        self.pipeline = VisionPipeline(
            small_width=self.small_width,
            motion_ratio_min=self.motion_ratio_min,
            motion_energy_min=self.motion_energy_min,
            burst_ratio=self.burst_ratio,
            preroll_frames=self.preroll_frames,
            trailing_quiet_frames=(
                self.trailing_quiet_frames
            ),
            min_glance_frames=self.min_glance_frames,
            max_glance_frames=self.max_glance_frames,
        )

        # Injectable like camera_factory; tests swap in fakes.
        self.camera_factory = lambda: CameraSource(
            device=self.device,
            width=self.frame_width,
            height=self.frame_height,
        )

        self._camera: Any = None

        self.face_factory = lambda: FaceAnalyzer(
            max_width=self.face_max_width,
        )

        self._faces: Any = None

        self._faces_failed = self.faces_enabled is False

        self.qr = QrScanner()

        self.ocr_models_dir = resolve_path(cfg.ocr_models_dir)

        self.ocr_factory = lambda: OcrReader(
            self.ocr_models_dir,
        )

        self._ocr: Any = None

        self._ocr_failed = self.ocr_enabled is False

        self.objects_model_path = (
            _paths.repo_root()
            / "models"
            / "vision"
            / "object"
            / "model.onnx"
        )

        self.objects_labels_path = (
            self.objects_model_path.parent / "labels.txt"
        )

        self.objects_factory = lambda: YoloOnnxDetector(
            self.objects_model_path,
            self.objects_labels_path,
        )

        self._objects: Any = None

        self._objects_failed = self.objects_enabled is False

        self.vlm_dir = resolve_path(cfg.vlm_dir)

        self.vlm_enabled = cfg.vlm_enabled

        self.vlm_factory = lambda: VlmCaptioner(
            self.vlm_dir,
            max_tokens=self.vlm_max_tokens,
        )

        self._vlm: Any = None

        # Missing weights auto-download during provisioning; a
        # failed download raises out of start() (Facade retry).
        self._vlm_failed = self.vlm_enabled is False

        self._last_vlm_at = 0.0

        self._last_caption: str = ""

        self._state_lock = threading.Lock()

        self._seen: deque[dict[str, Any]] = deque(
            maxlen=self.see_history,
        )

        self._turn_marks: deque[float] = deque(maxlen=20)

        self._last_ocr_at = 0.0

        self._last_ocr_text: str = ""

        self._last_objects_at = 0.0

        self._last_qr_text: str = ""

        self._last_qr_at = 0.0

        self._person_in_view = False

        self._stats: dict[str, Any] = {
            "available": False,
            "reason": "not started yet",
            "motion_active": False,
            "motion_energy": 0.0,
            "quiet_s": None,
            "camera_motion": False,
            "occluded": False,
            "flicker": False,
            "brightness": 0.0,
            "contrast": 0.0,
            "sharpness": 0.0,
            "edges": 0.0,
            "entropy": 0.0,
            "colorfulness": 0.0,
            "cuts_total": 0,
            "light_total": 0,
            "glances_total": 0,
            "dropped_glances_total": 0,
            "dropped_frames_total": 0,
            "load_error": None,
            "last_motion_ts": None,
            "last_cut_ts": None,
            "last_light_ts": None,
            "last_glance_ts": None,
            "person_in_view": False,
            "last_person": None,
            "last_person_dist": None,
            "enters_total": 0,
            "leaves_total": 0,
            "last_enter_ts": None,
            "last_leave_ts": None,
            "faces_backend": (
                "disabled" if not self.faces_enabled
                else "not loaded"
            ),
            "faces_total": 0,
            "persons_known": 0,
            "persons_session": 0,
            "qr_total": 0,
            "last_qr": None,
            "ocr_backend": (
                "disabled" if not self.ocr_enabled
                else "not loaded"
            ),
            "ocr_total": 0,
            "last_ocr": None,
            "objects_backend": (
                "disabled"
                if self._objects_failed
                else "not loaded"
            ),
            "objects_total": 0,
            "last_objects": [],
            "vlm_backend": (
                "disabled"
                if self._vlm_failed
                else "not loaded (auto-download on first use)"
            ),
            "vlm_total": 0,
            "last_caption": None,
        }

        self._stop_event = threading.Event()

        self._glance_queue: queue.Queue[Any] = queue.Queue(
            maxsize=8,
        )

        self._threads: list[threading.Thread] = []

        self._publish_task: Any = None

    # ==================================================================
    # Lifecycle
    # ==================================================================

    async def start(self) -> None:
        logger.info(
            "vision module starting (registry={})",
            self.registry_path,
        )

        self._load_registry()

        self._provision_backends()

        # A missing camera is a provisioning failure too: open it
        # here so absence crashes start() and the Facade retries
        # with backoff until hardware shows up.
        self._open_camera()

        for target, name in (
            (self._capture_loop, "vision-capture"),
            (self._inference_loop, "vision-inference"),
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

        self._close_camera()

        logger.info("vision module stopped")

    # ==================================================================
    # Backend provisioning
    # ==================================================================

    def _provision_backends(self) -> None:
        """
        Load every enabled model-backed backend before the loops
        start.

        First-run weight downloads (and model RAM residency)
        happen here, in the service lifetime phase -- never
        inside the tick loops. A failing backend raises out of
        start(): the Facade marks the module DOWN with the error
        and retries with backoff, so missing weights come up
        loudly failed and revive once they land.
        """
        if self.objects_enabled and not (
            self.objects_model_path.is_file()
            and self.objects_labels_path.is_file()
        ):
            raise RuntimeError(
                "objects backend: drop model.onnx + labels.txt "
                "into models/vision/object/"
            )

        for attr, failed, stats_key, factory, label in (
            (
                "_faces",
                "_faces_failed",
                "faces_backend",
                self.face_factory,
                "face",
            ),
            (
                "_ocr",
                "_ocr_failed",
                "ocr_backend",
                self.ocr_factory,
                "ocr",
            ),
            (
                "_objects",
                "_objects_failed",
                "objects_backend",
                self.objects_factory,
                "object",
            ),
            (
                "_vlm",
                "_vlm_failed",
                "vlm_backend",
                self.vlm_factory,
                "vlm",
            ),
        ):
            if getattr(self, failed):
                continue

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
        frame_interval = 1.0 / max(1.0, self.target_fps)

        while not self._stop_event.is_set():
            try:
                self._run_capture_session(frame_interval)

            except Exception as exc:  # hardware errors
                with self._state_lock:
                    self._stats["available"] = False

                    self._stats["reason"] = str(exc)

                logger.warning(
                    "vision capture session failed: {}", exc
                )

            self._close_camera()

            self._stop_event.wait(self.retry_interval)

    def _run_capture_session(
        self,
        frame_interval: float,
    ) -> None:
        camera = self._open_camera()

        with self._state_lock:
            self._stats["available"] = True

            self._stats["reason"] = ""

        while not self._stop_event.is_set():
            started = time.monotonic()

            frame = camera.read(timeout=self.read_timeout)

            if frame is None:
                continue

            dropped_before = camera.dropped

            events = self.pipeline.process(frame)

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
                    camera.dropped - dropped_before
                )

            for event in events:
                kind = event["type"]

                if kind == "motion_start":
                    with self._state_lock:
                        self._stats["last_motion_ts"] = (
                            time.time()
                        )

                elif kind == "light_change":
                    with self._state_lock:
                        self._stats["last_light_ts"] = (
                            time.time()
                        )

                elif kind == "scene_cut":
                    with self._state_lock:
                        self._stats["last_cut_ts"] = (
                            time.time()
                        )

                elif kind == "glance":
                    self._enqueue_glance(event["glance"])

            # Hold the target frame rate; processing above is
            # already most of the budget at 15 fps.
            elapsed = time.monotonic() - started

            if elapsed < frame_interval:
                self._stop_event.wait(
                    frame_interval - elapsed,
                )

    def _open_camera(self) -> Any:
        if self._camera is None:
            camera = self.camera_factory()

            camera.start()

            self._camera = camera

        return self._camera

    def _close_camera(self) -> None:
        if self._camera is not None:
            try:
                self._camera.close()

            except Exception:

                pass

            self._camera = None

    def _enqueue_glance(self, glance: Any) -> None:
        try:
            self._glance_queue.put_nowait(glance)

            with self._state_lock:
                self._stats["glances_total"] += 1

                self._stats["last_glance_ts"] = time.time()

        except queue.Full:

            with self._state_lock:
                self._stats["dropped_glances_total"] += 1

    # ==================================================================
    # Inference: one glance keyframe at a time
    # ==================================================================

    def _inference_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                glance = self._glance_queue.get(
                    timeout=self.read_timeout,
                )

            except queue.Empty:

                continue

            try:
                self._analyze_glance(glance)

            except Exception as exc:
                logger.exception(
                    "vision analysis failed: {}", exc
                )

                with self._state_lock:
                    self._stats["load_error"] = str(exc)

    def _analyze_glance(self, glance: Any) -> None:
        entry: dict[str, Any] = {
            "ts": time.time(),
            "reason": glance.reason,
            "duration_ms": glance.duration_ms,
            "peak": glance.peak_energy,
        }

        faces = self._recognize_faces(glance, entry)

        self._presence(faces, entry)

        self._scan_qr(glance, entry)

        self._read_text(glance, entry)

        self._detect_objects(glance, entry)

        self._caption_glance(glance, entry)

        with self._state_lock:
            self._seen.append(entry)

    # ------------------------------------------------------------------
    # Faces + person registry (L5/L6)
    # ------------------------------------------------------------------

    def _get_faces(self) -> Any | None:
        """
        Provisioned in start(); a failure raises there, so any
        running session has a loaded analyzer.
        """
        return self._faces

    def _recognize_faces(
        self,
        glance: Any,
        entry: dict[str, Any],
    ) -> list[str]:
        analyzer = self._get_faces()

        if analyzer is None:
            return []

        found = analyzer.analyze(glance.keyframe)

        labels: list[str] = []

        for face in found:
            if face.get("encoding") is None:
                continue

            seen_ms = float(glance.duration_ms)

            label, dist, promoted = self.matcher.assign(
                face["encoding"],
                seen_ms=seen_ms,
            )

            self._registry_dirty = True

            labels.append(label)

            with self._state_lock:
                self._stats["faces_total"] += 1

                self._stats["last_person"] = label

                self._stats["last_person_dist"] = (
                    round(dist, 3)
                    if dist is not None
                    else None
                )

                self._sync_person_stats()

            if promoted:
                logger.info(
                    "person promoted to persistent "
                    "registry: {}",
                    label,
                )

            self._maybe_save_registry(promoted)

        if labels:
            entry["faces"] = labels

        return labels

    def _presence(
        self,
        faces: list[str],
        entry: dict[str, Any],
    ) -> None:
        """
        Presence transitions: a glance with faces after one
        without is an entrance; the reverse is a departure.
        """
        now = time.time()

        with self._state_lock:
            if faces and not self._person_in_view:
                self._person_in_view = True

                self._stats["person_in_view"] = True

                self._stats["enters_total"] += 1

                self._stats["last_enter_ts"] = now

                entry["event"] = "entered"

            elif not faces and self._person_in_view:
                self._person_in_view = False

                self._stats["person_in_view"] = False

                self._stats["leaves_total"] += 1

                self._stats["last_leave_ts"] = now

                entry["event"] = "left"

    def _load_registry(self) -> None:
        """
        Restore the auto-enrolled person registry at boot.

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
                "person registry corrupt ({}); quarantined "
                "as {}; starting empty",
                exc,
                corrupt.name,
            )

            self.matcher = FaceMatcher(
                threshold=self.face_distance_threshold,
                promote_min_sightings=(
                    self.promote_min_sightings
                ),
                promote_min_seen_ms=(
                    self.promote_min_seen_ms
                ),
            )

        with self._state_lock:
            self._sync_person_stats()

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

    def _sync_person_stats(self) -> None:
        self._stats["persons_known"] = (
            self.matcher.known_count()
        )

        self._stats["persons_session"] = len(
            self.matcher.labels()
        )

    # ------------------------------------------------------------------
    # QR / OCR / objects (L6)
    # ------------------------------------------------------------------

    def _scan_qr(
        self,
        glance: Any,
        entry: dict[str, Any],
    ) -> None:
        if not self.qr_enabled:
            return

        text = self.qr.decode(glance.keyframe)

        if not text:
            return

        now = time.time()

        with self._state_lock:
            if (
                text == self._last_qr_text
                and now - self._last_qr_at
                <= self.ocr_dedup_window_s
            ):
                return

            self._last_qr_text = text

            self._last_qr_at = now

            self._stats["qr_total"] += 1

            self._stats["last_qr"] = text

        entry["qr"] = text

    def _read_text(
        self,
        glance: Any,
        entry: dict[str, Any],
    ) -> None:
        if self._ocr_failed:
            return

        now = time.monotonic()

        if (
            glance.duration_ms < self.ocr_min_glance_ms
            or now - self._last_ocr_at < self.ocr_interval_s
        ):
            return

        if self._ocr is None:
            return

        self._last_ocr_at = now

        try:
            lines = self._ocr.read(glance.keyframe)

        except Exception as exc:
            logger.warning("ocr read failed: {}", exc)

            return

        text = " | ".join(
            line for line, _ in lines[:4]
        ).strip()

        if not text:
            return

        with self._state_lock:
            if text == self._last_ocr_text:
                return

            self._last_ocr_text = text

            self._stats["ocr_total"] += 1

            self._stats["last_ocr"] = text

        entry["ocr"] = text

    def _detect_objects(
        self,
        glance: Any,
        entry: dict[str, Any],
    ) -> None:
        if self._objects_failed:
            return

        now = time.monotonic()

        if now - self._last_objects_at < (
            self.objects_interval_s
        ):
            return

        if self._objects is None:
            return

        self._last_objects_at = now

        try:
            objects = self._objects.detect(glance.keyframe)

        except Exception as exc:
            logger.warning("object detection failed: {}", exc)

            return

        if not objects:
            return

        summary = [
            f"{item['name']} {item['conf']:.2f}"
            for item in objects[:4]
        ]

        with self._state_lock:
            self._stats["objects_total"] += len(objects)

            self._stats["last_objects"] = summary

        entry["objects"] = summary

    def _caption_glance(
        self,
        glance: Any,
        entry: dict[str, Any],
    ) -> None:
        """
        L7 content: one-sentence VLM caption of the keyframe,
        interval-gated exactly like OCR. Weights are provisioned
        in start(); a failed download or load raises there.
        """
        if self._vlm_failed:
            return

        now = time.monotonic()

        if now - self._last_vlm_at < self.vlm_interval_s:
            return

        if self._vlm is None:
            return

        self._last_vlm_at = now

        try:
            caption = self._vlm.caption(glance.keyframe)

        except Exception as exc:
            logger.warning("vlm caption failed: {}", exc)

            return

        if not caption:
            return

        with self._state_lock:
            if caption == self._last_caption:
                return

            self._last_caption = caption

            self._stats["vlm_total"] += 1

            self._stats["last_caption"] = caption

        entry["caption"] = caption

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

    async def on_turn(self, record: Turn) -> None:
        with self._state_lock:
            self._turn_marks.append(record.started_at)

    async def query(self, turn: Turn) -> str | None:
        with self._state_lock:
            stats = dict(self._stats)

            seen = list(self._seen)

        # One module, one territory: everything lives under the
        # module's own header; no module may mint global-looking
        # sections inside <module>.
        lines = ["[Vision]"]

        if not stats.get("available"):
            reason = stats.get("reason") or "no camera"

            lines.append(f"- input unavailable: {reason}")

            return "\n".join(lines)

        if stats.get("occluded"):
            lines.append("- seeing: lens covered")

        elif stats.get("motion_active"):
            lines.append("- seeing: motion in view")

            if stats.get("camera_motion"):
                lines.append("- camera itself is moving")

        quiet = stats.get("quiet_s")

        if (
            not stats.get("motion_active")
            and quiet is not None
        ):
            lines.append(
                f"- seeing: still {self._fmt_span(quiet)}"
            )

        # The scene, in one line.
        lines.append(
            f"- scene: bright {stats.get('brightness')}, "
            f"sharp {stats.get('sharpness')}, "
            f"edges {self._fmt_pct(stats.get('edges'))}, "
            f"colors {stats.get('colorfulness')}"
        )

        if stats.get("person_in_view"):
            label = stats.get("last_person") or "person"

            lines.append(f"- present: {label}")

        cut_ts = stats.get("last_cut_ts")

        if cut_ts and time.time() - cut_ts < 60.0:
            lines.append(
                "- scene changed at "
                f"{datetime.fromtimestamp(cut_ts).strftime('%H:%M')}"
            )

        light_ts = stats.get("last_light_ts")

        if light_ts and time.time() - light_ts < 60.0:
            lines.append(
                "- lighting changed at "
                f"{datetime.fromtimestamp(light_ts).strftime('%H:%M')}"
            )

        seen_lines = self._render_seen(seen)

        lines.extend(seen_lines)

        if (
            not seen_lines
            and not stats.get("motion_active")
            and (quiet is None or quiet >= self.quiet_report_after_s)
            and stats.get("glances_total", 0) == 0
        ):
            # Nothing has ever been seen, the scene is dead and
            # nothing is in flight: stay silent instead of
            # spamming empty ambience.
            return None

        return "\n".join(lines)

    def _render_seen(
        self,
        seen: list[dict[str, Any]],
    ) -> list[str]:
        out: list[str] = []

        for item in reversed(seen):
            tags = ""

            faces = item.get("faces")

            if faces:
                tags += f" ({', '.join(faces)})"

            event = item.get("event")

            if event:
                tags += f" [{event}]"

            if item.get("qr"):
                tags += f" [qr: {item['qr']}]"

            if item.get("ocr"):
                preview = item["ocr"][: self.see_preview_cap]

                tags += f' [text: "{preview}"]'

            objects = item.get("objects")

            if objects:
                tags += f" [obj: {', '.join(objects)}]"

            if item.get("caption"):
                preview = item["caption"][
                    : self.see_preview_cap
                ]

                tags += f' [caption: "{preview}"]'

            clock = datetime.fromtimestamp(
                item["ts"],
            ).strftime("%H:%M")

            span = self._fmt_span(
                max(0.0, item["duration_ms"] / 1000.0),
            )

            out.append(
                f"- saw {clock} ({item['reason']}, {span}){tags}"
            )

            if len(out) >= self.see_render_limit:
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

    def _fmt_pct(self, value: Any) -> str:
        if not isinstance(value, (int, float)):
            return "?"

        return f"{value * 100:.0f}%"

    # ==================================================================
    # Persistence (counters only; senses reset on reboot)
    # ==================================================================

    def serialize_state(self) -> dict[str, Any]:
        with self._state_lock:
            return {
                "glances_total": self._stats[
                    "glances_total"
                ],
                "faces_total": self._stats[
                    "faces_total"
                ],
                "qr_total": self._stats["qr_total"],
                "ocr_total": self._stats["ocr_total"],
                "objects_total": self._stats[
                    "objects_total"
                ],
                "vlm_total": self._stats["vlm_total"],
                "cuts_total": self._stats["cuts_total"],
                "light_total": self._stats["light_total"],
                "enters_total": self._stats[
                    "enters_total"
                ],
                "leaves_total": self._stats[
                    "leaves_total"
                ],
            }

    def restore_state(self, state: Any) -> None:
        if not isinstance(state, dict):
            raise TypeError(
                "vision private state must be an object",
            )

        total_keys = (
            "glances_total",
            "faces_total",
            "qr_total",
            "ocr_total",
            "objects_total",
            "vlm_total",
            "cuts_total",
            "light_total",
            "enters_total",
            "leaves_total",
        )

        for key in total_keys:
            value = state.get(key, 0)

            if not isinstance(value, int):
                raise TypeError(
                    f"{key} must be an integer",
                )

            self._stats[key] = value
