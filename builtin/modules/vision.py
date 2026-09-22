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
import shutil
import sys
import threading
import time
import asyncio
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import easyocr
import face_recognition
import huggingface_hub
import numpy as np
import onnxruntime as ort
import torch
import yaml
from loguru import logger
from PIL import Image
from pydantic import BaseModel
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
)

from nan_itself.modules.model import Module, Turn

from nan_itself.utils import paths as _paths


# ============================================================================
# VISION TOOLKIT (inlined from backend/nan_itself/utils/vision.py)
#
# Vision toolkit for the seeing module.
#
# Everything here is deliberately free of async and of the Module
# framework: pure functions, pure state machines, and thin hardware
# adapters behind small interfaces -- the same stance as utils/audio.py.
#
# Blocks:
#
#     frame_stats           L1 photometric statistics of one frame
#     MotionTracker         L2 frame-diff energy / cuts / light jumps
#     optical_flow_summary  L3 Farneback flow: ego-motion vs scene motion
#     motion_blob           L3 largest connected motion region
#     spectral_saliency     L3 frequency-domain attention peak
#     GlanceSegmenter       L4 event-driven slicing of the frame stream
#     CameraSource          default camera source (OpenCV/AVFoundation)
#     FaceAnalyzer          L5 face_recognition adapter (detect+encode)
#     FaceMatcher           L6 auto-enrolling person registry
#     QrScanner             L6 QR decode via OpenCV
#     OcrReader             L6 lazy easyocr adapter
#     YoloOnnxDetector      L5 lazy ONNX object detector (model drop-in)
#     VisionPipeline        composes the per-frame brain work
#
# Frame contract: BGR uint8 numpy frames as produced by OpenCV.
# Heavy work happens on small grayscale proxies (small_width); the
# original frame is kept only as a glance keyframe candidate.
# ============================================================================


# ============================================================================
# L1: photometric statistics (pure functions over one frame)
# ============================================================================


def to_small_gray(
    frame: Any,
    small_width: int = 160,
) -> Any:
    """
    Downscaled grayscale proxy used by every temporal computation.
    """
    height, width = frame.shape[:2]

    if width <= small_width:
        small = frame

    else:
        small = cv2.resize(
            frame,
            (small_width, max(1, round(height * small_width / width))),
            interpolation=cv2.INTER_AREA,
        )

    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)


def frame_stats(frame: Any) -> dict[str, float]:
    """
    L1 photometric statistics of one BGR frame.

    brightness  mean luminance 0..255
    contrast    luminance std
    clipped     fraction of over/under-exposed pixels (>=250 / <=5)
    colorfulness Hasler-Süsstrunk metric
    sharpness   Laplacian variance (focus measure)
    noise       robust high-frequency estimate (MAD of Laplacian)
    entropy     luminance histogram entropy (bits)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    f = gray.astype(np.float32)

    clipped_over = float(
        (gray >= 250).mean()
    )

    clipped_under = float(
        (gray <= 5).mean()
    )

    b = frame[:, :, 0].astype(np.float32)
    g = frame[:, :, 1].astype(np.float32)
    r = frame[:, :, 2].astype(np.float32)

    rg = r - g
    yb = 0.5 * (r + g) - b

    root_mean_sq = (
        (rg.std() ** 2 + yb.std() ** 2) ** 0.5
    )

    root_mean = (
        (rg.mean() ** 2 + yb.mean() ** 2) ** 0.5
    )

    colorfulness = root_mean_sq + 0.3 * root_mean

    lap = cv2.Laplacian(
        f,
        cv2.CV_32F,
        ksize=1,
    )

    sharpness = float(lap.var())

    mad = float(
        np.median(np.abs(lap - np.median(lap)))
    )

    noise = 1.4826 * mad

    hist = np.bincount(
        gray.ravel(),
        minlength=256,
    ).astype(np.float64)

    p = hist / max(1.0, hist.sum())

    p = p[p > 0]

    entropy = float(-(p * np.log2(p)).sum())

    return {
        "brightness": round(float(f.mean()), 1),
        "contrast": round(float(f.std()), 1),
        "clipped_over": round(clipped_over, 4),
        "clipped_under": round(clipped_under, 4),
        "colorfulness": round(float(colorfulness), 1),
        "sharpness": round(sharpness, 1),
        "noise": round(noise, 2),
        "entropy": round(entropy, 2),
    }


def edge_density(
    small_gray: Any,
    canny_low: int = 60,
    canny_high: int = 150,
) -> float:
    """
    Fraction of edge pixels in the small gray proxy (0..1).
    """
    edges = cv2.Canny(
        small_gray,
        canny_low,
        canny_high,
    )

    return round(float((edges > 0).mean()), 4)


def spectral_saliency(
    small_gray: Any,
) -> dict[str, float]:
    """
    Frequency-domain saliency (spectral residual).

    A Hann window kills the frame-boundary artifacts that would
    otherwise drag the centroid toward the middle of the frame.
    Returns the normalized centroid (0..1) of the saliency map
    and its peak value -- where the eye would go first.
    """
    height, width = small_gray.shape[:2]

    windowed = small_gray.astype(np.float32)

    if height >= 8 and width >= 8:
        windowed = windowed * cv2.createHanningWindow(
            (width, height),
            cv2.CV_32F,
        )

    f = np.fft.fft2(windowed)

    log_amp = np.log(np.abs(f) + 1e-8)

    phase = np.angle(f)

    avg = cv2.blur(log_amp, (3, 3))

    residual = log_amp - avg

    sal = np.abs(
        np.fft.ifft2(np.exp(residual + 1j * phase)),
    ) ** 2

    sal = cv2.GaussianBlur(sal, (9, 9), 2.5)

    peak = float(sal.max())

    if peak <= 0:
        return {
            "salient_x": 0.5,
            "salient_y": 0.5,
            "salient_peak": 0.0,
        }

    ys, xs = np.nonzero(sal >= 0.8 * peak)

    return {
        "salient_x": round(
            float(xs.mean()) / width,
            3,
        ),
        "salient_y": round(
            float(ys.mean()) / height,
            3,
        ),
        "salient_peak": round(peak, 1),
    }


# ============================================================================
# L2: temporal motion statistics
# ============================================================================


class MotionTracker:
    """
    Frame-diff bookkeeping over the small gray stream.

    One feed() per frame; scalars only. Emits flags consumed by the
    pipeline as events:

        motion_active   scene content moved (ratio + energy gates)
        scene_cut       structural correlation collapsed while diff spiked
        light_change    uniform luminance jump (lamp/flash), no cut
        flicker         >= flicker_count light changes in the window
    """

    def __init__(
        self,
        pixel_diff: int = 18,
        motion_ratio_min: float = 0.02,
        motion_energy_min: float = 2.0,
        cut_corr: float = 0.3,
        light_jump: float = 25.0,
        light_cooldown_s: float = 1.0,
        light_diff_std_max: float = 12.0,
        flicker_count: int = 4,
        flicker_window_s: float = 8.0,
    ) -> None:
        self.pixel_diff = pixel_diff

        self.motion_ratio_min = motion_ratio_min

        self.motion_energy_min = motion_energy_min

        self.cut_corr = cut_corr

        self.light_jump = light_jump

        self.light_cooldown_s = light_cooldown_s

        self.light_diff_std_max = light_diff_std_max

        self.flicker_count = flicker_count

        self.flicker_window_s = flicker_window_s

        self.prev: Any = None

        self.mask: Any = None

        self.motion_active = False

        self.was_active = False

        self.cuts_total = 0

        self.light_total = 0

        self.last_cut_monotonic: float | None = None

        self.last_light_monotonic: float | None = None

        self.last_motion_monotonic: float | None = None

        self._light_times: deque[float] = deque()

        self._last_light_at = -1e9

    def feed(
        self,
        small_gray: Any,
        now: float,
    ) -> dict[str, Any]:
        if self.prev is None:
            self.prev = small_gray

            return {
                "motion_energy": 0.0,
                "motion_ratio": 0.0,
                "motion_active": False,
                "motion_start": False,
                "scene_cut": False,
                "light_change": False,
                "flicker": False,
            }

        diff = cv2.absdiff(small_gray, self.prev)

        self.mask = cv2.threshold(
            diff,
            self.pixel_diff,
            255,
            cv2.THRESH_BINARY,
        )[1]

        energy = round(float(diff.mean()), 2)

        ratio = round(
            float((self.mask > 0).mean()),
            4,
        )

        active = (
            ratio >= self.motion_ratio_min
            and energy >= self.motion_energy_min
        )

        # Light change: the WHOLE frame shifted brightness nearly
        # uniformly -- a lamp switch, not content movement. The
        # per-pixel diff of a lamp switch is a flat offset (tiny
        # std); content movement produces unstructured diffs.
        mean_now = float(small_gray.mean())

        mean_prev = float(self.prev.mean())

        light_change = (
            abs(mean_now - mean_prev) >= self.light_jump
            and ratio > 0.5
            and float(diff.std()) < self.light_diff_std_max
            and now - self._last_light_at
            >= self.light_cooldown_s
        )

        if light_change:
            self._last_light_at = now

            self.light_total += 1

            self.last_light_monotonic = now

            self._light_times.append(now)

            while (
                self._light_times
                and self._light_times[0]
                < now - self.flicker_window_s
            ):
                self._light_times.popleft()

        # Scene cut: the frame's STRUCTURE was replaced. Pearson
        # correlation is affine-invariant, so a lamp switch (same
        # structure, new brightness) keeps it near 1 while a real
        # cut collapses it. A flat frame carries no structural
        # evidence either way and is treated as unchanged.
        var_prev = float(self.prev.std())

        var_cur = float(small_gray.std())

        if var_prev < 1e-6 or var_cur < 1e-6:
            structure_corr = 1.0

        else:
            structure_corr = float(
                np.corrcoef(
                    self.prev.ravel(),
                    small_gray.ravel(),
                )[0, 1],
            )

        scene_cut = (
            structure_corr < self.cut_corr
            and energy >= self.motion_energy_min * 4
            and not light_change
        )

        if scene_cut:
            self.cuts_total += 1

            self.last_cut_monotonic = now

        if active or scene_cut or light_change:
            self.last_motion_monotonic = now

        self.motion_active = active

        rising = active and not self.was_active

        self.was_active = active

        self.prev = small_gray

        return {
            "motion_energy": energy,
            "motion_ratio": ratio,
            "motion_active": active,
            "motion_start": rising,
            "scene_cut": scene_cut,
            "light_change": light_change,
            "flicker": (
                len(self._light_times) >= self.flicker_count
            ),
        }


# ============================================================================
# L3: structure and flow
# ============================================================================


def optical_flow_summary(
    prev_gray: Any,
    gray: Any,
    camera_mag_min: float = 0.35,
    camera_uni_min: float = 0.55,
) -> dict[str, Any]:
    """
    Dense Farneback flow on small proxies.

    uniformity ~ mean / (mean + std) of vector magnitudes:
    everything drifting the same way is the CAMERA moving, while
    heterogeneous flow is the SCENE moving.
    """
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        gray,
        None,
        0.5,
        2,
        15,
        2,
        5,
        1.2,
        0,
    )

    vx = flow[:, :, 0]

    vy = flow[:, :, 1]

    mag = np.sqrt(vx**2 + vy**2)

    mean_mag = float(mag.mean())

    std_mag = float(mag.std())

    uniformity = mean_mag / (mean_mag + std_mag + 1e-6)

    return {
        "flow_dx": round(float(vx.mean()), 3),
        "flow_dy": round(float(vy.mean()), 3),
        "flow_mag": round(mean_mag, 3),
        "flow_uniformity": round(float(uniformity), 3),
        "camera_motion": bool(
            mean_mag >= camera_mag_min
            and uniformity >= camera_uni_min
        ),
    }


def motion_blob(
    mask: Any,
) -> dict[str, float] | None:
    """
    Largest connected motion region as normalized fractions
    (small-proxy coordinates). None when nothing moved.
    """
    if mask is None or not mask.any():
        return None

    count, _, stats, _ = (
        cv2.connectedComponentsWithStats(mask, 8)
    )

    if count <= 1:
        return None

    # Component 0 is the background; skip it.
    areas = stats[1:, cv2.CC_STAT_AREA]

    best = int(np.argmax(areas)) + 1

    x = stats[best, cv2.CC_STAT_LEFT]

    y = stats[best, cv2.CC_STAT_TOP]

    w = stats[best, cv2.CC_STAT_WIDTH]

    h = stats[best, cv2.CC_STAT_HEIGHT]

    height, width = mask.shape[:2]

    return {
        "blob_x": round(x / width, 3),
        "blob_y": round(y / height, 3),
        "blob_w": round(w / width, 3),
        "blob_h": round(h / height, 3),
        "blob_area": round(
            float(stats[best, cv2.CC_STAT_AREA])
            / float(width * height),
            4,
        ),
    }


# ============================================================================
# L4: glance slicing
# ============================================================================


@dataclass
class Glance:
    """
    One event-driven look at the scene (the visual utterance).

    keyframe is the highest-motion original frame seen while the
    glance was open; everything else is bookkeeping.
    """

    keyframe: Any = None

    started_at: float = 0.0

    duration_ms: int = 0

    peak_energy: float = 0.0

    bursts: int = 0

    reason: str = "motion"


class GlanceSegmenter:
    """
    Slices the continuous frame stream into glances.

    Feed one (frame, energy, active, trigger, ts) per call;
    returns zero or more completed glances. State machine:

        IDLE  --(motion or trigger)--> OPEN   (pre-roll keyframe)
        OPEN  --trailing quiet--> emit
        OPEN  --max length--> force emit

    The buffer keeps references, not copies: one keyframe
    (peak-energy frame) is retained, never the whole stream.
    """

    def __init__(
        self,
        preroll_frames: int = 8,
        trailing_quiet_frames: int = 20,
        min_glance_frames: int = 6,
        max_glance_frames: int = 450,
        burst_ratio: float = 0.08,
    ) -> None:
        self.preroll_frames = preroll_frames

        self.trailing_limit = trailing_quiet_frames

        self.min_frames = min_glance_frames

        self.max_total = max_glance_frames

        self.burst_ratio = burst_ratio

        self._preroll: deque[tuple[Any, float]] = deque(
            maxlen=max(1, preroll_frames),
        )

        self._open = False

        self._keyframe: Any = None

        self._keyframe_energy = 0.0

        self._started_at = 0.0

        self._frames = 0

        self._quiet_run = 0

        self._bursts = 0

        # Eventful frames (motion or trigger) seen while open:
        # the minimum-length gate counts THESE, not the total --
        # a 2-frame flicker riding a long quiet tail is noise,
        # exactly like a clap between silences is not speech.
        self._active_count = 0

        self._reason = "motion"

    def feed(
        self,
        frame: Any,
        energy: float,
        active: bool,
        trigger: str | None,
        ts: float,
    ) -> list[Glance]:
        """
        trigger is "light" or "cut" when the tracker flagged one on
        this frame (opens a glance even without content motion).
        """
        if not self._open:
            self._preroll.append((frame, energy))

            if active or trigger:
                self._begin(
                    trigger or "motion",
                    ts,
                )

            return []

        self._frames += 1

        self._quiet_run = (
            0 if active else self._quiet_run + 1
        )

        if active or trigger:
            self._active_count += 1

        if trigger:
            self._bursts += 1

        if energy > self._keyframe_energy:
            self._keyframe = frame

            self._keyframe_energy = energy

        closed = False

        if self._quiet_run >= self.trailing_limit:
            closed = True

        elif self._frames >= self.max_total:
            closed = True

        if closed:
            return self._emit()

        return []

    def flush(self) -> list[Glance]:
        """Force-close whatever is open (shutdown path)."""
        if not self._open:
            return []

        return self._emit()

    def _begin(
        self,
        reason: str,
        ts: float,
    ) -> None:
        self._open = True

        self._reason = reason

        self._started_at = ts

        self._frames = 1

        self._quiet_run = 0

        self._bursts = 1 if reason != "motion" else 0

        self._active_count = 1

        self._keyframe = None

        self._keyframe_energy = -1.0

        # Pre-roll: the most energetic frame seen just before the
        # glance opened, so the onset is never missed.
        for frame, energy in self._preroll:
            if energy > self._keyframe_energy:
                self._keyframe = frame

                self._keyframe_energy = energy

        self._preroll.clear()

    def _emit(self) -> list[Glance]:
        keyframe = self._keyframe

        active = self._active_count

        peak = self._keyframe_energy

        reason = self._reason

        self._open = False

        self._keyframe = None

        self._keyframe_energy = 0.0

        self._frames = 0

        self._quiet_run = 0

        self._bursts = 0

        self._active_count = 0

        self._reason = "motion"

        if active < self.min_frames or keyframe is None:
            return []

        return [
            Glance(
                keyframe=keyframe,
                started_at=self._started_at,
                duration_ms=0,
                peak_energy=round(peak, 2),
                bursts=self._bursts,
                reason=reason,
            )
        ]


# ============================================================================
# Hardware: default camera source
# ============================================================================


class CameraSource:
    """
    Default camera source built on OpenCV.

    A reader thread continuously drains the driver into the LATEST
    slot (overwriting unread frames): vision always wants the
    freshest frame, never a backlog. Overwrites are counted as
    drops, mirroring the mic source contract.
    """

    def __init__(
        self,
        device: int | str | None = None,
        width: int = 640,
        height: int = 480,
    ) -> None:
        self.device = device

        self.width = width

        self.height = height

        self.dropped = 0

        self.failed_reads = 0

        self._latest: Any = None

        self._lock = threading.Lock()

        self._stop = threading.Event()

        self._thread: threading.Thread | None = None

        self._cap: Any = None

    def start(self) -> None:
        device = self.device

        if (
            device is not None
            and isinstance(device, str)
            and device.isdigit()
        ):
            device = int(device)

        if device is None:
            device = 0

        backend = cv2.CAP_AVFOUNDATION if (
            sys_darwin()
        ) else cv2.CAP_ANY

        cap = cv2.VideoCapture(device, backend)

        if not cap.isOpened():
            cap.release()

            raise RuntimeError(
                f"camera {device!r} could not be opened"
            )

        cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            self.width,
        )

        cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            self.height,
        )

        self._cap = cap

        self._thread = threading.Thread(
            target=self._reader_loop,
            name="camera-reader",
            daemon=True,
        )

        self._thread.start()

    def _reader_loop(self) -> None:
        while not self._stop.is_set():
            try:
                ok, frame = self._cap.read()

            except Exception:
                ok, frame = False, None

            if not ok or frame is None:
                self.failed_reads += 1

                self._stop.wait(0.05)

                continue

            with self._lock:
                if self._latest is not None:
                    self.dropped += 1

                self._latest = frame

    def read(
        self,
        timeout: float = 0.5,
    ) -> Any | None:
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            with self._lock:
                frame = self._latest

                self._latest = None

            if frame is not None:
                return frame

            self._stop.wait(0.02)

        return None

    def close(self) -> None:
        self._stop.set()

        thread = self._thread

        if thread is not None:
            thread.join(timeout=2.0)

            self._thread = None

        cap = self._cap

        self._cap = None

        if cap is not None:
            try:
                cap.release()

            except Exception:

                pass


def sys_darwin() -> bool:
    return sys.platform == "darwin"


# ============================================================================
# L5/L6: model-backed analyzers
# ============================================================================


class FaceAnalyzer:
    """
    face_recognition adapter: detect faces and produce 128-d
    encodings. The import happens at module load; dlib is a hard
    dependency of the vision stack.
    """

    def __init__(
        self,
        max_width: int = 640,
        model: str = "hog",
    ) -> None:
        self.max_width = max_width

        self.model = model

        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return

        self._loaded = True

    def analyze(self, frame: Any) -> list[dict]:
        """
        Returns [{"box": (top, right, bottom, left),
        "encoding": ndarray|None}, ...] in ORIGINAL frame
        coordinates.
        """
        self.load()

        height, width = frame.shape[:2]

        scale = 1.0

        rgb_frame = frame

        if width > self.max_width:
            scale = self.max_width / width

            rgb_frame = cv2.resize(
                frame,
                (self.max_width, round(height * scale)),
                interpolation=cv2.INTER_AREA,
            )

        rgb = cv2.cvtColor(
            rgb_frame,
            cv2.COLOR_BGR2RGB,
        )

        locations = face_recognition.face_locations(
            rgb,
            model=self.model,
        )

        if not locations:
            return []

        encodings = face_recognition.face_encodings(
            rgb,
            locations,
            num_jitters=1,
        )

        faces: list[dict] = []

        for location, encoding in zip(
            locations,
            encodings,
        ):
            top, right, bottom, left = location

            inv = 1.0 / scale

            faces.append(
                {
                    "box": (
                        int(top * inv),
                        int(right * inv),
                        int(bottom * inv),
                        int(left * inv),
                    ),
                    "encoding": encoding,
                },
            )

        return faces


class FaceMatcher:
    """
    Auto-enrolling person registry for a permanently running eye.

    No agent, no tool, no manual file ever enrolls a person:
    clusters open on first sight (person-N, numbering monotonic
    across restarts), absorb nearest matches with an encoding
    gallery (a few exemplars per person, distance-gated for
    diversity), and PROMOTE into the persistent registry once they
    carry enough evidence. Only promoted persons survive a reboot.
    Serialization is JSON-native, human-inspectable.

    Distance is the face_recognition euclidean convention: <=0.5
    is the same person, >=0.6 almost never is.
    """

    MAX_ENCODINGS = 4

    def __init__(
        self,
        threshold: float = 0.5,
        promote_min_sightings: int = 5,
        promote_min_seen_ms: float = 30000.0,
    ) -> None:
        self.threshold = threshold

        self.promote_min_sightings = promote_min_sightings

        self.promote_min_seen_ms = promote_min_seen_ms

        self.persons: dict[str, dict] = {}

        self.persisted: set[str] = set()

        self.next_id = 1

    def labels(self) -> list[str]:
        return list(self.persons)

    def known_count(self) -> int:
        return len(self.persisted & set(self.persons))

    def assign(
        self,
        encoding,
        seen_ms: float = 0.0,
    ) -> tuple[str, float | None, bool]:
        """
        Route one face encoding to a person.

        Returns (label, best_distance, promoted_now). The
        distance is the nearest gallery entry even when it falls
        below the threshold (a new person opens then).
        """
        vec = np.asarray(encoding, dtype=np.float32)

        now = time.time()

        best_label: str | None = None

        best_dist = float("inf")

        for label, person in self.persons.items():
            for stored in person["encodings"]:
                dist = float(
                    np.linalg.norm(stored - vec)
                )

                if dist < best_dist:
                    best_label = label

                    best_dist = dist

        promoted_now = False

        if (
            best_label is not None
            and best_dist <= self.threshold
        ):
            person = self.persons[best_label]

            person["count"] += 1

            person["seen_ms"] += seen_ms

            person["last_seen"] = now

            # Add a diverse exemplar: far enough from everything
            # already stored (pose/lighting variation), capped.
            if len(person["encodings"]) < self.MAX_ENCODINGS:
                min_dist = min(
                    float(
                        np.linalg.norm(stored - vec)
                    )
                    for stored in person["encodings"]
                )

                if min_dist > self.threshold * 0.7:
                    person["encodings"].append(
                        vec.copy(),
                    )

        else:
            best_label = f"person-{self.next_id}"

            self.next_id += 1

            self.persons[best_label] = {
                "encodings": [vec.copy()],
                "count": 1,
                "seen_ms": float(seen_ms),
                "created_at": now,
                "last_seen": now,
            }

            best_dist = min(best_dist, 1.0)

        person = self.persons[best_label]

        if (
            best_label not in self.persisted
            and person["count"] >= self.promote_min_sightings
            and person["seen_ms"]
            >= self.promote_min_seen_ms
        ):
            self.persisted.add(best_label)

            promoted_now = True

        return best_label, best_dist, promoted_now

    def serialize(self) -> dict:
        persons: dict[str, dict] = {}

        for label in sorted(self.persisted):
            person = self.persons.get(label)

            if person is None:
                continue

            persons[label] = {
                "encodings": [
                    [round(float(x), 5) for x in enc]
                    for enc in person["encodings"]
                ],
                "count": person["count"],
                "seen_ms": round(person["seen_ms"], 1),
                "created_at": person["created_at"],
                "last_seen": person["last_seen"],
            }

        return {
            "next_id": self.next_id,
            "persons": persons,
        }

    def restore(self, data: Any) -> None:
        if not isinstance(data, dict):
            raise TypeError(
                "person registry must be an object",
            )

        persons = data.get("persons", {})

        next_id = data.get("next_id", 1)

        if not isinstance(persons, dict):
            raise TypeError("persons must be an object")

        if not isinstance(next_id, int) or next_id < 1:
            raise TypeError(
                "next_id must be a positive int",
            )

        self.persons.clear()

        self.persisted.clear()

        self.next_id = next_id

        for label, payload in persons.items():
            encodings = payload.get("encodings", [])

            if not isinstance(encodings, list):
                continue

            vectors = [
                np.asarray(enc, dtype=np.float32)
                for enc in encodings
                if isinstance(enc, list) and enc
            ]

            if not vectors:
                continue

            self.persons[label] = {
                "encodings": vectors,
                "count": int(payload.get("count", 1)),
                "seen_ms": float(
                    payload.get("seen_ms", 0.0),
                ),
                "created_at": float(
                    payload.get("created_at", 0.0),
                ),
                "last_seen": float(
                    payload.get("last_seen", 0.0),
                ),
            }

            self.persisted.add(label)

        used = [
            int(label.split("-")[1])
            for label in self.persons
            if label.startswith("person-")
            and label.split("-")[1].isdigit()
        ]

        if used:
            self.next_id = max(
                self.next_id,
                max(used) + 1,
            )


class QrScanner:
    """
    QR decode via OpenCV; cheap enough to run on every keyframe.
    """

    def __init__(self) -> None:
        self._detector: Any = None

    def _get(self) -> Any:
        if self._detector is None:
            self._detector = cv2.QRCodeDetector()

        return self._detector

    def decode(self, frame: Any) -> str | None:
        try:
            text, _, _ = self._get().detectAndDecode(frame)

        except Exception:

            return None

        return text or None


class OcrReader:
    """
    Lazy easyocr adapter. Model weights are expected under the
    repo models/easyocr directory; init is heavy (seconds), so
    the module gates calls behind its own interval.
    """

    def __init__(
        self,
        models_dir: Path,
        languages: tuple[str, ...] = ("ch_sim", "en"),
        gpu: bool = False,
    ) -> None:
        self.models_dir = Path(models_dir)

        self.languages = tuple(languages)

        self.gpu = gpu

        self._reader: Any = None

    def load(self) -> None:
        if self._reader is not None:
            return

        self._reader = easyocr.Reader(
            list(self.languages),
            model_storage_directory=str(self.models_dir),
            gpu=self.gpu,
            verbose=False,
        )

    def read(self, frame: Any) -> list[tuple[str, float]]:
        """
        Returns [(text, confidence), ...] sorted by confidence.
        """
        self.load()

        results = self._reader.readtext(frame)

        entries = [
            (str(text), float(conf))
            for _, text, conf in results
            if str(text).strip()
        ]

        entries.sort(key=lambda item: -item[1])

        return entries


class YoloOnnxDetector:
    """
    Lazy ONNX object detector following the YOLOv8/v11 export
    contract (output [1, 4+classes, N]).

    Activates with the auto-downloaded COCO YOLOv8n at
    models/vision/object/model.onnx (labels.txt written alongside,
    one name per line, background not listed). A custom export can
    replace both files by hand.
    Missing model -> load() raises -> the failure propagates out
    of the module's start(), which the Facade answers with a DOWN
    state and backoff retries until the weights land.
    """

    INPUT_SIZE = 640

    def __init__(
        self,
        model_path: Path,
        labels_path: Path,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
    ) -> None:
        self.model_path = Path(model_path)

        self.labels_path = Path(labels_path)

        self.conf_threshold = conf_threshold

        self.iou_threshold = iou_threshold

        self._session: Any = None

        self._labels: list[str] = []

        self._input_name = ""

    def load(self) -> None:
        if self._session is not None:
            return

        self._labels = [
            line.strip()
            for line in self.labels_path.read_text(
                encoding="utf-8",
            ).splitlines()
            if line.strip()
        ]

        self._session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )

        self._input_name = (
            self._session.get_inputs()[0].name
        )

    @staticmethod
    def nms(
        boxes: list[tuple[float, float, float, float]],
        scores: list[float],
        iou_threshold: float,
    ) -> list[int]:
        """
        Greedy NMS over (x1, y1, x2, y2); pure for testability.
        """
        order = sorted(
            range(len(scores)),
            key=lambda i: -scores[i],
        )

        keep: list[int] = []

        for i in order:
            if any(
                YoloOnnxDetector._iou(boxes[i], boxes[j])
                > iou_threshold
                for j in keep
            ):
                continue

            keep.append(i)

        return keep

    @staticmethod
    def _iou(
        a: tuple[float, float, float, float],
        b: tuple[float, float, float, float],
    ) -> float:
        x1 = max(a[0], b[0])

        y1 = max(a[1], b[1])

        x2 = min(a[2], b[2])

        y2 = min(a[3], b[3])

        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)

        area_a = (a[2] - a[0]) * (a[3] - a[1])

        area_b = (b[2] - b[0]) * (b[3] - b[1])

        union = area_a + area_b - inter

        return inter / union if union > 0 else 0.0

    def decode(
        self,
        output: Any,
        scale: float,
        pad_x: float,
        pad_y: float,
    ) -> list[dict]:
        """
        Pure decode of one YOLOv8-style output tensor
        ([1, 4+classes, N]) back into ORIGINAL frame coordinates.
        """
        predictions = np.asarray(output)[0]

        # (4+classes, N) -> (N, 4+classes)
        predictions = predictions.T

        boxes_xywh = predictions[:, :4]

        class_scores = predictions[:, 4:]

        class_ids = class_scores.argmax(axis=1)

        confidences = class_scores.max(axis=1)

        candidates = [
            i
            for i, conf in enumerate(confidences)
            if conf >= self.conf_threshold
            and class_ids[i] < len(self._labels)
        ]

        if not candidates:
            return []

        boxes: list[tuple[float, float, float, float]] = []

        scores: list[float] = []

        names: list[str] = []

        for i in candidates:
            cx, cy, w, h = boxes_xywh[i]

            # Undo letterbox, then undo scale.
            x1 = (cx - w / 2 - pad_x) / scale

            y1 = (cy - h / 2 - pad_y) / scale

            x2 = (cx + w / 2 - pad_x) / scale

            y2 = (cy + h / 2 - pad_y) / scale

            boxes.append((x1, y1, x2, y2))

            scores.append(float(confidences[i]))

            names.append(self._labels[int(class_ids[i])])

        detections: list[dict] = []

        for i in self.nms(
            boxes,
            scores,
            self.iou_threshold,
        ):
            x1, y1, x2, y2 = boxes[i]

            detections.append(
                {
                    "name": names[i],
                    "conf": round(scores[i], 2),
                    "box": [
                        round(x1, 1),
                        round(y1, 1),
                        round(x2 - x1, 1),
                        round(y2 - y1, 1),
                    ],
                },
            )

        return detections

    def detect(self, frame: Any) -> list[dict]:
        self.load()

        height, width = frame.shape[:2]

        scale = min(
            self.INPUT_SIZE / width,
            self.INPUT_SIZE / height,
        )

        resized = cv2.resize(
            frame,
            (round(width * scale), round(height * scale)),
            interpolation=cv2.INTER_LINEAR,
        )

        pad_x = (self.INPUT_SIZE - resized.shape[1]) / 2

        pad_y = (self.INPUT_SIZE - resized.shape[0]) / 2

        canvas = cv2.copyMakeBorder(
            resized,
            int(pad_y),
            int(pad_y),
            int(pad_x),
            int(pad_x),
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )

        blob = (
            canvas[:, :, ::-1]
            .astype(np.float32)
            .transpose(2, 0, 1)[None] / 255.0
        )

        output = self._session.run(
            None,
            {self._input_name: blob},
        )[0]

        return self.decode(output, scale, pad_x, pad_y)


class VlmCaptioner:
    """
    Lazy SmolVLM2 caption adapter (the content layer, L7).

    Deliberately the most cross-platform path available: plain
    transformers/torch runs on any OS (CPU), CUDA (NVIDIA) and
    MPS (Apple Silicon) -- no MLX, no GGUF toolchain, no per-OS
    builds. Weights live under models/vision/vlm/<snapshot>;
    a missing directory is downloaded from the Hub during module
    provisioning in start() (respects HF_ENDPOINT / proxy env
    vars), never inside the tick loops; a failing download
    raises out of start(), so the Facade marks the module DOWN
    and retries with backoff until the weights land.

    Scope is honest: this class captions ONE keyframe. Change
    descriptions, VQA and cross-modal grounding are future
    specialized additions, not hidden features here.
    """

    DEFAULT_PROMPT = (
        "Briefly describe what is happening in this image "
        "in one sentence."
    )

    DEFAULT_REPO_ID = (
        "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    )

    def __init__(
        self,
        model_path: Path,
        max_tokens: int = 48,
        repo_id: str | None = None,
    ) -> None:
        self.model_path = Path(model_path)

        self.max_tokens = max_tokens

        self.repo_id = repo_id or self.DEFAULT_REPO_ID

        self._processor: Any = None

        self._model: Any = None

    def _weights_present(self) -> bool:
        return self.model_path.is_dir() and any(
            self.model_path.iterdir()
        )

    def _download(self) -> None:
        logger.info(
            "vlm weights missing at {}; downloading {} "
            "(~1 GB, honors HF_ENDPOINT and proxy envs)",
            self.model_path,
            self.repo_id,
        )

        huggingface_hub.snapshot_download(
            repo_id=self.repo_id,
            local_dir=self.model_path,
        )

    def load(self) -> None:
        if self._model is not None:
            return

        if not self._weights_present():
            self._download()

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

        self._processor = AutoProcessor.from_pretrained(
            path,
        )

        self._model = (
            AutoModelForImageTextToText.from_pretrained(
                path,
                dtype=dtype,
            )
            .to(device)
            .eval()
        )

        logger.info(
            "vlm captioner loaded from {} on {}",
            path,
            device,
        )

    def caption(
        self,
        frame: Any,
        prompt: str | None = None,
    ) -> str:
        """
        One-sentence description of one BGR frame.
        """
        self.load()

        image = Image.fromarray(
            cvt_rgb(frame),
        )

        question = prompt or self.DEFAULT_PROMPT

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            },
        ]

        text = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )

        inputs = self._processor(
            text=text,
            images=[image],
            return_tensors="pt",
        ).to(self._model.device)

        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=False,
            )

        return self._processor.batch_decode(
            output,
            skip_special_tokens=True,
        )[0].strip()


def cvt_rgb(frame: Any) -> Any:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# ============================================================================
# Composition
# ============================================================================


class VisionPipeline:
    """
    One object owns the full per-frame brain work.

    process() returns typed events:

        {"type": "motion_start", "ratio": float}
        {"type": "light_change"}
        {"type": "scene_cut"}
        {"type": "occlusion"} / {"type": "recovery"}
        {"type": "glance", "glance": Glance}

    Statistics live here too so query-side rendering never
    recomputes anything.
    """

    def __init__(
        self,
        small_width: int = 160,
        motion_pixel_diff: int = 18,
        motion_ratio_min: float = 0.02,
        motion_energy_min: float = 2.0,
        cut_corr: float = 0.55,
        light_jump: float = 25.0,
        light_cooldown_s: float = 1.0,
        flicker_count: int = 4,
        flicker_window_s: float = 8.0,
        camera_mag_min: float = 0.35,
        camera_uni_min: float = 0.55,
        saliency_interval_s: float = 1.0,
        occlusion_brightness: float = 6.0,
        occlusion_frames: int = 10,
        preroll_frames: int = 8,
        trailing_quiet_frames: int = 20,
        min_glance_frames: int = 6,
        max_glance_frames: int = 450,
        burst_ratio: float = 0.08,
    ) -> None:
        self.small_width = small_width

        self.camera_mag_min = camera_mag_min

        self.camera_uni_min = camera_uni_min

        self.saliency_interval_s = saliency_interval_s

        self.occlusion_brightness = occlusion_brightness

        self.occlusion_frames = occlusion_frames

        self.tracker = MotionTracker(
            pixel_diff=motion_pixel_diff,
            motion_ratio_min=motion_ratio_min,
            motion_energy_min=motion_energy_min,
            cut_corr=cut_corr,
            light_jump=light_jump,
            light_cooldown_s=light_cooldown_s,
            flicker_count=flicker_count,
            flicker_window_s=flicker_window_s,
        )

        self.segmenter = GlanceSegmenter(
            preroll_frames=preroll_frames,
            trailing_quiet_frames=trailing_quiet_frames,
            min_glance_frames=min_glance_frames,
            max_glance_frames=max_glance_frames,
            burst_ratio=burst_ratio,
        )

        self.glance_open = False

        self.occluded = False

        self._occlusion_run = 0

        self._prev_small: Any = None

        self._last_saliency_at = -1e9

        self.latest_stats: dict[str, Any] = {}

    def process(
        self,
        frame: Any,
        now: float | None = None,
    ) -> list[dict]:
        now = now if now is not None else time.monotonic()

        events: list[dict] = []

        small = to_small_gray(frame, self.small_width)

        motion = self.tracker.feed(small, now)

        stats = frame_stats(frame)

        stats.update(motion)

        stats["cuts_total"] = self.tracker.cuts_total

        stats["light_total"] = self.tracker.light_total

        stats["edges"] = edge_density(small)

        if motion["motion_start"]:
            events.append(
                {
                    "type": "motion_start",
                    "ratio": motion["motion_ratio"],
                },
            )

        if motion["light_change"]:
            events.append(
                {"type": "light_change"},
            )

        if motion["scene_cut"]:
            events.append({"type": "scene_cut"})

        # ----------------------------------------------------------
        # Occlusion: a lens covered by a hand/fabric reads as a
        # dark (or flat) frame for several consecutive frames.
        # ----------------------------------------------------------
        if stats["brightness"] < self.occlusion_brightness:
            self._occlusion_run += 1

        else:
            self._occlusion_run = 0

        was_occluded = self.occluded

        self.occluded = (
            self._occlusion_run >= self.occlusion_frames
        )

        if self.occluded and not was_occluded:
            events.append({"type": "occlusion"})

            # A covered lens ends the current look: emit it now
            # instead of freezing the segmenter mid-glance.
            for glance in self.segmenter.flush():
                glance.duration_ms = int(
                    (now - glance.started_at) * 1000,
                )

                events.append(
                    {"type": "glance", "glance": glance},
                )

        elif was_occluded and not self.occluded:
            events.append({"type": "recovery"})

        stats["occluded"] = self.occluded

        # ----------------------------------------------------------
        # Flow + motion blob: only while motion is present.
        # ----------------------------------------------------------
        if (
            self._prev_small is not None
            and motion["motion_energy"] > 0
        ):
            flow = optical_flow_summary(
                self._prev_small,
                small,
                self.camera_mag_min,
                self.camera_uni_min,
            )

            stats.update(flow)

            blob = motion_blob(self.tracker.mask)

            if blob is not None:
                stats.update(blob)

        self._prev_small = small

        # Saliency is slow relative to the rest: gate it.
        if now - self._last_saliency_at >= (
            self.saliency_interval_s
        ):
            self._last_saliency_at = now

            stats.update(spectral_saliency(small))

        stats["glance_open"] = self.segmenter._open

        self.latest_stats = stats

        trigger = (
            "light"
            if motion["light_change"]
            else "cut"
            if motion["scene_cut"]
            else None
        )

        if self.occluded:
            trigger = None

            # Never slice glances out of a covered lens.
            return events

        for glance in self.segmenter.feed(
            frame,
            motion["motion_energy"],
            motion["motion_active"],
            trigger,
            now,
        ):
            glance.duration_ms = int(
                (now - glance.started_at) * 1000
            )

            events.append(
                {"type": "glance", "glance": glance},
            )

        self.glance_open = self.segmenter._open

        return events

    def quiet_seconds(
        self,
        now: float | None = None,
    ) -> float | None:
        now = now if now is not None else time.monotonic()

        if self.tracker.last_motion_monotonic is None:
            return None

        return max(
            0.0,
            now - self.tracker.last_motion_monotonic,
        )


class VisionConfig(BaseModel):
    """
    Module-private config: config/modules/vision.yaml over these
    defaults. Path-valued fields are repo-relative strings
    (_resolve_path); `device` is the camera index/name.
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


def _load_config() -> VisionConfig:
    """
    Module-private config: config/modules/vision.yaml over the
    VisionConfig defaults. Missing or empty file = pure
    defaults; anything unparsable is a loud construction
    failure.
    """
    path = (
        _paths.repo_root()
        / "config"
        / "modules"
        / "vision.yaml"
    )

    if not path.is_file():
        return VisionConfig()

    data = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )

    if data is None:
        return VisionConfig()

    if not isinstance(data, dict):
        raise ValueError(
            f"Invalid module config YAML: {path}"
        )

    return VisionConfig(**data)


# Standard COCO-80 class names for the auto-downloaded YOLOv8n
# weights (order matches the model's class indices; background
# not listed, per labels.txt convention).
COCO80_LABELS: tuple[str, ...] = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie",
    "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog",
    "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
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

    # First-run weight source for the objects backend (standard
    # COCO-80 pretrained YOLOv8n). Missing weights are downloaded
    # into models/vision/object/ during provisioning.
    objects_repo: str = "kshitijjjjjjjjjjjjjjjj/yolov8n-coco-onnx"

    objects_filename: str = "yolov8n.onnx"

    vlm_max_tokens: int = 48

    vlm_interval_s: float = 30.0

    def __init__(self) -> None:
        cfg = _load_config()

        self.device: int | str | None = cfg.device

        self.faces_enabled = cfg.faces_enabled

        self.ocr_enabled = cfg.ocr_enabled

        self.registry_path = _resolve_path(cfg.faces_registry)

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

        self.ocr_models_dir = _resolve_path(cfg.ocr_models_dir)

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

        self.vlm_dir = _resolve_path(cfg.vlm_dir)

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

    def _provision_objects(self) -> None:
        """
        Ensure the objects backend has weights before load().

        Missing model.onnx is downloaded from self.objects_repo
        during first-run provisioning; the matching COCO-80
        labels.txt is written alongside it when absent. A
        user-dropped labels.txt is never overwritten (it defines
        the class semantics). A failing download raises out of
        start(): DOWN + backoff, never a silent limbo.
        """
        if self.objects_model_path.is_file():
            if self.objects_labels_path.is_file():
                return

            # Model without labels: the operator is mid-setup or
            # swapped in a custom export; refuse to guess class
            # names for an unknown model.
            raise RuntimeError(
                f"objects backend: model present but "
                f"{self.objects_labels_path} is missing; add a "
                f"labels.txt (one class name per line) or remove "
                f"the model to re-download the COCO default"
            )

        import huggingface_hub

        logger.info(
            "objects backend: downloading {} from {}",
            self.objects_filename,
            self.objects_repo,
        )

        downloaded = Path(
            huggingface_hub.hf_hub_download(
                repo_id=self.objects_repo,
                filename=self.objects_filename,
            )
        )

        self.objects_model_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copyfile(downloaded, self.objects_model_path)

        if not self.objects_labels_path.is_file():
            self.objects_labels_path.write_text(
                "\n".join(COCO80_LABELS) + "\n",
                encoding="utf-8",
            )

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
        if self.objects_enabled:
            self._provision_objects()

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
