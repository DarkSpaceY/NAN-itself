"""
Vision toolkit for the seeing module.

Everything here is deliberately free of async and of the Module
framework: pure functions, pure state machines, and thin hardware
adapters behind small interfaces -- the same stance as utils/audio.py.

Blocks:

    frame_stats           L1 photometric statistics of one frame
    MotionTracker         L2 frame-diff energy / cuts / light jumps
    optical_flow_summary  L3 Farneback flow: ego-motion vs scene motion
    motion_blob           L3 largest connected motion region
    spectral_saliency     L3 frequency-domain attention peak
    GlanceSegmenter       L4 event-driven slicing of the frame stream
    CameraSource          default camera source (OpenCV/AVFoundation)
    FaceAnalyzer          L5 lazy face_recognition adapter (detect+encode)
    FaceMatcher           L6 auto-enrolling person registry
    QrScanner             L6 QR decode via OpenCV
    OcrReader             L6 lazy easyocr adapter
    YoloOnnxDetector      L5 lazy ONNX object detector (model drop-in)
    VisionPipeline        composes the per-frame brain work

Frame contract: BGR uint8 numpy frames as produced by OpenCV.
Heavy work happens on small grayscale proxies (small_width); the
original frame is kept only as a glance keyframe candidate.
"""

from __future__ import annotations

import time
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger


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
    import cv2

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
    import cv2
    import numpy as np

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
    import cv2

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
    import cv2
    import numpy as np

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
        import cv2
        import numpy as np

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
    import cv2
    import numpy as np

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
    import cv2
    import numpy as np

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
        import cv2

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
    import sys

    return sys.platform == "darwin"


# ============================================================================
# L5/L6: model-backed analyzers (all lazy, all optional)
# ============================================================================


class FaceAnalyzer:
    """
    Lazy face_recognition adapter: detect faces and produce 128-d
    encodings. import happens on first use so units without dlib
    still collect cleanly.
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

        import face_recognition  # noqa: F401

        self._loaded = True

    def analyze(self, frame: Any) -> list[dict]:
        """
        Returns [{"box": (top, right, bottom, left),
        "encoding": ndarray|None}, ...] in ORIGINAL frame
        coordinates.
        """
        import cv2
        import face_recognition

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
        import numpy as np

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
        import numpy as np

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
            import cv2

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

        import easyocr

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

    Activates only when a model has been dropped in at
    models/vision/object/model.onnx with a labels file next to it
    (labels.txt, one name per line, background not listed).
    Missing model -> load() raises -> the module degrades the
    backend to unavailable, exactly like the audio tagger.
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

        import onnxruntime as ort

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
        import numpy as np

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
        import cv2
        import numpy as np

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
    a missing directory is auto-downloaded from the Hub on first
    use (respects HF_ENDPOINT / proxy env vars), and a failing
    download keeps the backend marked unavailable without
    touching any other layer.

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
        from huggingface_hub import snapshot_download

        logger.info(
            "vlm weights missing at {}; downloading {} "
            "(~1 GB, honors HF_ENDPOINT and proxy envs)",
            self.model_path,
            self.repo_id,
        )

        snapshot_download(
            repo_id=self.repo_id,
            local_dir=self.model_path,
        )

    def load(self) -> None:
        if self._model is not None:
            return

        if not self._weights_present():
            self._download()

        import torch

        from transformers import (
            AutoModelForVision2Seq,
            AutoProcessor,
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

        self._processor = AutoProcessor.from_pretrained(
            path,
        )

        self._model = (
            AutoModelForVision2Seq.from_pretrained(
                path,
                torch_dtype=dtype,
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
        import torch

        from PIL import Image

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
    import cv2

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
