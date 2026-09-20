"""
Vision toolkit + module tests.

Everything here runs on synthetic frames; no camera, no dlib,
no easyocr. The model-backed adapters are exercised only through
their pure parts (matcher, NMS) exactly like audio tests never
load whisper.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import cv2
import numpy as np

from nan_itself.modules.model import Turn
from nan_itself.utils.vision import (
    FaceMatcher,
    Glance,
    GlanceSegmenter,
    MotionTracker,
    VisionPipeline,
    VlmCaptioner,
    YoloOnnxDetector,
    frame_stats,
    motion_blob,
    optical_flow_summary,
    spectral_saliency,
    to_small_gray,
)

from builtin.modules.vision import VisionModule


# ============================================================================
# Helpers
# ============================================================================


def black_frame(
    width: int = 640,
    height: int = 480,
) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def scene_frame(
    x: int,
    y: int,
    size: int = 80,
    value: int = 255,
    bg: int = 40,
    width: int = 640,
    height: int = 480,
) -> np.ndarray:
    """
    A square on a mid-gray background: dark enough to look like a
    real room, bright enough to never trip the occlusion gate.
    """
    frame = np.full(
        (height, width, 3),
        bg,
        dtype=np.uint8,
    )

    frame[y : y + size, x : x + size] = value

    return frame


def static_scene() -> np.ndarray:
    # Anchored at the same position the motion test starts from,
    # so the first "moving" frame continues the scene seamlessly.
    return scene_frame(x=0, y=200)


def frame_with_square(
    x: int,
    y: int,
    size: int = 80,
    value: int = 255,
    width: int = 640,
    height: int = 480,
) -> np.ndarray:
    frame = black_frame(width, height)

    frame[y : y + size, x : x + size] = value

    return frame


def noise_frame(
    seed: int,
    width: int = 160,
    height: int = 120,
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    return rng.integers(
        0,
        256,
        (height, width, 3),
        dtype=np.uint8,
    )


def feed_static(
    pipeline: VisionPipeline,
    count: int,
    start: float = 0.0,
    step: float = 1 / 15,
) -> float:
    now = start

    for _ in range(count):
        pipeline.process(static_scene(), now)

        now += step

    return now


def _turn() -> Turn:
    return Turn(
        agent_hash="hash",
        parent_hash=None,
        depth=0,
        task=None,
        world={},
    )


# ============================================================================
# L1: photometric statistics
# ============================================================================


def test_frame_stats_black_and_bright():
    dark = frame_stats(black_frame())

    assert dark["brightness"] == 0.0

    assert dark["clipped_under"] == 1.0

    assert dark["entropy"] == 0.0

    bright = frame_stats(
        np.full((480, 640, 3), 255, dtype=np.uint8),
    )

    assert bright["brightness"] == 255.0

    assert bright["clipped_over"] == 1.0

    # Uniform frames have no texture: zero contrast and zero
    # sharpness.
    assert bright["contrast"] == 0.0

    assert bright["sharpness"] == 0.0


def test_frame_stats_gradient_has_entropy():
    ramp = np.tile(
        np.linspace(0, 255, 640, dtype=np.uint8),
        (480, 1),
    )

    frame = cv2.cvtColor(ramp, cv2.COLOR_GRAY2BGR)

    stats = frame_stats(frame)

    assert stats["entropy"] > 4.0

    assert stats["contrast"] > 50.0


def test_spectral_saliency_finds_bright_spot():
    left = np.zeros((120, 160), dtype=np.uint8)

    left[40:80, 20:60] = 255

    result = spectral_saliency(left)

    assert result["salient_peak"] > 0

    # Hann-windowed spectral residual localizes but still smears
    # toward the center: assert the side, not the exact spot.
    assert result["salient_x"] < 0.45

    right = np.zeros((120, 160), dtype=np.uint8)

    right[40:80, 100:140] = 255

    mirrored = spectral_saliency(right)

    assert mirrored["salient_x"] > 0.55


# ============================================================================
# L2: temporal motion
# ============================================================================


def test_motion_tracker_static_then_motion():
    tracker = MotionTracker()

    now = 0.0

    for _ in range(5):
        stats = tracker.feed(
            to_small_gray(black_frame()),
            now,
        )

        now += 1 / 15

        assert stats["motion_active"] is False

        assert stats["scene_cut"] is False

    moved = frame_with_square(x=0, y=0)

    stats = tracker.feed(to_small_gray(moved), now)

    assert stats["motion_active"] is True

    assert stats["motion_ratio"] > 0.02

    assert tracker.last_motion_monotonic == now


def test_motion_tracker_light_change():
    tracker = MotionTracker()

    now = 0.0

    dim = np.full((120, 160), 60, dtype=np.uint8)

    bright = np.full((120, 160), 140, dtype=np.uint8)

    tracker.feed(dim, now)

    now += 1 / 15

    stats = tracker.feed(bright, now)

    assert stats["light_change"] is True

    assert tracker.light_total == 1

    # Cooldown suppresses the immediate next jump.
    now += 1 / 15

    stats = tracker.feed(dim, now)

    assert stats["light_change"] is False


def test_motion_tracker_scene_cut():
    tracker = MotionTracker()

    now = 0.0

    for _ in range(3):
        tracker.feed(
            to_small_gray(noise_frame(1)),
            now,
        )

        now += 1 / 15

    stats = tracker.feed(
        to_small_gray(noise_frame(2)),
        now,
    )

    assert stats["scene_cut"] is True

    assert tracker.cuts_total == 1


def test_optical_flow_detects_translation():
    # A textured patch shifted between two frames flows right.
    base = noise_frame(7)

    base_gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)

    shifted = np.roll(base_gray, 4, axis=1)

    summary = optical_flow_summary(base_gray, shifted)

    assert summary["flow_dx"] > 0.2

    assert summary["flow_mag"] > 0.2

    # A pure shift is uniform: the camera seems to move.
    assert summary["flow_uniformity"] > 0.5


def test_motion_blob_localizes_patch():
    import cv2 as _cv2

    prev = np.zeros((120, 160), dtype=np.uint8)

    cur = np.zeros((120, 160), dtype=np.uint8)

    cur[30:90, 40:110] = 255

    mask = _cv2.threshold(
        _cv2.absdiff(cur, prev),
        18,
        255,
        _cv2.THRESH_BINARY,
    )[1]

    blob = motion_blob(mask)

    assert blob is not None

    assert 0.2 <= blob["blob_x"] <= 0.3

    assert 0.2 <= blob["blob_y"] <= 0.3

    assert blob["blob_area"] > 0.1


# ============================================================================
# L4: glance slicing
# ============================================================================


def test_glance_segmenter_slices_motion():
    segmenter = GlanceSegmenter(
        preroll_frames=4,
        trailing_quiet_frames=6,
        min_glance_frames=3,
        max_glance_frames=100,
    )

    now = 0.0

    # Static: nothing opens.
    for _ in range(4):
        assert segmenter.feed(black_frame(), 0.0, False, None, now) == []

        now += 1 / 15

    # Motion opens the glance; the square must keep moving or
    # the quiet tail closes the glance mid-loop.
    first_moving = frame_with_square(x=100, y=100)

    emitted: list[Glance] = []

    for i in range(10):
        frame = (
            first_moving
            if i == 0
            else frame_with_square(x=100 + 40 * i, y=100)
        )

        emitted.extend(
            segmenter.feed(frame, 30.0, True, None, now),
        )

        now += 1 / 15

    assert emitted == []

    # Quiet closes it after the trailing window.
    for _ in range(10):
        emitted.extend(
            segmenter.feed(black_frame(), 0.0, False, None, now),
        )

        now += 1 / 15

    assert len(emitted) == 1

    glance = emitted[0]

    assert glance.reason == "motion"

    assert glance.peak_energy == 30.0

    # The onset frame is retained as the keyframe.
    assert glance.keyframe is first_moving

    assert glance.duration_ms == 0  # filled by the pipeline


def test_glance_segmenter_drops_micro_flickers():
    segmenter = GlanceSegmenter(
        preroll_frames=4,
        trailing_quiet_frames=6,
        min_glance_frames=6,
        max_glance_frames=100,
    )

    now = 0.0

    moving = frame_with_square(x=50, y=50)

    # Two motion frames: below min_glance_frames -> dropped.
    segmenter.feed(moving, 30.0, True, None, now)

    now += 1 / 15

    segmenter.feed(moving, 30.0, True, None, now)

    now += 1 / 15

    for _ in range(10):
        emitted = segmenter.feed(
            black_frame(),
            0.0,
            False,
            None,
            now,
        )

        now += 1 / 15

        assert emitted == []


def test_glance_segmenter_trigger_opens_without_motion():
    segmenter = GlanceSegmenter(
        preroll_frames=4,
        trailing_quiet_frames=4,
        min_glance_frames=2,
        max_glance_frames=100,
    )

    now = 0.0

    # A light jump opens a glance even with no content motion.
    emitted: list[Glance] = []

    for _ in range(6):
        emitted.extend(
            segmenter.feed(
                black_frame(),
                0.5,
                False,
                "light",
                now,
            ),
        )

        now += 1 / 15

    # Stays open while the trigger keeps firing; force-close via
    # quiet frames without trigger.
    for _ in range(10):
        emitted.extend(
            segmenter.feed(black_frame(), 0.0, False, None, now),
        )

        now += 1 / 15

    assert len(emitted) == 1

    assert emitted[0].reason == "light"


# ============================================================================
# L5/L6: matcher + NMS (pure parts only)
# ============================================================================


def test_face_matcher_assign_and_promote():
    rng = np.random.default_rng(42)

    base = rng.standard_normal(128).astype(np.float32)

    # 128-dim noise: the perturbation must stay well inside the
    # 0.5 threshold (0.03 * sqrt(128) ~= 0.34).
    again = base + rng.standard_normal(128).astype(
        np.float32,
    ) * 0.03

    stranger = rng.standard_normal(128).astype(np.float32)

    matcher = FaceMatcher(
        threshold=0.5,
        promote_min_sightings=3,
        promote_min_seen_ms=0.0,
    )

    label, _, promoted = matcher.assign(base, seen_ms=100.0)

    assert label == "person-1"

    assert promoted is False

    label, dist, _ = matcher.assign(again, seen_ms=100.0)

    assert label == "person-1"

    assert dist < 0.5

    label, _, _ = matcher.assign(stranger, seen_ms=100.0)

    assert label == "person-2"

    # Third sighting promotes person-1 (evidence thresholds met).
    _, _, promoted = matcher.assign(again, seen_ms=100.0)

    assert promoted is True

    assert matcher.known_count() == 1


def test_face_matcher_serialize_restore_roundtrip():
    rng = np.random.default_rng(7)

    matcher = FaceMatcher(
        threshold=0.5,
        promote_min_sightings=1,
        promote_min_seen_ms=0.0,
    )

    for _ in range(2):
        matcher.assign(
            rng.standard_normal(128).astype(np.float32),
            seen_ms=1000.0,
        )

    payload = matcher.serialize()

    assert len(payload["persons"]) == 2

    restored = FaceMatcher()

    restored.restore(payload)

    assert restored.next_id == matcher.next_id

    assert restored.known_count() == 2

    for label, person in matcher.persons.items():
        assert (
            len(restored.persons[label]["encodings"])
            == len(person["encodings"])
        )


def test_yolo_nms_suppresses_overlaps():
    boxes = [
        (0.0, 0.0, 10.0, 10.0),
        (1.0, 1.0, 11.0, 11.0),  # overlaps the first
        (50.0, 50.0, 70.0, 70.0),  # disjoint
    ]

    scores = [0.9, 0.8, 0.7]

    keep = YoloOnnxDetector.nms(boxes, scores, 0.45)

    assert keep == [0, 2]


# ============================================================================
# Pipeline composition
# ============================================================================


def test_vision_pipeline_emits_motion_and_glance():
    pipeline = VisionPipeline(
        preroll_frames=4,
        trailing_quiet_frames=8,
        min_glance_frames=3,
        max_glance_frames=200,
        saliency_interval_s=1000.0,
    )

    now = feed_static(pipeline, 4)

    # The square must keep MOVING in connected 40px steps: a
    # static patch dies after one frame, and a teleporting patch
    # is structurally a scene cut, not motion.
    frames = [
        static_scene(),
        scene_frame(x=40, y=200),
        scene_frame(x=80, y=200),
        scene_frame(x=120, y=200),
        scene_frame(x=160, y=200),
        scene_frame(x=200, y=200),
        scene_frame(x=240, y=200),
        scene_frame(x=280, y=200),
    ]

    second_moving = frames[1]

    kinds: list[str] = []

    for frame in frames:
        kinds.extend(
            event["type"]
            for event in pipeline.process(frame, now)
        )

        now += 1 / 15

    assert "motion_start" in kinds

    stats = pipeline.latest_stats

    assert stats["motion_active"] is True

    assert stats["brightness"] > 0

    assert stats["camera_motion"] is False  # local blob, not ego

    # The scene rests where motion ended; a teleport back would
    # be a legitimate higher-energy frame and steal the keyframe.
    rest = scene_frame(x=280, y=200)

    events: list[dict] = []

    for _ in range(12):
        events.extend(
            pipeline.process(rest, now),
        )

        now += 1 / 15

    glances = [
        event for event in events
        if event["type"] == "glance"
    ]

    assert len(glances) == 1

    glance = glances[0]["glance"]

    assert glance.reason == "motion"

    assert glance.duration_ms > 0

    # Equal-energy motion frames keep the first one seen.
    assert glance.keyframe is second_moving

    assert pipeline.quiet_seconds(now) is not None


def test_vision_pipeline_light_and_cut_events():
    pipeline = VisionPipeline(
        preroll_frames=2,
        trailing_quiet_frames=4,
        min_glance_frames=2,
        saliency_interval_s=1000.0,
    )

    now = feed_static(pipeline, 3)

    # Uniform brightness jump -> light_change.
    bright = np.full((480, 640, 3), 140, dtype=np.uint8)

    kinds: list[str] = []

    kinds.extend(
        event["type"]
        for event in pipeline.process(bright, now)
    )

    assert "light_change" in kinds

    now += 1 / 15

    # Two independent noise frames -> scene_cut.
    kinds.extend(
        event["type"]
        for event in pipeline.process(
            noise_frame(1, 640, 480),
            now,
        )
    )

    now += 1 / 15

    kinds.extend(
        event["type"]
        for event in pipeline.process(
            noise_frame(2, 640, 480),
            now,
        )
    )

    assert "scene_cut" in kinds

    # Only the noise->noise pair is a structural cut: black and
    # the uniform bright frame carry no structure, so the flat
    # rule treats their transitions as non-cuts.
    assert pipeline.tracker.cuts_total == 1


def test_vision_pipeline_occlusion_events():
    pipeline = VisionPipeline(
        occlusion_brightness=6.0,
        occlusion_frames=3,
        preroll_frames=2,
        trailing_quiet_frames=4,
        min_glance_frames=2,
        saliency_interval_s=1000.0,
    )

    now = feed_static(pipeline, 2)

    # A covered lens reads as a black frame: occlusion rises once.
    kinds: list[str] = []

    for _ in range(4):
        kinds.extend(
            event["type"]
            for event in pipeline.process(
                black_frame(),
                now,
            )
        )

        now += 1 / 15

    assert kinds.count("occlusion") == 1

    assert pipeline.latest_stats["occluded"] is True

    # Light returns: exactly one recovery.
    kinds = []

    bright = np.full((480, 640, 3), 90, dtype=np.uint8)

    kinds.extend(
        event["type"]
        for event in pipeline.process(bright, now)
    )

    assert "recovery" in kinds


# ============================================================================
# Module contract (query projection + persistence)
# ============================================================================


def test_vision_module_query_unavailable():
    module = VisionModule()

    module._stats["available"] = False

    module._stats["reason"] = "no camera"

    text = asyncio.run(module.query(_turn()))

    assert text is not None

    assert "[Vision]" in text

    assert "no camera" in text


def test_vision_module_query_renders_seen():
    module = VisionModule()

    with module._state_lock:
        module._stats["available"] = True

        module._stats["motion_active"] = False

        module._stats["quiet_s"] = 12.0

        module._stats["brightness"] = 110.0

        module._stats["sharpness"] = 42.0

        module._stats["edges"] = 0.07

        module._stats["colorfulness"] = 30.0

        module._stats["glances_total"] = 1

        module._stats["person_in_view"] = True

        module._stats["last_person"] = "person-1"

        module._seen.append(
            {
                "ts": 1758000000.0,
                "reason": "motion",
                "duration_ms": 2300,
                "peak": 30.0,
                "faces": ["person-1"],
                "event": "entered",
                "ocr": "EXIT",
            },
        )

    text = asyncio.run(module.query(_turn()))

    assert "[Vision]" in text

    assert "- seeing: still 12s" in text

    assert "- scene: bright 110.0" in text

    assert "- present: person-1" in text

    assert "(motion, 2s) (person-1) [entered]" in text

    assert '[text: "EXIT"]' in text


def test_vision_module_query_silent_when_never_seen():
    module = VisionModule()

    with module._state_lock:
        module._stats["available"] = True

        module._stats["quiet_s"] = 9999.0

        module._stats["brightness"] = 0.0

        module._stats["sharpness"] = 0.0

        module._stats["edges"] = 0.0

        module._stats["colorfulness"] = 0.0

    text = asyncio.run(module.query(_turn()))

    assert text is None


def test_vision_module_state_roundtrip():
    module = VisionModule()

    module._stats["glances_total"] = 5

    module._stats["faces_total"] = 3

    payload = module.serialize_state()

    fresh = VisionModule()

    fresh.restore_state(payload)

    assert fresh._stats["glances_total"] == 5

    assert fresh._stats["faces_total"] == 3

    try:
        fresh.restore_state({"glances_total": "many"})

    except TypeError:
        pass

    else:
        raise AssertionError("TypeError expected")


# ============================================================================
# L7: VLM captioning (fake backend; real weights are drop-in)
# ============================================================================


class FakeCaptioner:
    def __init__(self, text: str = "a white square on gray.") -> None:
        self.text = text

        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def caption(self, frame: Any) -> str:
        return self.text


def _motion_glance() -> Glance:
    return Glance(
        keyframe=scene_frame(x=0, y=200),
        started_at=0.0,
        duration_ms=2000,
        peak_energy=5.0,
        bursts=0,
        reason="motion",
    )


def test_vlm_captioner_construction_is_lazy():
    captioner = VlmCaptioner(
        Path("models/vision/vlm/never-here"),
    )

    assert captioner._model is None

    assert captioner._processor is None

    assert captioner.max_tokens == 48

    assert (
        captioner.repo_id
        == VlmCaptioner.DEFAULT_REPO_ID
    )


def test_vlm_missing_weights_do_not_fail_init():
    module = VisionModule()

    if module.vlm_dir.is_dir():
        # Weights were dropped in since this test was written;
        # the degradation contract is covered by the fake tests.
        return

    # Missing weights are no longer an init failure: the
    # captioner auto-downloads on first use, so the backend
    # only degrades if that download/load later throws
    # (covered by test_vlm_caption_failure_marks_backend).
    assert module._vlm_failed is False

    assert "not loaded" in module._stats["vlm_backend"]


def test_vlm_captioner_auto_downloads_missing_weights(
    monkeypatch: Any,
):
    calls: list[dict[str, Any]] = []

    def fake_download(
        repo_id: str,
        local_dir: Path,
        **kwargs: Any,
    ) -> None:
        calls.append(
            {"repo_id": repo_id, "local_dir": local_dir},
        )

        raise RuntimeError("download interrupted (fake)")

    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        fake_download,
    )

    captioner = VlmCaptioner(
        Path("models/vision/vlm/never-here"),
    )

    try:
        captioner.load()

    except RuntimeError as exc:
        assert "fake" in str(exc)

    assert calls == [
        {
            "repo_id": VlmCaptioner.DEFAULT_REPO_ID,
            "local_dir": Path("models/vision/vlm/never-here"),
        },
    ]


def test_vlm_caption_glance_via_fake_backend():
    module = VisionModule()

    module._vlm_failed = False

    module.vlm_factory = lambda: FakeCaptioner()

    module._provision_backends()

    entry: dict[str, Any] = {}

    module._caption_glance(_motion_glance(), entry)

    assert entry["caption"] == "a white square on gray."

    assert module._stats["vlm_total"] == 1

    assert (
        module._stats["last_caption"]
        == "a white square on gray."
    )

    assert module._stats["vlm_backend"] == "FakeCaptioner"

    # Identical caption (same scene) is deduplicated away.
    dedup_entry: dict[str, Any] = {}

    module._last_vlm_at = 0.0

    module._caption_glance(_motion_glance(), dedup_entry)

    assert "caption" not in dedup_entry

    # Interval gate blocks the next caption immediately.
    gated_entry: dict[str, Any] = {}

    module._last_caption = ""

    module._last_vlm_at = time.monotonic()

    module._caption_glance(_motion_glance(), gated_entry)

    assert "caption" not in gated_entry


def test_vlm_caption_failure_marks_backend():
    module = VisionModule()

    module._vlm_failed = False

    class Broken:
        def load(self) -> None:
            raise RuntimeError("no weights")

    module.vlm_factory = lambda: Broken()

    module._provision_backends()

    entry: dict[str, Any] = {}

    module._caption_glance(_motion_glance(), entry)

    assert "caption" not in entry

    assert module._vlm_failed is True

    assert "no weights" in module._stats["vlm_backend"]


def test_vlm_caption_renders_in_query():
    module = VisionModule()

    with module._state_lock:
        module._stats["available"] = True

        module._stats["motion_active"] = True

        module._stats["glances_total"] = 1

        module._seen.append(
            {
                "ts": 1758000000.0,
                "reason": "motion",
                "duration_ms": 1500,
                "peak": 5.0,
                "caption": "a person at a desk",
            },
        )

    text = asyncio.run(module.query(_turn()))

    assert '[caption: "a person at a desk"]' in text
