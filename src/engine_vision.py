"""
src/engine_vision.py
--------------------
Phase 2 dynamic subject tracking and cinematic reframing engine.

Detection pipeline:
  - Primary:  InsightFace (SCRFD) with 5-point facial landmarks (buffalo_sc / buffalo_l)
  - Secondary: Ultralytics YOLOv11 ("yolo11n.pt") on GPU with CPU fallback
  - Fallback: OpenCV Haar Cascade / center-frame static position

Smoothing & Framing:
  - SmoothGlideTracker: 1D Kalman Filter (Position + Velocity) with EWMA smoothing,
    deadband suppression, momentum coasting on subject exit, and gentle centering glide.
  - Anchors strictly to the anatomical eye/nose landmark center rather than jittery torso boxes.

Zoom keyframes:
  - Detects high-motion segments and injects subtle zoom pulses (1.0x → 1.12x)
    every 4–6 seconds to reset viewer attention.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
import logging
import os
import statistics
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
try:
    import torch
except Exception:
    torch = None

logger = logging.getLogger("engine_vision")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_CROP_W = 1080    # 9:16 output crop width
TARGET_CROP_H = 1540    # 9:16 output crop height (within 1920 canvas)
VERTICAL_AR_THRESHOLD = 1.0   # Videos with aspect ratio <= 1.0 (height >= width) are portrait/square

ALPHA_SMOOTH = 0.15     # Exponential smoothing factor (lower = smoother but laggier)
SAMPLE_STEP_SEC = 1.0   # Trajectory keyframe export & frame sample interval (1.0 second)
MAX_KEYFRAMES = 500     # Maximum keyframes exported per slice (raised to 500 to prevent freeze)
ZOOM_MIN = 1.0
ZOOM_MAX = 1.12
ZOOM_INTERVAL_SEC = (4, 6)   # Range of seconds between zoom pulses
ZOOM_DURATION_SEC = 1.5      # Duration of each zoom ramp

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_BASE_DIR, ".."))


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------
@dataclass
class FaceDetection:
    """
    Rich face detection record produced by InsightFace SCRFD.
    Contains bounding box (x, y, w, h), 5-point facial landmarks,
    and anatomically stabilized eye/nose landmark focal center.
    """
    box: Tuple[int, int, int, int]           # (x, y, w, h)
    landmarks: Optional[np.ndarray] = None   # 5x2 array: [[eye_l_x, eye_l_y], [eye_r_x, eye_r_y], [nose_x, nose_y], ...]
    landmark_center: Optional[Tuple[float, float]] = None  # (focal_x, focal_y)
    score: float = 1.0

    @property
    def x(self) -> int:
        return self.box[0]

    @property
    def y(self) -> int:
        return self.box[1]

    @property
    def w(self) -> int:
        return self.box[2]

    @property
    def h(self) -> int:
        return self.box[3]


# ---------------------------------------------------------------------------
# InsightFace (SCRFD) loader (GPU CUDA with CPU fallback)
# ---------------------------------------------------------------------------
_insightface_app = None
_insightface_available = False
_insightface_device = "cpu"


def _get_device() -> str | int:
    """Returns 0 if GPU (CUDA) is available, else 'cpu'."""
    try:
        if torch is not None and torch.cuda.is_available():
            return 0
    except Exception:
        pass
    return "cpu"


def _try_load_insightface() -> bool:
    """
    Attempts to load InsightFace (SCRFD) model configured for GPU if available, else CPU.
    Loads lightweight 'buffalo_sc' or 'buffalo_l' with detection-only module for maximal inference speed.
    """
    global _insightface_app, _insightface_available, _insightface_device
    if _insightface_available and _insightface_app is not None:
        return True
    try:
        import insightface
        from insightface.app import FaceAnalysis

        use_cuda = False
        try:
            if torch is not None and torch.cuda.is_available():
                use_cuda = True
        except Exception:
            pass

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_cuda else ['CPUExecutionProvider']
        ctx_id = 0 if use_cuda else -1

        app = None
        for model_name in ["buffalo_sc", "buffalo_l"]:
            try:
                app = FaceAnalysis(name=model_name, allowed_modules=['detection'], providers=providers)
                app.prepare(ctx_id=ctx_id, det_size=(640, 640))
                break
            except Exception as model_err:
                logger.debug(f"[engine_vision] Could not load InsightFace model {model_name}: {model_err}")
                app = None

        if app is None:
            app = FaceAnalysis(allowed_modules=['detection'], providers=providers)
            app.prepare(ctx_id=ctx_id, det_size=(640, 640))

        _insightface_app = app
        _insightface_available = True
        _insightface_device = "cuda" if use_cuda else "cpu"
        logger.info(f"[engine_vision] Loaded InsightFace SCRFD on {_insightface_device}")
        return True
    except Exception as e:
        logger.debug(f"[engine_vision] InsightFace SCRFD unavailable in current environment: {e}")
        _insightface_app = None
        _insightface_available = False
        return False


def purge_insightface_model() -> None:
    """Explicitly deletes the cached InsightFace model from memory."""
    global _insightface_app, _insightface_available
    if _insightface_app is not None:
        try:
            del _insightface_app
        except Exception:
            pass
        _insightface_app = None
        _insightface_available = False
    gc.collect()
    try:
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    logger.debug("[engine_vision] Purged InsightFace model from VRAM.")


# ---------------------------------------------------------------------------
# YOLO loader (Secondary / Legacy fallback)
# ---------------------------------------------------------------------------
_yolo_model = None
_yolo_available = False
_yolo_device: str | int = "cpu"


def _try_load_yolo() -> bool:
    """Attempts to load YOLOv11 model configured for GPU if available, else CPU."""
    global _yolo_model, _yolo_available, _yolo_device
    if _yolo_available and _yolo_model is not None:
        return True
    try:
        from ultralytics import YOLO  # type: ignore

        _yolo_device = _get_device()
        model_candidates = [
            os.path.join(_PROJECT_ROOT, "yolo11n.pt"),
            os.path.join(_PROJECT_ROOT, "assets", "models", "yolo11n.pt"),
            "yolo11n.pt",
        ]
        loaded_model = None
        for candidate in model_candidates:
            if os.path.isfile(candidate):
                loaded_model = YOLO(candidate)
                break
        if loaded_model is None:
            loaded_model = YOLO("yolo11n.pt")

        _yolo_model = loaded_model
        _yolo_available = True
        return True
    except Exception as e:
        logger.debug(f"[engine_vision] YOLO fallback unavailable: {e}")
        _yolo_model = None
        _yolo_available = False
        return False


def purge_yolo_model() -> None:
    """
    Explicitly deletes the cached vision models from memory and flushes PyTorch CUDA cache.
    Prevents VRAM leaks before launching FFmpeg NVENC.
    """
    global _yolo_model, _yolo_available
    purge_insightface_model()
    if _yolo_model is not None:
        try:
            del _yolo_model
        except Exception:
            pass
        _yolo_model = None
        _yolo_available = False

    gc.collect()
    try:
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    logger.debug("[engine_vision] Purged vision models from VRAM.")


purge_vision_models = purge_yolo_model
evict_yolo_model = purge_yolo_model


# ---------------------------------------------------------------------------
# Core detection helpers
# ---------------------------------------------------------------------------

def _detect_insightface(frame: np.ndarray) -> List[FaceDetection]:
    """
    Runs InsightFace SCRFD on a frame.
    Extracts bounding boxes and 5 facial landmarks (eyes, nose, mouth corners).
    Calculates stable eye/nose anatomical focal center.
    """
    if not _insightface_available or _insightface_app is None:
        return []
    try:
        faces = _insightface_app.get(frame)
        results: List[FaceDetection] = []
        for face in faces:
            bbox = face.bbox
            x1, y1, x2, y2 = map(int, bbox[:4])
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            score = float(face.det_score) if hasattr(face, "det_score") else 1.0

            kps = face.kps if hasattr(face, "kps") and face.kps is not None else None
            if kps is not None and len(kps) >= 3:
                # Landmark anchors:
                # kps[0] = left eye, kps[1] = right eye, kps[2] = nose
                eye_l = kps[0]
                eye_r = kps[1]
                nose = kps[2]
                eye_mid_x = (float(eye_l[0]) + float(eye_r[0])) / 2.0
                eye_mid_y = (float(eye_l[1]) + float(eye_r[1])) / 2.0
                # Weighted center: 50% eye midpoint + 50% nose for horizontal anchor
                focal_x = 0.5 * eye_mid_x + 0.5 * float(nose[0])
                focal_y = 0.4 * eye_mid_y + 0.6 * float(nose[1])
            else:
                focal_x = float(x1 + w / 2.0)
                focal_y = float(y1 + h * 0.38)

            results.append(FaceDetection(
                box=(x1, y1, w, h),
                landmarks=kps,
                landmark_center=(focal_x, focal_y),
                score=score
            ))
        return results
    except Exception as e:
        logger.debug(f"[engine_vision] InsightFace inference error: {e}")
        return []


def _detect_yolo(frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Runs YOLO on a frame.
    Returns list of (x, y, w, h) bounding boxes for detected humans (person class 0).
    """
    if not _yolo_available or _yolo_model is None:
        return []
    try:
        results = _yolo_model(frame, device=_yolo_device, classes=[0], verbose=False)
        boxes = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                boxes.append((x1, y1, x2 - x1, y2 - y1))
        return boxes
    except Exception:
        if _yolo_device != "cpu":
            try:
                results = _yolo_model(frame, device="cpu", classes=[0], verbose=False)
                boxes = []
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        boxes.append((x1, y1, x2 - x1, y2 - y1))
                return boxes
            except Exception:
                pass
        return []


def detect_faces_or_subjects(frame: np.ndarray, prefer_insightface: bool = True) -> List[FaceDetection]:
    """
    Detects faces/human subjects in a video frame.
    Primary: InsightFace SCRFD with eye/nose landmarks.
    Fallback: YOLOv11 person detection -> Haar Cascade -> empty.
    """
    if prefer_insightface and _try_load_insightface():
        faces = _detect_insightface(frame)
        if faces:
            return faces

    # Fallback 1: YOLO
    if _try_load_yolo():
        yolo_boxes = _detect_yolo(frame)
        if yolo_boxes:
            return [
                FaceDetection(
                    box=b,
                    landmarks=None,
                    landmark_center=(float(b[0] + b[2] / 2.0), float(b[1] + b[3] * 0.35)),
                    score=0.9
                )
                for b in yolo_boxes
            ]

    # Fallback 2: OpenCV Haar Cascade
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if os.path.isfile(cascade_path):
            face_cascade = cv2.CascadeClassifier(cascade_path)
            haar_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
            if len(haar_faces) > 0:
                return [
                    FaceDetection(
                        box=(int(x), int(y), int(w), int(h)),
                        landmarks=None,
                        landmark_center=(float(x + w / 2.0), float(y + h * 0.4)),
                        score=0.8
                    )
                    for (x, y, w, h) in haar_faces
                ]
    except Exception:
        pass

    return []


def detect_subjects(frame: np.ndarray, use_yolo: bool = True) -> List[Tuple[int, int, int, int]]:
    """
    Detects faces/human subjects in a video frame.
    Uses InsightFace SCRFD as primary detector, falling back to YOLO or Haar.
    Returns list of (x, y, w, h) bounding boxes.
    """
    detections = detect_faces_or_subjects(frame)
    if detections:
        return [d.box for d in detections]
    return []


# ---------------------------------------------------------------------------
# Scene Cut Detection (scenedetect)
# ---------------------------------------------------------------------------

def detect_scene_cuts(
    video_path: str,
    start_time: float = 0.0,
    end_time: Optional[float] = None,
    threshold: float = 27.0,
) -> List[float]:
    """
    Detects hard scene/shot transitions in a video within [start_time, end_time]
    using PySceneDetect (SceneManager, open_video, ContentDetector).

    Args:
        video_path: Path to video file.
        start_time: Start offset in seconds (default 0.0).
        end_time: Optional end offset in seconds.
        threshold: ContentDetector sensitivity threshold (default 27.0).

    Returns:
        List of cut timestamps in seconds, relative to start_time (timestamp offsets within the slice).
        If scenedetect is unavailable or an error occurs, logs the error and returns [].
    """
    if not video_path or not os.path.isfile(video_path):
        return []

    try:
        from scenedetect import SceneManager, open_video
        from scenedetect.detectors import ContentDetector

        video = open_video(video_path)
        base_start = max(0.0, float(start_time))
        if base_start > 0.0:
            try:
                video.seek(base_start)
            except Exception as seek_err:
                logger.debug(f"[engine_vision] Could not seek video to {base_start}s: {seek_err}")

        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=threshold))

        detect_kwargs: Dict[str, Any] = {}
        if end_time is not None and float(end_time) > base_start:
            detect_kwargs["end_time"] = float(end_time)

        scene_manager.detect_scenes(video, **detect_kwargs)
        scene_list = scene_manager.get_scene_list()

        cuts: List[float] = []
        for scene in scene_list:
            cut_sec = scene[0].get_seconds()
            rel_sec = cut_sec - base_start if base_start > 0.0 else cut_sec
            if rel_sec > 0.05:  # Ignore cuts directly at 0 boundary
                cuts.append(round(rel_sec, 4))

        logger.info(f"[engine_vision] Detected {len(cuts)} scene cut(s) in {os.path.basename(video_path)}: {cuts}")
        return cuts

    except Exception as exc:
        logger.error(f"[engine_vision] Scene cut detection failed for {video_path}: {exc}", exc_info=True)
        return []


# ---------------------------------------------------------------------------
# Pan offset calculation
# ---------------------------------------------------------------------------

def calculate_pan_offset(
    detections: List[Any],
    source_w: int,
    target_crop_w: int = TARGET_CROP_W,
    focal_box: Optional[Tuple[int, int, int, int]] = None,
    focal_center_x: Optional[float] = None,
) -> int:
    """
    Calculates the horizontal x_offset to frame the focal subject
    inside a `target_crop_w` wide vertical crop window.

    Supports direct focal_center_x (from eye/nose landmarks) or focal_box.
    Returns: x_offset (clamped to [0, max(0, source_w - target_crop_w)])
    """
    max_x = max(0, source_w - target_crop_w)
    center_x = max(0, (source_w - target_crop_w) // 2)

    # 1. Direct facial landmark anchor center
    if focal_center_x is not None:
        x_offset = int(round(focal_center_x - target_crop_w / 2.0))
        return max(0, min(x_offset, max_x))

    if not detections and not focal_box:
        return center_x  # Safe center-crop fallback

    if focal_box:
        fx, fy, fw, fh = focal_box[:4]
    else:
        # Check if detections are FaceDetection objects or raw tuples
        if detections and hasattr(detections[0], "box"):
            focal_obj = max(detections, key=lambda d: d.box[2] * d.box[3])
            if focal_obj.landmark_center is not None:
                x_offset = int(round(focal_obj.landmark_center[0] - target_crop_w / 2.0))
                return max(0, min(x_offset, max_x))
            fx, fy, fw, fh = focal_obj.box
        else:
            fx, fy, fw, fh = max(detections, key=lambda b: b[2] * b[3])[:4]

    # Center the crop window on the focal subject's horizontal midpoint
    subject_center_x = fx + fw // 2
    x_offset = subject_center_x - target_crop_w // 2

    return max(0, min(x_offset, max_x))


# ---------------------------------------------------------------------------
# Smooth-Glide Kalman Filter + EWMA Tracker
# ---------------------------------------------------------------------------

class SmoothGlideTracker:
    """
    Combines a 1D Kalman Filter (Position + Velocity state) with EWMA smoothing
    to create a cinematic, smooth-glide camera tracking experience.

    Key Features:
    1. Landmark-anchored tracking:
       Locks to the eye/nose landmark center rather than shifting torso boxes.
    2. Deadband threshold:
       Suppresses micro-jitter from head gestures or speaking cadence.
    3. Smooth-glide momentum:
       When a subject temporarily steps out of view or turns away, the camera
       does not snap. It coasts with damped momentum.
    4. Gentle centering glide:
       If the subject steps out for prolonged duration (>1.5s), the camera
       gently glides back towards neutral center at a controlled cruising velocity.
    5. Velocity clamping:
       Restricts maximum pan speed to prevent dizzying whip-pans.
    6. Instant scene-cut reset:
       Precludes moving-average drag across shot transitions.
    """

    def __init__(
        self,
        source_w: int,
        target_crop_w: int = TARGET_CROP_W,
        dt: float = 1.0,
        process_noise: float = 3.0,
        measurement_noise: float = 15.0,
        ewma_alpha: float = 0.18,
        deadband: float = 16.0,
        max_velocity: float = 180.0,
    ):
        self.source_w = source_w
        self.target_crop_w = target_crop_w
        self.max_x = max(0, source_w - target_crop_w)
        self.neutral_x = self.max_x // 2
        self.dt = dt
        self.alpha = ewma_alpha
        self.deadband = deadband
        self.max_v = max_velocity

        self.x = float(self.neutral_x)
        self.v = 0.0
        self.p00 = 100.0
        self.p01 = 0.0
        self.p10 = 0.0
        self.p11 = 50.0

        self.q_pos = process_noise
        self.q_vel = process_noise * 2.0
        self.r = measurement_noise

        self.smoothed_x = float(self.neutral_x)
        self.frames_without_detection = 0
        self.initialized = False

    def reset(self, new_x: Optional[float] = None) -> None:
        """Resets tracker state instantaneously for scene cuts."""
        reset_pos = new_x if new_x is not None else float(self.neutral_x)
        reset_pos = max(0.0, min(float(self.max_x), float(reset_pos)))
        self.x = reset_pos
        self.v = 0.0
        self.p00 = 50.0
        self.p01 = 0.0
        self.p10 = 0.0
        self.p11 = 20.0
        self.smoothed_x = reset_pos
        self.frames_without_detection = 0
        self.initialized = True

    def update(self, target_pan_x: Optional[float], dt: Optional[float] = None) -> int:
        """
        Updates tracker state and returns smooth clamped integer pan offset.
        target_pan_x: Desired pan offset from eye/nose landmark center (None if no detection).
        """
        step = dt if dt is not None and dt > 0 else self.dt

        if not self.initialized:
            start_x = target_pan_x if target_pan_x is not None else self.neutral_x
            self.reset(start_x)
            return int(round(self.smoothed_x))

        # 1. Kalman Predict Step
        x_pred = self.x + self.v * step
        v_pred = self.v

        p00_pred = self.p00 + step * (self.p10 + self.p01) + (step ** 2) * self.p11 + self.q_pos
        p01_pred = self.p01 + step * self.p11
        p10_pred = self.p10 + step * self.p11
        p11_pred = self.p11 + self.q_vel

        # 2. Measurement Update or Missing Subject Smooth-Glide
        if target_pan_x is not None:
            self.frames_without_detection = 0
            meas_x = max(0.0, min(float(self.max_x), float(target_pan_x)))

            # Deadband check: if within deadband of current position, suppress jitter
            if abs(meas_x - self.x) < self.deadband:
                meas_x = self.x

            # Kalman gain
            s = p00_pred + self.r
            k0 = p00_pred / max(1e-4, s)
            k1 = p10_pred / max(1e-4, s)

            # State update
            y = meas_x - x_pred
            self.x = x_pred + k0 * y
            self.v = v_pred + k1 * y

            # Covariance update
            self.p00 = p00_pred * (1.0 - k0)
            self.p01 = p01_pred * (1.0 - k0)
            self.p10 = -k1 * p00_pred + p10_pred
            self.p11 = -k1 * p01_pred + p11_pred

        else:
            # Subject stepped out of view or undetected
            self.frames_without_detection += 1

            if self.frames_without_detection <= 2:
                # Brief absence (step out or turn around): coast with decaying momentum
                self.v *= 0.70
                self.x = x_pred
            else:
                # Prolonged absence: gracefully glide towards neutral center
                dist_to_center = self.neutral_x - self.x
                glide_dir = 1.0 if dist_to_center > 0 else -1.0
                glide_speed = min(45.0, abs(dist_to_center) * 0.35)
                self.v = glide_dir * glide_speed
                self.x += self.v * step
                if abs(self.neutral_x - self.x) < 4.0:
                    self.x = float(self.neutral_x)
                    self.v = 0.0

            self.p00 = p00_pred
            self.p01 = p01_pred
            self.p10 = p10_pred
            self.p11 = p11_pred

        # Clamp velocity
        self.v = max(-self.max_v, min(self.max_v, self.v))

        # Clamp position
        self.x = max(0.0, min(float(self.max_x), self.x))

        # 3. EWMA Smoothing
        self.smoothed_x = self.alpha * self.x + (1.0 - self.alpha) * self.smoothed_x
        self.smoothed_x = max(0.0, min(float(self.max_x), self.smoothed_x))

        return int(round(self.smoothed_x))



# ---------------------------------------------------------------------------
# Exponential smoothing with scene cut awareness
# ---------------------------------------------------------------------------

def smooth_trajectory(
    raw_x_offsets: List[int],
    alpha: float = ALPHA_SMOOTH,
    timestamps: Optional[List[float]] = None,
    scene_cuts: Optional[List[float]] = None,
) -> List[int]:
    """
    Applies exponential smoothing to a raw per-sample x_offset list.
    smoothed[i] = alpha * raw[i] + (1 - alpha) * smoothed[i-1]
    Uses alpha=0.15 for buttery-smooth camera pans.

    If scene_cuts are provided, whenever a sample interval crosses a scene cut boundary,
    the filter state is hard-reset (smoothed[i] = raw_x_offsets[i]) to prevent
    smoothing or dragging camera positions across disparate scenes.
    """
    if not raw_x_offsets:
        return []

    smoothed = [raw_x_offsets[0]]
    cuts = sorted([float(c) for c in (scene_cuts or []) if c > 0])

    for i in range(1, len(raw_x_offsets)):
        x = raw_x_offsets[i]
        reset_filter = False

        if cuts and timestamps and i < len(timestamps):
            t_prev = timestamps[i - 1]
            t_curr = timestamps[i]
            for cut in cuts:
                if (t_prev < cut <= t_curr) or abs(t_curr - cut) <= 0.05 or abs(t_prev - cut) <= 0.05:
                    reset_filter = True
                    break

        if reset_filter:
            # Hard-reset filter state on scene transition boundary: no moving average across cuts
            smoothed.append(int(round(x)))
        else:
            smoothed.append(int(round(alpha * x + (1.0 - alpha) * smoothed[-1])))

    return smoothed


# ---------------------------------------------------------------------------
# Zoom keyframe generation
# ---------------------------------------------------------------------------

def get_zoom_keyframes(
    x_offsets: List[int],
    timestamps: List[float],
    fps: float,
    interval_sec: Tuple[float, float] = ZOOM_INTERVAL_SEC,
    zoom_max: float = ZOOM_MAX,
    zoom_duration: float = ZOOM_DURATION_SEC,
    scene_cuts: Optional[List[float]] = None,
) -> List[Dict[str, float]]:
    """
    Generates subtle zoom-in keyframes (1.0x → zoom_max) on high-energy
    segments to reset viewer attention every interval_sec seconds.

    High-energy segments are detected by large x_offset deltas between
    adjacent samples (active camera motion / subject movement).
    Suppresses false-motion zoom triggers across scene cuts.

    Returns: List of {time, zoom, duration} dicts, sorted by time.
    """
    if not x_offsets or not timestamps or len(x_offsets) < 2:
        return []

    cuts = sorted([float(c) for c in (scene_cuts or []) if c > 0])
    keyframes: List[Dict[str, float]] = []
    min_interval = interval_sec[0]
    max_interval = interval_sec[1]
    last_zoom_t = -max_interval  # Allow zoom immediately at t=0 range

    # Compute per-sample motion magnitude, filtering out scene-transition cuts
    deltas = []
    for i in range(1, len(x_offsets)):
        t_prev = timestamps[i - 1] if i - 1 < len(timestamps) else 0.0
        t_curr = timestamps[i] if i < len(timestamps) else 0.0
        crosses_cut = False
        if cuts:
            for c in cuts:
                if t_prev < c <= t_curr:
                    crosses_cut = True
                    break
        if crosses_cut:
            deltas.append(0)
        else:
            deltas.append(abs(x_offsets[i] - x_offsets[i - 1]))

    if not deltas:
        return []

    # Median delta = baseline motion; high-energy = delta > median
    median_delta = statistics.median(deltas)
    threshold = max(8, median_delta * 1.5)  # At least 8px shift to qualify

    for i, ts in enumerate(timestamps):
        if ts < last_zoom_t + min_interval:
            continue

        # Suppress zoom pulse if within 0.3s of a scene cut
        if cuts and any(abs(ts - c) < 0.3 for c in cuts):
            continue

        # i-1 maps to deltas index. deltas has len(x_offsets)-1 elements.
        is_high_energy = (i > 0 and (i - 1) < len(deltas) and deltas[i - 1] >= threshold)
        if ts > last_zoom_t + max_interval or is_high_energy:
            # Schedule a zoom pulse at this timestamp
            keyframes.append({
                "time": round(ts, 3),
                "zoom": round(zoom_max, 4),
                "duration": zoom_duration,
            })
            last_zoom_t = ts

    return keyframes


# ---------------------------------------------------------------------------
# Trajectory downsampling & clamping
# ---------------------------------------------------------------------------

def downsample_trajectory(
    timestamps: List[float],
    x_offsets: List[int],
    sample_step: float = SAMPLE_STEP_SEC,
    max_keyframes: int = MAX_KEYFRAMES,
    scene_cuts: Optional[List[float]] = None,
) -> Tuple[List[float], List[int]]:
    """
    Downsamples dense trajectory keyframes to `sample_step` second intervals
    (e.g., 1.0s), strictly preserving shot-boundary keyframe pairs (pre-cut and post-cut),
    and ensures the total count stays within `max_keyframes` (default 500).
    Preserves start, end, and scene-cut keyframes for smooth framing throughout the slice.
    """
    if not timestamps or not x_offsets:
        return timestamps, x_offsets
    if len(timestamps) <= 1:
        return timestamps[:max_keyframes], x_offsets[:max_keyframes]

    min_len = min(len(timestamps), len(x_offsets))
    timestamps = list(timestamps[:min_len])
    x_offsets = list(x_offsets[:min_len])

    cuts = sorted([float(c) for c in (scene_cuts or []) if c > 0])

    # Identify indices that surround scene cut boundaries (strictly preserve pre-cut and post-cut pairs)
    cut_boundary_indices = set()
    if cuts:
        for c in cuts:
            pre_idx = None
            post_idx = None
            for idx, ts in enumerate(timestamps):
                if ts < c:
                    pre_idx = idx
                elif ts >= c:
                    post_idx = idx
                    break
            if pre_idx is not None:
                cut_boundary_indices.add(pre_idx)
            if post_idx is not None:
                cut_boundary_indices.add(post_idx)

    downsampled_t = [timestamps[0]]
    downsampled_x = [x_offsets[0]]
    last_t = timestamps[0]

    for i in range(1, len(timestamps)):
        ts = timestamps[i]
        x = x_offsets[i]

        is_cut_boundary = (i in cut_boundary_indices)

        if is_cut_boundary or (ts - last_t >= sample_step - 1e-4):
            downsampled_t.append(round(ts, 4))
            downsampled_x.append(x)
            last_t = ts

    # Ensure the final sample is included if not already close to the end
    if timestamps[-1] - downsampled_t[-1] > 0.1:
        downsampled_t.append(round(timestamps[-1], 4))
        downsampled_x.append(x_offsets[-1])

    # Safety clamp: if keyframes still exceed max_keyframes, subsample uniformly
    if len(downsampled_t) > max_keyframes:
        if max_keyframes > 1:
            n = len(downsampled_t)
            step = (n - 1) / (max_keyframes - 1)
            indices = sorted(list(set(int(round(j * step)) for j in range(max_keyframes))))
            downsampled_t = [downsampled_t[j] for j in indices]
            downsampled_x = [downsampled_x[j] for j in indices]
        else:
            downsampled_t = downsampled_t[:max_keyframes]
            downsampled_x = downsampled_x[:max_keyframes]

    return downsampled_t[:max_keyframes], downsampled_x[:max_keyframes]


# ---------------------------------------------------------------------------
# Main trajectory computation
# ---------------------------------------------------------------------------

def detect_gaming_layout(video_path: str, num_frames: int = 5) -> Tuple[bool, Tuple[int, int, int, int]]:
    """
    Samples frames from the beginning of the video.
    If a subject is consistently found in a specific quadrant (e.g., top-left, top-right)
    and is small enough (width < 35% of source), flags as gaming layout.
    Returns (is_gaming, (x, y, w, h)).
    """
    if not _yolo_available or _yolo_model is None:
        _try_load_yolo()
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, (0, 0, 0, 0)
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps): fps = 30.0
    source_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if source_w <= 0 or source_h <= 0:
        cap.release()
        return False, (0, 0, 0, 0)
        
    # sample frames every 0.5 seconds
    step = max(1, int(fps * 0.5))
    
    webcam_boxes = []
    frames_read = 0
    max_frames = num_frames * step
    
    gray_frames = []
    
    while frames_read < max_frames:
        ret, frame = cap.read()
        if not ret: break
        
        if frames_read % step == 0:
            gray_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            boxes = detect_subjects(frame, use_yolo=True)
            for box in boxes:
                x, y, w, h = box
                cx = x + w // 2
                cy = y + h // 2
                # Check if it's a corner webcam (top 40% or bottom 40%, and side 40%)
                # and small enough (width < 40% of screen)
                is_corner = (cy < source_h * 0.4 or cy > source_h * 0.6) and (cx < source_w * 0.4 or cx > source_w * 0.6)
                is_small = w < source_w * 0.4 and h < source_h * 0.4
                if is_corner and is_small:
                    webcam_boxes.append(box)
                    break # just take the first one that fits
        frames_read += 1
        
    cap.release()
    
    if len(webcam_boxes) >= max(1, num_frames - 2):
        # Found consistent webcam
        med_x = int(np.median([b[0] for b in webcam_boxes]))
        med_y = int(np.median([b[1] for b in webcam_boxes]))
        med_w = int(np.median([b[2] for b in webcam_boxes]))
        med_h = int(np.median([b[3] for b in webcam_boxes]))
        
        # Verify it's actually a gaming layout by checking motion in the non-webcam region
        # Or by checking the variance of the webcam box itself (webcam borders are perfectly static)
        centers_x = [b[0] + b[2]//2 for b in webcam_boxes]
        centers_y = [b[1] + b[3]//2 for b in webcam_boxes]
        var_x = np.var(centers_x) if len(centers_x) > 1 else 0
        var_y = np.var(centers_y) if len(centers_y) > 1 else 0
        
        # Also check motion score
        motion_scores = []
        for i in range(1, len(gray_frames)):
            diff = cv2.absdiff(gray_frames[i], gray_frames[i-1])
            diff[med_y:med_y+med_h, med_x:med_x+med_w] = 0
            motion_scores.append(diff.mean())
            
        avg_motion = np.mean(motion_scores) if motion_scores else 0
        
        # A real webcam box shouldn't move around. Panning shots will have high variance.
        corner_detections = len(webcam_boxes)
        is_gaming = bool((var_x + var_y) < 250.0) # Highly static box = true webcam overlay
        
        logger.info(f"[engine_vision] detect_gaming_layout: corner_detections={corner_detections}/{len(gray_frames)}, box_variance={var_x+var_y:.2f}, non_webcam_motion={avg_motion:.2f}, is_gaming={is_gaming}")
        print(f"[engine_vision] detect_gaming_layout: corner_detections={corner_detections}/{len(gray_frames)}, box_variance={var_x+var_y:.2f}, non_webcam_motion={avg_motion:.2f}, is_gaming={is_gaming}")
        
        if is_gaming:
            return True, (med_x, med_y, med_w, med_h)
        
    return False, (0, 0, 0, 0)


# ---------------------------------------------------------------------------
# Dual-Speaker Podcast / 1v1 Split-Stack Layout Detection
# ---------------------------------------------------------------------------

def detect_multispeaker_framing(
    frame_detections: List[Any],
    speaker_turns: Optional[List[Dict[str, Any]]] = None,
    source_w: int = 1920,
    source_h: int = 1080,
) -> Optional[Dict[str, Any]]:
    """
    Detects whether a slice should use dual-speaker split-stack layout (top/bottom 9:8 stack).

    Requirements:
    1. Pyannote speaker turns: at least 2 distinct speakers exist with alternating dialogue,
       and neither speaker dominates (>85% of total spoken duration).
    2. Spatial YOLO clusters: bounding boxes form two distinct spatial clusters on opposite
       frame halves (Left: center X < 48% width; Right: center X > 52% width) with sufficient
       presence (detected in >= 20% of sampled frames).
    3. Calculates two stable 9:8 crop windows:
       - top_crop: Centered horizontally on Speaker A (Left cluster / first speaker)
       - bottom_crop: Centered horizontally on Speaker B (Right cluster / second speaker)
       Aspect ratio 9:8: target_w = int(source_h * (9/16) * (16/8)) = int(source_h * 9 / 8),
       forced even, clamped within [0, source_w].

    Returns layout metadata dictionary if dual speaker conditions are met, else None.
    """
    if source_w <= 0 or source_h <= 0:
        return None

    # Skip split-stack if source is already portrait or square (aspect ratio <= 1.0)
    if (source_w / float(source_h)) <= VERTICAL_AR_THRESHOLD:
        return None

    if not speaker_turns or len(speaker_turns) < 2:
        return None

    # 1. Analyze Diarization (speaker_turns)
    valid_turns = [
        s for s in speaker_turns
        if isinstance(s, dict) and s.get("speaker") and float(s.get("end", 0)) > float(s.get("start", 0))
    ]
    if len(valid_turns) < 2:
        return None

    speaker_durations: Dict[str, float] = {}
    for turn in valid_turns:
        spk = str(turn["speaker"])
        dur = float(turn["end"]) - float(turn["start"])
        speaker_durations[spk] = speaker_durations.get(spk, 0.0) + dur

    active_speakers = [spk for spk, dur in speaker_durations.items() if dur >= 0.5]
    if len(active_speakers) < 2:
        return None

    total_talk_time = sum(speaker_durations.values())
    if total_talk_time <= 0:
        return None

    # Fall back if one dominant speaker takes > 85% of total dialogue time
    top_speaker_pct = max(speaker_durations.values()) / total_talk_time
    if top_speaker_pct > 0.85:
        return None

    # Check for dialogue alternation (at least 1 speaker transition)
    transitions = 0
    last_spk = None
    for turn in valid_turns:
        spk = str(turn["speaker"])
        if last_spk is not None and spk != last_spk:
            transitions += 1
        last_spk = spk

    if transitions < 1:
        return None

    # 2. Analyze Spatial Bounding-Box Clusters
    if not frame_detections:
        return None

    all_frames_boxes: List[List[Any]] = []
    for item in frame_detections:
        if isinstance(item, (tuple, list)):
            if len(item) == 2 and isinstance(item[0], (int, float)) and isinstance(item[1], list):
                all_frames_boxes.append(item[1])
            elif len(item) == 4 and all(isinstance(v, (int, float)) for v in item):
                all_frames_boxes.append([item])
            else:
                all_frames_boxes.append(list(item))
        elif isinstance(item, dict):
            all_frames_boxes.append([item])

    total_frames = len(all_frames_boxes)
    if total_frames < 2:
        return None

    left_cluster_centers: List[float] = []
    right_cluster_centers: List[float] = []
    frames_with_left = 0
    frames_with_right = 0

    left_bound = source_w * 0.48
    right_bound = source_w * 0.52

    for boxes in all_frames_boxes:
        has_left = False
        has_right = False
        for b in boxes:
            if isinstance(b, (tuple, list)) and len(b) >= 4:
                bx, by, bw, bh = b[:4]
            elif isinstance(b, dict):
                bx = b.get("x", 0)
                by = b.get("y", 0)
                bw = b.get("w", b.get("width", 0))
                bh = b.get("h", b.get("height", 0))
            else:
                continue

            # Filter out tiny noise detections
            if bw < 20 or bh < 20:
                continue

            cx = bx + bw / 2.0
            if cx < left_bound:
                left_cluster_centers.append(cx)
                has_left = True
            elif cx > right_bound:
                right_cluster_centers.append(cx)
                has_right = True

        if has_left:
            frames_with_left += 1
        if has_right:
            frames_with_right += 1

    # Both left and right clusters must have consistent presence (>= 20% of sampled frames)
    min_presence = max(1, int(total_frames * 0.20))
    if frames_with_left < min_presence or frames_with_right < min_presence:
        return None

    if not left_cluster_centers or not right_cluster_centers:
        return None

    median_left = float(statistics.median(left_cluster_centers))
    median_right = float(statistics.median(right_cluster_centers))

    # Ensure Left and Right clusters are well separated (at least 20% of frame width apart)
    if (median_right - median_left) < (source_w * 0.20):
        return None

    # 3. Calculate 9:8 Crop Windows
    # 9:8 aspect ratio: target_w = int(source_h * (9/16) * (16/8)) = int(source_h * 9 / 8)
    target_w = int(round(source_h * 9.0 / 8.0))
    target_w = max(2, int(target_w // 2) * 2)
    target_w = min(source_w, target_w)
    target_h = max(2, int(source_h // 2) * 2)

    # Center horizontally on Left Speaker (Speaker A)
    x_a = int(round(median_left - target_w / 2.0))
    x_a = max(0, min(x_a, source_w - target_w))
    x_a = int(x_a // 2) * 2

    # Center horizontally on Right Speaker (Speaker B)
    x_b = int(round(median_right - target_w / 2.0))
    x_b = max(0, min(x_b, source_w - target_w))
    x_b = int(x_b // 2) * 2

    top_crop = {
        "x": x_a,
        "y": 0,
        "w": target_w,
        "h": target_h,
    }
    bottom_crop = {
        "x": x_b,
        "y": 0,
        "w": target_w,
        "h": target_h,
    }

    return {
        "layout": "dual_speaker_split",
        "is_dual_speaker": True,
        "top_crop": top_crop,
        "bottom_crop": bottom_crop,
        "speaker_a_center": round(median_left, 1),
        "speaker_b_center": round(median_right, 1),
        "transitions": transitions,
    }


def detect_dual_speaker_layout(
    detections_per_frame: List[Any],
    diarization_segments: Optional[List[Dict[str, Any]]] = None,
    source_w: int = 1920,
    source_h: int = 1080,
) -> Optional[Dict[str, Any]]:
    """Backward compatibility wrapper for detect_multispeaker_framing."""
    return detect_multispeaker_framing(
        frame_detections=detections_per_frame,
        speaker_turns=diarization_segments,
        source_w=source_w,
        source_h=source_h,
    )


def calculate_tracking_trajectory(
    video_path: str,
    sample_every_n_frames: Optional[int] = None,
    alpha: float = ALPHA_SMOOTH,
    use_yolo: bool = True,
    sample_step: float = SAMPLE_STEP_SEC,
    max_keyframes: int = MAX_KEYFRAMES,
    speaker_segments: Optional[List[Dict[str, Any]]] = None,
    slice_start_time: float = 0.0,
    scene_cuts: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Opens a video, samples frames at 1.0-second intervals (sample_step=1.0) and at scene cuts,
    runs subject detection using YOLOv11 (GPU with CPU fallback),
    applies exponential smoothing with scene-cut boundary resets, generates zoom keyframes,
    and strictly clamps the keyframes array to a maximum of 30 points per slice.
    """
    if scene_cuts is None:
        scene_cuts = detect_scene_cuts(video_path)
    clean_cuts = sorted([float(c) for c in (scene_cuts or []) if c > 0])

    # Safe empty trajectory for fallback
    def _center_fallback(source_w: int = 1920, source_h: int = 1080) -> Dict[str, Any]:
        is_vert = (source_w / source_h) <= VERTICAL_AR_THRESHOLD if source_h > 0 else False
        target_crop_w = min(source_w, TARGET_CROP_W)
        best_x = max(0, (source_w - target_crop_w) // 2)
        res = {
            "fps": 30.0,
            "frame_count": 0,
            "source_w": source_w,
            "source_h": source_h,
            "is_vertical": is_vert,
            "x_offsets": [best_x],
            "sample_timestamps": [0.0],
            "keyframes": [{"time": 0.0, "x_offset": best_x}],
            "best_x_offset": best_x,
            "zoom_keyframes": [],
            "scene_cuts": clean_cuts,
        }
        if not is_vert:
            res["no_crop_scale_fit"] = True
        return res

    if not os.path.isfile(video_path):
        print(f"[engine_vision] Warning: File not found: {video_path}. Using center fallback.")
        return _center_fallback()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[engine_vision] Warning: Cannot open video: {video_path}. Using center fallback.")
        return _center_fallback()

    try:
        f_fps = float(cap.get(cv2.CAP_PROP_FPS))
        import math
        if math.isnan(f_fps) or f_fps <= 0:
            f_fps = 30.0
    except Exception:
        f_fps = 30.0
    fps = max(1.0, f_fps)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if source_w <= 0 or source_h <= 0:
        return _center_fallback()

    is_vertical = (source_w / source_h) <= VERTICAL_AR_THRESHOLD
    target_crop_w = min(source_w, TARGET_CROP_W)
    center_x = max(0, (source_w - target_crop_w) // 2)

    is_gaming_layout, webcam_region = detect_gaming_layout(video_path)
    if is_gaming_layout:
        return {
            "fps": fps,
            "frame_count": frame_count,
            "source_w": source_w,
            "source_h": source_h,
            "is_vertical": is_vertical,
            "x_offsets": [center_x],
            "sample_timestamps": [0.0],
            "keyframes": [{"time": 0.0, "x_offset": center_x}],
            "best_x_offset": center_x,
            "zoom_keyframes": [],
            "is_gaming_layout": True,
            "webcam_region": webcam_region,
            "scene_cuts": clean_cuts,
        }

    # For vertical sources, no panning needed — return static center
    if is_vertical:
        return {
            "fps": fps,
            "frame_count": frame_count,
            "source_w": source_w,
            "source_h": source_h,
            "is_vertical": True,
            "x_offsets": [0],
            "sample_timestamps": [0.0],
            "keyframes": [{"time": 0.0, "x_offset": 0}],
            "best_x_offset": 0,
            "zoom_keyframes": [],
            "scene_cuts": clean_cuts,
        }

    # --- Process frames ---
    frame_step = max(1, sample_every_n_frames) if sample_every_n_frames is not None else max(1, int(round(fps * sample_step)))

    # Frame indices for scene transitions to ensure instantaneous cut keyframing
    cut_frame_indices = set()
    if clean_cuts:
        for c in clean_cuts:
            c_idx = int(round(c * fps))
            if c_idx > 0:
                cut_frame_indices.add(c_idx - 1)  # Frame immediately preceding cut
                cut_frame_indices.add(c_idx)      # Cut frame

    raw_x_offsets: List[int] = []
    sample_timestamps: List[float] = []
    all_frame_detections: List[Tuple[float, List[Tuple[int, int, int, int]]]] = []

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    has_any_detection = False
    
    speaker_to_box_idx: Dict[str, int] = {}
    tracker = SmoothGlideTracker(
        source_w=source_w,
        target_crop_w=target_crop_w,
        dt=sample_step,
        ewma_alpha=alpha,
        deadband=16.0,
        max_velocity=180.0,
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        should_sample = (frame_idx % frame_step == 0) or (frame_idx in cut_frame_indices)
        if should_sample:
            timestamp = frame_idx / fps
            orig_t = timestamp + slice_start_time

            # Reset speaker-to-box mapping on scene cut transition frame
            is_scene_cut_frame = frame_idx in cut_frame_indices and any(abs(timestamp - c) <= 1.0 / fps for c in clean_cuts)
            if is_scene_cut_frame:
                speaker_to_box_idx.clear()
            
            active_speaker = None
            if speaker_segments:
                for seg in speaker_segments:
                    if seg["start"] <= orig_t <= seg["end"]:
                        active_speaker = seg["speaker"]
                        break

            # Subsample frame to 640px wide for faster inference
            h, w = frame.shape[:2]
            if w <= 0 or h <= 0:
                frame_idx += 1
                continue

            scale = max(1e-3, min(1.0, 640.0 / float(w)))
            if scale < 1.0:
                small = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))))
            else:
                small = frame

            # Primary: InsightFace SCRFD with eye/nose landmarks (falls back gracefully if uninstalled)
            face_detections = detect_faces_or_subjects(small, prefer_insightface=True)
            if face_detections:
                has_any_detection = True

            # Scale detection boxes and landmark centers back to source coordinate space
            if scale < 1.0 and face_detections:
                inv = 1.0 / max(scale, 1e-4)
                scaled_detections = []
                for fd in face_detections:
                    bx, by, bw, bh = fd.box
                    sb = (int(bx * inv), int(by * inv), int(bw * inv), int(bh * inv))
                    sc = (fd.landmark_center[0] * inv, fd.landmark_center[1] * inv) if fd.landmark_center else None
                    scaled_detections.append(FaceDetection(box=sb, landmarks=None, landmark_center=sc, score=fd.score))
                face_detections = scaled_detections

            all_frame_detections.append((timestamp, [fd.box for fd in face_detections]))
                
            focal_face = None
            if face_detections:
                if len(face_detections) >= 2:
                    # Sort detections left to right across the canvas
                    face_detections.sort(key=lambda d: d.box[0])
                    
                    if active_speaker:
                        if active_speaker in speaker_to_box_idx and speaker_to_box_idx[active_speaker] < len(face_detections):
                            focal_face = face_detections[speaker_to_box_idx[active_speaker]]
                        else:
                            known_speakers = list(dict.fromkeys(seg["speaker"] for seg in (speaker_segments or []) if seg.get("speaker")))
                            if active_speaker in known_speakers:
                                spk_idx = known_speakers.index(active_speaker) % len(face_detections)
                                speaker_to_box_idx[active_speaker] = spk_idx
                                focal_face = face_detections[spk_idx]

                    if focal_face is None:
                        focal_face = max(face_detections, key=lambda d: d.box[2] * d.box[3])
                else:
                    focal_face = face_detections[0]

            # Calculate desired pan offset anchored to eye/nose landmark center (or None if subject stepped out)
            if focal_face is not None:
                if focal_face.landmark_center is not None:
                    target_pan_x = focal_face.landmark_center[0] - target_crop_w / 2.0
                else:
                    target_pan_x = (focal_face.box[0] + focal_face.box[2] / 2.0) - target_crop_w / 2.0
            else:
                target_pan_x = None  # Subject stepped out / out of view

            # Instantaneous reset on hard scene cuts to prevent camera drag across shots
            if is_scene_cut_frame:
                tracker.reset(target_pan_x)

            # Smooth glide update: Kalman Filter predict + update with momentum coasting & gentle centering
            dt_step = (timestamp - sample_timestamps[-1]) if sample_timestamps else sample_step
            x_offset = tracker.update(target_pan_x, dt=dt_step)

            raw_x_offsets.append(x_offset)
            sample_timestamps.append(round(timestamp, 4))

        frame_idx += 1

    cap.release()

    if not raw_x_offsets:
        return _center_fallback(source_w, source_h)
        
    if not has_any_detection:
        return {
            "fps": fps,
            "frame_count": frame_count,
            "source_w": source_w,
            "source_h": source_h,
            "is_vertical": False,
            "is_dual_speaker": False,
            "no_crop_scale_fit": True,
            "x_offsets": [center_x],
            "sample_timestamps": [0.0],
            "keyframes": [{"time": 0.0, "x_offset": center_x}],
            "best_x_offset": center_x,
            "zoom_keyframes": [],
            "scene_cuts": clean_cuts,
        }

    # Evaluate multi-speaker split-stack layout if two active alternating participants exist
    multispeaker_meta = detect_multispeaker_framing(
        frame_detections=all_frame_detections,
        speaker_turns=speaker_segments,
        source_w=source_w,
        source_h=source_h,
    )
    if multispeaker_meta is not None:
        best_x = max(0, (source_w - target_crop_w) // 2)
        print(
            f"[engine_vision] Dual-speaker split layout activated | "
            f"top_crop={multispeaker_meta['top_crop']} | bottom_crop={multispeaker_meta['bottom_crop']} | "
            f"transitions={multispeaker_meta.get('transitions', 0)}"
        )
        return {
            "fps": fps,
            "frame_count": frame_count,
            "source_w": source_w,
            "source_h": source_h,
            "is_vertical": False,
            "is_dual_speaker": True,
            "layout": "dual_speaker_split",
            "dual_speaker_layout": multispeaker_meta,
            "x_offsets": [best_x],
            "sample_timestamps": [0.0],
            "keyframes": [{"time": 0.0, "x_offset": best_x}],
            "best_x_offset": best_x,
            "zoom_keyframes": [],
            "scene_cuts": clean_cuts,
        }

    # Apply exponential smoothing with scene transition boundary hard-resets
    smoothed = smooth_trajectory(
        raw_x_offsets,
        alpha=alpha,
        timestamps=sample_timestamps,
        scene_cuts=clean_cuts,
    )

    # Best static fallback position (median of smoothed, clamped within source bounds)
    best_x = int(round(statistics.median(smoothed)))
    best_x = max(0, min(best_x, source_w - target_crop_w))

    # Generate zoom keyframes from smoothed trajectory (suppressing pulses on cut frames)
    zoom_kfs = get_zoom_keyframes(
        x_offsets=smoothed,
        timestamps=sample_timestamps,
        fps=fps,
        scene_cuts=clean_cuts,
    )

    # Downsample trajectory keyframes to 1.0s increments, preserving scene cut boundaries
    export_timestamps, export_x_offsets = downsample_trajectory(
        timestamps=sample_timestamps,
        x_offsets=smoothed,
        sample_step=sample_step,
        max_keyframes=max_keyframes,
        scene_cuts=clean_cuts,
    )

    # Strict clamp to cap keyframes to a maximum of 30 points per slice (avoids FFmpeg ARG_MAX overflow)
    if len(export_timestamps) > max_keyframes:
        export_timestamps = export_timestamps[:max_keyframes]
        export_x_offsets = export_x_offsets[:max_keyframes]
        print(f"[WARN] Downsampled keyframes exceeded limit; clamped to {max_keyframes}")

    keyframes_array = [
        {"time": t, "x_offset": x}
        for t, x in zip(export_timestamps, export_x_offsets)
    ]

    print(
        f"[engine_vision] Trajectory computed: {len(export_x_offsets)} keyframes "
        f"(downsampled from {len(smoothed)} raw @ {sample_step}s, capped to {max_keyframes}) | "
        f"best_x={best_x} | zoom_keyframes={len(zoom_kfs)} | scene_cuts={len(clean_cuts)} | is_vertical={is_vertical}"
    )

    return {
        "fps": fps,
        "frame_count": frame_count,
        "source_w": source_w,
        "source_h": source_h,
        "is_vertical": is_vertical,
        "is_dual_speaker": False,
        "x_offsets": export_x_offsets,
        "sample_timestamps": export_timestamps,
        "keyframes": keyframes_array,
        "best_x_offset": best_x,
        "zoom_keyframes": zoom_kfs,
        "scene_cuts": clean_cuts,
    }
