"""
src/engine_vision.py
--------------------
Phase 2 dynamic subject tracking and cinematic reframing engine.

Detection pipeline:
  - Primary:  ultralytics YOLOv11 ("yolo11n.pt") on GPU (device=0) with CPU fallback
  - Fallback: center-frame static position (center_x = (w - TARGET_CROP_W) // 2)

Smoothing:
  Exponential temporal filter (alpha=0.15) for butter-smooth pans.
  smoothed_x[i] = alpha * raw_x[i] + (1 - alpha) * smoothed_x[i-1]

Zoom keyframes:
  Detects high-motion segments and injects subtle zoom pulses (1.0x → 1.12x)
  every 4–6 seconds to reset viewer attention.
"""

from __future__ import annotations

import gc
import logging
import os
import statistics
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

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
# YOLO loader (lazy, guarded, GPU with CPU fallback)
# ---------------------------------------------------------------------------
_yolo_model = None
_yolo_available = False
_yolo_device: str | int = "cpu"


def _get_device() -> str | int:
    """Returns 0 if GPU (CUDA) is available, else 'cpu'."""
    try:
        if torch.cuda.is_available():
            return 0
    except Exception:
        pass
    return "cpu"


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
        print(f"[engine_vision] Warning: Could not load YOLOv11 model: {e}")
        _yolo_model = None
        _yolo_available = False
        return False


def purge_yolo_model() -> None:
    """
    Explicitly deletes the cached YOLO model from memory and flushes PyTorch CUDA cache.
    Prevents VRAM leaks before launching FFmpeg NVENC.
    """
    global _yolo_model, _yolo_available
    if _yolo_model is not None:
        try:
            del _yolo_model
        except Exception:
            pass
        _yolo_model = None
        _yolo_available = False

    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    logger.debug("[engine_vision] Purged YOLO model from VRAM.")


evict_yolo_model = purge_yolo_model


# ---------------------------------------------------------------------------
# Core detection helpers
# ---------------------------------------------------------------------------

def _detect_yolo(frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Runs YOLOv11 on a frame.
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
        # If GPU inference fails at runtime, gracefully fallback to CPU
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


def detect_subjects(frame: np.ndarray, use_yolo: bool = True) -> List[Tuple[int, int, int, int]]:
    """
    Detects human subjects in a video frame using YOLOv11.
    Returns list of (x, y, w, h) bounding boxes. Empty list = no human detections.
    """
    if use_yolo and _try_load_yolo():
        return _detect_yolo(frame)
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
    detections: List[Tuple[int, int, int, int]],
    source_w: int,
    target_crop_w: int = TARGET_CROP_W,
    focal_box: Optional[Tuple[int, int, int, int]] = None
) -> int:
    """
    Calculates the horizontal x_offset to frame the focal subject
    inside a `target_crop_w` wide vertical crop window.

    If no detections (no humans found, e.g. POV gaming footage) → returns center offset:
        center_x = (source_w - target_crop_w) // 2
    If multiple detections → uses the focal_box if provided, else largest bounding box.

    Returns: x_offset (clamped to [0, max(0, source_w - target_crop_w)])
    """
    max_x = max(0, source_w - target_crop_w)
    center_x = max(0, (source_w - target_crop_w) // 2)

    if not detections and not focal_box:
        return center_x  # Safe center-crop fallback

    if focal_box:
        focal = focal_box
    else:
        # Pick the largest box (most prominent subject)
        focal = max(detections, key=lambda b: b[2] * b[3])
        
    fx, fy, fw, fh = focal

    # Center the crop window on the focal subject's horizontal midpoint
    subject_center_x = fx + fw // 2
    x_offset = subject_center_x - target_crop_w // 2

    return max(0, min(x_offset, max_x))


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
    yolo_checked = use_yolo
    has_any_detection = False
    
    speaker_to_box_idx: Dict[str, int] = {}
    prev_gray: Optional[np.ndarray] = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        should_sample = (frame_idx % frame_step == 0) or (frame_idx in cut_frame_indices)
        if should_sample:
            timestamp = frame_idx / fps
            orig_t = timestamp + slice_start_time

            # Reset speaker-to-box mapping on scene cut transition frame
            if frame_idx in cut_frame_indices and any(abs(timestamp - c) <= 1.0 / fps for c in clean_cuts):
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

            detections = detect_subjects(small, use_yolo=yolo_checked)
            if detections:
                has_any_detection = True

            # Scale detection boxes back to source coordinate space
            if scale < 1.0 and detections:
                inv = 1.0 / max(scale, 1e-4)
                detections = [
                    (int(x * inv), int(y * inv), int(bw * inv), int(bh * inv))
                    for x, y, bw, bh in detections
                ]

            all_frame_detections.append((timestamp, list(detections)))
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
            focal_box = None
            if detections:
                if len(detections) >= 2:
                    # Sort detections left to right across the canvas
                    detections.sort(key=lambda b: b[0])
                    
                    if active_speaker:
                        # Check if this speaker already has a bound box index
                        if active_speaker in speaker_to_box_idx and speaker_to_box_idx[active_speaker] < len(detections):
                            focal_box = detections[speaker_to_box_idx[active_speaker]]
                        else:
                            # Map by order of known speaker appearances in diarization
                            known_speakers = list(dict.fromkeys(seg["speaker"] for seg in (speaker_segments or []) if seg.get("speaker")))
                            if active_speaker in known_speakers:
                                spk_idx = known_speakers.index(active_speaker) % len(detections)
                                speaker_to_box_idx[active_speaker] = spk_idx
                                focal_box = detections[spk_idx]

                    # If no active speaker or mapping could not be established, select primary foreground subject
                    if focal_box is None:
                        focal_box = max(detections, key=lambda b: b[2] * b[3])
                else:
                    focal_box = detections[0]

            prev_gray = curr_gray

            x_offset = calculate_pan_offset(detections, source_w, target_crop_w, focal_box=focal_box)
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
