"""
src/camera_framing.py
---------------------
Human-in-the-Loop (HITL) camera framing data structures and automatic zone suggestion engine.
Stores user-defined or auto-detected camera zones (Host, Guest, Wide, Facecam, Gameplay)
for intelligent multi-speaker vertical reframing and virtual TV director cutting.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2

logger = logging.getLogger("camera_framing")

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.abspath(os.path.join(_SRC_DIR, ".."))
CAMERA_ZONES_DIR = os.path.join(_ROOT_DIR, "memory", "camera_zones")
os.makedirs(CAMERA_ZONES_DIR, exist_ok=True)


def _sanitize_filename(name: str) -> str:
    """Sanitizes filename for config storage."""
    base = os.path.basename(str(name).strip())
    safe = re.sub(r'[^a-zA-Z0-9_.-]', '_', base)
    return safe.strip('._') or "default"


@dataclass
class CameraZone:
    id: str                                  # e.g. "zone_host", "zone_guest", "zone_wide", "zone_screencast"
    label: str                               # e.g. "Host (Left)", "Guest (Right)", "Screencast"
    x: int                                   # X offset in source video
    y: int                                   # Y offset in source video
    width: int                               # Bounding box width
    height: int                              # Bounding box height
    speaker_label: Optional[str] = None     # e.g. "Host", "Guest", "SPEAKER_00"
    is_wide: bool = False
    is_facecam: bool = False
    is_gameplay: bool = False
    is_screencast: bool = False
    color: str = "#3b82f6"                   # Hex color for UI canvas rendering

    def __post_init__(self):
        # Guarantee non-negative coordinates and even dimensions for FFmpeg compliance
        self.x = max(0, int(round(self.x)))
        self.y = max(0, int(round(self.y)))
        self.width = max(2, int(round(self.width / 2.0)) * 2)
        self.height = max(2, int(round(self.height / 2.0)) * 2)

    def clamp(self, max_w: int = 1920, max_h: int = 1080):
        self.x = max(0, min(max_w - 40, self.x))
        self.y = max(0, min(max_h - 40, self.y))
        self.width = max(40, min(max_w - self.x, self.width))
        self.height = max(40, min(max_h - self.y, self.height))
        self.width = (self.width // 2) * 2
        self.height = (self.height // 2) * 2

    @property
    def aspect_ratio(self) -> float:
        return self.width / max(1.0, float(self.height))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CameraZone":
        return cls(
            id=str(data.get("id", "zone_default")),
            label=str(data.get("label", "Camera Zone")),
            x=int(data.get("x", 0)),
            y=int(data.get("y", 0)),
            width=int(data.get("width", 1080)),
            height=int(data.get("height", 1080)),
            speaker_label=data.get("speaker_label"),
            is_wide=bool(data.get("is_wide", False)),
            is_facecam=bool(data.get("is_facecam", False)),
            is_gameplay=bool(data.get("is_gameplay", False)),
            is_screencast=bool(data.get("is_screencast", False)),
            color=str(data.get("color", "#3b82f6")),
        )


@dataclass
class CameraFramingConfig:
    file_name: str
    source_width: int = 1920
    source_height: int = 1080
    mode: str = "podcast"                      # "podcast" | "gaming" | "solo"
    split_preference: str = "split_stack"      # "split_stack" (9:8) | "wide"
    zones: List[CameraZone] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_name": self.file_name,
            "source_width": self.source_width,
            "source_height": self.source_height,
            "mode": self.mode,
            "split_preference": self.split_preference,
            "zones": [z.to_dict() for z in self.zones],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CameraFramingConfig":
        raw_zones = data.get("zones", [])
        zones = [CameraZone.from_dict(z) for z in raw_zones if isinstance(z, dict)]
        return cls(
            file_name=str(data.get("file_name", "unknown")),
            source_width=int(data.get("source_width", 1920)),
            source_height=int(data.get("source_height", 1080)),
            mode=str(data.get("mode", "podcast")),
            split_preference=str(data.get("split_preference", "split_stack")),
            zones=zones,
        )

    def get_zone(self, zone_id: str) -> Optional[CameraZone]:
        for z in self.zones:
            if z.id == zone_id:
                return z
        return None

    def get_zone_by_speaker(self, speaker: Optional[str]) -> Optional[CameraZone]:
        if not speaker:
            return None
        norm = speaker.strip().lower()
        # Direct match
        for z in self.zones:
            if z.speaker_label and z.speaker_label.strip().lower() == norm:
                return z
        # Fuzzy match (host, guest, speaker_00, etc.)
        for z in self.zones:
            if norm in z.id.lower() or norm in z.label.lower():
                return z
        return None

    def get_host_zone(self) -> Optional[CameraZone]:
        for z in self.zones:
            if "host" in z.id.lower() or "host" in z.label.lower():
                return z
        return self.zones[0] if self.zones else None

    def get_guest_zone(self, index: int = 1) -> Optional[CameraZone]:
        guests = [z for z in self.zones if "guest" in z.id.lower() or "guest" in z.label.lower()]
        if guests and index <= len(guests):
            return guests[index - 1]
        if len(self.zones) > 1:
            return self.zones[1]
        return None

    def get_wide_zone(self) -> Optional[CameraZone]:
        for z in self.zones:
            if z.is_wide or "wide" in z.id.lower() or "wide" in z.label.lower():
                return z
        return None

    def get_screencast_zone(self) -> Optional[CameraZone]:
        for z in self.zones:
            if z.is_screencast or "screen" in z.id.lower() or "screen" in z.label.lower() or "slide" in z.label.lower():
                return z
        return None

    def save(self, config_dir: Optional[str] = None) -> str:
        out_dir = config_dir or CAMERA_ZONES_DIR
        os.makedirs(out_dir, exist_ok=True)
        safe_name = _sanitize_filename(self.file_name)
        out_path = os.path.join(out_dir, f"{safe_name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"[camera_framing] Saved camera zones for {self.file_name} to {out_path}")
        return out_path

    @classmethod
    def load(cls, file_name: str, config_dir: Optional[str] = None) -> Optional["CameraFramingConfig"]:
        out_dir = config_dir or CAMERA_ZONES_DIR
        safe_name = _sanitize_filename(file_name)
        path = os.path.join(out_dir, f"{safe_name}.json")
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls.from_dict(data)
        except Exception as e:
            logger.warning(f"[camera_framing] Failed to load config from {path}: {e}")
            return None


def suggest_camera_zones(video_path: str) -> CameraFramingConfig:
    """
    Intelligently analyzes source video to suggest camera framing boxes:
    1. Probes dimensions and reads keyframe at ~1.5s (or middle).
    2. Detects subjects/faces using YOLO or OpenCV.
    3. Auto-configures Host (Left), Guest (Right), Wide Two-Shot, or Gaming layout.
    """
    file_name = os.path.basename(video_path)
    sw, sh = 1920, 1080

    if not video_path or not os.path.isfile(video_path):
        return _build_default_podcast_config(file_name, sw, sh)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return _build_default_podcast_config(file_name, sw, sh)

    try:
        sw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
        sh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)

        # Multi-sample across first 10 seconds to robustly detect all speakers
        sample_timestamps = [1.5, 3.0, 5.0, 8.0, 12.0]
        frames_to_check = []
        for ts in sample_timestamps:
            f_idx = min(max(0, int(fps * ts)), max(0, total_frames - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, fr = cap.read()
            if ret and fr is not None:
                frames_to_check.append(fr)
    finally:
        cap.release()

    if not frames_to_check:
        return _build_default_podcast_config(file_name, sw, sh)

    # Detect human subjects/faces via engine_vision (InsightFace SCRFD primary)
    all_face_dets: List[Any] = []
    try:
        from engine_vision import detect_faces_or_subjects
        for fr in frames_to_check:
            dets = detect_faces_or_subjects(fr, prefer_insightface=True)
            if dets:
                all_face_dets.extend(dets)
    except Exception as e:
        logger.debug(f"[camera_framing] InsightFace/vision detection fallback: {e}")

    # Fallback to Haar Cascade if 0 faces found
    if not all_face_dets:
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if os.path.isfile(cascade_path):
                face_cascade = cv2.CascadeClassifier(cascade_path)
                for fr in frames_to_check:
                    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
                    for (x, y, w, h) in faces:
                        from engine_vision import FaceDetection
                        all_face_dets.append(FaceDetection(
                            box=(int(x), int(y), int(w), int(h)),
                            landmark_center=(float(x + w / 2.0), float(y + h * 0.4))
                        ))
        except Exception as e:
            logger.debug(f"[camera_framing] Haar detection error: {e}")

    # Filter out tiny spurious detections (less than 8% frame height)
    min_h = int(sh * 0.08)
    valid_faces = [f for f in all_face_dets if f.box[3] >= min_h]

    # Cluster detections into Left (Host) and Right (Guest)
    left_faces = []
    right_faces = []
    left_bound = sw * 0.48
    right_bound = sw * 0.52

    for f in valid_faces:
        cx = f.landmark_center[0] if getattr(f, "landmark_center", None) is not None else (f.box[0] + f.box[2] / 2.0)
        if cx < left_bound:
            left_faces.append(f)
        elif cx > right_bound:
            right_faces.append(f)

    if left_faces and right_faces:
        def _get_coords(f: Any) -> Tuple[float, float, float, float]:
            bx, by, bw, bh = f.box
            if getattr(f, "landmark_center", None) is not None:
                return f.landmark_center[0], f.landmark_center[1], bw, bh
            return bx + bw / 2.0, by + bh / 2.0, bw, bh

        import statistics
        l_coords = [_get_coords(f) for f in left_faces]
        r_coords = [_get_coords(f) for f in right_faces]

        med_lx = statistics.median([c[0] for c in l_coords])
        med_ly = statistics.median([c[1] for c in l_coords])
        med_lbw = statistics.median([c[2] for c in l_coords])
        med_lbh = statistics.median([c[3] for c in l_coords])

        med_rx = statistics.median([c[0] for c in r_coords])
        med_ry = statistics.median([c[1] for c in r_coords])
        med_rbw = statistics.median([c[2] for c in r_coords])
        med_rbh = statistics.median([c[3] for c in r_coords])

        hz = _box_to_zone(
            "zone_host", "Host (Left)",
            (int(med_lx - med_lbw / 2), int(med_ly - med_lbh / 2), int(med_lbw), int(med_lbh)),
            sw, sh, speaker="SPEAKER_00", color="#3b82f6", landmark_center=(med_lx, med_ly)
        )
        gz = _box_to_zone(
            "zone_guest", "Guest (Right)",
            (int(med_rx - med_rbw / 2), int(med_ry - med_rbh / 2), int(med_rbw), int(med_rbh)),
            sw, sh, speaker="SPEAKER_01", color="#10b981", landmark_center=(med_rx, med_ry)
        )
        wz = CameraZone(
            id="zone_wide", label="Wide 2-Shot", x=0, y=0, width=sw, height=sh, is_wide=True, color="#8b5cf6"
        )
        return CameraFramingConfig(
            file_name=file_name, source_width=sw, source_height=sh, mode="podcast", split_preference="wide", zones=[hz, gz, wz]
        )

    elif valid_faces:
        f0 = valid_faces[0]
        hz = _box_to_zone(
            "zone_host", "Solo Speaker", f0.box, sw, sh,
            speaker="SPEAKER_00", color="#3b82f6",
            landmark_center=getattr(f0, "landmark_center", None)
        )
        wz = CameraZone(
            id="zone_wide",
            label="Full Wide Shot",
            x=0,
            y=0,
            width=sw,
            height=sh,
            is_wide=True,
            color="#8b5cf6",
        )
        return CameraFramingConfig(
            file_name=file_name,
            source_width=sw,
            source_height=sh,
            mode="solo",
            split_preference="wide",
            zones=[hz, wz],
        )

    # 0 detected faces: smart default podcast layout
    return _build_default_podcast_config(file_name, sw, sh)


def _box_to_zone(
    zone_id: str,
    label: str,
    box: Tuple[int, int, int, int],
    sw: int,
    sh: int,
    speaker: Optional[str] = None,
    color: str = "#3b82f6",
    landmark_center: Optional[Tuple[float, float]] = None,
) -> CameraZone:
    """Expands a face/subject bounding box or landmark center into a tight, isolated 9:8 portrait camera framing zone."""
    bx, by, bw, bh = box
    if landmark_center is not None:
        cx, cy = landmark_center
    else:
        cx = bx + bw / 2.0
        cy = by + bh / 2.0

    # Tighter 9:8 vertical framing (~0.68 of height or bounded by face height)
    target_h = max(int(sh * 0.60), min(int(sh * 0.78), int(bh * 3.5)))
    target_w = max(2, int(round(target_h * 9.0 / 8.0 / 2.0)) * 2)
    # Ensure target_w does not exceed 45% of source width to prevent capturing neighbor
    if target_w > int(sw * 0.45):
        target_w = max(2, int(round(int(sw * 0.45) / 2.0)) * 2)
        target_h = max(2, int(round(target_w * 8.0 / 9.0 / 2.0)) * 2)
    target_w = max(2, int(target_w // 2) * 2)
    target_h = max(2, int(target_h // 2) * 2)

    # Clamp to frame, anchoring on anatomical eye/nose center
    x1 = max(0, min(sw - target_w, int(cx - target_w / 2.0)))
    y1 = max(0, min(sh - target_h, int(cy - target_h * 0.35)))
    x1 = (x1 // 2) * 2
    y1 = (y1 // 2) * 2

    return CameraZone(
        id=zone_id,
        label=label,
        x=x1,
        y=y1,
        width=min(sw - x1, target_w),
        height=min(sh - y1, target_h),
        speaker_label=speaker,
        color=color,
    )


def _build_default_podcast_config(file_name: str, sw: int, sh: int) -> CameraFramingConfig:
    """Builds standard podcast left/right split and wide zones with isolated 9:8 margins."""
    target_h = int(round(sh * 0.70 / 2.0)) * 2
    target_w = int(round(target_h * 9.0 / 8.0 / 2.0)) * 2
    if target_w > int(sw * 0.42):
        target_w = int(round(int(sw * 0.42) / 2.0)) * 2
        target_h = int(round(target_w * 8.0 / 9.0 / 2.0)) * 2
    
    host_x = int(round(sw * 0.05 / 2.0)) * 2
    guest_x = max(0, int(round((sw * 0.95 - target_w) / 2.0)) * 2)
    top_y = max(0, min(sh - target_h, int(round(sh * 0.15 / 2.0)) * 2))

    return CameraFramingConfig(
        file_name=file_name,
        source_width=sw,
        source_height=sh,
        mode="podcast",
        split_preference="wide",
        zones=[
            CameraZone(
                id="zone_host",
                label="Host (Left)",
                x=host_x,
                y=top_y,
                width=target_w,
                height=target_h,
                speaker_label="SPEAKER_00",
                color="#3b82f6",
            ),
            CameraZone(
                id="zone_guest",
                label="Guest (Right)",
                x=guest_x,
                y=top_y,
                width=target_w,
                height=target_h,
                speaker_label="SPEAKER_01",
                color="#10b981",
            ),
            CameraZone(
                id="zone_wide",
                label="Wide 2-Shot",
                x=0,
                y=0,
                width=sw,
                height=sh,
                is_wide=True,
                color="#8b5cf6",
            ),
        ]
    )
