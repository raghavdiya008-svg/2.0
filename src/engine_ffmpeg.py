"""
src/engine_ffmpeg.py
--------------------
Single-pass FFmpeg compositor with aspect-ratio preservation, duration hardening,
VFR-drift normalization (30 FPS constant grid), NVENC hardware acceleration,
and resilient subtitle / text overlay support.
"""

from __future__ import annotations

import os
import random
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import functools
import uuid
import logging

logger = logging.getLogger("engine_ffmpeg")

@functools.lru_cache(maxsize=1)
def get_cfr_args() -> list:
    """
    Returns FFmpeg arguments enforcing Constant Frame Rate (CFR) at 30fps.
    Locks variable frame rate (VFR) inputs (common in YouTube and phone cameras)
    into an exact 30fps timeline to completely prevent lip-sync drift and subtitle misalignment.
    """
    try:
        res = subprocess.run(["ffmpeg", "-fps_mode", "cfr", "-version"], capture_output=True, text=True)
        if res.returncode == 0:
            return ["-fps_mode", "cfr", "-r", "30"]
    except Exception:
        pass
    return ["-vsync", "cfr", "-r", "30"]

_NVENC_FORCE_DISABLED = False

def is_nvenc_available() -> bool:
    global _NVENC_FORCE_DISABLED
    if _NVENC_FORCE_DISABLED:
        return False
    return _probe_nvenc()

@functools.lru_cache(maxsize=1)
def _probe_nvenc() -> bool:
    try:
        res = subprocess.run(["ffmpeg", "-encoders"], capture_output=True, text=True)
        return "h264_nvenc" in res.stdout
    except Exception:
        return False

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_BASE_DIR, ".."))
EMOJIS_DIR = os.path.join(_PROJECT_ROOT, "assets", "emojis")

TARGET_WIDTH = 1080
TARGET_HEIGHT = 1920
SPEED_FACTOR = 1.0


@dataclass
class RenderOptions:
    speed: float = SPEED_FACTOR
    crf: int = 20
    cq: int = 20
    preset: str = "p4"
    tune: str = "hq"
    rc: str = "vbr"
    bitrate: str = "8M"
    maxrate: str = "12M"
    bufsize: str = "24M"
    threads: int = 4
    audio_bitrate: str = "192k"
    bg_color: str = "white"
    auto_adjust: float = 0.0
    auto_color_correct: float = 0.0
    strip_social_ui: bool = False
    speed_factor: Optional[float] = None
    headline_text: Optional[str] = None
    brand_logo_path: Optional[str] = None
    watermark_logo_path: Optional[str] = None


def _has_dialogue_events(ass_path: Optional[str]) -> bool:
    """
    Checks if an ASS file exists, is non-empty, and contains at least one dialogue event.
    Returns False if the file has no dialogue events (e.g. empty words), is 0 bytes, or cannot be read.
    """
    if not ass_path or not os.path.isfile(ass_path):
        return False
    try:
        if os.path.getsize(ass_path) == 0:
            return False
        with open(ass_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.strip().startswith("Dialogue:"):
                    return True
        return False
    except Exception:
        return False


has_dialogue_events = _has_dialogue_events


def detect_letterbox_bounds(video_path: str) -> Tuple[int, int, int, int]:
    """Detects active content bounding box by reading a frame at 2.0s."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, 2000)
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return 0, 0, 0, 0

    fh, fw = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = (gray > 15) & (gray < 240)
    y_indices, x_indices = np.where(mask)

    if len(y_indices) > 0 and len(x_indices) > 0:
        return int(x_indices.min()), int(x_indices.max()), int(y_indices.min()), int(y_indices.max())
    return 0, fw, 0, fh


def escape_ffmpeg_drawtext(text: str) -> str:
    """Escapes single quotes, backslashes, colons, brackets, semicolons, and percent signs for FFmpeg's drawtext filter."""
    text = text.replace("\\", "\\\\")
    text = text.replace("'", "'\\''")
    text = text.replace(":", "\\:")
    text = text.replace("[", "\\[")
    text = text.replace("]", "\\]")
    text = text.replace(";", "\\;")
    text = text.replace("%", "\\%")
    return text


@functools.lru_cache(maxsize=1)
def get_system_font_path() -> Optional[str]:
    """Finds an available bold sans-serif TrueType font for drawtext across Linux/Kaggle, Windows, and macOS."""
    candidates = [
        # Project-level custom fonts
        os.path.join(_PROJECT_ROOT, "fonts", "Anton.ttf"),
        os.path.join(_PROJECT_ROOT, "fonts", "TheBoldFont.ttf"),
        # Linux / Kaggle / Colab / Debian / Ubuntu
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        # Windows
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        # macOS
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial Bold.ttf",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def wrap_headline_text(text: str, max_chars_per_line: int = 32) -> str:
    """
    Wraps headline text into clean, centered multi-line text (max 2 lines)
    so titles format elegantly in minimalist style without awkward 2-word breaks.
    """
    words = str(text or "").strip().split()
    if not words:
        return ""
    lines = []
    cur_line = []
    cur_len = 0
    for w in words:
        if cur_len + len(w) + (1 if cur_line else 0) <= max_chars_per_line:
            cur_line.append(w)
            cur_len += len(w) + (1 if len(cur_line) > 1 else 0)
        else:
            if cur_line:
                lines.append(" ".join(cur_line))
            cur_line = [w]
            cur_len = len(w)
    if cur_line:
        lines.append(" ".join(cur_line))
    return "\n".join(lines[:2])


# ---------------------------------------------------------------------------
# Phase 2 - Dynamic trajectory filter expression builders
# ---------------------------------------------------------------------------

def _build_x_crop_expr(
    timestamps: List[float],
    x_offsets: List[int],
    max_x: int,
    max_keyframes: int = 50,
    scene_cuts: Optional[List[float]] = None,
) -> str:
    """
    Builds a discrete, step-locked FFmpeg crop-x expression from trajectory keyframes.

    Generates: clip(x0 + dx1*gte(t,t1) + dx2*gte(t,t2) + ..., 0, max_x)

    Uses gte(t, {t_i}) evaluations so that on scene cuts and speaker focus changes,
    the crop jumps instantaneously rather than sliding linearly across the canvas.
    Guards against empty or single-point arrays with safe numeric fallbacks.
    Clamps the final expression safely within [0, max_x].
    """
    if not timestamps or not x_offsets:
        return "0"

    min_len = min(len(timestamps), len(x_offsets))
    timestamps = list(timestamps[:min_len])
    x_offsets = list(x_offsets[:min_len])

    if min_len == 0:
        return "0"

    clamped = [max(0, min(int(round(x)), max_x)) for x in x_offsets]
    if len(clamped) == 1 or len(timestamps) == 1:
        return str(clamped[0])

    # Safety clamp: if trajectory contains more than max_keyframes, subsample uniformly
    if len(timestamps) > max_keyframes:
        n = len(timestamps)
        step = (n - 1) / (max_keyframes - 1)
        indices = [int(round(i * step)) for i in range(max_keyframes)]
        timestamps = [timestamps[i] for i in indices]
        clamped = [clamped[i] for i in indices]

    expr_parts = [str(clamped[0])]
    for i in range(len(timestamps) - 1):
        t1 = timestamps[i + 1]
        dx = clamped[i + 1] - clamped[i]
        if dx == 0:
            continue
        sign = "+" if dx > 0 else ""
        expr_parts.append(f"{sign}{dx}*gte(t,{t1:.4f})")

    if len(expr_parts) == 1:
        return str(clamped[0])

    inner_expr = "".join(expr_parts)
    return f"clip({inner_expr},0,{max_x})"


def _build_zoom_expr(zoom_keyframes: List[dict]) -> str:
    """
    Builds an FFmpeg zoompan z-expression from zoom keyframes.

    Each keyframe contributes a sine-ramp pulse:
        delta_z * sin(3.14159265 * clip((time - t0) / duration, 0, 1))

    Produces smooth ease-in/ease-out zooms that create natural attention resets.
    """
    if not zoom_keyframes:
        return "1.0"

    parts = []
    for kf in zoom_keyframes:
        t0 = float(kf.get("time", 0.0))
        z = float(kf.get("zoom", 1.0))
        dur = float(kf.get("duration", 1.5))
        if dur <= 0:
            continue
        safe_dur = max(dur, 1e-4)
        delta_z = z - 1.0
        if abs(delta_z) < 0.001:
            continue
        parts.append(
            f"{delta_z:.4f}*sin(3.14159265*clip((time-{t0:.4f})/{safe_dur:.4f},0,1))"
        )

    if not parts:
        return "1.0"
    return "1.0+" + "+".join(parts)


def _resolve_font_path(font_path: Optional[str] = None) -> str:
    """Finds a valid font file on the system or falls back to system paths."""
    import glob as _glob

    if font_path and os.path.isfile(font_path):
        return font_path

    candidates = [
        os.path.join(_PROJECT_ROOT, "assets", "fonts", "LiberationSans-Bold.ttf"),
        r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibrib.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c

    for search_root in ["/usr/share/fonts", "/usr/local/share/fonts"]:
        found = _glob.glob(os.path.join(search_root, "**", "*.ttf"), recursive=True)
        if found:
            return found[0]

    return ""


# ---------------------------------------------------------------------------
# Slicing pass helper
# ---------------------------------------------------------------------------

def build_slice_command(
    input_path: str,
    output_path: str,
    start: float,
    duration: float,
    preset: str = "p4",
) -> List[str]:
    """
    Builds the FFmpeg command for the initial slicing pass.
    Uses fast seek (-ss before -i) + single-pass NVENC hardware cutting (-cq 18 -preset p4)
    with libx264 fast fallback to completely avoid keyframe/black-frame corruption and desync.
    """
    if is_nvenc_available():
        return [
            "ffmpeg", "-y",
            "-ss", str(start), "-t", str(duration),
            "-i", input_path,
            "-vf", "scale='max(2,trunc(iw/2)*2)':'max(2,trunc(ih/2)*2)'",
            "-c:v", "h264_nvenc", "-preset", preset,
            "-cq", "18", "-b:v", "10M", "-maxrate", "14M", "-bufsize", "20M",
            *get_cfr_args(),
            "-c:a", "aac",
            "-b:a", "192k",
            "-ar", "48000",
            "-af", "aresample=async=1",
            output_path,
        ]
    else:
        return [
            "ffmpeg", "-y",
            "-ss", str(start), "-t", str(duration),
            "-i", input_path,
            "-vf", "scale='max(2,trunc(iw/2)*2)':'max(2,trunc(ih/2)*2)'",
            "-c:v", "libx264", "-preset", "fast",
            "-crf", "18",
            *get_cfr_args(),
            "-c:a", "aac",
            "-b:a", "192k",
            "-ar", "48000",
            "-af", "aresample=async=1",
            output_path,
        ]


def slice_clip(
    input_path: str,
    output_path: str,
    start: float,
    duration: float,
    preset: str = "p4",
    check: bool = True,
    timeout: Optional[float] = 600.0,
) -> subprocess.CompletedProcess:
    """
    Executes the initial slicing pass with NVENC -preset p4 (or libx264 fast)
    to bypass keyframe corruption traps and timestamp desync.
    Includes explicit subprocess timeout to prevent worker deadlock.
    """
    cmd = build_slice_command(input_path, output_path, start, duration, preset=preset)
    try:
        return subprocess.run(cmd, check=check, capture_output=True, text=True, timeout=timeout)
    except subprocess.CalledProcessError as e:
        err_msg = (e.stderr or "") + (e.stdout or "")
        is_nvenc_err = any(
            k in err_msg.lower()
            for k in ["h264_nvenc", "nvenc", "nvcuda", "cuda", "cannot load nvcuda", "device creation failed"]
        )
        if is_nvenc_err and "h264_nvenc" in cmd:
            fallback_cmd = [
                "ffmpeg", "-y",
                "-ss", str(start), "-t", str(duration),
                "-i", input_path,
                "-vf", "scale='max(2,trunc(iw/2)*2)':'max(2,trunc(ih/2)*2)'",
                "-c:v", "libx264", "-preset", "ultrafast",
                "-crf", "18",
                *get_cfr_args(),
                "-c:a", "aac",
                "-af", "aresample=async=1",
                output_path,
            ]
            return subprocess.run(fallback_cmd, check=check, capture_output=True, text=True, timeout=timeout)
        raise
    except subprocess.TimeoutExpired as e:
        print(f"[engine_ffmpeg] Error: Slice pass timed out after {timeout}s: {' '.join(cmd)}")
        raise


# ---------------------------------------------------------------------------
# Master filter_complex command builder
# ---------------------------------------------------------------------------

def build_ffmpeg_command(
    input_path: str,
    output_path: str,
    video_coords: dict,
    text_coords: dict,
    logo_path: Optional[str] = None,
    logo_coords: Optional[dict] = None,
    emojis: List[dict] = [],
    options: Optional[RenderOptions] = None,
    ass_path: Optional[str] = None,
    trajectory: Optional[dict] = None,
    headline_text: Optional[str] = None,
    brand_logo_path: Optional[str] = None,
    watermark_logo_path: Optional[str] = None,
    speed_factor: Optional[float] = None,
) -> Tuple[List[str], Optional[str]]:
    """
    Constructs the single-pass filter_complex FFmpeg command as a list of args
    with aspect-ratio preserving scaling, duration hardening, and subtitle rendering.

    Standardizes input video to 30 FPS ([0:v]fps=30,setpts=PTS-STARTPTS...)
    to lock Variable Frame Rate (VFR) clips to a constant timestamp grid.
    Encodes via hardware-accelerated NVENC (h264_nvenc) with optimized preset p4 and -cq 19.
    """
    opts = options or RenderOptions()
    pad_color = opts.bg_color.lower() if opts.bg_color and opts.bg_color.lower() in ["white", "black"] else "white"

    # Resolve speed factor
    speed_val = speed_factor if speed_factor is not None else (opts.speed_factor if opts.speed_factor is not None else opts.speed)
    speed = max(0.5, min(3.0, float(speed_val)))

    # Resolve branding & headline parameters
    brand_path = brand_logo_path if brand_logo_path is not None else opts.brand_logo_path
    watermark_path = watermark_logo_path if watermark_logo_path is not None else opts.watermark_logo_path
    headline = headline_text if headline_text is not None else opts.headline_text

    cmd = ["ffmpeg", "-y", "-i", input_path]
    next_input_idx = 1

    # Optional Legacy Logo Input
    logo_input_idx = None
    if logo_path and str(logo_path).strip() and logo_coords:
        cmd.extend(["-loop", "1", "-i", str(logo_path).strip()])
        logo_input_idx = next_input_idx
        next_input_idx += 1

    # Optional Brand Logo Input
    brand_input_idx = None
    if brand_path and str(brand_path).strip():
        cmd.extend(["-loop", "1", "-i", str(brand_path).strip()])
        brand_input_idx = next_input_idx
        next_input_idx += 1

    # Optional Watermark Logo Input
    watermark_input_idx = None
    if watermark_path and str(watermark_path).strip():
        cmd.extend(["-loop", "1", "-i", str(watermark_path).strip()])
        watermark_input_idx = next_input_idx
        next_input_idx += 1

    # Emoji Inputs completely removed as per Phase 3 configuration

    # Target Box on Canvas
    target_box_w = int(round(video_coords.get("width", 1080)))
    target_box_h = int(round(video_coords.get("height", 1540)))
    video_x = int(round(video_coords.get("x", 0)))
    video_y = int(round(video_coords.get("y", (TARGET_HEIGHT - target_box_h) // 2)))

    # Force even integer dimensions with safe floor for FFmpeg encoder
    target_box_w = max(2, int(round(target_box_w / 2.0)) * 2)
    target_box_h = max(2, int(round(target_box_h / 2.0)) * 2)

    # Phase 5: Subtle color grading for anti-detection pixel hash alteration
    eq_filter = "eq=contrast=1.04:brightness=0.01:saturation=1.08:gamma=1.02"

    filter_complex_parts = []

    layout = trajectory.get("layout", "") if trajectory else ""
    is_gaming_layout = trajectory.get("is_gaming_layout", False) if trajectory else False
    shots = trajectory.get("shot_timeline", []) if trajectory else []
    has_shot_timeline = bool(shots) and len(shots) > 0
    is_dual_speaker = (
        layout == "dual_speaker_split"
        or bool(trajectory.get("is_dual_speaker", False) if trajectory else False)
        or any(s.get("type") == "split_stack" for s in shots)
    )

    split_src = "[0:v]"

    if has_shot_timeline:
        # Multi-Stream Virtual TV Broadcast Director Timeline
        print(f"[engine_ffmpeg] Building Virtual TV Director Shot Timeline ({len(shots)} cuts)")
        split_spec = "".join(f"[raw_s{i}]" for i in range(len(shots)))
        filter_complex_parts.append(f"{split_src}split={len(shots)}{split_spec}")

        for i, shot in enumerate(shots):
            s_start = max(0.0, float(shot.get("start", 0.0)))
            s_end = float(shot.get("end", 0.0))
            shot_type = shot.get("type", "single")
            trim_filter = f"trim=start={s_start:.2f}:end={s_end:.2f},setpts=PTS-STARTPTS"

            if shot_type == "split_stack":
                top_z = shot.get("top_zone") or {}
                bot_z = shot.get("bot_zone") or {}
                tx = max(0, int(round(top_z.get("x", 0))))
                ty = max(0, int(round(top_z.get("y", 0))))
                tw = max(2, int(round(top_z.get("width", 1080) / 2.0)) * 2)
                th = max(2, int(round(top_z.get("height", 1080) / 2.0)) * 2)

                bx = max(0, int(round(bot_z.get("x", 0))))
                by = max(0, int(round(bot_z.get("y", 0))))
                bw = max(2, int(round(bot_z.get("width", 1080) / 2.0)) * 2)
                bh = max(2, int(round(bot_z.get("height", 1080) / 2.0)) * 2)

                filter_complex_parts.append(f"[raw_s{i}]{trim_filter},split=2[s{i}_top_raw][s{i}_bot_raw]")
                filter_complex_parts.append(
                    f"[s{i}_top_raw]crop=w='min(iw,{tw})':h='min(ih,{th})':x='max(0,min({tx},iw-min(iw,{tw})))':y='max(0,min({ty},ih-min(ih,{th})))',"
                    f"scale=1080:960:force_original_aspect_ratio=increase,crop=1080:960:(in_w-1080)/2:(in_h-960)/2[s{i}_top]"
                )
                filter_complex_parts.append(
                    f"[s{i}_bot_raw]crop=w='min(iw,{bw})':h='min(ih,{bh})':x='max(0,min({bx},iw-min(iw,{bw})))':y='max(0,min({by},ih-min(ih,{bh})))',"
                    f"scale=1080:960:force_original_aspect_ratio=increase,crop=1080:960:(in_w-1080)/2:(in_h-960)/2[s{i}_bot]"
                )
                filter_complex_parts.append(
                    f"[s{i}_top][s{i}_bot]vstack=inputs=2,drawbox=x=0:y=958:w=1080:h=4:color=black@0.6:t=fill,setsar=1,format=yuv420p[v_shot_{i}]"
                )
            elif shot_type == "gaming":
                cam_z = shot.get("facecam_zone") or shot.get("top_zone") or {}
                cx = max(0, int(round(cam_z.get("x", 0))))
                cy = max(0, int(round(cam_z.get("y", 0))))
                cw = max(2, int(round(cam_z.get("width", 640) / 2.0)) * 2)
                ch = max(2, int(round(cam_z.get("height", 360) / 2.0)) * 2)

                filter_complex_parts.append(f"[raw_s{i}]{trim_filter},split=2[s{i}_cam_raw][s{i}_game_raw]")
                filter_complex_parts.append(
                    f"[s{i}_cam_raw]crop=w='min(iw,{cw})':h='min(ih,{ch})':x='max(0,min({cx},iw-min(iw,{cw})))':y='max(0,min({cy},ih-min(ih,{ch})))',"
                    f"scale=1080:672:force_original_aspect_ratio=increase,crop=1080:672:(in_w-1080)/2:(in_h-672)/2[s{i}_cam]"
                )
                filter_complex_parts.append(
                    f"[s{i}_game_raw]scale=1080:1248:force_original_aspect_ratio=increase,crop=1080:1248:(in_w-1080)/2:(in_h-1248)/2[s{i}_game]"
                )
                filter_complex_parts.append(
                    f"[s{i}_cam][s{i}_game]vstack=inputs=2,setsar=1,format=yuv420p[v_shot_{i}]"
                )
            elif shot_type == "blur_box":
                # Multi-Face / Banter Wide shot: Centered 16:9 (1080x608) at y=656 over Gaussian blur background
                filter_complex_parts.append(f"[raw_s{i}]{trim_filter},split=2[s{i}_bg_raw][s{i}_fg_raw]")
                filter_complex_parts.append(
                    f"[s{i}_bg_raw]scale=1080:1920:force_original_aspect_ratio=increase,scale='max(2,trunc(iw/2)*2)':'max(2,trunc(ih/2)*2)',"
                    f"crop=1080:1920:(in_w-1080)/2:(in_h-1920)/2,boxblur=25:5[s{i}_bg]"
                )
                filter_complex_parts.append(
                    f"[s{i}_fg_raw]scale=1080:608:force_original_aspect_ratio=increase,crop=1080:608:(in_w-1080)/2:(in_h-608)/2[s{i}_fg]"
                )
                filter_complex_parts.append(
                    f"[s{i}_bg][s{i}_fg]overlay=0:656:shortest=1,setsar=1,format=yuv420p[v_shot_{i}]"
                )
            else:
                zone = shot.get("zone") or {}
                zx = max(0, int(round(zone.get("x", 0))))
                zy = max(0, int(round(zone.get("y", 0))))
                zw = max(2, int(round(zone.get("width", 1080) / 2.0)) * 2)
                zh = max(2, int(round(zone.get("height", 1080) / 2.0)) * 2)
                filter_complex_parts.append(
                    f"[raw_s{i}]{trim_filter},"
                    f"crop=w='min(iw,{zw})':h='min(ih,{zh})':x='max(0,min({zx},iw-min(iw,{zw})))':y='max(0,min({zy},ih-min(ih,{zh})))',"
                    f"scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920:(in_w-1080)/2:(in_h-1920)/2,setsar=1,format=yuv420p[v_shot_{i}]"
                )

        concat_inputs = "".join(f"[v_shot_{i}]" for i in range(len(shots)))
        filter_complex_parts.append(f"{concat_inputs}concat=n={len(shots)}:v=1:a=0[comp_concat]")
        filter_complex_parts.append(
            f"[comp_concat]fps=30,setpts=PTS-STARTPTS,{eq_filter},format=yuv420p[comp0]"
        )

    elif is_gaming_layout:
        # Phase 2 Gaming Split-Screen Stack
        print("[engine_ffmpeg] Building Split-Screen Gaming Layout")
        webcam_rect = trajectory.get("webcam_region", (0, 0, 1920, 1080))
        wx, wy, ww, wh = webcam_rect
        wx, wy = max(0, wx), max(0, wy)
        ww, wh = max(2, int(ww / 2) * 2), max(2, int(wh / 2) * 2)

        filter_complex_parts.append(f"{split_src}split=2[webcam_raw][game_raw]")
        
        webcam_chain = (
            f"[webcam_raw]fps=30,setpts=PTS-STARTPTS,"
            f"{eq_filter},"
            f"crop={ww}:{wh}:{wx}:{wy},"
            f"scale=1080:672:force_original_aspect_ratio=increase,"
            f"crop=1080:672:(in_w-1080)/2:(in_h-672)/2[webcam]"
        )
        
        game_chain = (
            f"[game_raw]fps=30,setpts=PTS-STARTPTS,"
            f"{eq_filter},"
            f"scale=1080:1248:force_original_aspect_ratio=increase,"
            f"crop=1080:1248:(in_w-1080)/2:(in_h-1248)/2[game]"
        )
        
        filter_complex_parts.append(webcam_chain)
        filter_complex_parts.append(game_chain)
        filter_complex_parts.append("[webcam][game]vstack=inputs=2,format=yuv420p[comp0]")

    elif is_dual_speaker:
        # Dual-Speaker Podcast / 1v1 Vertical Split Stack (1080x1920 stacked 9:8)
        print("[engine_ffmpeg] Building Dual-Speaker Split Layout (1080x1920 stacked 9:8)")
        dual_meta = trajectory.get("dual_speaker_layout", {})
        top_crop = dual_meta.get("top_crop", {"x": 0, "y": 0, "w": 1215, "h": 1080})
        bot_crop = dual_meta.get("bottom_crop", {"x": 705, "y": 0, "w": 1215, "h": 1080})

        tx = max(0, int(round(top_crop.get("x", 0))))
        ty = max(0, int(round(top_crop.get("y", 0))))
        tw = max(2, int(round(top_crop.get("w", 1215))))
        th = max(2, int(round(top_crop.get("h", 1080))))
        tw = int(tw // 2) * 2
        th = int(th // 2) * 2

        bx = max(0, int(round(bot_crop.get("x", 0))))
        by = max(0, int(round(bot_crop.get("y", 0))))
        bw = max(2, int(round(bot_crop.get("w", 1215))))
        bh = max(2, int(round(bot_crop.get("h", 1080))))
        bw = int(bw // 2) * 2
        bh = int(bh // 2) * 2

        filter_complex_parts.append(f"{split_src}split=2[top_raw][bot_raw]")

        top_chain = (
            f"[top_raw]fps=30,setpts=PTS-STARTPTS,"
            f"{eq_filter},"
            f"crop=w='min(iw,{tw})':h='min(ih,{th})':x='max(0,min({tx},iw-min(iw,{tw})))':y='max(0,min({ty},ih-min(ih,{th})))',"
            f"scale=1080:960:force_original_aspect_ratio=increase,"
            f"crop=1080:960:(in_w-1080)/2:(in_h-960)/2[top]"
        )

        bot_chain = (
            f"[bot_raw]fps=30,setpts=PTS-STARTPTS,"
            f"{eq_filter},"
            f"crop=w='min(iw,{bw})':h='min(ih,{bh})':x='max(0,min({bx},iw-min(iw,{bw})))':y='max(0,min({by},ih-min(ih,{bh})))',"
            f"scale=1080:960:force_original_aspect_ratio=increase,"
            f"crop=1080:960:(in_w-1080)/2:(in_h-960)/2[bottom]"
        )

        filter_complex_parts.append(top_chain)
        filter_complex_parts.append(bot_chain)

        # Combine both streams using vstack=inputs=2[stacked]
        # Draw a clean 4px subtle divider line at the horizontal boundary (y=958:h=4:color=black@0.6)
        filter_complex_parts.append(
            "[top][bottom]vstack=inputs=2[stacked]"
        )
        filter_complex_parts.append(
            "[stacked]drawbox=x=0:y=958:w=1080:h=4:color=black@0.6:t=fill,"
            "format=yuv420p[comp0]"
        )

    elif layout == "camera_zone":
        # Single creator-defined camera zone: Punch in to fill 1080x1920 vertical shorts
        zone = trajectory.get("zone", {}) if trajectory else {}
        zx = max(0, int(round(zone.get("x", 0))))
        zy = max(0, int(round(zone.get("y", 0))))
        zw = max(2, int(round(zone.get("width", 1080) / 2.0)) * 2)
        zh = max(2, int(round(zone.get("height", 1080) / 2.0)) * 2)

        filter_complex_parts.append(
            f"{split_src}fps=30,setpts=PTS-STARTPTS,"
            f"{eq_filter},"
            f"crop=w='min(iw,{zw})':h='min(ih,{zh})':x='max(0,min({zx},iw-min(iw,{zw})))':y='max(0,min({zy},ih-min(ih,{zh})))',"
            f"scale=1080:1920:force_original_aspect_ratio=increase,"
            f"crop=1080:1920:(in_w-1080)/2:(in_h-1920)/2,"
            f"format=yuv420p[comp0]"
        )

    else:
        # 1. Base blurred background canvas locked strictly to input video duration (prevents infinite stream bug)
        filter_complex_parts.append(f"{split_src}split=2[bg_raw][vid_raw]")
        filter_complex_parts.append(
            f"[bg_raw]fps=30,setpts=PTS-STARTPTS,"
            f"scale=1080:1920:force_original_aspect_ratio=increase,scale='max(2,trunc(iw/2)*2)':'max(2,trunc(ih/2)*2)',"
            f"crop=1080:1920:(in_w-1080)/2:(in_h-1920)/2,boxblur=25:5[bg]"
        )

        use_trajectory = (
            trajectory is not None
            and not trajectory.get("is_vertical", False)
            and not trajectory.get("no_crop_scale_fit", False)
            and len(trajectory.get("x_offsets", [])) > 0
        )

        if use_trajectory:
            traj_timestamps = trajectory["sample_timestamps"]
            traj_offsets = list(trajectory["x_offsets"])
            source_w = max(1, int(trajectory.get("source_w", 1920) or 1920))
            source_h = max(1, int(trajectory.get("source_h", 1080) or 1080))
            scaled_w = trajectory.get("scaled_w")
            if scaled_w is None:
                scaled_w = max(target_box_w, int(round(source_w * (target_box_h / max(1.0, float(source_h))) / 2.0)) * 2)
            max_x = max(0, scaled_w - target_box_w)
            if "scaled_w" not in trajectory and scaled_w > source_w and source_w > 0:
                s_factor = scaled_w / float(source_w)
                unscaled_target_w = min(source_w, target_box_w)
                traj_offsets = [
                    int(round((x + unscaled_target_w / 2.0) * s_factor - target_box_w / 2.0))
                    for x in traj_offsets
                ]
            traj_cuts = trajectory.get("scene_cuts", [])
            x_crop_expr = _build_x_crop_expr(
                traj_timestamps, traj_offsets, max_x, scene_cuts=traj_cuts
            )
            traj_fps = trajectory.get("fps", 30.0)
            zoom_kfs = trajectory.get("zoom_keyframes", [])

            vid_chain = (
                f"[vid_raw]fps=30,setpts=PTS-STARTPTS,"
                f"{eq_filter},"
                f"scale=-2:{target_box_h},"
                f"crop={target_box_w}:{target_box_h}:{x_crop_expr}:0"
            )

            if zoom_kfs:
                zoom_expr = _build_zoom_expr(zoom_kfs)
                vid_chain += (
                    f",zoompan=z='{zoom_expr}'"
                    f":x='iw/2-(iw/zoom/2)'"
                    f":y='ih/2-(ih/zoom/2)'"
                    f":d=1:s={target_box_w}x{target_box_h}:fps={traj_fps:.3f}"
                )

            vid_chain += "[vid]"
            filter_complex_parts.append(vid_chain)
            print(
                f"[engine_ffmpeg] Phase 2 dynamic tracking active | "
                f"samples={len(traj_offsets)} | zoom_pulses={len(zoom_kfs)}"
            )

            filter_complex_parts.append(f"[bg][vid]overlay={video_x}:{video_y}:shortest=1[comp0]")
        elif layout == "blur_box" or (trajectory and trajectory.get("no_crop_scale_fit")):
            # Multi-Face or wide full-canvas centered 16:9 frame over Gaussian blur background
            filter_complex_parts.append(
                f"[vid_raw]fps=30,setpts=PTS-STARTPTS,"
                f"{eq_filter},"
                f"scale=1080:608:force_original_aspect_ratio=increase,"
                f"crop=1080:608:(in_w-1080)/2:(in_h-608)/2,format=yuv420p[vid_box]"
            )
            filter_complex_parts.append(f"[bg][vid_box]overlay=0:656:shortest=1[comp0]")
        else:
            filter_complex_parts.append(
                f"[vid_raw]fps=30,setpts=PTS-STARTPTS,"
                f"{eq_filter},"
                f"scale=1080:1920:force_original_aspect_ratio=increase,"
                f"crop=1080:1920:(in_w-1080)/2:(in_h-1920)/2,format=yuv420p[vid_fill]"
            )
            filter_complex_parts.append(f"[bg][vid_fill]overlay=0:0:shortest=1[comp0]")

    current_label = "[comp0]"

    # 4. Optional Legacy Logo Composite
    if logo_input_idx is not None and logo_coords:
        lx = int(round(logo_coords.get("x", 0)))
        ly = int(round(logo_coords.get("y", 0)))
        lw = max(2, int(round(logo_coords.get("width", 100) * logo_coords.get("scale_x", 1.0) / 2.0)) * 2)
        lh = max(2, int(round(logo_coords.get("height", 100) * logo_coords.get("scale_y", 1.0) / 2.0)) * 2)

        filter_complex_parts.append(f"[{logo_input_idx}:v]scale={lw}:{lh}[scaled_logo]")
        filter_complex_parts.append(
            f"{current_label}[scaled_logo]overlay={lx}:{ly}[comp_logo]"
        )
        current_label = "[comp_logo]"

    # 4b. Optional Brand Logo Composite (Top-right, 40px padding)
    if brand_input_idx is not None:
        filter_complex_parts.append(f"[{brand_input_idx}:v]scale='min(200,iw)':-1,format=rgba[brand_logo]")
        filter_complex_parts.append(
            f"{current_label}[brand_logo]overlay=W-w-40:40[comp_brand]"
        )
        current_label = "[comp_brand]"

    # 4c. Optional Watermark Logo Composite (Bottom-right, 30px padding, 50% opacity)
    if watermark_input_idx is not None:
        filter_complex_parts.append(
            f"[{watermark_input_idx}:v]scale='min(250,iw)':-1,format=rgba,colorchannelmixer=aa=0.5[wm_semi]"
        )
        filter_complex_parts.append(
            f"{current_label}[wm_semi]overlay=W-w-30:H-h-30[comp_wm]"
        )
        current_label = "[comp_wm]"

    # 5. Optional Hook Headline Banner (Centered multi-line pill with auto-wrapping)
    temp_text_file = None
    if headline and str(headline).strip():
        wrapped_title = wrap_headline_text(str(headline).strip(), max_chars_per_line=32)
        line_count = len(wrapped_title.split("\n"))
        f_size = 42 if line_count > 1 else 46

        temp_dir = os.path.join(_PROJECT_ROOT, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        temp_text_file = os.path.join(temp_dir, f"headline_{uuid.uuid4().hex[:8]}.txt")
        try:
            with open(temp_text_file, "w", encoding="utf-8") as f:
                f.write(wrapped_title)
        except Exception as e:
            logger.warning(f"[engine_ffmpeg] Failed writing headline textfile: {e}")
            temp_text_file = None

        font_path = get_system_font_path()
        font_opt = ""
        if font_path:
            font_esc = os.path.abspath(font_path).replace("\\", "/").replace(":", "\\:").replace("'", "'\\\\\\''")
            font_opt = f"fontfile='{font_esc}':"

        if temp_text_file and os.path.isfile(temp_text_file):
            txt_esc = os.path.abspath(temp_text_file).replace("\\", "/").replace(":", "\\:").replace("'", "'\\\\\\''")
            filter_complex_parts.append(
                f"{current_label}drawtext={font_opt}textfile='{txt_esc}':"
                f"x=(w-text_w)/2:y=90:fontsize={f_size}:fontcolor=white:line_spacing=10:"
                f"box=1:boxcolor=black@0.65:boxborderw=14:enable='between(t,0,4.5)'[comp_title]"
            )
        else:
            escaped_title = escape_ffmpeg_drawtext(wrapped_title)
            filter_complex_parts.append(
                f"{current_label}drawtext={font_opt}text='{escaped_title}':"
                f"x=(w-text_w)/2:y=90:fontsize={f_size}:fontcolor=white:line_spacing=10:"
                f"box=1:boxcolor=black@0.65:boxborderw=14:enable='between(t,0,4.5)'[comp_title]"
            )
        current_label = "[comp_title]"

    # Emojis Composite completely removed as per Phase 3 configuration

    # 6. Burn ASS Subtitles (Purged debug banners: only render styled .ass karaoke subtitles)
    has_valid_subtitles = (
        ass_path is not None
        and bool(str(ass_path).strip())
        and os.path.isfile(ass_path)
        and os.path.getsize(ass_path) > 0
        and _has_dialogue_events(ass_path)
    )
    if has_valid_subtitles:
        # Full absolute path properly escaped so subprocess working directory changes never break subtitle loading
        abs_ass = os.path.abspath(ass_path).replace("\\", "/")
        ass_path_escaped = abs_ass.replace(":", "\\:").replace("'", "'\\\\\\''")
        fonts_dir = os.path.join(_PROJECT_ROOT, "fonts")
        fonts_dir_esc = os.path.abspath(fonts_dir).replace("\\", "/").replace(":", "\\:").replace("'", "'\\\\\\''")
        fonts_arg = f":fontsdir='{fonts_dir_esc}'" if os.path.isdir(fonts_dir) else ""

        # MarginV = 440 safe-zone lower third is standard across all layouts
        filter_complex_parts.append(
            f"{current_label}subtitles='{ass_path_escaped}'{fonts_arg},format=yuv420p[vout]"
        )
    else:
        # Static drawtext header banners are completely purged
        filter_complex_parts.append(f"{current_label}null,format=yuv420p[vout]")

    filter_complex = ";".join(filter_complex_parts)

    # Audio filter chain locked strictly to native playback without atempo retiming
    audio_filter_str = "aresample=async=1"

    preset_val = opts.preset if opts.preset.startswith("p") else "p6"
    cq_val = getattr(opts, "cq", 20)
    tune_val = getattr(opts, "tune", "hq")
    rc_val = getattr(opts, "rc", "vbr")
    bitrate_val = getattr(opts, "bitrate", "8M")
    maxrate_val = getattr(opts, "maxrate", "12M")
    bufsize_val = getattr(opts, "bufsize", "24M")

    cmd.extend([
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", "0:a?",
        "-filter:a", audio_filter_str,
        *get_cfr_args(),
    ])
    
    if is_nvenc_available():
        cmd.extend([
            "-c:v", "h264_nvenc",
            "-preset", str(preset_val),
            "-tune", str(tune_val),
            "-rc", str(rc_val),
            "-cq", str(cq_val),
            "-b:v", str(bitrate_val),
            "-maxrate", str(maxrate_val),
            "-bufsize", str(bufsize_val),
        ])
    else:
        cmd.extend([
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", str(getattr(opts, "crf", 20)),
            "-b:v", str(bitrate_val),
            "-maxrate", str(maxrate_val),
            "-bufsize", str(bufsize_val),
        ])

    cmd.extend([
        "-pix_fmt", "yuv420p",
        "-threads", str(opts.threads),
        "-c:a", "aac",
        "-b:a", opts.audio_bitrate,
        "-ar", "48000",
        "-shortest",
        "-avoid_negative_ts", "make_zero",
        "-movflags", "+faststart",
        output_path,
    ])

    return cmd, temp_text_file


def render_clip(
    input_path: str,
    output_path: str,
    video_coords: dict,
    text_coords: dict,
    logo_path: Optional[str] = None,
    logo_coords: Optional[dict] = None,
    emojis: List[dict] = [],
    options: Optional[RenderOptions] = None,
    check: bool = True,
    ass_path: Optional[str] = None,
    trajectory: Optional[dict] = None,
    timeout: Optional[float] = 600.0,
    headline_text: Optional[str] = None,
    brand_logo_path: Optional[str] = None,
    watermark_logo_path: Optional[str] = None,
    speed_factor: Optional[float] = None,
) -> subprocess.CompletedProcess:
    """
    Executes the master FFmpeg rendering command for one clip.
    Raises subprocess.CalledProcessError if check=True and FFmpeg fails.

    Hardware-accelerated with NVENC, with automatic CPU fallback if NVENC
    driver or CUDA device is absent. Includes timeout protection.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input video not found: {input_path}")
    if logo_path and not os.path.isfile(logo_path):
        raise FileNotFoundError(f"Logo file not found: {logo_path}")
    if brand_logo_path and not os.path.isfile(brand_logo_path):
        raise FileNotFoundError(f"Brand logo file not found: {brand_logo_path}")
    if watermark_logo_path and not os.path.isfile(watermark_logo_path):
        raise FileNotFoundError(f"Watermark logo file not found: {watermark_logo_path}")

    # Normalize empty or missing subtitle file path to None
    if ass_path is not None:
        if not str(ass_path).strip() or not os.path.isfile(ass_path) or os.path.getsize(ass_path) == 0:
            ass_path = None

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    cmd, temp_text_file = build_ffmpeg_command(
        input_path=input_path,
        output_path=output_path,
        video_coords=video_coords,
        text_coords=text_coords,
        logo_path=logo_path,
        logo_coords=logo_coords,
        emojis=emojis,
        options=options,
        ass_path=ass_path,
        trajectory=trajectory,
        headline_text=headline_text,
        brand_logo_path=brand_logo_path,
        watermark_logo_path=watermark_logo_path,
        speed_factor=speed_factor,
    )

    try:
        proc = subprocess.run(cmd, check=check, capture_output=True, text=True, timeout=timeout, cwd=_PROJECT_ROOT)
        return proc
    except subprocess.CalledProcessError as e:
        # Graceful CPU fallback if NVENC is unavailable (e.g. non-NVIDIA host/local dev)
        err_msg = (e.stderr or "") + (e.stdout or "")
        is_nvenc_err = any(
            k in err_msg.lower()
            for k in [
                "cannot load nvcuda",
                "nvcuda.dll",
                "libnvcuda",
                "device creation failed",
                "unknown encoder 'h264_nvenc'",
                "no nvenc capable devices",
                "failed to create nvenc",
                "driver does not support nvenc",
            ]
        ) or (e.returncode == 8 and "cuda" in err_msg.lower())
        if is_nvenc_err and "-c:v" in cmd and "h264_nvenc" in cmd:
            global _NVENC_FORCE_DISABLED
            _NVENC_FORCE_DISABLED = True
            print("[engine_ffmpeg] Warning: NVENC unavailable. Falling back to CPU libx264...")
            fallback_cmd = list(cmd)
            idx = fallback_cmd.index("h264_nvenc")
            fallback_cmd[idx] = "libx264"
            if "-cq" in fallback_cmd:
                cq_idx = fallback_cmd.index("-cq")
                fallback_cmd[cq_idx] = "-crf"
                fallback_cmd[cq_idx + 1] = str(getattr(options, "crf", 20) if options else 20)
            if "-preset" in fallback_cmd:
                p_idx = fallback_cmd.index("-preset")
                fallback_cmd[p_idx + 1] = "fast"
            # Replace tune with zerolatency for CPU gaming fallback
            if "-tune" in fallback_cmd:
                t_idx = fallback_cmd.index("-tune")
                fallback_cmd[t_idx + 1] = "zerolatency"
            else:
                fallback_cmd.extend(["-tune", "zerolatency"])
            
            # Strip NVENC-specific rc option not accepted by libx264
            if "-rc" in fallback_cmd:
                r_idx = fallback_cmd.index("-rc")
                fallback_cmd.pop(r_idx)
                fallback_cmd.pop(r_idx)
                
            try:
                proc = subprocess.run(fallback_cmd, check=check, capture_output=True, text=True, timeout=timeout, cwd=_PROJECT_ROOT)
                return proc
            except subprocess.CalledProcessError as fallback_e:
                fb_err = (fallback_e.stderr or "").strip() or (fallback_e.stdout or "").strip()
                raise RuntimeError(f"FFmpeg CPU fallback failed (exit {fallback_e.returncode}): {fb_err}") from fallback_e

        clean_err = (e.stderr or "").strip() or (e.stdout or "").strip()
        raise RuntimeError(f"FFmpeg render failed (exit {e.returncode}): {clean_err}") from e
    except subprocess.TimeoutExpired as e:
        print(f"[engine_ffmpeg] Error: Render timed out after {timeout}s: {' '.join(cmd)}")
        raise
    finally:
        if temp_text_file and os.path.exists(temp_text_file):
            try:
                os.remove(temp_text_file)
            except OSError:
                pass


def render_batch(
    jobs: List[dict],
    options: Optional[RenderOptions] = None,
    timeout: Optional[float] = 600.0,
) -> List[dict]:
    """Renders a list of approved jobs sequentially without aborting on isolated errors."""
    results = []
    for job in jobs:
        try:
            proc = render_clip(
                input_path=job["input_path"],
                output_path=job["output_path"],
                video_coords=job["video_coords"],
                text_coords=job["text_coords"],
                logo_path=job.get("logo_path"),
                logo_coords=job.get("logo_coords"),
                emojis=job.get("emojis", []),
                options=options,
                check=True,
                ass_path=job.get("ass_path"),
                trajectory=job.get("trajectory"),
                timeout=timeout,
                headline_text=job.get("headline_text"),
                brand_logo_path=job.get("brand_logo_path"),
                watermark_logo_path=job.get("watermark_logo_path"),
                speed_factor=job.get("speed_factor"),
            )
            results.append({"job": job, "success": True, "stderr": proc.stderr})
        except Exception as e:
            err = e.stderr if isinstance(e, subprocess.CalledProcessError) else str(e)
            results.append({"job": job, "success": False, "stderr": err})
    return results


def create_zip_archive(files_or_dir, output_zip_path: str = "final_reels.zip") -> str:
    """
    Compresses rendered reels into a final ZIP archive using shutil.make_archive.
    Returns the absolute path to the generated .zip file.
    """
    import shutil
    import tempfile

    abs_out = os.path.abspath(output_zip_path)
    if not abs_out.endswith(".zip"):
        abs_out += ".zip"

    out_dir = os.path.dirname(abs_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    base_name = abs_out[:-4]  # shutil.make_archive appends .zip

    if isinstance(files_or_dir, str) and os.path.isdir(files_or_dir):
        shutil.make_archive(base_name, "zip", root_dir=files_or_dir)
        return abs_out

    files = files_or_dir if isinstance(files_or_dir, (list, tuple)) else [files_or_dir]
    temp_pack_dir = tempfile.mkdtemp(prefix="reels_pack_")
    try:
        for f in files:
            if isinstance(f, str) and os.path.isfile(f):
                shutil.copy2(f, os.path.join(temp_pack_dir, os.path.basename(f)))
        shutil.make_archive(base_name, "zip", root_dir=temp_pack_dir)
    finally:
        shutil.rmtree(temp_pack_dir, ignore_errors=True)

    return abs_out
