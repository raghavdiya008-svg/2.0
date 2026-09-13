"""
src/engine_vad.py
-----------------
Voice Activity Detection (VAD) and dead-air pruning engine using Silero VAD v5.
Tightens video clips by eliminating non-vocal pauses (>500ms) with phoneme-safe padding.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
import uuid
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import soundfile as sf
import torch
import engine_ffmpeg

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Silero VAD Model Loader (Guarded, Lazy)
# ---------------------------------------------------------------------------
_vad_model = None
_vad_utils = None


def load_silero_vad():
    """
    Loads Silero VAD v5 model and utilities from torch.hub.
    Caches model globally to prevent repeated initialization overhead.
    """
    global _vad_model, _vad_utils
    if _vad_model is not None and _vad_utils is not None:
        return _vad_model, _vad_utils

    try:
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )
        _vad_model = model
        _vad_utils = utils
        logger.info("Silero VAD v5 model loaded successfully via torch.hub")
        return _vad_model, _vad_utils
    except Exception as e:
        logger.error(f"Failed to load Silero VAD model via torch.hub: {e}")
        raise RuntimeError(f"Silero VAD initialization failed: {e}") from e


# ---------------------------------------------------------------------------
# Audio Conversion Helper
# ---------------------------------------------------------------------------

def _load_audio_16k(audio_path: str) -> torch.Tensor:
    """
    Loads audio from a WAV file or video container and returns a 1D float32
    torch.Tensor normalized to 16kHz mono.
    """
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio/video file not found: {audio_path}")

    temp_wav = None
    try:
        # Check if file is already a valid WAV that soundfile can read directly
        is_wav = audio_path.lower().endswith(".wav")
        if is_wav:
            try:
                data, sr = sf.read(audio_path)
                if len(data.shape) > 1:
                    data = data.mean(axis=1)
                if sr == 16000:
                    return torch.from_numpy(data.astype(np.float32))
            except Exception:
                pass  # Fall through to ffmpeg extraction

        # Extract 16kHz mono PCM via FFmpeg for any container format (MP4, MOV, etc.)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            temp_wav = tf.name

        cmd = [
            "ffmpeg", "-y",
            "-i", audio_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            temp_wav,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError:
            return torch.empty(0, dtype=torch.float32)

        data, sr = sf.read(temp_wav)
        if len(data.shape) > 1:
            data = data.mean(axis=1)

        # Convert 16-bit PCM integer or float to float32 tensor
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        else:
            data = data.astype(np.float32)

        return torch.from_numpy(data)
    finally:
        if temp_wav and os.path.isfile(temp_wav):
            try:
                os.remove(temp_wav)
            except OSError:
                pass


def _get_media_duration(file_path: str) -> float:
    """Probes media duration in seconds using ffprobe with OpenCV fallback."""
    if not os.path.isfile(file_path):
        return 0.0

    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json",
            file_path,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        dur = float(json.loads(res.stdout)["format"]["duration"])
        if dur > 0:
            return dur
    except Exception:
        pass

    try:
        cap = cv2.VideoCapture(file_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        cap.release()
        if fps > 0 and frame_count > 0:
            return float(frame_count / fps)
    except Exception as e:
        logger.debug(f"[engine_vad] cv2 duration fallback failed: {e}")

    logger.warning(f"[engine_vad] Could not determine video duration for: {file_path}")
    return 0.0


# ---------------------------------------------------------------------------
# Core VAD Detection
# ---------------------------------------------------------------------------

def detect_speech_segments(
    audio_path: str,
    min_silence_duration_ms: int = 500,
    threshold: float = 0.5,
) -> List[Dict[str, float]]:
    """
    Detects active human speech segments in an audio or video file using Silero VAD v5.

    Args:
        audio_path: Path to input audio or video container.
        min_silence_duration_ms: Minimum silence duration (ms) required to split speech chunks (default: 500ms).
        threshold: Speech probability threshold (default: 0.5).

    Returns:
        List of active speech intervals [{"start": float, "end": float}] in seconds.
    """
    model, utils = load_silero_vad()
    get_speech_timestamps = utils[0]

    wav_tensor = _load_audio_16k(audio_path)
    if wav_tensor.numel() == 0:
        return []

    # get_speech_timestamps natively accepts min_silence_duration_ms and return_seconds
    timestamps = get_speech_timestamps(
        wav_tensor,
        model,
        sampling_rate=16000,
        threshold=threshold,
        min_silence_duration_ms=min_silence_duration_ms,
        return_seconds=True,
    )

    speech_segments = []
    for ts in timestamps:
        start_sec = round(float(ts["start"]), 3)
        end_sec = round(float(ts["end"]), 3)
        if end_sec > start_sec:
            speech_segments.append({"start": start_sec, "end": end_sec})

    return speech_segments


# ---------------------------------------------------------------------------
# Dead-Air Silence Pruning
# ---------------------------------------------------------------------------

def _merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Merges overlapping or adjacent time intervals."""
    if not intervals:
        return []
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_intervals[0]]
    for cur_start, cur_end in sorted_intervals[1:]:
        prev_start, prev_end = merged[-1]
        if cur_start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, cur_end))
        else:
            merged.append((cur_start, cur_end))
    return merged


def strip_dead_air(
    video_path: str,
    output_path: str,
    speech_segments: List[Dict[str, float]],
    padding_buffer_ms: float = 80.0,
    min_silence_pruned_sec: float = 1.5,
) -> Tuple[str, List[Tuple[float, float]], float]:
    """
    Removes dead air from video_path according to speech_segments.

    Rules:
      - If total silence pruned < 1.5 seconds or zero segments are detected,
        skip filtering and copy directly to output_path.
      - Otherwise, build an FFmpeg `select` and `aselect` filtergraph concatenating
        only speech chunks, with an additional 80ms padding buffer before and after
        each segment to avoid clipping phonemes.

    Args:
        video_path: Path to source input video.
        output_path: Target path for the tightened video.
        speech_segments: Detected speech segments [{"start": float, "end": float}].
        padding_buffer_ms: Padding buffer in milliseconds (default: 80ms).
        min_silence_pruned_sec: Threshold under which pruning is bypassed (default: 1.5s).

    Returns:
        Tuple containing:
        - Path to output_path.
        - List of preserved intervals (original start, original end). Empty if no pruning.
        - Total duration of pruned silence in seconds.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    total_duration = _get_media_duration(video_path)

    # If zero segments are detected, skip filtering and copy directly
    if not speech_segments or total_duration <= 0.0:
        logger.info("[engine_vad] Zero speech segments detected. Preserving original clip.")
        shutil.copy2(video_path, output_path)
        return output_path, [], 0.0

    # Apply 80ms padding buffer before and after each segment
    pad_sec = padding_buffer_ms / 1000.0
    raw_intervals = [
        (
            max(0.0, float(seg["start"]) - pad_sec),
            min(total_duration, float(seg["end"]) + pad_sec),
        )
        for seg in speech_segments
    ]

    merged_intervals = _merge_intervals(raw_intervals)
    total_speech_sec = sum(end - start for start, end in merged_intervals)
    silence_pruned_sec = max(0.0, total_duration - total_speech_sec)

    # If total silence pruned is < 1.5s, skip filtering and copy directly
    if silence_pruned_sec < min_silence_pruned_sec:
        logger.info(
            f"[engine_vad] Silence pruned ({silence_pruned_sec:.2f}s) < {min_silence_pruned_sec}s threshold. "
            f"Skipping filtering and copying directly."
        )
        shutil.copy2(video_path, output_path)
        return output_path, [], 0.0

    logger.info(
        f"[engine_vad] Pruning {silence_pruned_sec:.2f}s of dead air across "
        f"{len(merged_intervals)} speech interval(s) (+{padding_buffer_ms}ms buffer)."
    )

    # Build select and aselect filter expressions
    select_parts = [
        f"between(t,{start:.4f},{end:.4f})"
        for start, end in merged_intervals
    ]
    select_expr = "+".join(select_parts)

    filter_complex = (
        f"[0:v]select='{select_expr}',setpts=N/FRAME_RATE/TB,scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p[v];"
        f"[0:a]aselect='{select_expr}',asetpts=N/SR/TB,aresample=async=1[a]"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-map", "[a]",
    ]

    if engine_ffmpeg.is_nvenc_available():
        cmd.extend([
            "-c:v", "h264_nvenc", "-preset", "p1",
            "-cq", "18", "-b:v", "10M", "-maxrate", "14M", "-bufsize", "20M",
        ])
    else:
        cmd.extend([
            "-c:v", "libx264", "-preset", "ultrafast",
            "-crf", "18",
        ])

    cmd.extend([
        *engine_ffmpeg.get_cfr_args(),
        "-c:a", "aac",
        output_path,
    ])

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        err_msg = (e.stderr or "") + (e.stdout or "")
        is_nvenc_err = any(
            k in err_msg.lower()
            for k in ["h264_nvenc", "nvenc", "nvcuda", "cuda", "cannot load nvcuda", "device creation failed", "unknown encoder"]
        ) or e.returncode == 8
        if is_nvenc_err and "-c:v" in cmd and "h264_nvenc" in cmd:
            logger.warning("[engine_vad] NVENC unavailable for dead-air pruning. Falling back to CPU libx264...")
            fallback_cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-filter_complex", filter_complex,
                "-map", "[v]",
                "-map", "[a]",
                "-c:v", "libx264", "-preset", "ultrafast",
                "-crf", "18",
                *engine_ffmpeg.get_cfr_args(),
                "-c:a", "aac",
                output_path,
            ]
            subprocess.run(fallback_cmd, check=True, capture_output=True, text=True)
        else:
            raise

    return output_path, merged_intervals, silence_pruned_sec
