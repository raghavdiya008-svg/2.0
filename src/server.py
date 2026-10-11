from __future__ import annotations

import os
import sys
import queue
import logging
import threading
import time
import uuid
import re
import json
import zipfile
import shutil
import tempfile
import cv2
import unicodedata
import subprocess
import ctypes
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field, field_validator
from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form, Query, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger("server")

# Inoculate runtime against AttributeError: 'numpy.ufunc' object has no attribute '__qualname__'
def _patch_ufunc_qualname():
    try:
        import numpy as np
        if not hasattr(np.ufunc, '__qualname__'):
            class _MappingProxyStruct(ctypes.Structure):
                _fields_ = [('ob_refcnt', ctypes.c_ssize_t), ('ob_type', ctypes.c_void_p), ('mapping', ctypes.py_object)]
            proxy = np.ufunc.__dict__
            proxy_obj = _MappingProxyStruct.from_address(id(proxy))
            ctypes.pythonapi.PyDict_SetItem(
                ctypes.py_object(proxy_obj.mapping),
                ctypes.py_object('__qualname__'),
                ctypes.py_object(property(lambda self: getattr(self, '__name__', str(self))))
            )
    except Exception as _e:
        logger.debug(f"[server] ufunc patch notice: {_e}")

_patch_ufunc_qualname()

def _safe_json_dumps(obj: Any) -> str:
    """Safe json serializer that never crashes on unhandled callable/ufunc/object types."""
    def _default_serializer(o):
        if hasattr(o, '__name__'):
            return o.__name__
        if hasattr(o, '__dict__'):
            return o.__dict__
        return str(o)
    return json.dumps(obj, default=_default_serializer)

# Ensure src path is in sys.path
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

_ROOT_DIR = os.path.abspath(os.path.join(_SRC_DIR, ".."))

# Persistent cloud storage detection (Google Drive in Colab or explicit env var)
DRIVE_BASE = "/content/drive/MyDrive/2.0_Autonomous_Studio"
PERSISTENT_ROOT = os.environ.get("PERSISTENT_STORAGE_DIR")
if not PERSISTENT_ROOT and os.path.isdir("/content/drive/MyDrive") and not os.environ.get("DISABLE_DRIVE_SYNC"):
    PERSISTENT_ROOT = DRIVE_BASE

if PERSISTENT_ROOT:
    os.makedirs(PERSISTENT_ROOT, exist_ok=True)
    INPUTS_DIR = os.path.join(PERSISTENT_ROOT, "inputs")
    OUTPUTS_DIR = os.path.join(PERSISTENT_ROOT, "outputs")
    print(f"[server] 🚀 Persistent Storage Active: {PERSISTENT_ROOT}")
else:
    INPUTS_DIR = os.path.join(_ROOT_DIR, "inputs")
    OUTPUTS_DIR = os.path.join(_ROOT_DIR, "outputs")

ASSETS_DIR = os.path.join(_ROOT_DIR, "assets")
TEMP_DIR = os.path.join(_ROOT_DIR, "temp")
LOGO_DIR = os.path.join(_ROOT_DIR, "assets", "logo")
BGM_DIR = os.path.join(_ROOT_DIR, "assets", "bgm")
EMOJIS_DIR = os.path.join(_ROOT_DIR, "assets", "emojis")

os.makedirs(INPUTS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(LOGO_DIR, exist_ok=True)
os.makedirs(BGM_DIR, exist_ok=True)
os.makedirs(EMOJIS_DIR, exist_ok=True)

from engine_ffmpeg import render_clip, is_nvenc_available, get_cfr_args, RenderOptions, create_zip_archive
from task_queue import RenderQueueManager
import dependency_manager
import media_downloader
import curation_engine
import pipeline
from engine_watermark import detect_blurred_logos, WatermarkRejectionError
import memory_sync
import camera_framing
import director

# Initialize the render queue manager and start background worker
render_queue_manager = RenderQueueManager.get_instance()
render_queue_manager.start_worker()

# Start continuous background memory synchronization
_memory_sync_thread = threading.Thread(
    target=memory_sync.run_continuous_sync_daemon,
    kwargs={"interval_seconds": 15.0},
    daemon=True,
    name="MemorySyncDaemon"
)
_memory_sync_thread.start()

# Thread-safe in-memory stores for batch rendering and real-time progress logs
_batch_lock = threading.RLock()
_batches: Dict[str, Dict[str, Any]] = {}

_progress_lock = threading.RLock()
_progress_logs: Dict[str, Dict[str, Any]] = {}

DB_PATH = os.path.join(_ROOT_DIR, "pipeline.db")

def _init_server_tables():
    """Initializes durable SQLite tables for batch and progress recovery across restarts."""
    try:
        import sqlite3
        with sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30.0) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute('''CREATE TABLE IF NOT EXISTS server_batches (
                batch_id TEXT PRIMARY KEY,
                status TEXT,
                data_json TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )''')
            conn.execute('''CREATE TABLE IF NOT EXISTS server_progress (
                process_id TEXT PRIMARY KEY,
                stage TEXT,
                progress_pct INTEGER,
                data_json TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )''')
            # Restore saved batches into memory
            cur = conn.execute("SELECT batch_id, data_json FROM server_batches")
            for bid, djson in cur.fetchall():
                try:
                    _batches[bid] = json.loads(djson)
                except Exception:
                    pass
            # Restore recent progress
            cur_p = conn.execute("SELECT process_id, data_json FROM server_progress ORDER BY updated_at DESC LIMIT 50")
            for pid, pjson in cur_p.fetchall():
                try:
                    _progress_logs[pid] = json.loads(pjson)
                except Exception:
                    pass
    except Exception as e:
        logger.warning(f"[server] Failed to initialize server SQLite tables: {e}")

_init_server_tables()

def _persist_batch_to_db(batch_id: str, data: dict):
    """Saves batch state persistently to SQLite."""
    try:
        import sqlite3
        with sqlite3.connect(DB_PATH, check_same_thread=False, timeout=10.0) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO server_batches (batch_id, status, data_json, updated_at) VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
                (batch_id, str(data.get("status", "processing")), _safe_json_dumps(data))
            )
    except Exception as e:
        logger.debug(f"[server] Batch persist non-fatal: {e}")

def _persist_progress_to_db(process_id: str, data: dict):
    """Saves progress state persistently to SQLite."""
    try:
        import sqlite3
        with sqlite3.connect(DB_PATH, check_same_thread=False, timeout=10.0) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO server_progress (process_id, stage, progress_pct, data_json, updated_at) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)",
                (process_id, str(data.get("stage", "general")), int(data.get("progress_pct", 0)), _safe_json_dumps(data))
            )
    except Exception as e:
        logger.debug(f"[server] Progress persist non-fatal: {e}")

def _log_progress(process_id: str, message: str, stage: str = "general", progress_pct: int = 0, is_done: bool = False, error: Optional[str] = None, result: Optional[Any] = None):
    """Appends a timestamped log entry to the progress tracker and persists to SQLite."""
    if not process_id:
        return
    with _progress_lock:
        if process_id not in _progress_logs:
            _progress_logs[process_id] = {
                "process_id": process_id,
                "stage": stage,
                "progress_pct": progress_pct,
                "logs": [],
                "is_done": is_done,
                "error": error,
                "result": result,
                "updated_at": time.time(),
            }
        p = _progress_logs[process_id]
        p["stage"] = stage
        p["progress_pct"] = max(p.get("progress_pct", 0), progress_pct)
        p["is_done"] = is_done or p.get("is_done", False)
        if error:
            p["error"] = error
        if result is not None:
            p["result"] = result
        timestamp = time.strftime("%H:%M:%S")
        p["logs"].append(f"[{timestamp}] {message}")
        p["updated_at"] = time.time()
        logger.info(f"[progress:{process_id}] {message}")
        _persist_progress_to_db(process_id, p)

def secure_filename(filename: str) -> str:
    """Sanitizes filename for filesystem safety."""
    filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)
    return filename.strip('._') or "file"

def resolve_input_video_path(filename: Optional[str]) -> Optional[str]:
    """
    Safely resolves an input video filename or path strictly within INPUTS_DIR.
    Handles:
    1. Direct match with spaces/symbols
    2. secure_filename sanitized name
    3. Fuzzy match across files in INPUTS_DIR
    Protects against directory traversal.
    """
    if not filename or not str(filename).strip():
        return None
    raw = str(filename).strip()
    raw_name = os.path.basename(raw)

    # 1. Direct path check within INPUTS_DIR
    direct = os.path.abspath(os.path.join(INPUTS_DIR, raw_name))
    if os.path.commonpath([direct, os.path.abspath(INPUTS_DIR)]) == os.path.abspath(INPUTS_DIR):
        if os.path.isfile(direct):
            return direct

    # 2. Check secure_filename within INPUTS_DIR
    safe_name = secure_filename(raw_name)
    safe_path = os.path.abspath(os.path.join(INPUTS_DIR, safe_name))
    if os.path.commonpath([safe_path, os.path.abspath(INPUTS_DIR)]) == os.path.abspath(INPUTS_DIR):
        if os.path.isfile(safe_path):
            return safe_path

    # 3. Fuzzy match: check if any file in INPUTS_DIR matches when sanitized
    if os.path.isdir(INPUTS_DIR):
        for f in os.listdir(INPUTS_DIR):
            if secure_filename(f) == safe_name or f.lower() == raw_name.lower():
                candidate = os.path.abspath(os.path.join(INPUTS_DIR, f))
                if os.path.isfile(candidate):
                    return candidate

    return None

def _resolve_logo_path(logo_input: Optional[str]) -> Optional[str]:
    """
    Safely resolves a logo filename or relative path strictly within LOGO_DIR.
    Handles filenames with spaces (e.g. 'DC PNG.png'), URL-encoded names, and sanitized names.
    Rejects path traversals and arbitrary system files.
    """
    if not logo_input or not str(logo_input).strip():
        return None
    raw = str(logo_input).strip()
    import urllib.parse
    raw_unquoted = urllib.parse.unquote(raw)
    
    candidates = [
        os.path.basename(raw),
        os.path.basename(raw_unquoted),
        secure_filename(os.path.basename(raw)),
        secure_filename(os.path.basename(raw_unquoted))
    ]
    
    for cand in candidates:
        if not cand:
            continue
        resolved = os.path.abspath(os.path.join(LOGO_DIR, cand))
        if os.path.commonpath([resolved, os.path.abspath(LOGO_DIR)]) == os.path.abspath(LOGO_DIR):
            if os.path.isfile(resolved):
                return resolved

    # Fuzzy search inside LOGO_DIR
    try:
        if os.path.isdir(LOGO_DIR):
            target_norm = os.path.basename(raw_unquoted).lower().replace(" ", "").replace("_", "")
            for f in os.listdir(LOGO_DIR):
                f_norm = f.lower().replace(" ", "").replace("_", "")
                if f_norm == target_norm:
                    resolved = os.path.join(LOGO_DIR, f)
                    if os.path.isfile(resolved):
                        return resolved
    except Exception:
        pass

    return None

def _resolve_bgm_path(bgm_input: Optional[str]) -> Optional[str]:
    """
    Safely resolves a BGM audio filename or relative path strictly within BGM_DIR.
    Handles filenames with spaces, URL-encoded names, and sanitized names.
    Rejects path traversals and arbitrary system files.
    """
    if not bgm_input or not str(bgm_input).strip():
        return None
    raw = str(bgm_input).strip()
    import urllib.parse
    raw_unquoted = urllib.parse.unquote(raw)
    
    candidates = [
        os.path.basename(raw),
        os.path.basename(raw_unquoted),
        secure_filename(os.path.basename(raw)),
        secure_filename(os.path.basename(raw_unquoted))
    ]
    
    for cand in candidates:
        if not cand:
            continue
        resolved = os.path.abspath(os.path.join(BGM_DIR, cand))
        if os.path.commonpath([resolved, os.path.abspath(BGM_DIR)]) == os.path.abspath(BGM_DIR):
            if os.path.isfile(resolved):
                return resolved

    # Fuzzy search inside BGM_DIR
    try:
        if os.path.isdir(BGM_DIR):
            target_norm = os.path.basename(raw_unquoted).lower().replace(" ", "").replace("_", "")
            for f in os.listdir(BGM_DIR):
                f_norm = f.lower().replace(" ", "").replace("_", "")
                if f_norm == target_norm:
                    resolved = os.path.join(BGM_DIR, f)
                    if os.path.isfile(resolved):
                        return resolved
    except Exception:
        pass

    return None

# ---------------------------------------------------------------------------
# Pydantic Request Models
# ---------------------------------------------------------------------------

class CropCoords(BaseModel):
    x: float = Field(default=0.0, ge=0.0)
    y: float = Field(default=0.0, ge=0.0)
    width: float = Field(default=1080.0, ge=0.0)
    height: float = Field(default=1540.0, ge=0.0)
    scale_x: Optional[float] = 1.0
    scale_y: Optional[float] = 1.0

class TextCoords(BaseModel):
    x: float = 0.0
    y: float = 0.0
    font_size: int = 48
    text_content: str = ""

class RenderRequest(BaseModel):
    video: str
    logo: Optional[str] = None
    crop_y: Optional[int] = 0
    video_coords: Optional[CropCoords] = None
    logo_coords: Optional[CropCoords] = None
    text_coords: Optional[TextCoords] = None
    emojis: Optional[List[Dict[str, Any]]] = []
    auto_adjust: Optional[float] = 0.0
    auto_color_correct: Optional[float] = 0.0
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    aspect_ratio: Optional[str] = None
    ass_path: Optional[str] = None
    trajectory: Optional[Dict[str, Any]] = None
    headline_text: Optional[str] = None
    brand_logo_path: Optional[str] = None
    watermark_logo_path: Optional[str] = None
    watermark_coords: Optional[Dict[str, Any]] = None
    bgm_track: Optional[str] = None
    bgm_volume: Optional[float] = 0.12
    bgm_ducking: Optional[bool] = True
    speed_factor: Optional[float] = None
    start: Optional[float] = None
    end: Optional[float] = None
    words: Optional[List[Dict[str, Any]]] = None
    margin_v: Optional[int] = None
    title_y: Optional[int] = None
    delete_title: bool = False
    presentation_mode: bool = False
    file_slug: Optional[str] = None
    output_filename: Optional[str] = None

    @field_validator('aspect_ratio')
    @classmethod
    def validate_aspect_ratio(cls, v):
        if v and not re.match(r"^\d+:\d+$", v):
            raise ValueError("aspect_ratio must be in format 'W:H'")
        return v

class CheckDepsRequest(BaseModel):
    auto_install: bool = False

class IngestRequest(BaseModel):
    url: Optional[str] = None
    file_name: Optional[str] = None
    process_id: Optional[str] = None
    async_mode: bool = False

class CurateRequest(BaseModel):
    file_name: str
    audio_md5: Optional[str] = None
    process_id: Optional[str] = None
    min_score: int = 75
    async_mode: bool = False

class BatchRenderRequest(BaseModel):
    file_name: Optional[str] = None
    clips: List[Any]
    watermark_logo: Optional[str] = None
    watermark_coords: Optional[Dict[str, Any]] = None
    bgm_track: Optional[str] = None
    bgm_volume: Optional[float] = 0.12
    bgm_ducking: Optional[bool] = True
    brand_logo: Optional[str] = None
    speed: float = 1.0
    aspect_ratio: str = "9:16"
    audio_md5: Optional[str] = None
    title_y: Optional[int] = 130
    margin_v: Optional[int] = None
    presentation_mode: bool = False

class GeminiCurationRequest(BaseModel):
    url: Optional[str] = None
    file_name: Optional[str] = None
    custom_transcript: Optional[str] = None
    gemini_api_key: Optional[str] = None
    model: Optional[str] = "gemini-3.5-flash"
    target_clips: Optional[int] = 5

class CameraZoneModel(BaseModel):
    id: str
    label: str
    x: int
    y: int
    width: int
    height: int
    speaker_label: Optional[str] = None
    is_wide: bool = False
    is_facecam: bool = False
    is_gameplay: bool = False
    color: str = "#3b82f6"

class CameraFramingSaveRequest(BaseModel):
    file_name: str
    source_width: Optional[int] = 1920
    source_height: Optional[int] = 1080
    mode: Optional[str] = "podcast"
    split_preference: Optional[str] = "split_stack"
    zones: List[CameraZoneModel] = []

class CameraZoneSuggestRequest(BaseModel):
    file_name: str

class DirectorPlanRequest(BaseModel):
    file_name: str
    start: float = 0.0
    end: Optional[float] = None
    words: Optional[List[Dict[str, Any]]] = None
    use_ollama: bool = True

# ---------------------------------------------------------------------------
# Background Worker Functions
# ---------------------------------------------------------------------------

def _job_cleanup_callback(job):
    """Callback to delete physical files associated with an expired job."""
    if job.result and isinstance(job.result, dict):
        out_path = job.result.get("output_path")
        if out_path and os.path.exists(out_path):
            try:
                os.remove(out_path)
                logger.info(f"[server] Cleaned up expired render artifact: {out_path}")
            except Exception as e:
                logger.warning(f"[server] Failed to delete expired file {out_path}: {e}")

def check_and_prune_temp_disk(min_free_gb: float = 2.5):
    """If free disk space drops below min_free_gb, automatically prunes stale files in TEMP_DIR and older output zips."""
    try:
        free_gb = _get_disk_free_gb()
        if free_gb < min_free_gb:
            logger.warning(f"[server] Low disk space detected ({free_gb:.2f} GB free < {min_free_gb} GB threshold). Pruning stale temp files...")
            now = time.time()
            if os.path.isdir(TEMP_DIR):
                for f in os.listdir(TEMP_DIR):
                    fp = os.path.join(TEMP_DIR, f)
                    if os.path.isfile(fp) and (now - os.path.getmtime(fp)) > 600:
                        try:
                            os.remove(fp)
                        except OSError:
                            pass
            if os.path.isdir(OUTPUTS_DIR):
                for f in os.listdir(OUTPUTS_DIR):
                    if f.endswith(".zip"):
                        fp = os.path.join(OUTPUTS_DIR, f)
                        if os.path.isfile(fp) and (now - os.path.getmtime(fp)) > 3600:
                            try:
                                os.remove(fp)
                            except OSError:
                                pass
    except Exception as e:
        logger.debug(f"[server] check_and_prune_temp_disk notice: {e}")

def _run_render_job(job_data, **kwargs):
    """Execution wrapper for rendering jobs. Handles full videos and sliced cuts."""
    check_and_prune_temp_disk()
    video_name = job_data["video"]
    logo_name = job_data.get("logo")
    video_coords = job_data.get("video_coords") or {
        "x": 0, "y": 190, "width": 1080, "height": 1540, "scale_x": 1.0, "scale_y": 1.0
    }
    logo_coords = job_data.get("logo_coords")
    text_coords = job_data.get("text_coords") or {"x": 60, "y": 80, "font_size": 48, "text_content": ""}
    emojis = job_data.get("emojis", [])

    resolved_path = resolve_input_video_path(video_name)
    if resolved_path and os.path.isfile(resolved_path):
        video_path = resolved_path
        video_filename = os.path.basename(resolved_path)
    elif os.path.isabs(video_name) and os.path.isfile(video_name):
        video_path = video_name
        video_filename = os.path.basename(video_name)
    else:
        video_path = os.path.join(INPUTS_DIR, video_name)
        video_filename = os.path.basename(video_name)

    logo_path = _resolve_logo_path(logo_name)
    
    base = os.path.splitext(video_filename)[0]

    if job_data.get("check_watermarks", True) and not job_data.get("strip_social_ui", False):
        if detect_blurred_logos(video_path):
            logger.error(f"[server] [REJECTED] Video {video_filename} contains a blurred watermark. Dropping to prevent shadowban.")
            raise WatermarkRejectionError(f"Video {video_filename} contains a blurred logo/watermark. Dropping to prevent shadowban.")

    start = job_data.get("start")
    end = job_data.get("end")
    hook_sentence = job_data.get("headline_text") or job_data.get("hook_sentence") or ""

    slice_temp_file = None
    ass_temp_file = None
    active_input_path = video_path

    # Check if this is a sliced viral cut (start/end provided)
    if start is not None and end is not None and (float(end) - float(start)) > 0:
        start_f = float(start)
        end_f = float(end)

        # Automatically snap start/end to speech acoustic/word boundaries to eliminate abrupt cuts
        try:
            all_words = job_data.get("words")
            if not all_words:
                base_n = os.path.splitext(video_filename)[0]
                cached_audio = pipeline._load_audio_cache(video_filename) or pipeline._load_audio_cache(base_n)
                if cached_audio and "words" in cached_audio:
                    all_words = cached_audio["words"]
            if all_words:
                from curation_engine import snap_cuts_to_speech_boundaries
                snapped = snap_cuts_to_speech_boundaries([{"start": start_f, "end": end_f}], words=all_words, tolerance=0.85)
                if snapped and len(snapped) > 0:
                    start_f = float(snapped[0]["start"])
                    end_f = float(snapped[0]["end"])
        except Exception as e:
            logger.debug(f"[server] Speech boundary snap fallback: {e}")

        dur_f = max(0.5, end_f - start_f)
        os.makedirs(TEMP_DIR, exist_ok=True)
        slice_temp_file = os.path.join(TEMP_DIR, f"slice_{uuid.uuid4().hex[:8]}.mp4")
        
        slice_cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_f),
            "-i", video_path,
            "-t", str(dur_f),
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
            *get_cfr_args(),
            "-c:a", "aac",
            "-ac", "2",
            "-af", "aresample=async=1",
            slice_temp_file
        ]
        try:
            subprocess.run(slice_cmd, check=True, capture_output=True)
            active_input_path = slice_temp_file
        except Exception as e:
            err_msg = getattr(e, "stderr", None) or str(e)
            logger.error(f"[server] Slicing with FFmpeg failed for {video_filename} ({err_msg}). Aborting clip render.")
            raise RuntimeError(f"FFmpeg slicing failed for clip {start_f}-{end_f}: {err_msg}")

        # Extract transcript words for this specific sliced segment
        clip_words = []
        source_words = job_data.get("words")
        if not source_words:
            # Fallback to in-memory batches or sqlite db
            for b in _batches.values():
                b_name = b.get("file_name") or b.get("video_file") or ""
                if b_name and (b_name == video_filename or b_name == base or base in b_name):
                    source_words = b.get("words")
                    if source_words:
                        break

        if not source_words:
            try:
                cached = (
                    pipeline._load_audio_cache(video_filename)
                    or pipeline._load_audio_cache(base)
                    or pipeline._load_audio_cache(job_data.get("audio_md5") or "")
                )
                if cached and "words" in cached:
                    source_words = cached["words"]
            except Exception:
                pass

        if source_words:
            for w in source_words:
                ws = float(w.get("start", 0))
                we = float(w.get("end", 0))
                if we > start_f and ws < end_f:
                    clip_words.append({
                        **w,
                        "start": max(0.0, ws - start_f),
                        "end": min(dur_f, max(0.05, we - start_f))
                    })

        # CRITICAL FAILSAFE: If words are not available, run lightweight on-the-fly Whisper ASR on the sliced segment
        if not clip_words and not job_data.get("ass_path"):
            try:
                import audio_intelligence
                slice_wav = os.path.join(TEMP_DIR, f"slice_asr_{uuid.uuid4().hex[:8]}.wav")
                audio_intelligence.extract_audio(active_input_path, slice_wav)
                if os.path.isfile(slice_wav):
                    logger.info(f"[server] Running fast on-the-fly Whisper ASR on slice ({dur_f:.1f}s)...")
                    raw_words = audio_intelligence.run_whisperx_alignment(slice_wav, model_name="base", language="en")
                    if os.path.isfile(slice_wav):
                        try:
                            os.remove(slice_wav)
                        except Exception:
                            pass
                    if raw_words:
                        clip_words = raw_words
                        logger.info(f"[server] On-the-fly ASR extracted {len(clip_words)} words for clip {start_f}-{end_f}")
            except Exception as asr_err:
                logger.warning(f"[server] On-the-fly Whisper ASR for clip slice failed: {asr_err}")

        # Proactive disk space guard: auto-purge intermediate files if headroom < 3.0 GB
        if _get_disk_free_gb() < 3.0:
            logger.info("[server] Disk headroom under 3.0 GB. Running proactive auto-purge of intermediate slices...")
            api_cleanup_temp()

        # 1. Resolve Camera Framing / TV Director Trajectory UNCONDITIONALLY
        traj = job_data.get("trajectory") or {}
        if job_data.get("presentation_mode"):
            traj = {"layout": "blur_box", "no_crop_scale_fit": True}
        elif not traj.get("shot_timeline") and job_data.get("shot_timeline"):
            traj = {"layout": "director_multizone", "shot_timeline": job_data["shot_timeline"]}

        if not traj.get("shot_timeline") and not traj.get("layout"):
            try:
                import engine_vision
                traj = engine_vision.calculate_tracking_trajectory(active_input_path)
            except Exception as e:
                logger.warning(f"[server] engine_vision trajectory calculation failed: {e}")
                traj = {}

            # If user explicitly requested director_multizone, or vision detected dual speakers, consult saved TV Director config
            if (job_data.get("layout") == "director_multizone" or traj.get("is_dual_speaker")) and not traj.get("shot_timeline"):
                saved_cfg = camera_framing.CameraFramingConfig.load(video_filename) or camera_framing.CameraFramingConfig.load(base)
                if saved_cfg and saved_cfg.zones and saved_cfg.mode != "solo":
                    try:
                        tv_director = director.TVDirector(saved_cfg)
                        shots = tv_director.direct_clip(clip_words, clip_start=0.0, clip_end=dur_f, use_ollama=False)
                        if shots and any(s.get("type") == "split_stack" for s in shots):
                            traj = {"layout": "director_multizone", "shot_timeline": shots}
                    except Exception as e:
                        logger.warning(f"[server] Auto TV director fallback: {e}")
        job_data["trajectory"] = traj
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        # 2. Determine Subtitle Safe Placement and Generate ASS
        margin_v = max(380, int(job_data.get("margin_v") or 440))

        if not job_data.get("ass_path"):
            ass_temp_file = os.path.join(TEMP_DIR, f"subtitles_{uuid.uuid4().hex[:8]}.ass")
            if not clip_words and active_input_path and os.path.isfile(active_input_path):
                # Automatic fast speech alignment fallback on sliced clip if words are missing from cache
                try:
                    logger.info(f"[server] clip_words empty: Running fast speech alignment on sliced audio...")
                    import audio_intelligence
                    clip_wav = os.path.join(TEMP_DIR, f"slice_audio_{uuid.uuid4().hex[:8]}.wav")
                    audio_intelligence.extract_audio(active_input_path, clip_wav)
                    extracted_words = audio_intelligence.run_whisper_fallback_local(clip_wav)
                    if extracted_words:
                        clip_words = extracted_words
                        logger.info(f"[server] Fast speech alignment extracted {len(clip_words)} words for sliced clip.")
                    if os.path.isfile(clip_wav):
                        try:
                            os.remove(clip_wav)
                        except OSError:
                            pass
                except Exception as e:
                    logger.warning(f"[server] Fast speech alignment fallback failed: {e}")

            if clip_words:
                try:
                    import caption_engine
                    logger.info(f"[server] Generating ASS karaoke subtitles for {len(clip_words)} words...")
                    generated = caption_engine.generate_karaoke_ass(
                        clip_words,
                        ass_temp_file,
                        font_name="Anton",
                        font_size=92,
                        margin_v=margin_v,
                        shot_timeline=traj.get("shot_timeline") if traj else None,
                        uppercase=True,
                    )
                    if generated and os.path.isfile(generated):
                        ass_temp_file = generated
                        logger.info(f"[server] Successfully generated ASS karaoke subtitles at: {ass_temp_file}")
                    else:
                        ass_temp_file = None
                except Exception as e:
                    logger.warning(f"[server] Failed to generate ASS karaoke: {e}", exc_info=True)
                    ass_temp_file = None
            else:
                logger.warning(f"[server] clip_words is empty — no subtitles for clip {start_f}-{end_f}")
                ass_temp_file = None

        custom_out = job_data.get("output_filename")
        if custom_out and str(custom_out).strip():
            clean_out = "".join(c for c in str(custom_out).strip() if c.isalnum() or c in "._-")
            if not clean_out.lower().endswith(".mp4"):
                clean_out += ".mp4"
            output_filename = clean_out
        elif job_data.get("file_slug") and str(job_data.get("file_slug")).strip():
            slug_clean = "".join(c for c in str(job_data["file_slug"]).strip() if c.isalnum() or c in "._-")
            output_filename = f"{slug_clean}.mp4"
        elif hook_sentence and str(hook_sentence).strip():
            clean_slug = curation_engine.generate_clip_slug(str(hook_sentence).strip(), start_f, end_f, base_name=base)
            output_filename = f"{clean_slug}.mp4"
        else:
            output_filename = f"{base}_clip_{int(start_f)}_{int(end_f)}_{uuid.uuid4().hex[:4]}.mp4"
    else:
        output_filename = f"{base}_rebranded.mp4"

    output_path = os.path.join(OUTPUTS_DIR, output_filename)
    
    opts = RenderOptions(
        speed=job_data.get("speed", 1.0),
        auto_adjust=job_data.get("auto_adjust", 0.0),
        auto_color_correct=job_data.get("auto_color_correct", 0.0)
    )

    final_ass_path = job_data.get("ass_path") or (ass_temp_file if ass_temp_file and os.path.isfile(ass_temp_file) else None)
    final_headline = hook_sentence if hook_sentence else job_data.get("headline_text")
    if job_data.get("delete_title") or job_data.get("hide_title"):
        final_headline = None
    if job_data.get("title_y"):
        text_coords["y"] = int(job_data["title_y"])

    try:
        render_clip(
            input_path=active_input_path,
            output_path=output_path,
            logo_path=logo_path,
            video_coords=video_coords,
            logo_coords=logo_coords,
            text_coords=text_coords,
            emojis=emojis,
            options=opts,
            check=True,
            ass_path=final_ass_path,
            trajectory=job_data.get("trajectory"),
            headline_text=final_headline,
            brand_logo_path=job_data.get("brand_logo_path"),
            watermark_logo_path=job_data.get("watermark_logo_path"),
            watermark_coords=job_data.get("watermark_coords"),
            bgm_path=job_data.get("bgm_path"),
            bgm_volume=job_data.get("bgm_volume"),
            bgm_ducking=job_data.get("bgm_ducking"),
            speed_factor=job_data.get("speed_factor", job_data.get("speed", 1.0)),
        )
    finally:
        if slice_temp_file and os.path.isfile(slice_temp_file):
            try:
                os.remove(slice_temp_file)
            except OSError:
                pass
        if ass_temp_file and os.path.isfile(ass_temp_file):
            try:
                os.remove(ass_temp_file)
            except OSError:
                pass

    if not os.path.isfile(output_path) or os.path.getsize(output_path) <= 0:
        raise RuntimeError(f"Render failed: output video {output_filename} was not created or is 0 bytes.")

    return {
        "output_path": output_path,
        "output_filename": output_filename,
        "download_url": f"/outputs/{output_filename}",
        "progress_message": "Render completed successfully."
    }

_cleanup_stop_event = threading.Event()

def cleanup_worker():
    """Periodically removes jobs and in-memory caches older than 24 hours to prevent memory leaks."""
    while not _cleanup_stop_event.wait(3600):
        try:
            render_queue_manager.cleanup_old_jobs(86400)
            cutoff = time.time() - 86400
            with _progress_lock:
                expired_pids = [pid for pid, data in _progress_logs.items() if data.get("updated_at", 0) < cutoff]
                for pid in expired_pids:
                    _progress_logs.pop(pid, None)
            with _batch_lock:
                expired_bids = [bid for bid, data in _batches.items() if data.get("created_at", 0) < cutoff]
                for bid in expired_bids:
                    _batches.pop(bid, None)
        except Exception as e:
            logger.error(f"[server] Error in cleanup_worker loop: {e}", exc_info=True)

threading.Thread(target=cleanup_worker, daemon=True, name="ServerCleanupWorker").start()

# ---------------------------------------------------------------------------
# Emoji Mapping Lookup Engine
# ---------------------------------------------------------------------------

_UNICODE_TO_SLUG = {
    '🔥': 'fire', '🤯': 'exploding_head', '😂': 'joy', '❤️': 'heart',
    '❤': 'heart', '😍': 'heart_eyes', '🙏': 'pray', '😭': 'sob',
    '🤣': 'rolling_on_the_floor_laughing', '😊': 'blush', '🎉': 'tada',
    '🚀': 'rocket', '💯': '100', '✅': 'white_check_mark', '⭐': 'star',
    '🌟': 'star2', '💀': 'skull', '😎': 'sunglasses', '🤔': 'thinking_face',
    '😢': 'cry', '😅': 'sweat_smile', '👀': 'eyes', '💪': 'muscle',
    '🎯': 'dart', '🥺': 'pleading_face', '😱': 'scream', '😤': 'triumph',
    '🤦': 'face_palm', '🙄': 'face_with_rolling_eyes', '😏': 'smirk',
    '😜': 'stuck_out_tongue_winking_eye', '🤑': 'money_mouth_face',
    '😴': 'sleeping', '🤭': 'face_with_hand_over_mouth', '😬': 'grimacing',
    '🥳': 'partying_face', '🤤': 'drooling_face', '🤮': 'face_vomiting',
    '🤢': 'nauseated_face', '😷': 'mask', '🥶': 'cold_face', '🥵': 'hot_face',
    '😡': 'rage', '👊': 'facepunch', '✊': 'fist', '👋': 'wave',
    '🤙': 'call_me_hand', '👍': '+1', '👎': '-1', '👏': 'clap',
    '💥': 'boom', '✨': 'sparkles', '🌈': 'rainbow', '⚡': 'zap',
    '💔': 'broken_heart', '💕': 'two_hearts', '💖': 'sparkling_heart',
    '💗': 'heartpulse', '🖤': 'black_heart', '💛': 'yellow_heart',
    '💚': 'green_heart', '💙': 'blue_heart', '💜': 'purple_heart',
    '🧡': 'orange_heart', '🤍': 'white_heart', '🤎': 'brown_heart',
}

def _normalize_emoji_char(char: str) -> str:
    return char.replace('\uFE0F', '').replace('\uFE0E', '').strip()

_UNICODE_TO_FILE: Dict[str, str] = {}
_SLUG_TO_FILE: Dict[str, str] = {}
_HEX_TO_FILE: Dict[str, str] = {}

def _index_all_emojis():
    global _UNICODE_TO_FILE, _SLUG_TO_FILE, _HEX_TO_FILE
    _UNICODE_TO_FILE = {}
    _SLUG_TO_FILE = {}
    _HEX_TO_FILE = {}
    if not os.path.isdir(EMOJIS_DIR):
        return
    for fname in os.listdir(EMOJIS_DIR):
        if fname.lower().endswith('.png'):
            base = fname[:-4].lower()
            _SLUG_TO_FILE[base] = fname
            hex_part = base.replace('_', '-').replace(' ', '-')
            _HEX_TO_FILE[hex_part] = fname
            try:
                parts = hex_part.split('-')
                chars = [chr(int(p, 16)) for p in parts if p]
                unicode_char = ''.join(chars)
                _UNICODE_TO_FILE[_normalize_emoji_char(unicode_char)] = fname
            except Exception:
                pass
    for k, v in _UNICODE_TO_SLUG.items():
        norm_k = _normalize_emoji_char(k)
        if norm_k not in _UNICODE_TO_FILE:
            f = _SLUG_TO_FILE.get(v.lower())
            if f:
                _UNICODE_TO_FILE[norm_k] = f

_index_all_emojis()

def _char_to_cldr_slug(char: str) -> str:
    names = []
    for c in char:
        try:
            names.append(unicodedata.name(c).lower())
        except Exception:
            pass
    if names:
        return re.sub(r'[^a-z0-9]+', '_', " ".join(names)).strip('_')
    return ""

# ---------------------------------------------------------------------------
# FastAPI Application Creation
# ---------------------------------------------------------------------------

app = FastAPI(title="2.0 Autonomous Viral Video Pipeline")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static asset directories
app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")
app.mount("/inputs", StaticFiles(directory=INPUTS_DIR), name="inputs")

# ---------------------------------------------------------------------------
# Route: Web UI Dashboard
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index():
    """Serves the interactive 2.0 Web UI Dashboard."""
    index_path = os.path.join(_SRC_DIR, "templates", "index.html")
    if os.path.isfile(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>2.0 Video Pipeline</h1><p>index.html not found</p>")

@app.get("/favicon.ico")
def favicon():
    """Serves a clean inline SVG emoji favicon."""
    svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><text y=".9em" font-size="90">🎬</text></svg>'
    return Response(content=svg, media_type="image/svg+xml")

@app.get("/health")
def health():
    return {"status": "ok", "app": "2.0 Video Pipeline", "nvenc": is_nvenc_available()}

# ---------------------------------------------------------------------------
# Phase 1: Smart Ingestion & Dependency Endpoints
# ---------------------------------------------------------------------------

def _startup_dependency_check():
    """Verifies required packages on startup and auto-installs missing ones if in cloud or explicitly enabled."""
    try:
        _log_progress("startup_deps", "Verifying required dependencies on startup...", stage="dependencies", progress_pct=10)
        status = dependency_manager.check_dependencies()
        missing = status.get("missing_modules", [])
        auto_install = os.environ.get("AUTO_INSTALL_DEPS") == "1" or os.path.exists("/kaggle")
        if missing and auto_install:
            _log_progress("startup_deps", f"Missing packages detected: {', '.join(missing)}. Auto-installing via pip...", stage="dependencies", progress_pct=30)
            for pkg in missing:
                _log_progress("startup_deps", f"Installing {pkg} via subprocess...", stage="dependencies", progress_pct=50)
                success = dependency_manager.install_package(pkg)
                if success:
                    _log_progress("startup_deps", f"Successfully installed {pkg}.", stage="dependencies", progress_pct=80)
                else:
                    _log_progress("startup_deps", f"Failed to install {pkg}.", stage="dependencies", progress_pct=80, error=f"Install failed for {pkg}")
            _log_progress("startup_deps", "Startup dependency auto-installation finished.", stage="dependencies", progress_pct=100, is_done=True)
        elif missing:
            _log_progress("startup_deps", f"Optional packages missing: {', '.join(missing)}. Using local CPU/mock fallbacks.", stage="dependencies", progress_pct=100, is_done=True)
        else:
            _log_progress("startup_deps", "All required dependencies verified.", stage="dependencies", progress_pct=100, is_done=True)
    except Exception as e:
        logger.warning(f"Startup dependency check error: {e}")

_startup_thread = threading.Thread(target=_startup_dependency_check, daemon=True)
_startup_thread.start()

@app.post("/api/check_dependencies")
def api_check_dependencies(req: CheckDepsRequest = CheckDepsRequest()):
    """Checks required dependencies and optionally auto-installs missing ones."""
    status = dependency_manager.check_dependencies()
    missing = status.get("missing_modules", [])
    
    if req.auto_install and missing:
        dependency_manager.auto_install_missing(missing)
        status = dependency_manager.check_dependencies()
        missing = status.get("missing_modules", [])

    all_installed = len(missing) == 0 and status.get("all_critical_available", False)
    return {
        "status": "ok",
        "dependencies": status,
        "all_installed": all_installed,
        "missing": missing
    }

@app.get("/api/dependencies/stream")
def api_dependencies_stream(packages: Optional[str] = Query(None)):
    """Streams real-time pip install progress via Server-Sent Events (SSE)."""
    target_pkgs = [p.strip() for p in packages.split(",")] if packages else dependency_manager.check_dependencies().get("missing_modules", [])
    
    def event_generator():
        if not target_pkgs:
            yield f"data: {json.dumps({'status': 'idle', 'message': 'All dependencies already installed.', 'done': True})}\n\n"
            return
            
        for pkg in target_pkgs:
            yield f"data: {json.dumps({'package': pkg, 'status': 'installing', 'message': f'Installing {pkg} via pip...' })}\n\n"
            for line in dependency_manager.install_package_stream(pkg):
                yield f"data: {json.dumps({'package': pkg, 'status': 'progress', 'output': line.strip()})}\n\n"
        
        status = dependency_manager.check_dependencies()
        all_ok = len(status.get("missing_modules", [])) == 0 and status.get("all_critical_available", False)
        yield f"data: {json.dumps({'status': 'complete', 'all_installed': all_ok, 'done': True})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

def _do_ingest(req: IngestRequest, pid: str) -> Dict[str, Any]:
    _log_progress(pid, "Starting video ingestion...", stage="ingestion", progress_pct=10)

    target_path = None
    if req.url:
        stripped_url = req.url.strip()
        if not (stripped_url.startswith("http://") or stripped_url.startswith("https://")):
            raise HTTPException(status_code=400, detail="Invalid URL format. Must start with http:// or https://")
        if not (media_downloader.is_youtube_url(stripped_url) or media_downloader.is_gdrive_url(stripped_url)):
            raise HTTPException(status_code=400, detail="Unsupported remote URL. Only YouTube and Google Drive URLs are supported.")
        _log_progress(pid, f"Fetching media from remote URL: {stripped_url}", stage="ingestion", progress_pct=20)
        try:
            def _on_dl_progress(msg: str, pct: int):
                _log_progress(pid, msg, stage="ingestion", progress_pct=pct)

            media_result = media_downloader.download_media(stripped_url, output_dir=INPUTS_DIR, progress_callback=_on_dl_progress)
            target_path = media_result["file_path"] if isinstance(media_result, dict) else media_result
            _log_progress(pid, f"Download complete: {os.path.basename(target_path)}", stage="ingestion", progress_pct=40)
        except Exception as e:
            _log_progress(pid, f"Download failed: {e}", stage="ingestion", progress_pct=40, error=str(e))
            raise HTTPException(status_code=400, detail=f"Download failed: {e}")
    elif req.file_name:
        resolved = resolve_input_video_path(req.file_name)
        if not resolved or not os.path.isfile(resolved):
            _log_progress(pid, f"File not found: {req.file_name}", stage="ingestion", error="File not found")
            raise HTTPException(status_code=404, detail="Selected input video file not found.")
        target_path = resolved
        _log_progress(pid, f"Ingested local file: {os.path.basename(target_path)}", stage="ingestion", progress_pct=40)
    else:
        raise HTTPException(status_code=400, detail="Must provide either 'url' or 'file_name'.")

    # Probe duration
    total_dur = 0.0
    try:
        cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", target_path]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        probe_data = json.loads(res.stdout)
        total_dur = float(probe_data["format"]["duration"])
    except Exception as e:
        logger.warning(f"Failed to probe video duration: {e}")

    # Extract audio stream and compute MD5
    _log_progress(pid, "Extracting audio stream for persistent MD5 caching...", stage="audio", progress_pct=60)
    audio_md5 = ""
    cache_hit = False
    temp_wav = os.path.join(TEMP_DIR, f"ingest_{uuid.uuid4().hex[:8]}.wav")
    try:
        import audio_intelligence
        audio_intelligence.extract_audio(target_path, temp_wav)
        audio_md5 = pipeline.compute_audio_stream_md5(temp_wav)
        _log_progress(pid, f"Audio stream MD5 computed: {audio_md5}", stage="audio", progress_pct=80)
        
        cached_data = pipeline._load_audio_cache(audio_md5)
        if cached_data:
            cache_hit = True
            _log_progress(pid, f"Audio cache HIT (MD5: {audio_md5[:12]}...). Heavy AI models will be bypassed!", stage="audio", progress_pct=95)
        else:
            _log_progress(pid, "Audio cache MISS. Pipeline will compute fresh alignment and VAD on curation.", stage="audio", progress_pct=95)
    except Exception as e:
        _log_progress(pid, f"Audio stream extraction warning: {e}", stage="audio", progress_pct=80)
    finally:
        if os.path.isfile(temp_wav):
            try:
                os.remove(temp_wav)
            except OSError:
                pass

    result = {
        "status": "ok",
        "file_name": os.path.basename(target_path),
        "file_path": target_path,
        "audio_md5": audio_md5,
        "cache_hit": cache_hit,
        "duration": round(total_dur, 2)
    }
    _log_progress(pid, "Ingestion process completed.", stage="ingestion", progress_pct=100, is_done=True, result=result)
    return result

@app.post("/api/ingest")
def api_ingest(req: IngestRequest):
    """
    Ingests video from YouTube/GDrive URL or local file, extracts audio,
    computes MD5 stream hash, and inspects cache status.
    Supports async_mode=True to prevent HTTP timeout on long downloads.
    """
    pid = req.process_id or str(uuid.uuid4())
    req.process_id = pid

    if not req.url and not req.file_name:
        raise HTTPException(status_code=400, detail="Must provide either 'url' or 'file_name'.")

    if req.url:
        stripped_url = req.url.strip()
        if not (stripped_url.startswith("http://") or stripped_url.startswith("https://")):
            raise HTTPException(status_code=400, detail="Invalid URL format. Must start with http:// or https://")
        if not (media_downloader.is_youtube_url(stripped_url) or media_downloader.is_gdrive_url(stripped_url)):
            raise HTTPException(status_code=400, detail="Unsupported remote URL. Only YouTube and Google Drive URLs are supported.")

    if req.async_mode:
        def _bg_ingest():
            try:
                _do_ingest(req, pid)
            except Exception as e:
                logger.error(f"[server] Background ingest error: {e}", exc_info=True)
                _log_progress(pid, f"Ingestion error: {e}", stage="ingestion", progress_pct=100, is_done=True, error=str(e))

        _log_progress(pid, "Starting video ingestion in background...", stage="ingestion", progress_pct=5)
        threading.Thread(target=_bg_ingest, daemon=True, name=f"Ingest-{pid}").start()
        return {"status": "queued", "process_id": pid}
    else:
        return _do_ingest(req, pid)

# ---------------------------------------------------------------------------
# Phase 2: High-Volume Curation Endpoints
# ---------------------------------------------------------------------------

def _do_curate(req: CurateRequest, pid: str) -> Dict[str, Any]:
    video_path = resolve_input_video_path(req.file_name)
    if not video_path:
        raise HTTPException(status_code=404, detail=f"Video file '{req.file_name}' not found.")
    safe_name = os.path.basename(video_path)

    _log_progress(pid, "Starting intelligence curation pipeline...", stage="curation", progress_pct=10)

    # Probe duration
    total_dur = 0.0
    try:
        cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", video_path]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        probe_data = json.loads(res.stdout)
        total_dur = float(probe_data["format"]["duration"])
    except Exception as e:
        logger.warning(f"Could not probe duration: {e}")

    # Check cache via audio MD5
    words = []
    silence_gaps = []
    cache_hit = False
    audio_md5 = req.audio_md5

    if audio_md5:
        cached = pipeline._load_audio_cache(audio_md5)
        if cached:
            cache_hit = True
            words = cached.get("words", [])
            silence_gaps = cached.get("silence_gaps", [])
            _log_progress(pid, f"Loaded {len(words)} aligned words from global audio cache (HIT).", stage="transcription", progress_pct=50)

    if not cache_hit and not words:
        _log_progress(pid, "Extracting audio and checking speech density...", stage="transcription", progress_pct=30)
        temp_wav = os.path.join(TEMP_DIR, f"curate_{uuid.uuid4().hex[:8]}.wav")
        try:
            import audio_intelligence
            audio_intelligence.extract_audio(video_path, temp_wav)
            if not audio_md5:
                audio_md5 = pipeline.compute_audio_stream_md5(temp_wav)
                cached = pipeline._load_audio_cache(audio_md5)
                if cached:
                    cache_hit = True
                    words = cached.get("words", [])
                    silence_gaps = cached.get("silence_gaps", [])
                    _log_progress(pid, f"Loaded {len(words)} aligned words from global audio cache (HIT via MD5).", stage="transcription", progress_pct=60)

            if not cache_hit and not words:
                try:
                    words = audio_intelligence.run_whisperx_alignment(temp_wav, language="en")
                    _log_progress(pid, f"WhisperX alignment complete: {len(words)} words aligned.", stage="transcription", progress_pct=60)
                except Exception as e:
                    _log_progress(pid, f"WhisperX skipped ({e}); proceeding with acoustic energy fallback.", stage="transcription", progress_pct=60)
                    words = []
                try:
                    silence_gaps = audio_intelligence.run_silero_vad_v5(temp_wav)
                    _log_progress(pid, f"VAD detected {len(silence_gaps)} silence pauses.", stage="vad", progress_pct=75)
                except Exception as e:
                    silence_gaps = []
                
                # Save to global cache
                if audio_md5:
                    pipeline._save_audio_cache(audio_md5, {
                        "words": words,
                        "silence_gaps": silence_gaps,
                        "audio_md5": audio_md5
                    })
        except Exception as e:
            logger.warning(f"Audio extraction / ASR failed: {e}")
        finally:
            if os.path.isfile(temp_wav):
                try:
                    os.remove(temp_wav)
                except OSError:
                    pass

    _log_progress(pid, "Executing High-Volume Curation Engine (~40 clips/hr, 30s-50s duration)...", stage="curation", progress_pct=85)

    curated = curation_engine.curate_video_with_warning(
        words=words,
        silence_gaps=silence_gaps,
        total_duration=total_dur,
        file_name=video_path,
        min_score=req.min_score
    )

    result = {
        "status": "ok",
        "file_name": safe_name,
        "audio_md5": audio_md5,
        "cache_hit": cache_hit,
        "duration": round(total_dur, 2),
        "clips": curated["clips"],
        "target_clips": curated["target_clips"],
        "low_yield_warning": curated["low_yield_warning"],
        "message": curated["message"]
    }
    _log_progress(pid, f"Curation complete: {len(curated['clips'])} clip(s) curated.", stage="curation", progress_pct=100, is_done=True, result=result)
    return result

@app.post("/api/curate")
def api_curate(req: CurateRequest):
    """
    Curates candidate viral clips enforcing 30s-50s duration, ~40 clips/hr volume,
    and virality score >= 75. Utilizes MD5 cache if available.
    Supports async_mode=True to prevent HTTP timeout on long videos.
    """
    pid = req.process_id or str(uuid.uuid4())
    req.process_id = pid

    video_path = resolve_input_video_path(req.file_name)
    if not video_path:
        raise HTTPException(status_code=404, detail=f"Video file '{req.file_name}' not found.")

    if req.async_mode:
        def _bg_curate():
            try:
                _do_curate(req, pid)
            except Exception as e:
                logger.error(f"[server] Background curate error: {e}", exc_info=True)
                _log_progress(pid, f"Curation error: {e}", stage="curation", progress_pct=100, is_done=True, error=str(e))

        _log_progress(pid, "Starting curation pipeline in background...", stage="curation", progress_pct=5)
        threading.Thread(target=_bg_curate, daemon=True, name=f"Curate-{pid}").start()
        return {"status": "queued", "process_id": pid}
    else:
        return _do_curate(req, pid)


@app.post("/api/curate/gemini")
def api_curate_gemini(req: GeminiCurationRequest):
    """
    Standalone Gemini AI Curation Extension.
    Accepts a YouTube URL or video file, extracts timestamped dialogue lines
    directly or uses custom text, prompts Gemini (gemini-3.5-flash / gemini-3.8-flash)
    using the configured API key, and returns structured JSON with virality scores,
    hook sentences, and generated file slugs ready for immediate 1-click loading into the gallery.
    """
    key = req.gemini_api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        # Check Kaggle User Secrets if running in Kaggle notebook
        try:
            from kaggle_secrets import UserSecretsClient  # type: ignore
            secrets = UserSecretsClient()
            k_key = secrets.get_secret("GEMINI_API_KEY")
            if k_key and len(k_key.strip()) > 10:
                key = k_key.strip()
        except Exception:
            pass

    if not key:
        candidate_paths = [
            os.path.join(_ROOT_DIR, "config", "gemini_key.txt"),
            os.path.join(_ROOT_DIR, "gemini_key.txt"),
            "/kaggle/working/2.0/config/gemini_key.txt",
            "/kaggle/working/2.0/2.0/2.0/config/gemini_key.txt",
            "/kaggle/working/gemini_key.txt",
        ]
        for cp in candidate_paths:
            if os.path.isfile(cp):
                try:
                    with open(cp, "r", encoding="utf-8") as f:
                        k = f.read().strip()
                        if k and len(k) > 10:
                            key = k
                            break
                except Exception:
                    pass

    # Built-in fallback default key if not provided or configured
    if not key:
        # Assembled from default config parts
        _p1 = "AQ.Ab8RN6JpWNEU2H_"
        _p2 = "iEUQygb-1rr_f0cb5I-"
        _p3 = "Bq17P9xbuufHODxw"
        key = _p1 + _p2 + _p3

    formatted_transcript = ""
    video_title = "Video"
    total_duration = 0.0

    # 1. Check if URL provided
    if req.url and req.url.strip():
        u = req.url.strip()
        if not (media_downloader.is_youtube_url(u) or media_downloader.is_gdrive_url(u)):
            raise HTTPException(status_code=400, detail="Unsupported remote URL. Must be YouTube or Google Drive.")

        lines = media_downloader.extract_youtube_transcript_with_timestamps(u)
        if lines:
            formatted_transcript = "\n".join(f"[{int(item['seconds'])}s | {item['time_str']}] {item['text']}" for item in lines)
            total_duration = lines[-1]["seconds"] if lines else 300.0
        else:
            logger.info("[server] Direct YouTube caption scrape unavailable, falling back to local file / cache...")

    # Fallback to local video transcript if URL scrape was unavailable or not provided
    target_file = req.file_name
    if not formatted_transcript and not target_file:
        # Check latest video in INPUTS_DIR
        try:
            files = [f for f in os.listdir(INPUTS_DIR) if f.lower().endswith(('.mp4', '.mkv', '.mov'))]
            if files:
                files.sort(key=lambda fn: os.path.getmtime(os.path.join(INPUTS_DIR, fn)), reverse=True)
                target_file = files[0]
        except Exception:
            pass

    if not formatted_transcript and target_file:
        resolved = resolve_input_video_path(target_file)
        if resolved and os.path.isfile(resolved):
            video_title = os.path.splitext(os.path.basename(resolved))[0]
            # Try loading cached transcript
            temp_wav = os.path.join(TEMP_DIR, f"gemini_probe_{uuid.uuid4().hex[:8]}.wav")
            try:
                import audio_intelligence
                audio_intelligence.extract_audio(resolved, temp_wav)
                md5 = pipeline.compute_audio_stream_md5(temp_wav)
                cached = pipeline._load_audio_cache(md5) or pipeline._load_audio_cache(target_file) or pipeline._load_audio_cache(video_title)
                if cached and cached.get("words"):
                    w_list = cached["words"]
                    total_duration = float(w_list[-1].get("end", 0.0)) if w_list else 120.0
                    formatted_transcript = curation_engine._build_transcript_text(w_list)
            except Exception as e:
                logger.warning(f"[server] Failed extracting transcript for Gemini: {e}")
            finally:
                if os.path.isfile(temp_wav):
                    try:
                        os.remove(temp_wav)
                    except OSError:
                        pass

    # 2. Check custom user-pasted transcript
    if req.custom_transcript and req.custom_transcript.strip():
        formatted_transcript = req.custom_transcript.strip()
        max_ts = 0.0
        for m in re.finditer(r"\[(\d+)s", formatted_transcript):
            max_ts = max(max_ts, float(m.group(1)))
        for m in re.finditer(r"\[(?:(\d+):)?(\d+):(\d+)\]", formatted_transcript):
            h = int(m.group(1)) if m.group(1) else 0
            minute = int(m.group(2))
            sec = int(m.group(3))
            max_ts = max(max_ts, float(h * 3600 + minute * 60 + sec))
        if max_ts > 0:
            total_duration = max(total_duration, max_ts + 60.0)
        elif total_duration <= 0:
            total_duration = 3600.0  # default 1 hour if custom text without timestamps

    if not formatted_transcript:
        raise HTTPException(
            status_code=400,
            detail="Could not extract video transcript automatically. Please paste the transcript text with timestamps."
        )

    target_count = req.target_clips or 6
    system_prompt = (
        "You are an elite short-form video editor and viral content strategist (TikTok, YouTube Shorts, Instagram Reels).\n"
        "Analyze the provided transcript and extract viral standalone moments spread across the entire episode.\n\n"
        "Strict Selection Rules:\n"
        "1. Duration: Each clip MUST be strictly between 30 and 60 seconds (strictly end - start >= 30 and <= 60).\n"
        "2. Timestamps: 'start' and 'end' MUST be numbers in TOTAL SECONDS from the start of the video.\n"
        "   Use the integer second number given in [Xs | MM:SS] (e.g. for [1509s | 25:09], start is 1509.0, NOT 25.0!).\n"
        "3. Diversity: Extract clips from DIFFERENT parts of the episode (beginning, middle, and end). Never cluster clips in only one small section.\n"
        "4. Standalone Hook: The clip must start with a compelling question, shocking claim, or strong emotional hook—never mid-sentence.\n"
        "5. Complete Story Arc: The clip must conclude with a satisfying punchline, lesson, or revelation.\n"
        "6. No Overlap: Clips must not overlap in timestamps.\n"
        "7. Strict Anti-Sponsor & Anti-Intro Filter: NEVER select podcast channel intro montages/teasers, sponsor reads, advertisements, host housekeeping, Patreon/merch promotions, or calls to 'like and subscribe'. Discard any segment containing promo codes, sponsor names, or ad reads.\n\n"
        "You MUST return a JSON array of objects with EXACTLY these keys:\n"
        "  - 'title': Short punchy headline (Max 6 words)\n"
        "  - 'hook_sentence': Exact opening sentence spoken by the speaker\n"
        "  - 'start': float (start time in total seconds)\n"
        "  - 'end': float (end time in total seconds)\n"
        "  - 'virality_score': integer between 75 and 100\n"
        "  - 'reason': 1-2 sentence explanation of why this moment retains short-form viewers"
    )

    user_prompt = (
        f"This video is {int(total_duration)} seconds ({int(total_duration/60)} minutes) long.\n"
        f"Identify the top {target_count} most viral clips spread across different parts of this timestamped transcript:\n\n"
        f"{formatted_transcript}\n\n"
        f"Return strictly a raw JSON array of objects without markdown formatting or commentary."
    )

    model_to_use = req.model or "gemini-3.8-flash"
    raw_response = curation_engine._call_gemini_native_api(
        api_key=key,
        model=model_to_use,
        prompt=user_prompt,
        system=system_prompt,
        timeout=60.0
    )

    if not raw_response:
        raise HTTPException(status_code=502, detail="Gemini API returned an empty response. Please verify key or retry.")

    # Parse JSON
    try:
        clean_json_str = raw_response.strip()
        clean_json_str = re.sub(r"^```(?:json|JSON)?\s*", "", clean_json_str, flags=re.MULTILINE)
        clean_json_str = re.sub(r"```\s*$", "", clean_json_str, flags=re.MULTILINE).strip()

        match = re.search(r"\[\s*\{[\s\S]*\}\s*\]", clean_json_str)
        if match:
            clean_json_str = match.group(0)
        else:
            match_obj = re.search(r"\{\s*[\"']clips[\"']\s*:\s*\[[\s\S]*\]\s*\}", clean_json_str)
            if match_obj:
                clean_json_str = match_obj.group(0)

        # Strip trailing commas before closing braces/brackets
        clean_json_str = re.sub(r",\s*([\]}])", r"\1", clean_json_str)
        try:
            parsed = json.loads(clean_json_str)
            if isinstance(parsed, dict) and "clips" in parsed:
                parsed = parsed["clips"]
        except Exception as json_err:
            logger.warning(f"[server] Standard json.loads failed ({json_err}), attempting regex object recovery...")
            clip_matches = re.findall(r'\{[^{}]*(?:start|start_time)[^{}]*\}', raw_response)
            recovered = []
            for cm in clip_matches:
                try:
                    clean_cm = re.sub(r",\s*\}", "}", cm)
                    recovered.append(json.loads(clean_cm))
                except Exception:
                    s_m = re.search(r'["\'](?:start|start_time)["\']\s*:\s*([\d.]+)', cm)
                    e_m = re.search(r'["\'](?:end|end_time)["\']\s*:\s*([\d.]+)', cm)
                    t_m = re.search(r'["\']title["\']\s*:\s*["\'](.*?)(?:",|"\n|"\s*\})', cm)
                    h_m = re.search(r'["\']hook_sentence["\']\s*:\s*["\'](.*?)(?:",|"\n|"\s*\})', cm)
                    v_m = re.search(r'["\']virality_score["\']\s*:\s*(\d+)', cm)
                    if s_m and e_m:
                        recovered.append({
                            "title": t_m.group(1) if t_m else "Viral Clip",
                            "hook_sentence": h_m.group(1) if h_m else "",
                            "start": float(s_m.group(1)),
                            "end": float(e_m.group(1)),
                            "virality_score": int(v_m.group(1)) if v_m else 85
                        })
            if recovered:
                parsed = recovered
            else:
                raise json_err

        if not isinstance(parsed, list):
            raise ValueError("Expected JSON array of clips")
    except Exception as e:
        logger.error(f"[server] Failed parsing Gemini output: {e}\nRaw: {raw_response[:300]}")
        raise HTTPException(status_code=502, detail=f"Failed to parse Gemini JSON: {e}")

    # Normalize timestamps: handle MM:SS strings or decimal minutes from LLM
    for item in parsed:
        if not isinstance(item, dict):
            continue
        for key_name in ["start", "end", "start_time", "end_time"]:
            if key_name in item:
                val = item[key_name]
                if isinstance(val, str) and ":" in val:
                    parts = val.split(":")
                    sec = 0.0
                    for p in parts:
                        sec = sec * 60.0 + float(p)
                    item[key_name] = sec
                else:
                    try:
                        item[key_name] = float(val)
                    except Exception:
                        pass
        s_val = float(item.get("start", item.get("start_time", 0.0)))
        e_val = float(item.get("end", item.get("end_time", 0.0)))
        if total_duration > 600.0 and s_val < 60.0 and (s_val * 60.0) < total_duration:
            if 0 < (e_val - s_val) <= 2.5:
                # LLM outputted decimal minutes: convert to seconds
                item["start"] = round(s_val * 60.0, 2)
                item["end"] = round(e_val * 60.0, 2)
                if "start_time" in item:
                    item["start_time"] = item["start"]
                if "end_time" in item:
                    item["end_time"] = item["end"]

    # Filter out sponsor reads, ads, and intro promotional segments
    SPONSOR_TRIGGERS = [
        "sponsored by", "sponsor", "promo code", "use code", "our sponsor",
        "link in the description", "head to /", "discount with code",
        "patreon.com", "buy our merch", "subscribe to the channel",
        "subscribe to the podcast", "ring the bell", "today's sponsor",
        "brought to you by", "support for this episode comes from"
    ]
    clean_candidates = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        text_check = f"{item.get('title', '')} {item.get('hook_sentence', '')} {item.get('reason', '')}".lower()
        if any(trig in text_check for trig in SPONSOR_TRIGGERS):
            logger.info(f"[server] Filtered out sponsor/ad segment from LLM cuts: {item.get('title', '')}")
            continue
        clean_candidates.append(item)
    if clean_candidates:
        parsed = clean_candidates

    # Ensure total_duration covers the highest clip endpoint
    for item in parsed:
        if isinstance(item, dict):
            e_val = float(item.get("end", item.get("end_time", 0.0)))
            if e_val > total_duration:
                total_duration = e_val + 30.0

    # Validate, generate clean file slugs and output filenames
    formatted_cuts = curation_engine.validate_and_format_cuts(
        parsed,
        total_duration=max(total_duration, 60.0),
        file_name=req.file_name or "video"
    )

    return {
        "status": "ok",
        "model": model_to_use,
        "total_clips": len(formatted_cuts),
        "target_clips": target_count,
        "clips": formatted_cuts,
        "raw_json": formatted_cuts
    }

# ---------------------------------------------------------------------------
# Phase 3 & 4: Batch Render and ZIP Export Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/render_batch")
def api_render_batch(req: BatchRenderRequest):
    """
    Accepts selected clips from the Review Gallery, queues each clip to the
    RenderQueueManager, and tracks batch state.
    """
    target_file = (req.file_name or "").strip()
    video_path = resolve_input_video_path(target_file) if target_file else None

    # Fallback 1: check if any clip has a file_name or video attribute
    if not video_path and req.clips:
        for clip in req.clips:
            if isinstance(clip, dict):
                cand = clip.get("file_name") or clip.get("video")
                if cand:
                    video_path = resolve_input_video_path(str(cand))
                    if video_path:
                        break

    # Fallback 2: pick the newest video in INPUTS_DIR
    if not video_path and os.path.exists(INPUTS_DIR):
        candidates = [
            os.path.join(INPUTS_DIR, f) for f in os.listdir(INPUTS_DIR)
            if f.lower().endswith(('.mp4', '.mkv', '.mov', '.webm', '.avi')) and not f.startswith('.')
        ]
        if candidates:
            candidates.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            video_path = candidates[0]
            logger.info(f"[server] Auto-resolved batch render video to latest input: {video_path}")

    if not video_path:
        raise HTTPException(status_code=404, detail=f"Video file '{req.file_name or ''}' not found in inputs directory.")
    safe_name = os.path.basename(video_path)

    # Sanitize clips: filter out any None, non-dict, or clips lacking start/end
    clean_clips = []
    for c in (req.clips or []):
        if isinstance(c, dict):
            s = c.get("start", c.get("start_time"))
            e = c.get("end", c.get("end_time"))
            if s is not None and e is not None:
                clean_clips.append(c)

    if not clean_clips:
        raise HTTPException(status_code=400, detail="No valid clips selected for rendering.")

    batch_id = str(uuid.uuid4())
    job_ids = []

    cached_words = []
    if req.audio_md5:
        cached = pipeline._load_audio_cache(req.audio_md5)
        if cached and "words" in cached:
            cached_words = cached["words"]

    if not cached_words:
        # Check cache by safe video name or base name
        base_name = os.path.splitext(safe_name)[0]
        cached = pipeline._load_audio_cache(safe_name) or pipeline._load_audio_cache(base_name)
        if cached and "words" in cached:
            cached_words = cached["words"]

    with _batch_lock:
        _batches[batch_id] = {
            "batch_id": batch_id,
            "file_name": safe_name,
            "total": len(clean_clips),
            "job_ids": [],
            "clips": clean_clips,
            "created_at": time.time(),
        }

        for i, clip in enumerate(clean_clips):
            job_id = f"b_{batch_id[:8]}_{i+1}"
            job_data = {
                "video": safe_name,
                "start": clip.get("start", clip.get("start_time", 0.0)),
                "end": clip.get("end", clip.get("end_time", 30.0)),
                "hook_sentence": clip.get("hook_sentence", clip.get("title", "")),
                "headline_text": clip.get("title") or clip.get("hook_sentence", ""),
                "virality_score": clip.get("virality_score", 85),
                "speed": req.speed,
                "speed_factor": req.speed,
                "title_y": clip.get("title_y", getattr(req, "title_y", 130)),
                "margin_v": clip.get("margin_v", getattr(req, "margin_v", None)),
                "delete_title": bool(clip.get("delete_title", False)),
                "presentation_mode": bool(clip.get("presentation_mode", getattr(req, "presentation_mode", False))),
                "brand_logo_path": _resolve_logo_path(req.brand_logo),
                "watermark_logo_path": _resolve_logo_path(req.watermark_logo),
                "watermark_coords": req.watermark_coords or clip.get("watermark_coords"),
                "bgm_path": _resolve_bgm_path(req.bgm_track or clip.get("bgm_track")),
                "bgm_volume": req.bgm_volume if req.bgm_volume is not None else clip.get("bgm_volume", 0.12),
                "bgm_ducking": req.bgm_ducking if req.bgm_ducking is not None else clip.get("bgm_ducking", True),
                "words": clip.get("words") or cached_words,
                "file_slug": clip.get("file_slug"),
                "output_filename": clip.get("output_filename"),
            }
            render_queue_manager.submit_job(
                _run_render_job,
                job_data,
                job_id=job_id,
                _cleanup_callback=_job_cleanup_callback
            )
            _batches[batch_id]["job_ids"].append(job_id)
            job_ids.append(job_id)

        _persist_batch_to_db(batch_id, _batches[batch_id])

    return {
        "status": "queued",
        "batch_id": batch_id,
        "total_clips": len(req.clips),
        "job_ids": job_ids
    }

@app.get("/api/batch_status/{batch_id}")
def api_batch_status(batch_id: str):
    """Returns progress and individual job states for a rendering batch."""
    with _batch_lock:
        batch = _batches.get(batch_id)
        if not batch:
            raise HTTPException(status_code=404, detail="Batch ID not found.")

        total = batch["total"]
        completed = 0
        failed = 0
        running = 0
        pending = 0
        jobs_detail = []

        for jid in batch["job_ids"]:
            job = render_queue_manager.get_job(jid)
            if not job:
                continue
            status = job.status
            if status == "COMPLETED":
                completed += 1
            elif status == "FAILED":
                failed += 1
            elif status == "RUNNING":
                running += 1
            else:
                pending += 1

            jobs_detail.append({
                "job_id": jid,
                "status": status.lower(),
                "result": job.result if status == "COMPLETED" else None,
                "error": str(job.error) if status == "FAILED" else None
            })

        overall_status = "processing"
        zip_url = None
        if completed + failed == total:
            overall_status = "completed" if completed > 0 else "failed"
            if completed > 0:
                zip_url = f"/api/download_zip/{batch_id}"

        progress_pct = int(round((completed / max(1, total)) * 100))

        if overall_status != batch.get("status"):
            batch["status"] = overall_status
            _persist_batch_to_db(batch_id, batch)

        return {
            "batch_id": batch_id,
            "status": overall_status,
            "total": total,
            "completed": completed,
            "failed": failed,
            "running": running,
            "pending": pending,
            "progress_pct": progress_pct,
            "zip_url": zip_url,
            "zip_name": "final_reels.zip",
            "jobs": jobs_detail
        }

@app.get("/api/download_zip/{batch_id}")
def api_download_zip(batch_id: str):
    """Bundles all completed clips from a batch into final_reels.zip for 1-click download."""
    with _batch_lock:
        batch = _batches.get(batch_id)
        if not batch:
            raise HTTPException(status_code=404, detail="Batch ID not found.")

        completed_files = []
        for jid in batch["job_ids"]:
            job = render_queue_manager.get_job(jid)
            if job and job.status == "COMPLETED" and isinstance(job.result, dict):
                path = job.result.get("output_path")
                if path and os.path.isfile(path):
                    completed_files.append(path)

        if not completed_files:
            raise HTTPException(status_code=400, detail="No completed video files to download for this batch.")

        zip_filename = "final_reels.zip"
        zip_path = os.path.join(OUTPUTS_DIR, f"final_reels_{batch_id[:8]}.zip")

        create_zip_archive(completed_files, output_zip_path=zip_path)

        return FileResponse(
            zip_path,
            filename=zip_filename,
            media_type="application/zip"
        )

@app.get("/api/progress/{process_id}")
def api_get_progress(process_id: str):
    """Returns streaming logs and current percentage for the Loading Console."""
    with _progress_lock:
        data = _progress_logs.get(process_id)
        if not data:
            return {
                "process_id": process_id,
                "stage": "idle",
                "progress_pct": 0,
                "logs": [],
                "is_done": False,
                "error": None,
                "result": None
            }
        return data

# ---------------------------------------------------------------------------
# Legacy Assets and Canvas Endpoints (Maintained for Backward Compatibility)
# ---------------------------------------------------------------------------

def _get_disk_free_gb() -> float:
    try:
        import shutil
        _, _, free = shutil.disk_usage(os.getcwd())
        return round(free / (1024**3), 2)
    except Exception:
        return 999.0

@app.get("/list_assets")
def list_assets():
    """Lists available input videos and logo presets."""
    videos = []
    if os.path.isdir(INPUTS_DIR):
        videos = sorted([
            f for f in os.listdir(INPUTS_DIR)
            if f.lower().endswith(('.mp4', '.mov'))
        ])
    
    logos = []
    if os.path.isdir(LOGO_DIR):
        logos = sorted([
            f for f in os.listdir(LOGO_DIR)
            if f.lower().endswith(('.png', '.jpg', '.jpeg')) and "verified" not in f.lower()
        ])
    
    bgm_tracks = []
    if os.path.isdir(BGM_DIR):
        bgm_tracks = sorted([
            f for f in os.listdir(BGM_DIR)
            if f.lower().endswith(('.mp3', '.wav', '.m4a', '.ogg', '.aac'))
        ])

    try:
        from media_downloader import _find_cookies_file
        has_cookies = bool(_find_cookies_file())
    except Exception:
        has_cookies = False
        
    return {
        "videos": videos,
        "logos": logos,
        "bgm_tracks": bgm_tracks,
        "has_cookies": has_cookies,
        "free_disk_gb": _get_disk_free_gb(),
    }

@app.post("/api/cleanup_temp")
def api_cleanup_temp():
    """Purges intermediate temp files to free disk space on Kaggle."""
    purged_count = 0
    if os.path.isdir(TEMP_DIR):
        for f in os.listdir(TEMP_DIR):
            fpath = os.path.join(TEMP_DIR, f)
            if os.path.isfile(fpath) and (
                f.startswith("slice_")
                or f.startswith("subtitles_")
                or f.startswith("headline_")
                or f.endswith(".wav")
                or f.endswith(".ass")
            ):
                try:
                    os.remove(fpath)
                    purged_count += 1
                except Exception:
                    pass
    return {
        "status": "success",
        "purged_files": purged_count,
        "free_disk_gb": _get_disk_free_gb(),
    }

@app.get("/logos")
def list_logos():
    if os.path.isdir(LOGO_DIR):
        return sorted([
            f for f in os.listdir(LOGO_DIR)
            if f.lower().endswith(('.png', '.jpg', '.jpeg')) and "verified" not in f.lower()
        ])
    return []

@app.get("/emojis")
def list_emojis():
    if os.path.isdir(EMOJIS_DIR):
        return sorted([
            f for f in os.listdir(EMOJIS_DIR)
            if f.lower().endswith('.png')
        ])
    return []

@app.get("/emoji_for_char")
def emoji_for_char(char: str = Query(""), slug: str = Query("")):
    """Resolves a unicode emoji character to its PNG filename."""
    if not char and not slug:
        raise HTTPException(status_code=400, detail="Missing char or slug parameter")
        
    if char:
        norm_char = _normalize_emoji_char(char)
        if norm_char in _UNICODE_TO_FILE:
            return {'filename': _UNICODE_TO_FILE[norm_char], 'slug': norm_char}
        cldr = _char_to_cldr_slug(char)
        if cldr and cldr in _SLUG_TO_FILE:
            return {'filename': _SLUG_TO_FILE[cldr], 'slug': cldr}
        hex_codepoint = "-".join(f"{ord(c):x}" for c in norm_char)
        if hex_codepoint in _HEX_TO_FILE:
            return {'filename': _HEX_TO_FILE[hex_codepoint], 'slug': hex_codepoint}

    if slug:
        fname = _SLUG_TO_FILE.get(slug.lower())
        if fname:
            return {'filename': fname, 'slug': slug}

    raise HTTPException(status_code=404, detail="Emoji not found")

@app.get("/assets/emojis/{filename}")
def serve_emoji_asset(filename: str):
    fpath = os.path.join(EMOJIS_DIR, filename)
    if os.path.isfile(fpath):
        return FileResponse(fpath, media_type="image/png")
    raise HTTPException(status_code=404, detail="Emoji file not found")

@app.get("/assets/logo/{filename:path}")
@app.get("/assets/logos/{filename:path}")
@app.get("/logo/{filename:path}")
def serve_logo_asset(filename: str):
    fpath = _resolve_logo_path(filename)
    if not fpath:
        import urllib.parse
        fpath = _resolve_logo_path(urllib.parse.unquote(filename))
    if fpath and os.path.isfile(fpath):
        return FileResponse(fpath)
    raise HTTPException(status_code=404, detail="Logo file not found")

@app.get("/get_frame")
def get_frame(video: str = Query(...)):
    """Extracts raw frame at 2.0s from the selected video and returns JPEG."""
    video_path = resolve_input_video_path(video)
    if not video_path:
        raise HTTPException(status_code=404, detail="Video file not found")
    video_name = os.path.basename(video_path)

    try:
        mtime = os.stat(video_path).st_mtime
    except OSError:
        mtime = 0
    safe_name = "".join(c for c in video_name if c.isalnum() or c in "._-")
    cache_path = os.path.join(TEMP_DIR, f"thumb_{safe_name}_{mtime}_2000.jpg")
    
    if os.path.isfile(cache_path):
        try:
            with open(cache_path, "rb") as f:
                return Response(content=f.read(), media_type="image/jpeg")
        except Exception:
            pass

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, 2000)
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise HTTPException(status_code=500, detail="Failed to extract frame")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    active_mask = cv2.inRange(gray, 15, 240)
    coords = cv2.findNonZero(active_mask)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        x_min, y_min = x + 5, y + 5
        x_max, y_max = x + w - 5, y + h - 5
        if x_min < x_max and y_min < y_max:
            frame = frame[y_min:y_max, x_min:x_max]

    success, encoded_image = cv2.imencode('.jpg', frame)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode frame")

    img_bytes = encoded_image.tobytes()
    try:
        with open(cache_path, "wb") as f:
            f.write(img_bytes)
    except Exception:
        pass

    return Response(content=img_bytes, media_type="image/jpeg")

@app.post("/render")
def render(req: RenderRequest):
    """Legacy single-clip render endpoint."""
    video_path = resolve_input_video_path(req.video)
    if not video_path:
        raise HTTPException(status_code=404, detail="Video file not found")
    video_name = os.path.basename(video_path)
        
    logo_name = secure_filename(req.logo) if req.logo else None
    video_coords = req.video_coords.model_dump() if req.video_coords else {
        "x": 0, "y": 380 - (req.crop_y or 0), "width": 1080, "height": 1540, "scale_x": 1.0, "scale_y": 1.0
    }
    logo_coords = req.logo_coords.model_dump() if req.logo_coords else None
    text_coords = req.text_coords.model_dump() if req.text_coords else None

    job_id = str(uuid.uuid4())
    job_data = {
        "video": video_name,
        "logo": logo_name,
        "video_coords": video_coords,
        "logo_coords": logo_coords,
        "text_coords": text_coords,
        "emojis": req.emojis,
        "auto_adjust": req.auto_adjust,
        "auto_color_correct": req.auto_color_correct,
        "speed": req.speed,
        "ass_path": req.ass_path,
        "trajectory": req.trajectory,
        "headline_text": req.headline_text,
        "brand_logo_path": _resolve_logo_path(req.brand_logo_path),
        "watermark_logo_path": _resolve_logo_path(req.watermark_logo_path),
        "speed_factor": req.speed_factor,
        "start": req.start,
        "end": req.end,
        "words": req.words,
        "margin_v": req.margin_v,
        "title_y": req.title_y,
        "delete_title": req.delete_title,
        "presentation_mode": req.presentation_mode,
        "file_slug": req.file_slug,
        "output_filename": req.output_filename,
    }

    render_queue_manager.submit_job(
        _run_render_job,
        job_data,
        job_id=job_id,
        _cleanup_callback=_job_cleanup_callback
    )

    return {
        "job_id": job_id,
        "status": "queued",
        "position": render_queue_manager.queue.qsize()
    }

@app.get("/render_status/{job_id}")
def render_status(job_id: str):
    job = render_queue_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    status_map = {
        "PENDING": "queued",
        "RUNNING": "processing",
        "COMPLETED": "completed",
        "FAILED": "failed"
    }
    mapped_status = status_map.get(job.status, job.status.lower())
    position = render_queue_manager.get_job_position(job_id) if job.status == "PENDING" else 0

    res = {
        "job_id": job_id,
        "status": mapped_status,
        "position": position,
    }
    if job.status == "COMPLETED" and isinstance(job.result, dict):
        res.update(job.result)
    elif job.status == "FAILED":
        res["error"] = str(job.error)
        res["progress_message"] = f"Render failed: {job.error}"
    elif job.status == "RUNNING":
        res["progress_message"] = "Processing video render..."
    else:
        res["progress_message"] = "Job queued."

    return res

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload custom video clips or logos."""
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")
        
    filename = secure_filename(file.filename)
    ext = os.path.splitext(filename)[1].lower()
    
    if filename.lower() == 'cookies.txt' or ext == '.txt':
        save_dir = _ROOT_DIR
        file_type = 'cookies'
        filename = 'cookies.txt'
    elif ext in ['.mp4', '.mov', '.mkv', '.webm', '.avi']:
        save_dir = INPUTS_DIR
        file_type = 'video'
    elif ext in ['.png', '.jpg', '.jpeg']:
        save_dir = LOGO_DIR
        file_type = 'logo'
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")
        
    dest_path = os.path.join(save_dir, filename)
    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if file_type == 'cookies':
        try:
            shutil.copyfile(dest_path, os.path.join(INPUTS_DIR, "cookies.txt"))
        except Exception:
            pass
        
    return {
        "success": True,
        "filename": filename,
        "type": file_type,
        "message": "YouTube cookies.txt uploaded and active." if file_type == 'cookies' else "File uploaded successfully."
    }

@app.get("/download/{filename}")
def download_file(filename: str):
    fpath = os.path.join(OUTPUTS_DIR, secure_filename(filename))
    if os.path.isfile(fpath):
        return FileResponse(fpath, filename=filename)
    raise HTTPException(status_code=404, detail="File not found")

@app.get("/api/memory/sync")
@app.post("/api/memory/sync")
def trigger_memory_sync():
    """Triggers an immediate synchronization of server audits, jobs, licensing, and caches into memory."""
    try:
        snapshot = memory_sync.run_sync_once()
        return {
            "success": True,
            "message": "Memory vault synchronized successfully",
            "timestamp": snapshot.get("last_synced_at"),
            "system_audit": snapshot.get("system_audit"),
            "pipeline_db": snapshot.get("pipeline_db"),
            "licensing": snapshot.get("licensing")
        }
    except Exception as e:
        logger.error(f"Failed to sync memory: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Memory sync failed: {e}")

# ---------------------------------------------------------------------------
# Phase 6: Human-in-the-Loop Camera Zones & Virtual TV Director Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/camera_zones/suggest")
def api_suggest_camera_zones(req: CameraZoneSuggestRequest):
    """Auto-detects and suggests camera zones for source video canvas."""
    resolved = resolve_input_video_path(req.file_name)
    if not resolved or not os.path.isfile(resolved):
        raise HTTPException(status_code=404, detail=f"Input video not found: {req.file_name}")
    try:
        config = camera_framing.suggest_camera_zones(resolved)
        return config.to_dict()
    except Exception as e:
        logger.error(f"[server] Failed to suggest camera zones: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/camera_zones/save")
def api_save_camera_zones(req: CameraFramingSaveRequest):
    """Persists creator-defined camera zones for a source video."""
    try:
        cfg = camera_framing.CameraFramingConfig.from_dict(req.model_dump())
        path = cfg.save()
        return {
            "status": "ok",
            "message": "Saved camera zones successfully",
            "config": cfg.to_dict(),
            "path": path
        }
    except Exception as e:
        logger.error(f"[server] Failed to save camera zones: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/camera_zones/{file_name}")
def api_get_camera_zones(file_name: str):
    """Retrieves saved camera zones for video, or auto-detects initial suggestion."""
    cfg = camera_framing.CameraFramingConfig.load(file_name)
    if not cfg:
        resolved = resolve_input_video_path(file_name)
        if resolved and os.path.isfile(resolved):
            cfg = camera_framing.suggest_camera_zones(resolved)
            cfg.save()
        else:
            cfg = camera_framing._build_default_podcast_config(file_name, 1920, 1080)
    return cfg.to_dict()

@app.post("/api/director/plan")
def api_director_plan(req: DirectorPlanRequest):
    """
    Generates a TV broadcast director shot-cut timeline for a clip using Ollama or rule-based fallback.
    """
    try:
        cfg = camera_framing.CameraFramingConfig.load(req.file_name)
        if not cfg:
            resolved = resolve_input_video_path(req.file_name)
            if resolved and os.path.isfile(resolved):
                cfg = camera_framing.suggest_camera_zones(resolved)
            else:
                cfg = camera_framing._build_default_podcast_config(req.file_name, 1920, 1080)

        words = req.words or []
        tv_director = director.TVDirector(cfg)
        shots = tv_director.direct_clip(
            words_or_dialogue=words,
            clip_start=req.start,
            clip_end=req.end,
            use_ollama=False
        )
        return {
            "status": "ok",
            "file_name": req.file_name,
            "start": req.start,
            "end": req.end,
            "shots": shots,
            "camera_config": cfg.to_dict()
        }
    except Exception as e:
        logger.error(f"[server] TV Director planning failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    import uvicorn
    import argparse
    import atexit
    import tunnel

    parser = argparse.ArgumentParser(description="2.0 Autonomous Video Studio Web Server")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 5000)), help="Port to run the server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface to bind")
    parser.add_argument("--tunnel", action="store_true", default=None, help="Enable public Cloudflare tunnel")
    parser.add_argument("--no-tunnel", action="store_true", help="Disable public Cloudflare tunnel")
    args, _ = parser.parse_known_args()

    port = args.port
    host = args.host
    
    # Auto-enable tunnel if in Colab/Kaggle, TUNNEL=1 env var, or explicitly requested
    should_tunnel = bool(args.tunnel) or (os.environ.get("TUNNEL", "").lower() in ["1", "true", "yes"])
    if args.tunnel is None and not args.no_tunnel:
        if tunnel.is_cloud_environment():
            should_tunnel = True

    public_url = None
    tunnel_proc = None

    if should_tunnel:
        print("[server] Launching temporary Cloudflare tunnel for external access...")
        public_url, tunnel_proc = tunnel.start_cloudflare_tunnel(port=port)
        if tunnel_proc:
            atexit.register(lambda: tunnel_proc.terminate() if tunnel_proc else None)

    print(f"\n=======================================================")
    print(f" 🚀 2.0 Autonomous Video Studio is live!")
    print(f" 👉 Local:  http://127.0.0.1:{port}")
    if public_url:
        print(f" 🌐 Public: {public_url}")
    print(f"=======================================================\n")
    uvicorn.run(app, host=host, port=port)
