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
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field, field_validator
from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form, Query, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger("server")

# Ensure src path is in sys.path
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

_ROOT_DIR = os.path.abspath(os.path.join(_SRC_DIR, ".."))

INPUTS_DIR = os.path.join(_ROOT_DIR, "inputs")
ASSETS_DIR = os.path.join(_ROOT_DIR, "assets")
TEMP_DIR = os.path.join(_ROOT_DIR, "temp")
OUTPUTS_DIR = os.path.join(_ROOT_DIR, "outputs")
LOGO_DIR = os.path.join(_ROOT_DIR, "assets", "logo")
EMOJIS_DIR = os.path.join(_ROOT_DIR, "assets", "emojis")

os.makedirs(INPUTS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(LOGO_DIR, exist_ok=True)
os.makedirs(EMOJIS_DIR, exist_ok=True)

from engine_ffmpeg import render_clip, is_nvenc_available, get_cfr_args, RenderOptions, create_zip_archive
from task_queue import RenderQueueManager
import dependency_manager
import media_downloader
import curation_engine
import pipeline
from engine_watermark import detect_blurred_logos, WatermarkRejectionError
import memory_sync

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

def _log_progress(process_id: str, message: str, stage: str = "general", progress_pct: int = 0, is_done: bool = False, error: Optional[str] = None):
    """Appends a timestamped log entry to the in-memory progress tracker for process_id."""
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
                "result": None,
                "updated_at": time.time(),
            }
        p = _progress_logs[process_id]
        p["stage"] = stage
        p["progress_pct"] = max(p.get("progress_pct", 0), progress_pct)
        p["is_done"] = is_done or p.get("is_done", False)
        if error:
            p["error"] = error
        timestamp = time.strftime("%H:%M:%S")
        p["logs"].append(f"[{timestamp}] {message}")
        p["updated_at"] = time.time()
        logger.info(f"[progress:{process_id}] {message}")

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
    Rejects path traversals and arbitrary system files.
    """
    if not logo_input or not str(logo_input).strip():
        return None
    raw = str(logo_input).strip()
    safe_base = secure_filename(os.path.basename(raw))
    resolved = os.path.abspath(os.path.join(LOGO_DIR, safe_base))
    if os.path.commonpath([resolved, os.path.abspath(LOGO_DIR)]) != os.path.abspath(LOGO_DIR):
        logger.warning(f"[server] Path traversal attempt in logo parameter rejected: {logo_input}")
        return None
    return resolved if os.path.isfile(resolved) else None

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
    speed: float = Field(default=1.12, ge=0.5, le=2.0)
    aspect_ratio: Optional[str] = None
    ass_path: Optional[str] = None
    trajectory: Optional[Dict[str, Any]] = None
    headline_text: Optional[str] = None
    brand_logo_path: Optional[str] = None
    watermark_logo_path: Optional[str] = None
    speed_factor: Optional[float] = None

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

class CurateRequest(BaseModel):
    file_name: str
    audio_md5: Optional[str] = None
    process_id: Optional[str] = None
    min_score: int = 75

class BatchRenderRequest(BaseModel):
    file_name: str
    clips: List[Dict[str, Any]]
    watermark_logo: Optional[str] = None
    brand_logo: Optional[str] = None
    speed: float = 1.12
    aspect_ratio: str = "9:16"

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

def _run_render_job(job_data, **kwargs):
    """Execution wrapper for rendering jobs. Handles full videos and sliced cuts."""
    video_name = job_data["video"]
    logo_name = job_data.get("logo")
    video_coords = job_data.get("video_coords") or {
        "x": 0, "y": 190, "width": 1080, "height": 1540, "scale_x": 1.0, "scale_y": 1.0
    }
    logo_coords = job_data.get("logo_coords")
    text_coords = job_data.get("text_coords") or {"x": 60, "y": 80, "font_size": 48, "text_content": ""}
    emojis = job_data.get("emojis", [])

    if os.path.isabs(video_name) and os.path.isfile(video_name):
        video_path = video_name
        video_filename = os.path.basename(video_name)
    else:
        video_path = os.path.join(INPUTS_DIR, video_name)
        video_filename = video_name

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
        dur_f = end_f - start_f
        os.makedirs(TEMP_DIR, exist_ok=True)
        slice_temp_file = os.path.join(TEMP_DIR, f"slice_{uuid.uuid4().hex[:8]}.mp4")
        
        slice_cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_f), "-t", str(dur_f),
            "-i", video_path,
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
            *get_cfr_args(),
            "-c:a", "aac",
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

        # Generate kinetic ASS subtitles if transcript words are provided
        if not job_data.get("ass_path") and job_data.get("words"):
            try:
                import caption_engine
                ass_temp_file = os.path.join(TEMP_DIR, f"sub_{uuid.uuid4().hex[:8]}.ass")
                clip_words = []
                for w in job_data["words"]:
                    ws = float(w.get("start", 0))
                    we = float(w.get("end", 0))
                    if ws >= start_f and we <= end_f:
                        clip_words.append({
                            **w,
                            "start": max(0.0, ws - start_f),
                            "end": max(0.1, we - start_f)
                        })
                if clip_words:
                    traj = job_data.get("trajectory") or {}
                    is_dual = traj.get("layout") == "dual_speaker_split" or bool(traj.get("is_dual_speaker", False))
                    margin_v = 960 if is_dual else 580
                    caption_engine.generate_karaoke_ass(clip_words, ass_temp_file, margin_v=margin_v)
            except Exception as e:
                logger.warning(f"[server] Failed to generate ASS karaoke: {e}")

        output_filename = f"{base}_clip_{int(start_f)}_{int(end_f)}_{uuid.uuid4().hex[:4]}.mp4"
    else:
        output_filename = f"{base}_rebranded.mp4"

    output_path = os.path.join(OUTPUTS_DIR, output_filename)
    
    opts = RenderOptions(
        speed=job_data.get("speed", 1.12),
        auto_adjust=job_data.get("auto_adjust", 0.0),
        auto_color_correct=job_data.get("auto_color_correct", 0.0)
    )

    final_ass_path = job_data.get("ass_path") or (ass_temp_file if ass_temp_file and os.path.isfile(ass_temp_file) else None)
    final_headline = hook_sentence if hook_sentence else job_data.get("headline_text")

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
            speed_factor=job_data.get("speed_factor", job_data.get("speed", 1.12)),
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

@app.post("/api/ingest")
def api_ingest(req: IngestRequest):
    """
    Ingests video from YouTube/GDrive URL or local file, extracts audio,
    computes MD5 stream hash, and inspects cache status.
    """
    pid = req.process_id or str(uuid.uuid4())
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
            media_result = media_downloader.download_media(stripped_url, output_dir=INPUTS_DIR)
            target_path = media_result["file_path"] if isinstance(media_result, dict) else media_result
            _log_progress(pid, f"Download complete: {os.path.basename(target_path)}", stage="ingestion", progress_pct=40)
        except Exception as e:
            _log_progress(pid, f"Download failed: {e}", stage="ingestion", progress_pct=40, error=str(e))
            raise HTTPException(status_code=400, detail=f"Download failed: {e}")
    elif req.file_name:
        safe_name = secure_filename(req.file_name)
        target_path = os.path.join(INPUTS_DIR, safe_name)
        if not os.path.isfile(target_path):
            _log_progress(pid, f"File not found: {safe_name}", stage="ingestion", error="File not found")
            raise HTTPException(status_code=404, detail="Selected input video file not found.")
        _log_progress(pid, f"Ingested local file: {safe_name}", stage="ingestion", progress_pct=40)
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

    _log_progress(pid, "Ingestion process completed.", stage="ingestion", progress_pct=100, is_done=True)

    return {
        "status": "ok",
        "file_name": os.path.basename(target_path),
        "file_path": target_path,
        "audio_md5": audio_md5,
        "cache_hit": cache_hit,
        "duration": round(total_dur, 2)
    }

# ---------------------------------------------------------------------------
# Phase 2: High-Volume Curation Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/curate")
def api_curate(req: CurateRequest):
    """
    Curates candidate viral clips enforcing 30s-50s duration, ~40 clips/hr volume,
    and virality score >= 75. Utilizes MD5 cache if available.
    """
    pid = req.process_id or str(uuid.uuid4())
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

    _log_progress(pid, f"Curation complete: {len(curated['clips'])} clip(s) curated.", stage="curation", progress_pct=100, is_done=True)

    return {
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

# ---------------------------------------------------------------------------
# Phase 3 & 4: Batch Render and ZIP Export Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/render_batch")
def api_render_batch(req: BatchRenderRequest):
    """
    Accepts selected clips from the Review Gallery, queues each clip to the
    RenderQueueManager, and tracks batch state.
    """
    video_path = resolve_input_video_path(req.file_name)
    if not video_path:
        raise HTTPException(status_code=404, detail=f"Video file '{req.file_name}' not found.")
    safe_name = os.path.basename(video_path)

    if not req.clips:
        raise HTTPException(status_code=400, detail="No clips selected for rendering.")

    batch_id = str(uuid.uuid4())
    job_ids = []

    cached_words = []
    try:
        temp_wav = os.path.join(TEMP_DIR, f"probe_{uuid.uuid4().hex[:8]}.wav")
        import audio_intelligence
        audio_intelligence.extract_audio(video_path, temp_wav)
        md5 = pipeline.compute_audio_stream_md5(temp_wav)
        if os.path.isfile(temp_wav):
            os.remove(temp_wav)
        cached = pipeline._load_audio_cache(md5)
        if cached and "words" in cached:
            cached_words = cached["words"]
    except Exception as e:
        logger.warning(f"[server] Could not probe audio cache for batch: {e}")

    with _batch_lock:
        _batches[batch_id] = {
            "batch_id": batch_id,
            "file_name": safe_name,
            "total": len(req.clips),
            "job_ids": [],
            "clips": req.clips,
            "created_at": time.time(),
        }

        for i, clip in enumerate(req.clips):
            job_id = f"b_{batch_id[:8]}_{i+1}"
            job_data = {
                "video": safe_name,
                "start": clip.get("start", clip.get("start_time", 0.0)),
                "end": clip.get("end", clip.get("end_time", 30.0)),
                "hook_sentence": clip.get("hook_sentence", clip.get("title", "")),
                "headline_text": clip.get("hook_sentence", clip.get("title", "")),
                "virality_score": clip.get("virality_score", 85),
                "speed": req.speed,
                "brand_logo_path": _resolve_logo_path(req.brand_logo),
                "watermark_logo_path": _resolve_logo_path(req.watermark_logo),
                "words": cached_words,
            }
            render_queue_manager.submit_job(
                _run_render_job,
                job_data,
                job_id=job_id,
                _cleanup_callback=_job_cleanup_callback
            )
            _batches[batch_id]["job_ids"].append(job_id)
            job_ids.append(job_id)

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
                "error": None
            }
        return data

# ---------------------------------------------------------------------------
# Legacy Assets and Canvas Endpoints (Maintained for Backward Compatibility)
# ---------------------------------------------------------------------------

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
        
    return {
        "videos": videos,
        "logos": logos
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

@app.get("/assets/logos/{filename}")
@app.get("/logo/{filename}")
def serve_logo_asset(filename: str):
    fpath = os.path.join(LOGO_DIR, filename)
    if os.path.isfile(fpath):
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
    
    if ext in ['.mp4', '.mov']:
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
        
    return {
        "success": True,
        "filename": filename,
        "type": file_type
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
