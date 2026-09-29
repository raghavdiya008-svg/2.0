#!/usr/bin/env python3
"""
src/media_downloader.py
-----------------------
Downloads media from YouTube and Google Drive URLs using yt-dlp and gdown.
Saves downloaded files into the inputs/ directory for pipeline processing.
"""

import os
import re
import sys
import json
import logging
import subprocess
from typing import Dict, Any, Optional

logger = logging.getLogger("media_downloader")

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SRC_DIR, ".."))
INPUTS_DIR = os.path.join(_PROJECT_ROOT, "inputs")


def is_youtube_url(url: str) -> bool:
    """Checks if the URL is a YouTube URL."""
    if not url:
        return False
    patterns = [
        r"^(https?://)?(www\.|m\.|music\.)?(youtube\.com|youtu\.be)/.+",
        r"^(https?://)?(www\.|m\.)?youtube\.com/shorts/.+",
    ]
    return any(re.match(p, url.strip()) for p in patterns)


def is_gdrive_url(url: str) -> bool:
    """Checks if the URL is a Google Drive URL."""
    if not url:
        return False
    return "drive.google.com" in url.strip() or "docs.google.com" in url.strip()


def sanitize_filename(name: str) -> str:
    """Removes problematic filesystem characters and normalizes for clean cross-platform safety."""
    clean = re.sub(r'[^a-zA-Z0-9_.-]', '_', name)
    clean = re.sub(r'_+', '_', clean).strip('._')
    return clean[:80] or "video"


def _probe_duration(file_path: str) -> float:
    """Returns video duration in seconds using ffprobe."""
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", file_path],
            capture_output=True, text=True
        )
        if probe.returncode == 0 and probe.stdout.strip():
            return float(probe.stdout.strip())
    except Exception:
        pass
    return 0.0


def download_youtube_video(url: str, output_dir: str = INPUTS_DIR) -> Dict[str, Any]:
    """
    Downloads a YouTube video via the yt-dlp CLI subprocess (not the Python library).
    Using the CLI binary bypasses the bot-detection that blocks Python API callers
    on cloud VMs (Kaggle, Colab, Docker).

    Returns dictionary with file_path, file_name, title, duration.
    Reuses existing file if already downloaded in output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Step 1: Probe metadata to check deduplication ────────────────────────
    title = "youtube_video"
    duration = 0.0
    try:
        result = subprocess.run(
            ["yt-dlp", "--no-playlist", "--print", "%(title)s\t%(duration)s", "--no-download", url],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split("\t")
            title = parts[0].strip() if parts else title
            duration = float(parts[1].strip()) if len(parts) > 1 else 0.0
    except Exception as e:
        logger.debug(f"[media_downloader] Metadata pre-check non-fatal: {e}")

    safe_base = sanitize_filename(title)
    safe_filename = os.path.join(output_dir, f"{safe_base}.mp4")

    # ── Step 2: Skip download if file already exists ─────────────────────────
    if os.path.isfile(safe_filename) and os.path.getsize(safe_filename) > 1024 * 1024:
        print(f"[media_downloader] Already downloaded: {safe_filename} ({duration:.1f}s) — skipping.")
        return {
            "file_path": safe_filename,
            "file_name": os.path.basename(safe_filename),
            "title": title,
            "duration": duration or _probe_duration(safe_filename),
            "source": "youtube",
            "url": url,
        }

    # Fuzzy match existing files in output_dir
    if os.path.isdir(output_dir):
        for f in os.listdir(output_dir):
            if f.lower().endswith(".mp4"):
                f_base = sanitize_filename(os.path.splitext(f)[0])
                if f_base == safe_base or f.startswith(safe_base[:30]):
                    candidate = os.path.join(output_dir, f)
                    if os.path.isfile(candidate) and os.path.getsize(candidate) > 1024 * 1024:
                        print(f"[media_downloader] Found cached file: {candidate} — skipping re-download.")
                        return {
                            "file_path": candidate,
                            "file_name": f,
                            "title": title,
                            "duration": duration or _probe_duration(candidate),
                            "source": "youtube",
                            "url": url,
                        }

    # ── Step 3: Download via yt-dlp CLI subprocess (user's proven approach) ──
    print(f"[media_downloader] Downloading YouTube video: {url}")
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", safe_filename,
        "--no-playlist",
        url,
    ]

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0 or not os.path.isfile(safe_filename):
        # Fallback: try without height constraint (catches age-restricted or geo-locked videos)
        print("[media_downloader] Primary format failed — retrying with relaxed format...")
        fallback_cmd = [
            "yt-dlp",
            "-f", "best[ext=mp4]/best",
            "--merge-output-format", "mp4",
            "-o", safe_filename,
            "--no-playlist",
            url,
        ]
        result = subprocess.run(fallback_cmd, capture_output=False)

    if not os.path.isfile(safe_filename):
        raise RuntimeError(
            f"yt-dlp failed to download '{url}'. "
            "If you see a bot/cookie error on Kaggle, export your YouTube cookies "
            "as 'cookies.txt' and place them in the project root."
        )

    final_duration = duration or _probe_duration(safe_filename)
    size_mb = os.path.getsize(safe_filename) / (1024 * 1024)
    print(f"[media_downloader] Download complete: {safe_filename} ({size_mb:.1f} MB, {final_duration:.1f}s)")

    return {
        "file_path": safe_filename,
        "file_name": os.path.basename(safe_filename),
        "title": title,
        "duration": final_duration,
        "source": "youtube",
        "url": url,
    }


def download_gdrive_video(url: str, output_dir: str = INPUTS_DIR) -> Dict[str, Any]:
    """
    Downloads a Google Drive file via gdown.
    """
    import gdown

    os.makedirs(output_dir, exist_ok=True)
    print(f"[media_downloader] Downloading Google Drive media from {url}...")
    
    # Let gdown download with fuzzy matching
    out_path = gdown.download(url, output=output_dir + os.sep, quiet=False, fuzzy=True)
    if not out_path or not os.path.isfile(out_path):
        raise RuntimeError(f"Failed to download Google Drive media from {url}")

    # Inspect downloaded file duration if video
    duration = 0.0
    try:
        probe = subprocess.run([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "csv=p=0", out_path
        ], capture_output=True, text=True)
        if probe.returncode == 0 and probe.stdout.strip():
            duration = float(probe.stdout.strip())
    except Exception as e:
        logger.warning(f"[media_downloader] ffprobe duration probe failed on {out_path}: {e}")

    file_name = os.path.basename(out_path)
    print(f"[media_downloader] GDrive download complete: {out_path} ({duration:.1f}s)")
    return {
        "file_path": out_path,
        "file_name": file_name,
        "title": os.path.splitext(file_name)[0],
        "duration": duration,
        "source": "gdrive",
        "url": url,
    }


def download_media(url_or_path: str, output_dir: str = INPUTS_DIR) -> Dict[str, Any]:
    """
    Main ingestion resolver. Handles YouTube URLs, GDrive URLs, or local filenames.
    """
    stripped = url_or_path.strip()

    if is_youtube_url(stripped):
        return download_youtube_video(stripped, output_dir=output_dir)
    elif is_gdrive_url(stripped):
        return download_gdrive_video(stripped, output_dir=output_dir)
    elif stripped.startswith("http://") or stripped.startswith("https://"):
        raise ValueError(f"Unsupported remote URL: {stripped}. Only YouTube and Google Drive URLs are supported.")
    else:
        # Check if it's already a local file in inputs/ or absolute path
        candidate = os.path.join(output_dir, stripped) if not os.path.isabs(stripped) else stripped
        if os.path.isfile(candidate):
            duration = 0.0
            try:
                probe = subprocess.run([
                    "ffprobe", "-v", "error", "-show_entries", "format=duration",
                    "-of", "csv=p=0", candidate
                ], capture_output=True, text=True)
                if probe.returncode == 0 and probe.stdout.strip():
                    duration = float(probe.stdout.strip())
            except Exception as e:
                logger.warning(f"[media_downloader] ffprobe duration probe failed on {candidate}: {e}")
            return {
                "file_path": candidate,
                "file_name": os.path.basename(candidate),
                "title": os.path.splitext(os.path.basename(candidate))[0],
                "duration": duration,
                "source": "local",
                "url": url_or_path,
            }
        else:
            raise FileNotFoundError(f"Source URL or local file could not be resolved: {url_or_path}")
