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
    """Removes problematic filesystem characters."""
    clean = re.sub(r'[\\/*?:"<>|]', "", name)
    clean = clean.replace(" ", "_").strip("._")
    return clean[:80] or "video"


def download_youtube_video(url: str, output_dir: str = INPUTS_DIR) -> Dict[str, Any]:
    """
    Downloads a YouTube video via yt_dlp.
    Returns dictionary with file_path, file_name, title, duration.
    """
    import yt_dlp

    os.makedirs(output_dir, exist_ok=True)
    out_template = os.path.join(output_dir, "%(title).80s.%(ext)s")

    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": out_template,
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": False,
        "no_warnings": False,
    }

    print(f"[media_downloader] Downloading YouTube video from {url}...")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get("title", "youtube_video")
        duration = float(info.get("duration", 0.0))
        ext = info.get("ext", "mp4")

        filename = ydl.prepare_filename(info)
        # In case merge format renamed to .mp4
        if not os.path.isfile(filename):
            base_no_ext = os.path.splitext(filename)[0]
            if os.path.isfile(f"{base_no_ext}.mp4"):
                filename = f"{base_no_ext}.mp4"

        print(f"[media_downloader] YouTube download complete: {filename} ({duration:.1f}s)")
        return {
            "file_path": filename,
            "file_name": os.path.basename(filename),
            "title": title,
            "duration": duration,
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
