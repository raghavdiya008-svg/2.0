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
import shutil
import subprocess
from typing import Dict, Any, Optional

logger = logging.getLogger("media_downloader")

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SRC_DIR, ".."))
INPUTS_DIR = os.path.join(_PROJECT_ROOT, "inputs")


def _ytdlp_cmd() -> list:
    """
    Returns the correct command prefix for yt-dlp.
    Prefers the standalone binary (yt-dlp) if it exists on PATH.
    Falls back to `python -m yt_dlp` when only the Python package is installed
    (common in venv/pip-only environments like Kaggle or Windows without PATH setup).
    """
    if shutil.which("yt-dlp"):
        return ["yt-dlp"]
    return [sys.executable, "-m", "yt_dlp"]


def _ytdlp_extra_args() -> list:
    """
    Returns extra resilience flags for yt-dlp to bypass bot-detection on cloud VMs
    (Kaggle, Colab, Docker).

    - player_client=android  -> Android client skips YouTube's JS n-challenge entirely.
      No Node.js or Deno runtime required. Works reliably on restricted cloud VMs.
    - --cookies cookies.txt  -> injected only when cookies.txt exists in the project root.
    - --no-check-certificates -> avoids TLS errors in restricted cloud networks.
    """
    extra = [
        "--extractor-args", "youtube:player_client=ios,mweb,android,tv,web",
        "--no-check-certificates",
    ]
    cookies_path = os.path.join(_PROJECT_ROOT, "cookies.txt")
    if os.path.isfile(cookies_path):
        extra += ["--cookies", cookies_path]
        logger.info(f"[media_downloader] Using cookies file: {cookies_path}")
    return extra


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
    Downloads a YouTube video using the yt-dlp Python API (primary) with subprocess fallback.

    PRIMARY path: `import yt_dlp` Python API — works on any environment where
    yt-dlp is pip-installed, regardless of whether the binary is on PATH.
    This makes it fully compatible with Kaggle, Colab, Docker, and Windows venvs.

    FALLBACK path: subprocess `yt-dlp` binary or `python -m yt_dlp`.

    Returns dictionary with file_path, file_name, title, duration.
    Reuses existing file if already downloaded in output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Step 1: Probe metadata (non-fatal, used for deduplication only) ───────
    title = "youtube_video"
    duration = 0.0
    try:
        result = subprocess.run(
            _ytdlp_cmd() + _ytdlp_extra_args() + [
                "--no-playlist", "--print", "%(title)s\t%(duration)s", "--no-download", url
            ],
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

    # ── Step 2: Skip download if file already exists ──────────────────────────
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

    print(f"[media_downloader] Downloading YouTube video: {url}")

    # ── Step 3: PRIMARY — yt-dlp Python API (no binary required) ─────────────
    # Works wherever `pip install yt-dlp` was run, no binary on PATH needed.
    _downloaded_ok = False
    try:
        import yt_dlp  # noqa: PLC0415

        cookies_path = os.path.join(_PROJECT_ROOT, "cookies.txt")
        ydl_opts = {
            "format": "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best",
            "merge_output_format": "mp4",
            "outtmpl": safe_filename,
            "noplaylist": True,
            "quiet": False,
            "no_warnings": False,
            "extractor_args": {"youtube": {"player_client": ["ios", "mweb", "android", "tv", "web"]}},
            "nocheckcertificate": True,
        }
        if os.path.isfile(cookies_path):
            ydl_opts["cookiefile"] = cookies_path
            logger.info(f"[media_downloader] Using cookies file: {cookies_path}")

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if info:
                title = info.get("title", title) or title
                duration = float(info.get("duration", duration) or duration)
                # yt-dlp may sanitize the filename differently than we did
                resolved = ydl.prepare_filename(info)
                for ext in (".webm", ".mkv", ".m4a"):
                    resolved = resolved.replace(ext, ".mp4")
                if os.path.isfile(resolved) and resolved != safe_filename:
                    safe_filename = resolved

        # Verify the file exists — yt-dlp sometimes places it with a slightly different name
        if not (os.path.isfile(safe_filename) and os.path.getsize(safe_filename) > 0):
            for f in os.listdir(output_dir):
                candidate = os.path.join(output_dir, f)
                if (f.lower().endswith(".mp4")
                        and os.path.getsize(candidate) > 1024 * 1024
                        and sanitize_filename(os.path.splitext(f)[0])[:20] == safe_base[:20]):
                    safe_filename = candidate
                    break

        if os.path.isfile(safe_filename) and os.path.getsize(safe_filename) > 0:
            _downloaded_ok = True
            print(f"[media_downloader] Python API download complete: {safe_filename}")
        else:
            logger.warning("[media_downloader] Python API reported success but output file not found.")

    except ImportError:
        logger.warning("[media_downloader] yt_dlp not importable — falling back to subprocess.")
    except Exception as e:
        logger.warning(f"[media_downloader] yt-dlp Python API error ({e}) — falling back to subprocess.")

    # ── Step 4: FALLBACK — subprocess (yt-dlp binary or python -m yt_dlp) ────
    if not _downloaded_ok:
        print("[media_downloader] Trying subprocess fallback...")
        cmd = _ytdlp_cmd() + _ytdlp_extra_args() + [
            "-f", "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best",
            "--merge-output-format", "mp4",
            "-o", safe_filename,
            "--no-playlist",
            url,
        ]
        result = subprocess.run(cmd, capture_output=False)

        if result.returncode != 0 or not os.path.isfile(safe_filename):
            print("[media_downloader] Primary format failed — retrying with relaxed format...")
            fallback_cmd = _ytdlp_cmd() + _ytdlp_extra_args() + [
                "-f", "best[ext=mp4]/best",
                "--merge-output-format", "mp4",
                "-o", safe_filename,
                "--no-playlist",
                url,
            ]
            subprocess.run(fallback_cmd, capture_output=False)

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
