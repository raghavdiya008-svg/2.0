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
import html
import logging
import shutil
import subprocess
import urllib.request
from typing import Dict, Any, Optional, Tuple, List

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


def _find_cookies_file() -> Optional[str]:
    """Finds cookies.txt across common root/working/dataset directory locations."""
    candidates = [
        "/kaggle/input/datasets/rapexx/reel-engine-bundle/config/cookies.txt",
        "/kaggle/input/datasets/rapexx/reel-engine-bundle/cookies.txt",
        "/kaggle/input/reel-engine-bundle/config/cookies.txt",
        "/kaggle/input/reel-engine-bundle/cookies.txt",
        "/kaggle/input/pipeline-bundle/config/cookies.txt",
        "/kaggle/input/pipeline-bundle/cookies.txt",
        os.path.join(_PROJECT_ROOT, "cookies.txt"),
        os.path.join(_PROJECT_ROOT, "config", "cookies.txt"),
        os.path.join(INPUTS_DIR, "cookies.txt"),
        os.path.join(os.getcwd(), "cookies.txt"),
        "/kaggle/working/cookies.txt",
        "/kaggle/working/2.0/cookies.txt",
    ]
    env_c = os.environ.get("YOUTUBE_COOKIES_PATH")
    if env_c and os.path.isfile(env_c):
        return env_c

    # Check for raw cookies content in environment variable
    raw_env_c = os.environ.get("YOUTUBE_COOKIES")
    if raw_env_c and len(raw_env_c.strip()) > 20:
        c_dest = os.path.join(_PROJECT_ROOT, "cookies.txt")
        try:
            with open(c_dest, "w", encoding="utf-8") as f:
                f.write(raw_env_c.strip())
            return c_dest
        except Exception:
            pass

    # Check for Kaggle User Secrets (Add-ons -> Secrets -> YOUTUBE_COOKIES)
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore
        secrets = UserSecretsClient()
        sec_c = secrets.get_secret("YOUTUBE_COOKIES")
        if sec_c and len(sec_c.strip()) > 20:
            c_dest = os.path.join(_PROJECT_ROOT, "cookies.txt")
            with open(c_dest, "w", encoding="utf-8") as f:
                f.write(sec_c.strip())
            logger.info("Loaded YouTube cookies from Kaggle User Secrets (YOUTUBE_COOKIES).")
            return c_dest
    except Exception:
        pass

    for path in candidates:
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            return path
    
    # Generic scan in /kaggle/input
    if os.path.isdir("/kaggle/input"):
        for root, _, files in os.walk("/kaggle/input"):
            if "cookies.txt" in files:
                c_path = os.path.join(root, "cookies.txt")
                if os.path.getsize(c_path) > 0:
                    return c_path
    return None


def _ytdlp_extra_args() -> list:
    """
    Returns extra resilience flags for yt-dlp.
    Injects multi-client fallback (android, ios, web) to bypass bot-detection on cloud VMs (Kaggle/Colab).
    Injects cookies.txt automatically when present.
    """
    extra = [
        "--extractor-args", "youtube:player_client=android,ios,web,visionos",
        "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "--no-check-certificates",
        "--retries", "3",
        "--fragment-retries", "3",
    ]
    cookies_path = _find_cookies_file()
    if cookies_path:
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


def _probe_resolution(file_path: str) -> Tuple[int, int]:
    """Returns (width, height) of video using ffprobe."""
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height",
             "-of", "csv=s=x:p=0", file_path],
            capture_output=True, text=True
        )
        if probe.returncode == 0 and probe.stdout.strip():
            w, h = map(int, probe.stdout.strip().split("x"))
            return w, h
    except Exception:
        pass
    return 0, 0


def extract_youtube_id(url: str) -> Optional[str]:
    """Extracts the 11-character YouTube video ID from standard or shortened URLs."""
    if not url:
        return None
    m = re.search(r"(?:v=|\/|youtu\.be\/|shorts\/)([0-9A-Za-z_-]{11})", url.strip())
    return m.group(1) if m else None


def _scrape_youtube_title(url: str) -> Optional[str]:
    """
    Lightweight HTTP scrape of YouTube video title (0.2s, never triggers bot detection).
    Extracts og:title or HTML title tag directly from the page.
    """
    try:
        req = urllib.request.Request(
            url.strip(),
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                "Accept-Language": "en-US,en;q=0.9",
            }
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            content = resp.read().decode("utf-8", errors="ignore")
            # Try OpenGraph og:title
            og_match = re.search(r'<meta\s+property=["\']og:title["\']\s+content=["\'](.*?)["\']', content)
            if og_match:
                candidate = og_match.group(1).strip()
                if candidate and candidate != "YouTube":
                    return html.unescape(candidate)
            # Try standard title tag
            t_match = re.search(r'<title>(.*?)</title>', content)
            if t_match:
                raw_t = t_match.group(1).replace("- YouTube", "").strip()
                if raw_t and raw_t != "YouTube":
                    return html.unescape(raw_t)
    except Exception as e:
        logger.debug(f"[media_downloader] HTML title scrape fallback: {e}")
    return None


def _ensure_in_output_dir(source_path: str, output_dir: str) -> str:
    """Copies source_path to output_dir if not already located there."""
    src_abs = os.path.abspath(source_path)
    out_abs = os.path.abspath(output_dir)
    if os.path.dirname(src_abs) == out_abs:
        return source_path
    os.makedirs(output_dir, exist_ok=True)
    dest_path = os.path.join(output_dir, os.path.basename(source_path))
    if not os.path.isfile(dest_path) or os.path.getsize(dest_path) != os.path.getsize(source_path):
        print(f"[media_downloader] Copying pre-loaded dataset file to inputs: {source_path} -> {dest_path}")
        shutil.copy2(source_path, dest_path)
    return dest_path


def _find_cached_video(output_dir: str, title: Optional[str] = None, v_id: Optional[str] = None) -> Optional[str]:
    """
    Searches for an already-downloaded or pre-loaded video matching the title or video ID.
    Checks output_dir, project inputs, and Kaggle dataset input directories.
    If found in a read-only directory (e.g. /kaggle/input), copies it to output_dir.
    """
    search_dirs = [output_dir, INPUTS_DIR]
    kaggle_dirs = [
        "/kaggle/input/datasets/rapexx/reel-engine-bundle/inputs",
        "/kaggle/input/datasets/rapexx/reel-engine-bundle",
        "/kaggle/input/reel-engine-bundle/inputs",
        "/kaggle/input/reel-engine-bundle",
        "/kaggle/input/pipeline-bundle/inputs",
        "/kaggle/input/pipeline-bundle",
    ]
    for kd in kaggle_dirs:
        if os.path.isdir(kd) and kd not in search_dirs:
            search_dirs.append(kd)

    safe_base = sanitize_filename(title or "").lower() if title else ""
    title_words = set(re.findall(r"[a-zA-Z0-9]{3,}", (title or "").lower())) if title else set()
    stop_words = {"youtube", "video", "official", "clip", "full", "audio", "watch"}
    sig_words = title_words - stop_words

    for sdir in search_dirs:
        if not os.path.isdir(sdir):
            continue
        try:
            for fname in os.listdir(sdir):
                if not fname.lower().endswith((".mp4", ".mov", ".mkv", ".webm")):
                    continue
                cand_path = os.path.join(sdir, fname)
                if not (os.path.isfile(cand_path) and os.path.getsize(cand_path) > 1024 * 1024):
                    continue

                f_lower = fname.lower()
                f_base = sanitize_filename(os.path.splitext(fname)[0]).lower()

                # 1. Match by video ID
                if v_id and len(v_id) == 11 and v_id.lower() in f_lower:
                    logger.info(f"[media_downloader] Video ID matched cached file: {cand_path}")
                    return _ensure_in_output_dir(cand_path, output_dir)

                # 2. Match by sanitized title
                if safe_base and safe_base != "youtube_video":
                    if f_base == safe_base or f_base.startswith(safe_base[:30]) or safe_base.startswith(f_base[:30]):
                        logger.info(f"[media_downloader] Title matched cached file: {cand_path}")
                        return _ensure_in_output_dir(cand_path, output_dir)

                # 3. Match by significant word overlap (>= 3 words or >= 50% match)
                if sig_words:
                    cand_words = set(re.findall(r"[a-zA-Z0-9]{3,}", f_lower))
                    overlap = sig_words & cand_words
                    if len(overlap) >= 3 or (len(sig_words) <= 3 and len(overlap) >= 2):
                        logger.info(f"[media_downloader] Significant word overlap ({overlap}) matched cached file: {cand_path}")
                        return _ensure_in_output_dir(cand_path, output_dir)
        except Exception as e:
            logger.debug(f"[media_downloader] Error scanning directory {sdir}: {e}")

    return None


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
    v_id = extract_youtube_id(url)

    # ── Step 1: Probe metadata (HTML scrape first, then yt-dlp android client) ──
    title = _scrape_youtube_title(url) or "youtube_video"
    duration = 0.0

    # If title is still default, try yt-dlp metadata probe with android client
    if title == "youtube_video":
        try:
            result = subprocess.run(
                _ytdlp_cmd() + _ytdlp_extra_args() + [
                    "--no-playlist", "--print", "%(title)s\t%(duration)s", "--no-download", url
                ],
                capture_output=True, text=True, timeout=25
            )
            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split("\t")
                title = parts[0].strip() if parts else title
                duration = float(parts[1].strip()) if len(parts) > 1 else 0.0
        except Exception as e:
            logger.debug(f"[media_downloader] Metadata pre-check non-fatal: {e}")

    # ── Step 2: Check cache / pre-loaded files before making network download ──
    cached_file = _find_cached_video(output_dir, title=title, v_id=v_id)
    if cached_file and os.path.isfile(cached_file) and os.path.getsize(cached_file) > 1024 * 1024:
        final_dur = duration or _probe_duration(cached_file)
        w, h = _probe_resolution(cached_file)
        print(f"[media_downloader] Reusing cached/pre-loaded file: {cached_file} ({final_dur:.1f}s, {w}x{h}) — skipping network download.")
        return {
            "file_path": cached_file,
            "file_name": os.path.basename(cached_file),
            "title": title if title != "youtube_video" else os.path.splitext(os.path.basename(cached_file))[0],
            "duration": final_dur,
            "width": w,
            "height": h,
            "source": "youtube",
            "url": url,
        }

    safe_base = sanitize_filename(title)
    safe_filename = os.path.join(output_dir, f"{safe_base}.mp4")

    print(f"[media_downloader] Downloading YouTube video: {url}")

    # ── Step 3: PRIMARY — yt-dlp Python API with resilient player clients ────
    _downloaded_ok = False
    last_error_msg = ""
    max_h = os.environ.get("MAX_DOWNLOAD_HEIGHT", "1440")
    format_spec = f"bestvideo[height<={max_h}]+bestaudio/bestvideo+bestaudio/best[height<={max_h}]/best"

    try:
        import yt_dlp  # noqa: PLC0415

        cookies_path = _find_cookies_file()
        ydl_opts = {
            "format": format_spec,
            "merge_output_format": "mp4",
            "outtmpl": safe_filename,
            "noplaylist": True,
            "quiet": False,
            "no_warnings": False,
            "extractor_args": {"youtube": {"player_client": ["android", "ios", "web", "visionos"]}},
            "nocheckcertificate": True,
        }
        if cookies_path:
            ydl_opts["cookiefile"] = cookies_path
            logger.info(f"[media_downloader] Using cookies file: {cookies_path}")

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if info:
                title = info.get("title", title) or title
                duration = float(info.get("duration", duration) or duration)
                resolved = ydl.prepare_filename(info)
                for ext in (".webm", ".mkv", ".m4a"):
                    resolved = resolved.replace(ext, ".mp4")
                if os.path.isfile(resolved) and resolved != safe_filename:
                    safe_filename = resolved

        # Verify output file
        if not (os.path.isfile(safe_filename) and os.path.getsize(safe_filename) > 0):
            for f in os.listdir(output_dir):
                candidate = os.path.join(output_dir, f)
                if (f.lower().endswith(".mp4")
                        and os.path.getsize(candidate) > 1024 * 1024
                        and (sanitize_filename(os.path.splitext(f)[0])[:20] == safe_base[:20]
                             or (v_id and v_id in f))):
                    safe_filename = candidate
                    break

        if os.path.isfile(safe_filename) and os.path.getsize(safe_filename) > 0:
            _downloaded_ok = True
            print(f"[media_downloader] Python API download complete: {safe_filename}")

    except ImportError:
        logger.warning("[media_downloader] yt_dlp not importable — falling back to subprocess.")
    except Exception as e:
        last_error_msg = str(e)
        logger.warning(f"[media_downloader] yt-dlp primary error ({e}) — attempting android single-stream client fallback.")

    # ── Step 3.5: Client fallback (android client with single-stream format) ─
    if not _downloaded_ok:
        try:
            import yt_dlp  # noqa: PLC0415
            print("[media_downloader] Retrying with Android client single-stream fallback...")
            ydl_opts_android = {
                "format": "best[ext=mp4]/best",
                "merge_output_format": "mp4",
                "outtmpl": safe_filename,
                "noplaylist": True,
                "quiet": False,
                "no_warnings": False,
                "extractor_args": {"youtube": {"player_client": ["android"]}},
                "nocheckcertificate": True,
            }
            cookies_path = _find_cookies_file()
            if cookies_path:
                ydl_opts_android["cookiefile"] = cookies_path
            with yt_dlp.YoutubeDL(ydl_opts_android) as ydl:
                info = ydl.extract_info(url, download=True)
                if info:
                    title = info.get("title", title) or title
                    duration = float(info.get("duration", duration) or duration)
                    resolved = ydl.prepare_filename(info)
                    for ext in (".webm", ".mkv", ".m4a"):
                        resolved = resolved.replace(ext, ".mp4")
                    if os.path.isfile(resolved) and resolved != safe_filename:
                        safe_filename = resolved

            if not (os.path.isfile(safe_filename) and os.path.getsize(safe_filename) > 0):
                for f in os.listdir(output_dir):
                    candidate = os.path.join(output_dir, f)
                    if (f.lower().endswith(".mp4")
                            and os.path.getsize(candidate) > 1024 * 1024
                            and (sanitize_filename(os.path.splitext(f)[0])[:20] == safe_base[:20]
                                 or (v_id and v_id in f))):
                        safe_filename = candidate
                        break

            if os.path.isfile(safe_filename) and os.path.getsize(safe_filename) > 0:
                _downloaded_ok = True
                print(f"[media_downloader] Android client fallback download complete: {safe_filename}")
        except Exception as android_err:
            last_error_msg = str(android_err)
            logger.warning(f"[media_downloader] Android client fallback error: {android_err}")

    # ── Step 4: FALLBACK — subprocess with android,ios,web client ────────────
    if not _downloaded_ok:
        print("[media_downloader] Trying subprocess fallback with multi-client args...")
        cmd = _ytdlp_cmd() + _ytdlp_extra_args() + [
            "-f", format_spec,
            "--merge-output-format", "mp4",
            "-o", safe_filename,
            "--no-playlist",
            url,
        ]
        result = subprocess.run(cmd, capture_output=False)

        if result.returncode != 0 or not os.path.isfile(safe_filename):
            print("[media_downloader] Multi-client subprocess failed — retrying with standard single stream...")
            fallback_cmd = _ytdlp_cmd() + [
                "-f", "best[ext=mp4]/best",
                "--merge-output-format", "mp4",
                "-o", safe_filename,
                "--no-playlist",
                url,
            ]
            cookies_path = _find_cookies_file()
            if cookies_path:
                fallback_cmd.extend(["--cookies", cookies_path])
            subprocess.run(fallback_cmd, capture_output=False)

    if not os.path.isfile(safe_filename):
        # Scan output_dir to see what local videos exist
        existing_local = [f for f in os.listdir(output_dir) if f.lower().endswith((".mp4", ".mov", ".mkv"))]
        local_hint = f" You can select a pre-loaded local video from the dropdown (available: {', '.join(existing_local[:3])})." if existing_local else ""
        err_hint = f" (Details: {last_error_msg})" if last_error_msg else ""
        raise RuntimeError(
            f"yt-dlp failed to download '{url}'{err_hint}.{local_hint} "
            "To bypass YouTube bot challenges on cloud VMs, add your YouTube cookies "
            "as 'cookies.txt' in the project root or Kaggle Secrets, or use a Google Drive URL / upload a local video."
        )

    final_duration = duration or _probe_duration(safe_filename)
    res_w, res_h = _probe_resolution(safe_filename)
    size_mb = os.path.getsize(safe_filename) / (1024 * 1024)
    print(f"[media_downloader] Download complete: {safe_filename} ({size_mb:.1f} MB, {final_duration:.1f}s, {res_w}x{res_h})")
    if res_h > 0 and res_h < 720:
        logger.warning(
            f"[media_downloader] Notice: Downloaded video resolution is {res_w}x{res_h} (<720p). "
            "For highest clarity 9:16 reels, import a 1080p source file or Google Drive link."
        )

    return {
        "file_path": safe_filename,
        "file_name": os.path.basename(safe_filename),
        "title": title,
        "duration": final_duration,
        "width": res_w,
        "height": res_h,
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
