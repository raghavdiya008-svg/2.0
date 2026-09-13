#!/usr/bin/env python3
"""
src/tunnel.py
-------------
Automated Zero-Config Cloudflare Quick Tunnel Manager.
Enables instant public HTTPS access in Google Colab, Kaggle, or Localhost
without requiring account login, tokens, or custom domains.
"""

import os
import sys
import re
import time
import shutil
import platform
import subprocess
import urllib.request
import threading
from typing import Optional, Tuple

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.abspath(os.path.join(_SRC_DIR, ".."))
_TEMP_DIR = os.path.join(_ROOT_DIR, "temp")


def is_cloud_environment() -> bool:
    """Detects if currently executing inside Google Colab or Kaggle."""
    if "COLAB_GPU" in os.environ or "google.colab" in sys.modules or os.path.exists("/content"):
        return True
    if os.path.exists("/kaggle") or "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
        return True
    return False


def get_cloudflared_binary() -> Optional[str]:
    """
    Locates or automatically downloads the appropriate standalone cloudflared binary.
    Returns path to the executable or None on failure.
    """
    # 1. Check system PATH
    system_path = shutil.which("cloudflared")
    if system_path:
        return system_path

    # 2. Check local temp directory
    os.makedirs(_TEMP_DIR, exist_ok=True)
    is_win = sys.platform.startswith("win")
    bin_name = "cloudflared.exe" if is_win else "cloudflared"
    local_bin = os.path.join(_TEMP_DIR, bin_name)

    if os.path.isfile(local_bin) and os.path.getsize(local_bin) > 10000:
        return local_bin

    # 3. Download standalone binary based on OS/Arch
    sys_arch = platform.machine().lower()
    if is_win:
        url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
    elif sys.platform.startswith("darwin"):
        url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-arm64" if "arm" in sys_arch else "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-amd64"
    else:  # Linux (Colab / Kaggle)
        url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"

    print(f"[tunnel] Downloading Cloudflare tunnel binary ({bin_name})...")
    try:
        urllib.request.urlretrieve(url, local_bin)
        if not is_win:
            os.chmod(local_bin, 0o755)
        print(f"[tunnel] Cloudflared ready at: {local_bin}")
        return local_bin
    except Exception as e:
        print(f"[tunnel] Failed to download cloudflared: {e}")
        return None


def start_cloudflare_tunnel(port: int = 5000, timeout: int = 30) -> Tuple[Optional[str], Optional[subprocess.Popen]]:
    """
    Launches a Cloudflare Quick Tunnel pointing to http://127.0.0.1:{port}.
    Returns (public_url, process_handle).
    """
    binary = get_cloudflared_binary()
    if not binary:
        print("[tunnel] Warning: cloudflared binary unavailable.")
        return None, None

    cmd = [binary, "tunnel", "--url", f"http://127.0.0.1:{port}"]

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
    except Exception as e:
        print(f"[tunnel] Failed to start cloudflared process: {e}")
        return None, None

    public_url: Optional[str] = None
    url_pattern = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")
    start_time = time.time()

    # Read output stream until tunnel URL is found
    while time.time() - start_time < timeout:
        if proc.poll() is not None:
            print(f"[tunnel] Process terminated unexpectedly with code {proc.returncode}")
            break

        line = proc.stdout.readline() if proc.stdout else ""
        if not line and proc.poll() is not None:
            break

        match = url_pattern.search(line)
        if match:
            public_url = match.group(0)
            break

    if public_url:
        return public_url, proc
    else:
        print("[tunnel] Tunnel startup timed out without finding a trycloudflare.com URL.")
        try:
            proc.terminate()
        except Exception:
            pass
        return None, None


if __name__ == "__main__":
    url, proc = start_cloudflare_tunnel(5000)
    if url:
        print(f"Public URL: {url}")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            if proc:
                proc.terminate()
