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


def ensure_executable(path: str) -> bool:
    """
    Ensures the specified binary has executable permissions on Unix-like systems.
    Returns True if the file exists and is executable.
    """
    if not path or not os.path.isfile(path):
        return False
    if sys.platform.startswith("win"):
        return True
    try:
        current_mode = os.stat(path).st_mode
        os.chmod(path, current_mode | 0o755)
    except Exception:
        pass
    try:
        subprocess.run(["chmod", "+x", path], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass
    return os.access(path, os.X_OK)


def get_cloudflared_binary() -> Optional[str]:
    """
    Locates or automatically downloads the appropriate standalone cloudflared binary.
    Ensures executable permissions are set on Linux/macOS before returning.
    Returns path to the executable or None on failure.
    """
    is_win = sys.platform.startswith("win")
    bin_name = "cloudflared.exe" if is_win else "cloudflared"

    # 1. Check system PATH
    system_path = shutil.which("cloudflared")
    if system_path:
        if is_win or ensure_executable(system_path):
            return system_path

    # 2. Check candidate locations
    os.makedirs(_TEMP_DIR, exist_ok=True)
    local_temp_bin = os.path.join(_TEMP_DIR, bin_name)

    candidates = [local_temp_bin]
    if not is_win:
        candidates.extend(["/tmp/cloudflared", "/usr/local/bin/cloudflared"])

    for cand in candidates:
        if os.path.isfile(cand) and os.path.getsize(cand) > 10000:
            if is_win or ensure_executable(cand):
                return cand
            # If filesystem has noexec, try copying to /tmp/cloudflared
            if not is_win and cand != "/tmp/cloudflared":
                try:
                    tmp_bin = "/tmp/cloudflared"
                    shutil.copyfile(cand, tmp_bin)
                    if ensure_executable(tmp_bin):
                        return tmp_bin
                except Exception:
                    pass

    # 3. Download standalone binary based on OS/Arch
    sys_arch = platform.machine().lower()
    if is_win:
        url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
    elif sys.platform.startswith("darwin"):
        url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-arm64" if "arm" in sys_arch else "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-amd64"
    else:  # Linux (Colab / Kaggle)
        url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"

    target_paths = [local_temp_bin]
    if not is_win:
        target_paths.append("/tmp/cloudflared")

    for target in target_paths:
        print(f"[tunnel] Downloading Cloudflare tunnel binary to {target}...")
        try:
            os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
            urllib.request.urlretrieve(url, target)
            if is_win or ensure_executable(target):
                print(f"[tunnel] Cloudflared ready at: {target}")
                return target
        except Exception as e:
            print(f"[tunnel] Warning: Failed writing {target}: {e}")
            continue

    return None


def start_cloudflare_tunnel(port: int = 5000, timeout: int = 35) -> Tuple[Optional[str], Optional[subprocess.Popen]]:
    """
    Launches a Cloudflare Quick Tunnel pointing to http://127.0.0.1:{port}.
    Returns (public_url, process_handle).
    """
    binary = get_cloudflared_binary()
    if not binary:
        print("[tunnel] Warning: cloudflared binary unavailable.")
        return None, None

    ensure_executable(binary)
    cmd = [binary, "tunnel", "--url", f"http://127.0.0.1:{port}"]

    proc = None
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
    except PermissionError as e:
        print(f"[tunnel] Permission denied executing {binary}: {e}. Retrying via /tmp/cloudflared...")
        if not sys.platform.startswith("win"):
            tmp_bin = "/tmp/cloudflared"
            try:
                if binary != tmp_bin:
                    shutil.copyfile(binary, tmp_bin)
                ensure_executable(tmp_bin)
                cmd = [tmp_bin, "tunnel", "--url", f"http://127.0.0.1:{port}"]
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
            except Exception as e2:
                print(f"[tunnel] Fallback execution failed: {e2}")
                return None, None
        else:
            return None, None
    except Exception as e:
        print(f"[tunnel] Failed to start cloudflared process: {e}")
        return None, None

    public_url: Optional[str] = None
    url_pattern = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")

    # Read output stream concurrently using a daemon thread to avoid pipe blocking
    def read_stream():
        nonlocal public_url
        if not proc or not proc.stdout:
            return
        try:
            for line in iter(proc.stdout.readline, ""):
                if not line:
                    break
                match = url_pattern.search(line)
                if match and not public_url:
                    public_url = match.group(0)
                    break
        except Exception:
            pass

    reader_thread = threading.Thread(target=read_stream, daemon=True)
    reader_thread.start()

    # Wait up to timeout seconds for public_url
    start_time = time.time()
    while time.time() - start_time < timeout:
        if public_url:
            break
        if proc.poll() is not None:
            print(f"[tunnel] Process terminated unexpectedly with code {proc.returncode}")
            break
        time.sleep(0.5)

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
