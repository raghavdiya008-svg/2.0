#!/usr/bin/env python3
"""
src/dependency_manager.py
-------------------------
Dynamic Dependency Checker and Auto-Installer for the 2.0 Video Pipeline.
Checks for required modules and binaries, and provides auto-installation
capabilities via pip subprocess calls.
"""

import sys
import subprocess
import shutil
import importlib
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger("dependency_manager")

REQUIRED_MODULES = [
    {"name": "ultralytics", "import_name": "ultralytics", "pip_name": "ultralytics", "critical": True},
    {"name": "yt-dlp", "import_name": "yt_dlp", "pip_name": "yt-dlp", "critical": True},
    {"name": "gdown", "import_name": "gdown", "pip_name": "gdown", "critical": True},
    {"name": "opencv-python", "import_name": "cv2", "pip_name": "opencv-python", "critical": True},
    {"name": "torch", "import_name": "torch", "pip_name": "torch", "critical": True},
    {"name": "fastapi", "import_name": "fastapi", "pip_name": "fastapi", "critical": True},
    {"name": "uvicorn", "import_name": "uvicorn", "pip_name": "uvicorn", "critical": True},
    {"name": "whisperx", "import_name": "whisperx", "pip_name": "whisperx", "critical": False},
]


def check_dependencies() -> Dict[str, Any]:
    """
    Checks if required python packages and binaries (ffmpeg, ffprobe) are installed.
    Returns status map with version and availability details.
    """
    results: Dict[str, Any] = {
        "modules": {},
        "binaries": {},
        "all_critical_available": True,
        "missing_modules": [],
    }

    # 1. Check Python modules
    for mod in REQUIRED_MODULES:
        import_name = mod["import_name"]
        pip_name = mod["pip_name"]
        critical = mod["critical"]

        try:
            m = importlib.import_module(import_name)
            ver = getattr(m, "__version__", "installed")
            results["modules"][pip_name] = {
                "installed": True,
                "version": str(ver),
                "critical": critical,
            }
        except (ImportError, OSError):
            results["modules"][pip_name] = {
                "installed": False,
                "version": None,
                "critical": critical,
            }
            results["missing_modules"].append(pip_name)
            if critical:
                results["all_critical_available"] = False

    # 2. Check external binaries
    for binary in ["ffmpeg", "ffprobe"]:
        path = shutil.which(binary)
        results["binaries"][binary] = {
            "available": path is not None,
            "path": path,
        }
        if path is None:
            results["all_critical_available"] = False

    return results


def install_package(pip_name: str, timeout: int = 300) -> bool:
    """
    Installs a single pip package using subprocess.
    """
    print(f"[dependency_manager] Installing {pip_name} via pip...")
    cmd = [sys.executable, "-m", "pip", "install", pip_name]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if res.returncode == 0:
            print(f"[dependency_manager] Successfully installed {pip_name}.")
            return True
        else:
            print(f"[dependency_manager] Error installing {pip_name}: {res.stderr[:200]}")
            return False
    except Exception as e:
        print(f"[dependency_manager] Exception installing {pip_name}: {e}")
        return False


def install_package_stream(pip_name: str):
    """
    Generator that installs a package via subprocess and yields stdout lines
    in real time for streaming to the UI.
    """
    cmd = [sys.executable, "-m", "pip", "install", pip_name]
    yield f"[install] Starting installation of {pip_name} via pip...\n"
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        if proc.stdout:
            for line in iter(proc.stdout.readline, ''):
                if line:
                    yield line
            proc.stdout.close()
        returncode = proc.wait()
        if returncode == 0:
            yield f"[install] Successfully installed {pip_name}.\n"
        else:
            yield f"[install] Error: pip exited with code {returncode}.\n"
    except Exception as e:
        yield f"[install] Exception during install: {e}\n"


def auto_install_missing(target_packages: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Identifies missing packages and installs them automatically.
    Returns a log list of action results.
    """
    status = check_dependencies()
    missing = status["missing_modules"]
    if target_packages:
        missing = [p for p in missing if p in target_packages]

    log: List[Dict[str, Any]] = []
    for pkg in missing:
        success = install_package(pkg)
        log.append({"package": pkg, "installed": success})

    return log


def is_installed(module_or_import_name: str) -> bool:
    """Checks if a python package or import name is currently available."""
    for mod in REQUIRED_MODULES:
        if module_or_import_name in (mod["name"], mod["import_name"], mod["pip_name"]):
            try:
                importlib.import_module(mod["import_name"])
                return True
            except (ImportError, OSError):
                return False

    try:
        importlib.import_module(module_or_import_name)
        return True
    except (ImportError, OSError):
        return False

