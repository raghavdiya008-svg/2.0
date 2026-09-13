#!/usr/bin/env python3
"""
src/licensing/hwid.py
---------------------
Hardware Machine Fingerprinting module.
Collects CPU ID/model, motherboard/system UUID, and MAC address via standard
libraries and OS queries, producing an immutable SHA-256 machine identifier.
Cross-platform compatible across Windows host and Linux (Docker / Colab / Kaggle).
"""

import os
import sys
import uuid
import platform
import hashlib
import subprocess
from typing import Dict, Any


def _get_cpu_id() -> str:
    """Retrieves CPU processor string / identifier."""
    cpu_str = platform.processor() or ""
    machine = platform.machine() or ""
    
    if sys.platform.startswith("win"):
        # On Windows, check environment or registry/wmic
        env_cpu = os.environ.get("PROCESSOR_IDENTIFIER", "")
        if env_cpu:
            return f"{cpu_str}_{machine}_{env_cpu}"
        try:
            out = subprocess.check_output(
                ["wmic", "cpu", "get", "processorid"],
                text=True, stderr=subprocess.DEVNULL, timeout=2
            ).splitlines()
            valid = [line.strip() for line in out if line.strip() and "processorid" not in line.lower()]
            if valid:
                return f"{cpu_str}_{machine}_{valid[0]}"
        except Exception:
            pass
    elif sys.platform.startswith("linux"):
        # On Linux, inspect /proc/cpuinfo
        try:
            if os.path.exists("/proc/cpuinfo"):
                with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if "model name" in line.lower() or "hardware" in line.lower():
                            return f"{cpu_str}_{machine}_{line.split(':', 1)[-1].strip()}"
        except Exception:
            pass

    return f"{cpu_str}_{machine}_{platform.version()}"


def _get_system_uuid() -> str:
    """Retrieves Motherboard / System UUID or container machine-id."""
    if sys.platform.startswith("win"):
        # First check Windows MachineGuid registry (instant, non-blocking, deterministic)
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Cryptography") as key:
                guid, _ = winreg.QueryValueEx(key, "MachineGuid")
                if guid and len(str(guid).strip()) > 8:
                    return str(guid).strip()
        except Exception:
            pass

        # Windows system UUID via powershell / wmic
        try:
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_ComputerSystemProduct).UUID"],
                text=True, stderr=subprocess.DEVNULL, timeout=5
            ).strip()
            if out and "error" not in out.lower() and len(out) > 8:
                return out
        except Exception:
            pass

        try:
            out = subprocess.check_output(
                ["wmic", "csproduct", "get", "uuid"],
                text=True, stderr=subprocess.DEVNULL, timeout=5
            ).splitlines()
            valid = [line.strip() for line in out if line.strip() and "uuid" not in line.lower()]
            if valid:
                return valid[0]
        except Exception:
            pass

        # Fallback to USERNAME + COMPUTERNAME
        comp = os.environ.get("COMPUTERNAME", "")
        user = os.environ.get("USERNAME", "")
        return f"WIN_FALLBACK_{comp}_{user}"

    elif sys.platform.startswith("linux"):
        # Linux / container system UUID
        for path in ["/sys/class/dmi/id/product_uuid", "/etc/machine-id", "/var/lib/dbus/machine-id"]:
            try:
                if os.path.exists(path):
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        val = f.read().strip()
                        if val:
                            return val
            except Exception:
                continue

    return f"GENERIC_UUID_{platform.node()}"


def _get_mac_address() -> str:
    """Retrieves primary MAC address via uuid.getnode()."""
    try:
        node = uuid.getnode()
        # Convert integer to standard MAC format
        mac = ":".join(f"{(node >> (8 * i)) & 0xFF:02x}" for i in reversed(range(6)))
        return mac
    except Exception:
        return "00:00:00:00:00:00"


def get_hardware_details() -> Dict[str, Any]:
    """Returns a structured dictionary of raw hardware parameters."""
    return {
        "os": platform.system(),
        "node": platform.node(),
        "cpu": _get_cpu_id(),
        "uuid": _get_system_uuid(),
        "mac": _get_mac_address(),
    }


_CACHED_HWID = None


def get_machine_hwid(force_refresh: bool = False) -> str:
    """
    Computes an immutable, deterministic SHA-256 hardware identifier (HWID)
    bound to the current machine.
    """
    global _CACHED_HWID
    if _CACHED_HWID is not None and not force_refresh:
        return _CACHED_HWID

    details = get_hardware_details()
    # Construct normalized deterministic fingerprint seed
    fingerprint_raw = (
        f"OS={details['os']}|"
        f"CPU={details['cpu']}|"
        f"UUID={details['uuid']}|"
        f"MAC={details['mac']}"
    )
    _CACHED_HWID = hashlib.sha256(fingerprint_raw.encode("utf-8")).hexdigest()
    return _CACHED_HWID


if __name__ == "__main__":
    hwid = get_machine_hwid()
    print(f"Machine HWID: {hwid}")
    print(f"Hardware Details: {get_hardware_details()}")
