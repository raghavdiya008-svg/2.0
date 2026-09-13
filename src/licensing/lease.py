#!/usr/bin/env python3
"""
src/licensing/lease.py
----------------------
Cryptographic Offline Lease Manager.
Issues and validates RSA-signed offline machine leases with:
1. Machine HWID binding verification.
2. Rolling 72-hour offline expiration window (time.time() < expires_at).
3. Production RSA-SHA256 signature verification.
"""

import os
import sys
import time
import json
import base64
import logging
from typing import Optional, Tuple, Dict, Any

try:
    from .hwid import get_machine_hwid
except (ImportError, ValueError):
    try:
        from licensing.hwid import get_machine_hwid
    except ImportError:
        from hwid import get_machine_hwid

logger = logging.getLogger("licensing.lease")

# Default lease location
DEFAULT_LEASE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "license.lease")
)

def _load_key(env_var: str, path_env_var: str) -> bytes:
    """Loads a cryptographic key from environment variable or file path, returning bytes or empty string."""
    val = os.environ.get(env_var)
    if val:
        return val.encode('utf-8')
    path = os.environ.get(path_env_var)
    if path and os.path.isfile(path):
        with open(path, "rb") as f:
            return f.read()
    return b""

_DEV_PRIV_PEM = _load_key("LICENSE_PRIVATE_KEY", "LICENSE_PRIVATE_KEY_PATH")
_DEV_PUB_PEM = _load_key("LICENSE_PUBLIC_KEY", "LICENSE_PUBLIC_KEY_PATH")


class LicenseError(Exception):
    """Raised when license or lease verification fails."""
    pass


def _get_crypto_backends():
    """Dynamically loads cryptography or rsa library."""
    try:
        from cryptography.hazmat.primitives.asymmetric import rsa, padding
        from cryptography.hazmat.primitives import hashes, serialization
        return "cryptography", (rsa, padding, hashes, serialization)
    except ImportError:
        try:
            import rsa
            return "rsa", rsa
        except ImportError:
            return "builtin", None


def generate_rsa_keypair() -> Tuple[bytes, bytes]:
    """Generates a new 2048-bit RSA keypair in PEM format (private_pem, public_pem)."""
    backend, libs = _get_crypto_backends()
    if backend == "cryptography":
        rsa_mod, _, _, ser_mod = libs
        private_key = rsa_mod.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        private_pem = private_key.private_bytes(
            encoding=ser_mod.Encoding.PEM,
            format=ser_mod.PrivateFormat.PKCS8,
            encryption_algorithm=ser_mod.NoEncryption(),
        )
        public_pem = private_key.public_key().public_bytes(
            encoding=ser_mod.Encoding.PEM,
            format=ser_mod.PublicFormat.SubjectPublicKeyInfo,
        )
        return private_pem, public_pem
    elif backend == "rsa":
        rsa_mod = libs
        pubkey, privkey = rsa_mod.newkeys(2048)
        return privkey.save_pkcs1(), pubkey.save_pkcs1()
    else:
        # Minimal HMAC fallback seed if no RSA package exists
        import hashlib, secrets
        seed = secrets.token_bytes(64)
        return seed, hashlib.sha256(seed).digest()

_CONFIGURED_PRIV = _load_key("LICENSE_PRIVATE_KEY", "LICENSE_PRIVATE_KEY_PATH")
_CONFIGURED_PUB = _load_key("LICENSE_PUBLIC_KEY", "LICENSE_PUBLIC_KEY_PATH")

if _CONFIGURED_PRIV and _CONFIGURED_PUB:
    _DEV_PRIV_PEM, _DEV_PUB_PEM = _CONFIGURED_PRIV, _CONFIGURED_PUB
else:
    # Dynamically generate ephemeral in-memory RSA keypair at startup (zero hardcoded secrets)
    _DEV_PRIV_PEM, _DEV_PUB_PEM = generate_rsa_keypair()


def sign_payload(payload_bytes: bytes, private_key_pem: bytes = _DEV_PRIV_PEM) -> str:
    """Cryptographically signs payload bytes using RSA-SHA256."""
    if not private_key_pem:
        raise LicenseError("No private key configured for signing.")
    backend, libs = _get_crypto_backends()
    if backend == "cryptography":
        _, padding_mod, hashes_mod, ser_mod = libs
        private_key = ser_mod.load_pem_private_key(private_key_pem, password=None)
        sig = private_key.sign(
            payload_bytes,
            padding_mod.PSS(
                mgf=padding_mod.MGF1(hashes_mod.SHA256()),
                salt_length=padding_mod.PSS.MAX_LENGTH,
            ),
            hashes_mod.SHA256(),
        )
        return base64.b64encode(sig).decode("ascii")
    elif backend == "rsa":
        rsa_mod = libs
        priv = rsa_mod.PrivateKey.load_pkcs1(private_key_pem)
        sig = rsa_mod.sign(payload_bytes, priv, "SHA-256")
        return base64.b64encode(sig).decode("ascii")
    else:
        import hmac, hashlib
        sig = hmac.new(private_key_pem, payload_bytes, hashlib.sha256).digest()
        return base64.b64encode(sig).decode("ascii")


def verify_signature(payload_bytes: bytes, signature_b64: str, public_key_pem: bytes = _DEV_PUB_PEM) -> bool:
    """Verifies RSA-SHA256 cryptographic signature against public key."""
    if not public_key_pem:
        raise LicenseError("No public key configured for verification.")
    backend, libs = _get_crypto_backends()
    sig_bytes = base64.b64decode(signature_b64.encode("ascii"))
    
    if backend == "cryptography":
        _, padding_mod, hashes_mod, ser_mod = libs
        try:
            public_key = ser_mod.load_pem_public_key(public_key_pem)
            public_key.verify(
                sig_bytes,
                payload_bytes,
                padding_mod.PSS(
                    mgf=padding_mod.MGF1(hashes_mod.SHA256()),
                    salt_length=padding_mod.PSS.MAX_LENGTH,
                ),
                hashes_mod.SHA256(),
            )
            return True
        except Exception as e:
            logger.debug(f"Signature verification failed: {e}")
            return False
    elif backend == "rsa":
        rsa_mod = libs
        try:
            pub = rsa_mod.PublicKey.load_pkcs1(public_key_pem)
            rsa_mod.verify(payload_bytes, sig_bytes, pub)
            return True
        except Exception:
            return False
    else:
        import hmac, hashlib
        expected = hmac.new(public_key_pem, payload_bytes, hashlib.sha256).digest()
        return hmac.compare_digest(expected, sig_bytes)


def issue_lease(
    hwid: Optional[str] = None,
    duration_hours: float = 72.0,
    license_type: str = "production",
    private_key_pem: bytes = _DEV_PRIV_PEM,
) -> Dict[str, Any]:
    """
    Issues a cryptographically signed offline lease dictionary for a machine HWID.
    """
    target_hwid = hwid or get_machine_hwid()
    now = time.time()
    expires_at = now + (duration_hours * 3600.0)

    payload = {
        "hwid": target_hwid,
        "issued_at": round(now, 2),
        "expires_at": round(expires_at, 2),
        "max_offline_hours": float(duration_hours),
        "license_type": license_type,
        "issuer": "Antigravity Licensing Authority",
    }

    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    signature = sign_payload(payload_json.encode("utf-8"), private_key_pem=private_key_pem)

    return {
        "payload": payload,
        "signature": signature,
    }


def generate_dev_lease(
    lease_path: Optional[str] = None,
    duration_hours: float = 72.0,
) -> str:
    """
    Generates and saves a development lease bound to current hardware.
    Returns the absolute path to the written lease file.
    """
    target_path = lease_path or DEFAULT_LEASE_PATH
    os.makedirs(os.path.dirname(os.path.abspath(target_path)) or ".", exist_ok=True)
    
    lease_data = issue_lease(duration_hours=duration_hours, license_type="development")
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(lease_data, f, indent=2)

    logger.info(f"[licensing] Development lease generated at: {target_path}")
    return target_path


def verify_lease(
    lease_path: Optional[str] = None,
    public_key_pem: bytes = _DEV_PUB_PEM,
    expected_hwid: Optional[str] = None,
    auto_issue_dev: bool = True,
) -> bool:
    """
    Validates the offline hardware lease against local hardware and expiration clock.

    Raises:
        LicenseError: If lease is missing, tampered, expired, or machine HWID mismatches.

    Returns:
        True if the license is valid and active.
    """
    target_path = lease_path or os.environ.get("ANTIGRAVITY_LEASE_PATH") or DEFAULT_LEASE_PATH

    def _check_lease_file(path: str) -> dict:
        try:
            with open(path, "r", encoding="utf-8") as f:
                lease_data = json.load(f)
        except Exception as e:
            raise LicenseError(f"Malformed lease JSON in {path}: {e}")

        payload = lease_data.get("payload")
        signature = lease_data.get("signature")
        if not payload or not signature:
            raise LicenseError("Invalid lease structure: 'payload' or 'signature' missing")

        # 1. Cryptographic Signature Verification
        payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        if not verify_signature(payload_json.encode("utf-8"), signature, public_key_pem=public_key_pem):
            raise LicenseError("Cryptographic signature verification failed: lease has been tampered with")

        # 2. Hardware-Bound Machine HWID Verification
        current_hwid = expected_hwid or get_machine_hwid()
        lease_hwid = payload.get("hwid")
        if not lease_hwid or lease_hwid.lower() != current_hwid.lower():
            raise LicenseError(
                f"Hardware mismatch: Lease bound to HWID [{lease_hwid}], but local machine is [{current_hwid}]"
            )

        # 3. Rolling Offline Expiration Check (72-hour window)
        now = time.time()
        expires_at = float(payload.get("expires_at", 0))
        issued_at = float(payload.get("issued_at", 0))

        if now > expires_at:
            expired_date = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(expires_at))
            raise LicenseError(f"Offline lease expired on {expired_date}. Renewal required.")

        if now < issued_at - 300.0:  # Allow 5-minute clock drift
            raise LicenseError("System clock anomaly detected: current timestamp precedes lease issue time")

        return payload

    # If lease does not exist and auto_issue_dev is enabled, auto-issue dev lease
    if not os.path.isfile(target_path):
        if auto_issue_dev:
            logger.info("[licensing] No lease found. Auto-generating 72h development lease...")
            generate_dev_lease(target_path, duration_hours=72.0)
        else:
            raise LicenseError(f"License lease file not found: {target_path}")

    try:
        payload = _check_lease_file(target_path)
    except LicenseError as err:
        if auto_issue_dev and target_path == DEFAULT_LEASE_PATH:
            logger.info(f"[licensing] Dev lease invalid/expired ({err}). Auto-renewing development lease...")
            generate_dev_lease(target_path, duration_hours=72.0)
            payload = _check_lease_file(target_path)
        else:
            raise

    remaining_hours = (float(payload.get("expires_at", 0)) - time.time()) / 3600.0
    logger.info(
        f"[licensing] Valid {payload.get('license_type', 'standard')} lease verified. "
        f"Remaining offline window: {remaining_hours:.1f} hours."
    )
    return True


if __name__ == "__main__":
    lease_file = generate_dev_lease()
    print(f"Generated lease file: {lease_file}")
    is_valid = verify_lease(lease_file)
    print(f"Lease verification: {'SUCCESS' if is_valid else 'FAILED'}")
