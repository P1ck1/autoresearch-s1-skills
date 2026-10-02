"""Private benchmark-release key loading and domain-separated derivation."""

from __future__ import annotations

import hashlib
import hmac
import os
import stat
from pathlib import Path


RELEASE_ID = "pcabench-eval-v3"


def read_math_key(path: str | Path) -> bytes:
    """Read one owner-private, no-follow 32-byte key encoded as lowercase hex."""
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise PermissionError("math key must be a singly linked regular file")
        if info.st_mode & 0o077:
            raise PermissionError("math key must not be accessible by group or other users")
        if hasattr(os, "geteuid") and os.geteuid() == 0 and info.st_uid != 0:
            raise PermissionError("deployed math key must be owned by root")
        payload = os.read(descriptor, 256)
        if os.read(descriptor, 1):
            raise ValueError("math key file is unexpectedly large")
    finally:
        os.close(descriptor)
    encoded = payload.strip()
    if len(encoded) != 64 or any(byte not in b"0123456789abcdef" for byte in encoded):
        raise ValueError("math key must be exactly 64 lowercase hexadecimal characters")
    return bytes.fromhex(encoded.decode("ascii"))


def derive_bytes(key: bytes, domain: str, *, length: int = 32) -> bytes:
    """Derive private deterministic bytes under a versioned, domain-separated label."""
    if len(key) != 32:
        raise ValueError("math key must contain exactly 32 bytes")
    if not domain or "\x00" in domain:
        raise ValueError("derivation domain must be a nonempty plain string")
    if not 1 <= length <= hashlib.sha256().digest_size:
        raise ValueError("derived length is outside SHA-256 output bounds")
    message = f"{RELEASE_ID}\0{domain}".encode("utf-8")
    return hmac.digest(key, message, "sha256")[:length]


def derive_seed(key: bytes, domain: str) -> int:
    """Derive a NumPy-compatible nonnegative 63-bit seed."""
    return int.from_bytes(derive_bytes(key, domain, length=8), "little") & 0x7FFF_FFFF_FFFF_FFFF


def key_commitment(key: bytes) -> str:
    """Return a nonsecret release-consistency commitment, never the key itself."""
    if len(key) != 32:
        raise ValueError("math key must contain exactly 32 bytes")
    return hashlib.sha256(RELEASE_ID.encode("ascii") + b"\0" + key).hexdigest()
