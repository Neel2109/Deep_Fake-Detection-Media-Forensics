"""SHA-256 file fingerprinting backed by the current evidence pipeline."""

from app.services.forensics import sha256_file

__all__ = ["sha256_file"]
