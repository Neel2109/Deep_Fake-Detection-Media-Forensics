"""Existing media validation contract exposed from the modular layout."""

from app.config import ALLOWED_EXTENSIONS
from app.services.forensics import detect_media_type

__all__ = ["ALLOWED_EXTENSIONS", "detect_media_type"]
