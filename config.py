"""Compatibility re-exports for the established application configuration."""

from app.config import (
    ALLOWED_EXTENSIONS,
    APP_TITLE,
    APP_VERSION,
    BASE_DIR,
    CORS_ORIGINS,
    DATABASE_PATH,
    MAX_UPLOAD_BYTES,
    REPORT_DIR,
    UPLOAD_DIR,
    WEIGHTS_DIR,
    XCEPTION_CHECKPOINT,
    ensure_storage_dirs,
)

__all__ = [
    "ALLOWED_EXTENSIONS",
    "APP_TITLE",
    "APP_VERSION",
    "BASE_DIR",
    "CORS_ORIGINS",
    "DATABASE_PATH",
    "MAX_UPLOAD_BYTES",
    "REPORT_DIR",
    "UPLOAD_DIR",
    "WEIGHTS_DIR",
    "XCEPTION_CHECKPOINT",
    "ensure_storage_dirs",
]
