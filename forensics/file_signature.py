"""Binary signature inspection backed by the current evidence pipeline."""

from app.services.forensics import inspect_file_signature

__all__ = ["inspect_file_signature"]
