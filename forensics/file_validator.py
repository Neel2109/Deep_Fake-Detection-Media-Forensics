"""File type/signature checks; this does not assert provenance or authenticity."""

from pathlib import Path

from app.services.forensics import detect_media_type, inspect_file_signature


def validate_file(path: str | Path, filename: str) -> dict:
    media_type = detect_media_type(filename)
    if media_type == "unsupported":
        raise ValueError(f"Unsupported file type: {filename}")
    signature = inspect_file_signature(path, filename)
    if not signature["signature_recognized"]:
        raise ValueError("Unrecognized media file signature")
    if not signature["extension_matches_signature"]:
        raise ValueError(
            f"Extension does not match detected content type "
            f"({signature['detected_mime_type']})"
        )
    return {"media_type": media_type, **signature}
