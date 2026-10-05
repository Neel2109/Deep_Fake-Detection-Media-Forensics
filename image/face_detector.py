"""Face candidate detection adapter; candidates are not authenticity predictions."""

from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from app.services.xception_detector import detect_largest_face_crop


def inspect_face_image(image: Image.Image) -> dict[str, Any]:
    _, review = detect_largest_face_crop(image.convert("RGB"))
    review["interpretation"] = (
        "The OpenCV Haar-cascade detector is unavailable in this runtime. No face count, "
        "authenticity score, landmarks, or manipulation localization was computed."
        if review["status"] == "unavailable"
        else (
            "A Haar-cascade candidate count is a basic face-detection observation, not a face "
            "authenticity score. Landmarks and manipulation localization were not computed."
        )
    )
    return review


def inspect_face_candidates(path: str | Path) -> dict[str, Any]:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.load()
    return inspect_face_image(image)


__all__ = ["detect_largest_face_crop", "inspect_face_candidates", "inspect_face_image"]
