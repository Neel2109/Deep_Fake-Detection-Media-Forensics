"""Static-image prediction adapter for the validated Xception checkpoint."""

from pathlib import Path

from app.config import XCEPTION_CHECKPOINT
from app.services.xception_detector import XceptionModelError, get_xception_detector


def predict_image(path: str | Path) -> dict:
    detector = get_xception_detector(XCEPTION_CHECKPOINT)
    if detector is None:
        raise XceptionModelError(
            "Image prediction requires a trained, validated Xception checkpoint."
        )
    return detector.predict(path)
