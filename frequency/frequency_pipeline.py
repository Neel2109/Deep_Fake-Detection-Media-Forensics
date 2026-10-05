"""Bounded descriptive frequency analysis; this is not an authenticity detector."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from frequency.fft import measure_frequency_spectrum

MAX_ANALYSIS_SIDE = 512


def analyze_image_frequency(path: str | Path) -> dict:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("L")
        image.thumbnail((MAX_ANALYSIS_SIDE, MAX_ANALYSIS_SIDE))
        grayscale = np.asarray(image, dtype=np.uint8)

    return {
        "status": "measured_descriptive",
        "measurements": measure_frequency_spectrum(grayscale),
        "interpretation": (
            "These are descriptive spectrum measurements of this decoded image. "
            "They are not a manipulation score and cannot distinguish original from deepfake."
        ),
    }
