"""Descriptive high-pass residual measurements for decoded images."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageFilter, ImageOps

MAX_ANALYSIS_SIDE = 512


def measure_high_pass_residual(grayscale: NDArray[np.uint8]) -> dict[str, float | int | str]:
    if grayscale.ndim != 2 or grayscale.size == 0:
        raise ValueError("Residual analysis requires a non-empty grayscale image")

    blurred = np.asarray(
        Image.fromarray(grayscale).filter(ImageFilter.GaussianBlur(radius=1.0)),
        dtype=np.float64,
    )
    residual = grayscale.astype(np.float64) - blurred
    absolute_residual = np.abs(residual)
    return {
        "method": "grayscale minus Gaussian blur (radius 1 px)",
        "width": int(grayscale.shape[1]),
        "height": int(grayscale.shape[0]),
        "residual_mean": float(residual.mean()),
        "residual_rms": float(np.sqrt(np.mean(residual ** 2))),
        "residual_standard_deviation": float(residual.std()),
        "absolute_residual_p95": float(np.percentile(absolute_residual, 95)),
    }


def analyze_image_noise(path: str | Path) -> dict:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("L")
        image.thumbnail((MAX_ANALYSIS_SIDE, MAX_ANALYSIS_SIDE))
        grayscale = np.asarray(image, dtype=np.uint8)

    return {
        "status": "measured_descriptive",
        "measurements": measure_high_pass_residual(grayscale),
        "interpretation": (
            "The residual combines edges, fine image detail, sensor noise, and compression effects. "
            "It is not a validated noise-forensics score or proof of manipulation."
        ),
    }
