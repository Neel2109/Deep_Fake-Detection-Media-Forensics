"""Descriptive two-dimensional FFT measurements for decoded images."""

import numpy as np
from numpy.typing import NDArray


def measure_frequency_spectrum(grayscale: NDArray[np.uint8]) -> dict[str, float | int | str]:
    if grayscale.ndim != 2 or grayscale.size == 0:
        raise ValueError("Frequency analysis requires a non-empty grayscale image")

    height, width = grayscale.shape
    signal = grayscale.astype(np.float64)
    centered = np.fft.fftshift(np.fft.fft2(signal))
    power = np.abs(centered) ** 2
    total_power = float(power.sum())
    if not np.isfinite(total_power):
        raise ValueError("Frequency spectrum contains non-finite signal power")

    y = np.linspace(-1.0, 1.0, height, endpoint=False)
    x = np.linspace(-1.0, 1.0, width, endpoint=False)
    radius = np.sqrt(y[:, None] ** 2 + x[None, :] ** 2) / np.sqrt(2.0)
    bands = {
        "low_frequency_power_fraction": radius < 0.15,
        "mid_frequency_power_fraction": (radius >= 0.15) & (radius < 0.5),
        "high_frequency_power_fraction": radius >= 0.5,
    }
    result: dict[str, float | int | str] = {
        "method": "2D FFT power-spectrum summary",
        "width": int(width),
        "height": int(height),
        "log_magnitude_mean": float(np.log1p(np.abs(centered)).mean()),
    }
    result.update(
        {
            name: float(power[band].sum() / total_power) if total_power else 0.0
            for name, band in bands.items()
        }
    )
    result["total_power"] = total_power
    result["has_signal_power"] = total_power > 0
    return result
