"""Audio stream properties and bounded signal summaries for analyst review."""

from pathlib import Path
from typing import Any

import numpy as np

MAX_DECODE_SECONDS = 30
WAVEFORM_BINS = 96
SPECTRUM_BANDS = 24
FFT_SIZE = 2048
FFT_HOP = 1024


def _audio_container(path: Path) -> dict[str, Any]:
    try:
        from mutagen import File as MutagenFile
        from mutagen import MutagenError
    except ImportError:
        return {"container": path.suffix.lstrip(".").upper() or "Unknown"}

    try:
        audio = MutagenFile(path)
    except (MutagenError, OSError) as error:
        return {
            "container": path.suffix.lstrip(".").upper() or "Unknown",
            "metadata_status": "partial",
            "metadata_limitation": f"Container metadata could not be parsed: {error}",
        }
    if audio is None or audio.info is None:
        return {"container": path.suffix.lstrip(".").upper() or "Unknown"}
    info = audio.info
    mime = getattr(audio, "mime", None) or []
    return {
        "container": path.suffix.lstrip(".").upper() or "Unknown",
        "codec_mime": mime[0] if mime else None,
        "duration_seconds": round(float(info.length), 4)
        if getattr(info, "length", None) is not None
        else None,
        "sample_rate_hz": int(info.sample_rate)
        if getattr(info, "sample_rate", None) is not None
        else None,
        "channels": int(info.channels)
        if getattr(info, "channels", None) is not None
        else None,
        "bitrate_bps": int(info.bitrate)
        if getattr(info, "bitrate", None) is not None
        else None,
    }


def _signal_measurements(samples: np.ndarray, sample_rate: int) -> dict[str, Any]:
    mono = samples.mean(axis=1, dtype=np.float64) if samples.ndim == 2 else samples.astype(
        np.float64, copy=False
    )
    if mono.size == 0:
        raise ValueError("Decoded audio contains no samples")
    if not np.isfinite(mono).all():
        raise ValueError("Decoded audio contains NaN or infinite samples")

    chunks = np.array_split(mono, min(WAVEFORM_BINS, mono.size))
    waveform = [
        {
            "minimum": float(chunk.min()),
            "maximum": float(chunk.max()),
            "rms": float(np.sqrt(np.mean(chunk**2))),
        }
        for chunk in chunks
        if chunk.size
    ]
    padded = mono if mono.size >= FFT_SIZE else np.pad(mono, (0, FFT_SIZE - mono.size))
    windows = np.lib.stride_tricks.sliding_window_view(padded, FFT_SIZE)[::FFT_HOP]
    spectrum = np.abs(np.fft.rfft(windows * np.hanning(FFT_SIZE), axis=1)) ** 2
    frequency_bins = np.fft.rfftfreq(FFT_SIZE, d=1.0 / sample_rate)
    mean_power_spectrum = spectrum.mean(axis=0)
    total_spectral_power = float(mean_power_spectrum.sum())
    if total_spectral_power > 0:
        spectral_centroid_hz = float(
            np.dot(frequency_bins, mean_power_spectrum) / total_spectral_power
        )
        positive_power = np.maximum(mean_power_spectrum, np.finfo(np.float64).tiny)
        spectral_flatness = float(
            np.exp(np.mean(np.log(positive_power)))
            / mean_power_spectrum.mean()
        )
    else:
        spectral_centroid_hz = 0.0
        spectral_flatness = 0.0
    band_edges = np.linspace(0, sample_rate / 2, SPECTRUM_BANDS + 1)
    band_power = []
    for lower, upper in zip(band_edges[:-1], band_edges[1:]):
        selected = (frequency_bins >= lower) & (frequency_bins < upper)
        band_power.append(float(spectrum[:, selected].mean()) if np.any(selected) else 0.0)

    peak = float(np.max(np.abs(mono)))
    return {
        "sample_count_analyzed": int(mono.size),
        "analysis_duration_seconds": round(mono.size / sample_rate, 4),
        "waveform_bins": waveform,
        "mean_amplitude": float(np.mean(mono)),
        "mean_absolute_amplitude": float(np.mean(np.abs(mono))),
        "zero_crossing_rate": float(
            np.count_nonzero(np.diff(np.signbit(mono))) / max(1, mono.size - 1)
        ),
        "rms_amplitude": float(np.sqrt(np.mean(mono**2))),
        "peak_amplitude": peak,
        "clipped_sample_fraction": float(np.mean(np.abs(mono) >= 0.999)),
        "spectrogram_summary": {
            "method": f"mean power in {SPECTRUM_BANDS} equal-width FFT bands",
            "fft_size": FFT_SIZE,
            "hop_size": FFT_HOP,
            "window_count": int(spectrum.shape[0]),
            "power_weighted_spectral_centroid_hz": spectral_centroid_hz,
            "spectral_flatness": spectral_flatness,
            "band_power": band_power,
        },
    }


def inspect_audio(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    metadata = _audio_container(source)
    try:
        import soundfile as sf
    except ImportError:
        return {
            **metadata,
            "metadata_status": "available" if metadata.get("duration_seconds") is not None else "partial",
            "signal_status": "decoder_unavailable",
            "metadata_limitation": "Install the declared soundfile dependency to decode audio signal measurements.",
            "audio_authenticity_status": "not_configured",
            "interpretation": "Container properties only; waveform and spectrogram measurements were not decoded.",
        }

    try:
        stream_info = sf.info(str(source))
        sample_rate = int(stream_info.samplerate)
        channels = int(stream_info.channels)
        limit_frames = max(1, sample_rate * MAX_DECODE_SECONDS)
        samples, decoded_rate = sf.read(
            str(source),
            frames=limit_frames,
            dtype="float32",
            always_2d=True,
        )
    except (RuntimeError, OSError, ValueError) as error:
        return {
            **metadata,
            "metadata_status": "available" if metadata.get("duration_seconds") is not None else "partial",
            "signal_status": "unsupported_decoder",
            "metadata_limitation": f"Container metadata was partially read, but signal decoding failed: {error}",
            "audio_authenticity_status": "not_configured",
            "interpretation": "No waveform or spectrogram signal was generated because decoding failed.",
        }

    if decoded_rate != sample_rate:
        raise RuntimeError("Audio decoder reported inconsistent sample rates for the same file")
    try:
        measurements = _signal_measurements(samples, sample_rate)
    except ValueError as error:
        return {
            **metadata,
            "sample_rate_hz": sample_rate,
            "channels": channels,
            "metadata_status": "available",
            "signal_status": "invalid_signal",
            "metadata_limitation": f"Audio was decoded, but signal measurements could not be calculated: {error}",
            "audio_authenticity_status": "not_configured",
            "interpretation": "Signal measurements are unavailable; no audio authenticity conclusion was made.",
        }
    return {
        **metadata,
        "sample_rate_hz": sample_rate,
        "channels": channels,
        "metadata_status": "available",
        "signal_status": "measured_descriptive",
        "analysis_limit_seconds": MAX_DECODE_SECONDS,
        "measurements": measurements,
        "audio_authenticity_status": "not_configured",
        "interpretation": (
            "Waveform, amplitude, and coarse spectral values describe decoded audio only. "
            "They are not a synthetic-voice detector and do not establish authenticity."
        ),
    }
