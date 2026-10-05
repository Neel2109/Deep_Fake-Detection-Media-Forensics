import numpy as np
import soundfile as sf

from audio.audio_loader import _signal_measurements, inspect_audio


def test_short_clip_is_padded_for_spectrum():
    measurements = _signal_measurements(np.array([0.1, -0.1, 0.2], dtype=np.float32), 8000)

    assert measurements["sample_count_analyzed"] == 3
    assert measurements["spectrogram_summary"]["window_count"] == 1
    assert len(measurements["spectrogram_summary"]["band_power"]) == 24
    assert 0 <= measurements["zero_crossing_rate"] <= 1
    assert measurements["spectrogram_summary"]["power_weighted_spectral_centroid_hz"] >= 0
    assert 0 <= measurements["spectrogram_summary"]["spectral_flatness"] <= 1


def test_silence_has_finite_zero_spectral_descriptors():
    measurements = _signal_measurements(np.zeros(4096, dtype=np.float32), 16000)

    assert measurements["zero_crossing_rate"] == 0
    assert measurements["spectrogram_summary"]["power_weighted_spectral_centroid_hz"] == 0
    assert measurements["spectrogram_summary"]["spectral_flatness"] == 0


def test_wav_inspection_returns_bounded_signal_summary(tmp_path):
    audio_path = tmp_path / "short.wav"
    samples = np.sin(np.linspace(0, 4 * np.pi, 512, dtype=np.float32))
    sf.write(audio_path, samples, 16000)

    result = inspect_audio(audio_path)

    assert result["signal_status"] == "measured_descriptive"
    assert result["audio_authenticity_status"] == "not_configured"
    assert result["measurements"]["sample_count_analyzed"] == 512
    assert result["measurements"]["spectrogram_summary"]["window_count"] == 1


def test_non_finite_audio_is_not_reported_as_a_measurement(tmp_path):
    audio_path = tmp_path / "non-finite.wav"
    sf.write(audio_path, np.array([0.0, np.nan, 0.2], dtype=np.float32), 8000, subtype="FLOAT")

    result = inspect_audio(audio_path)

    assert result["signal_status"] == "invalid_signal"
    assert "NaN or infinite" in result["metadata_limitation"]
