import numpy as np

from video import frame_extractor
from video.video_pipeline import analyze_video


class FakeCapture:
    def __init__(self, opened=True, frame_shape=(720, 1280, 3)):
        self.opened = opened
        self.frame_shape = frame_shape
        self.position_ms = 0.0
        self.released = False

    def isOpened(self):
        return self.opened

    def get(self, property_id):
        if property_id == frame_extractor.cv2.CAP_PROP_FRAME_COUNT:
            return 250
        if property_id == frame_extractor.cv2.CAP_PROP_FPS:
            return 25.0
        return 0

    def set(self, property_id, value):
        if property_id == frame_extractor.cv2.CAP_PROP_POS_MSEC:
            self.position_ms = value
        return True

    def read(self):
        return True, np.zeros(self.frame_shape, dtype=np.uint8)

    def release(self):
        self.released = True


def test_keyframes_are_bounded_and_evenly_sampled(monkeypatch, tmp_path):
    capture = FakeCapture()
    monkeypatch.setattr(frame_extractor.cv2, "VideoCapture", lambda _: capture)
    monkeypatch.setattr(
        frame_extractor,
        "inspect_face_image",
        lambda _: {"status": "no_face_detected", "interpretation": "Descriptive test."},
    )

    result = frame_extractor.extract_keyframes(tmp_path / "test.mp4")

    assert result["status"] == "measured_descriptive"
    assert len(result["frames"]) == frame_extractor.MAX_KEYFRAMES
    assert result["frames"][0]["timestamp_seconds"] == 0
    assert result["frames"][-1]["timestamp_seconds"] > 0
    assert max(frame["width"] for frame in result["frames"]) <= frame_extractor.MAX_FRAME_SIDE
    assert max(frame["height"] for frame in result["frames"]) <= frame_extractor.MAX_FRAME_SIDE
    assert all(frame["preview_data_url"].startswith("data:image/jpeg;base64,") for frame in result["frames"])
    assert capture.released


def test_unopenable_video_returns_explicit_unavailable_status(monkeypatch, tmp_path):
    capture = FakeCapture(opened=False)
    monkeypatch.setattr(frame_extractor.cv2, "VideoCapture", lambda _: capture)

    result = frame_extractor.extract_keyframes(tmp_path / "invalid.mp4")

    assert result["status"] == "unavailable"
    assert result["frames"] == []
    assert capture.released


def test_keyframes_can_include_ephemeral_rgb_images_for_inference(monkeypatch, tmp_path):
    capture = FakeCapture(frame_shape=(32, 48, 3))
    monkeypatch.setattr(frame_extractor.cv2, "VideoCapture", lambda _: capture)
    monkeypatch.setattr(
        frame_extractor,
        "inspect_face_image",
        lambda _: {"status": "no_face_detected", "interpretation": "Descriptive test."},
    )

    result = frame_extractor.extract_keyframes(
        tmp_path / "test.mp4",
        max_frames=1,
        include_images=True,
    )

    assert result["frames"][0]["_image"].mode == "RGB"
    assert result["frames"][0]["_image"].size == (48, 32)


def test_video_pipeline_records_frame_scores_without_a_clip_verdict(monkeypatch, tmp_path):
    capture = FakeCapture(frame_shape=(32, 48, 3))
    monkeypatch.setattr(frame_extractor.cv2, "VideoCapture", lambda _: capture)
    monkeypatch.setattr(
        frame_extractor,
        "inspect_face_image",
        lambda _: {"status": "no_face_detected", "interpretation": "Descriptive test."},
    )

    class FakeDetector:
        def predict_image(self, image):
            assert image.mode == "RGB"
            return {
                "status": "uncalibrated_prediction",
                "decision": "likely_fake",
                "deepfake_probability": 0.73,
                "authenticity_probability": 0.27,
                "model": {
                    "architecture": "test-xception",
                    "calibration": "uncalibrated raw softmax scores",
                },
            }

    result = analyze_video(
        tmp_path / "test.mp4",
        {"decoder_reported_fps": 25.0},
        FakeDetector(),
    )

    assert result["model_frame_analysis"]["analyzed_frame_count"] == frame_extractor.MAX_KEYFRAMES
    assert result["model_frame_analysis"]["video_level_decision"] == "withheld"
    assert result["frames"][0]["xception_estimate"]["deepfake_score"] == 0.73
    assert "_image" not in result["frames"][0]
    assert "must not be averaged" in result["interpretation"]
