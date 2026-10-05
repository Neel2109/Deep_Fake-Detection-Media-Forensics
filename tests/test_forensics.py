import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import soundfile as sf

from frequency.fft import measure_frequency_spectrum
from forensics.chain_of_custody import verify_audit_chain
from forensics.exif_analyzer import gps_decimal_coordinate
from app.services.forensics import (
    analyze_media,
    detect_media_type,
    inspect_file_signature,
)


class MediaSignatureTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)

    def test_all_supported_formats_match_their_file_signatures(self):
        mp4_header = b"\x00\x00\x00\x18ftypisom1234"
        mov_header = b"\x00\x00\x00\x18ftypqt  1234"
        mkv_header = b"\x1a\x45\xdf\xa3" + b"\x00" * 4 + b"matroska"
        webm_header = b"\x1a\x45\xdf\xa3" + b"\x00" * 4 + b"webm"
        m4a_header = b"\x00\x00\x00\x18ftypM4A 1234"
        cases = {
            ".jpg": (b"\xff\xd8\xff" + b"\x00" * 13, "image/jpeg", "image"),
            ".jpeg": (b"\xff\xd8\xff" + b"\x00" * 13, "image/jpeg", "image"),
            ".png": (b"\x89PNG\r\n\x1a\n" + b"\x00" * 8, "image/png", "image"),
            ".gif": (b"GIF89a" + b"\x00" * 10, "image/gif", "image"),
            ".webp": (b"RIFF" + b"\x00" * 4 + b"WEBP" + b"\x00" * 4, "image/webp", "image"),
            ".bmp": (b"BM" + b"\x00" * 14, "image/bmp", "image"),
            ".tif": (b"II*\x00" + b"\x00" * 12, "image/tiff", "image"),
            ".tiff": (b"MM\x00+" + b"\x00" * 12, "image/tiff", "image"),
            ".mp4": (mp4_header, "video/mp4", "video"),
            ".m4v": (mp4_header, "video/mp4", "video"),
            ".mov": (mov_header, "video/quicktime", "video"),
            ".avi": (b"RIFF" + b"\x00" * 4 + b"AVI " + b"\x00" * 4, "video/x-msvideo", "video"),
            ".mkv": (mkv_header, "video/x-matroska", "video"),
            ".webm": (webm_header, "video/webm", "video"),
            ".mp3": (b"ID3" + b"\x00" * 13, "audio/mpeg", "audio"),
            ".wav": (b"RIFF" + b"\x00" * 4 + b"WAVE" + b"\x00" * 4, "audio/wav", "audio"),
            ".flac": (b"fLaC" + b"\x00" * 12, "audio/flac", "audio"),
            ".m4a": (m4a_header, "audio/mp4", "audio"),
            ".aac": (b"\xff\xf1" + b"\x00" * 14, "audio/aac", "audio"),
        }

        for extension, (header, expected_mime, expected_media_type) in cases.items():
            with self.subTest(extension=extension):
                filename = f"evidence{extension}"
                path = self.root / filename
                path.write_bytes(header)

                signature = inspect_file_signature(path, filename)
                self.assertEqual(detect_media_type(filename), expected_media_type)
                self.assertEqual(signature["detected_mime_type"], expected_mime)
                self.assertTrue(signature["signature_recognized"])
                self.assertTrue(signature["extension_matches_signature"])

    def test_fft_measurements_handle_black_images_without_failing(self):
        measurements = measure_frequency_spectrum(
            np.zeros((12, 16), dtype=np.uint8)
        )

        self.assertEqual(measurements["total_power"], 0)
        self.assertFalse(measurements["has_signal_power"])
        self.assertEqual(measurements["low_frequency_power_fraction"], 0)
        self.assertEqual(measurements["mid_frequency_power_fraction"], 0)
        self.assertEqual(measurements["high_frequency_power_fraction"], 0)

    def test_signature_match_is_not_reported_as_full_integrity_verification(self):
        image_path = self.root / "sample.png"
        Image.new("RGB", (8, 5), "navy").save(image_path)

        with patch("app.services.forensics.get_xception_detector", return_value=None):
            report = analyze_media(image_path, image_path.name)

        self.assertEqual(report["file_integrity"]["status"], "signature_and_extension_match")
        self.assertEqual(report["file_integrity"]["content_validation"], "decoded")
        self.assertEqual(report["metadata_summary"]["width"], 8)
        self.assertEqual(report["metadata_summary"]["height"], 5)
        self.assertTrue(verify_audit_chain(report))
        self.assertNotIn("orientation", report["metadata_summary"])
        self.assertNotIn("software", report["metadata_summary"])
        self.assertNotIn("timestamp", report["metadata_summary"])

    def test_static_image_remains_inconclusive_without_a_checkpoint(self):
        image_path = self.root / "sample.png"
        Image.new("RGB", (8, 5), "navy").save(image_path)

        with patch("app.services.forensics.XCEPTION_CHECKPOINT", self.root / "missing.pth"):
            report = analyze_media(image_path, image_path.name)

        self.assertEqual(report["verdict"], "Insufficient evidence")
        self.assertEqual(report["decision_status"], "not_configured")
        self.assertIsNone(report["authenticity_probability"])
        self.assertIsNone(report["deepfake_probability"])
        self.assertEqual(report["model_results"], {})
        pipeline_by_name = {stage["name"]: stage for stage in report["model_pipeline"]}
        self.assertEqual(pipeline_by_name["Xception checkpoint"]["status"], "not_configured")
        self.assertEqual(pipeline_by_name["Xception inference"]["status"], "not_run")
        self.assertEqual(pipeline_by_name["Original / deepfake decision"]["status"], "withheld")
        self.assertEqual(pipeline_by_name["Calibration and operating thresholds"]["status"], "not_run")
        self.assertEqual(report["frequency_analysis"]["status"], "measured_descriptive")
        self.assertEqual(report["noise_analysis"]["status"], "measured_descriptive")
        self.assertIn(
            report["face_review"]["status"],
            {"unavailable", "no_face_detected", "face_detected"},
        )
        self.assertIn("not a manipulation score", report["frequency_analysis"]["interpretation"])
        self.assertIn("not a validated", report["noise_analysis"]["interpretation"])

    def test_uncalibrated_model_class_is_not_reported_as_calibrated(self):
        image_path = self.root / "uncalibrated.png"
        Image.new("RGB", (24, 16), "navy").save(image_path)
        result = {
            "status": "uncalibrated_prediction",
            "decision": "likely_fake",
            "verdict": "Likely deepfake",
            "deepfake_probability": 0.87,
            "authenticity_probability": 0.13,
            "model": {
                "architecture": "RamadhanZome/deepfake-xception",
                "inference_device": "cpu",
                "calibration": "uncalibrated raw softmax scores",
                "thresholds": None,
                "input_preprocessing": "Full image resized to 299x299 RGB.",
            },
            "face_review": {
                "detected_face_count": 0,
                "status": "no_face_detected",
                "selected_crop": "not_used_full_image_model_input",
                "alignment": "not_performed_landmarks_unavailable",
            },
            "explainability": {
                "status": "generated",
                "method": "Grad-CAM",
                "target": "fake-class logit",
                "target_layer": "test.conv",
                "overlay_data_url": "data:image/jpeg;base64,dGVzdA==",
                "overlay_width": 24,
                "overlay_height": 16,
                "interpretation": "Test-only model influence overlay.",
            },
            "limitations": ["Raw scores are uncalibrated."],
        }

        with patch("app.services.forensics.get_xception_detector") as detector_factory:
            detector_factory.return_value.predict.return_value = result
            report = analyze_media(image_path, image_path.name)

        self.assertEqual(report["decision_status"], "uncalibrated_prediction")
        self.assertEqual(report["verdict"], "Likely deepfake")
        self.assertEqual(
            report["model_results"]["xception"]["explainability"]["status"],
            "generated",
        )
        self.assertIn("raw, uncalibrated softmax", report["decision_basis"])
        self.assertIn("deepfake softmax score", report["evidence_assessment"][0]["interpretation"])
        pipeline_by_name = {stage["name"]: stage for stage in report["model_pipeline"]}
        self.assertEqual(pipeline_by_name["Calibration and operating thresholds"]["status"], "not_calibrated")
        self.assertEqual(pipeline_by_name["Original / deepfake decision"]["status"], "uncalibrated_prediction")
        self.assertIn("raw, uncalibrated", pipeline_by_name["Xception inference"]["detail"])
        self.assertEqual(pipeline_by_name["Grad-CAM explainability"]["status"], "generated")

    def test_conformal_abstention_withholds_a_threshold_selected_image_class(self):
        image_path = self.root / "conformal.png"
        Image.new("RGB", (24, 16), "navy").save(image_path)
        result = {
            "status": "conformal_abstention",
            "decision": "not_determined",
            "verdict": "Insufficient evidence",
            "deepfake_probability": 0.12,
            "authenticity_probability": 0.88,
            "conformal_prediction": {
                "prediction_set": [],
                "alpha": 0.1,
                "abstained": True,
                "scope": "Exchangeability assumed.",
            },
            "model": {
                "architecture": "timm:legacy_xception",
                "validation_auc": 0.8,
                "calibration": "held-out temperature scaling",
                "calibration_temperature": 1.0,
                "thresholds": {
                    "real_max_probability": 0.2,
                    "fake_min_probability": 0.8,
                },
            },
            "face_review": {
                "detected_face_count": 0,
                "status": "no_face_detected",
                "selected_crop": "full_image_fallback",
                "alignment": "not_performed_landmarks_unavailable",
            },
            "limitations": ["Domain shift is not covered."],
        }

        with patch("app.services.forensics.get_xception_detector") as detector_factory:
            detector_factory.return_value.predict.return_value = result
            report = analyze_media(image_path, image_path.name)

        stages = {stage["number"]: stage for stage in report["model_pipeline"]}
        assert report["verdict"] == "Insufficient evidence"
        assert report["decision_status"] == "conformal_abstention"
        assert stages[9]["status"] == "conformal_abstention"
        assert stages[10]["status"] == "withheld"
        assert "prediction set" in report["decision_basis"]

    def test_synthetic_demo_illustration_is_never_sent_to_the_classifier(self):
        image_path = (
            Path(__file__).resolve().parents[1]
            / "public"
            / "examples"
            / "sample_edited_illustration.png"
        )

        with patch(
            "app.services.forensics.get_xception_detector",
            side_effect=AssertionError("synthetic demo samples must bypass inference"),
        ):
            report = analyze_media(image_path, image_path.name)

        self.assertEqual(report["verdict"], "Insufficient evidence")
        self.assertEqual(report["decision_status"], "demo_sample_not_classified")
        self.assertIsNone(report["deepfake_probability"])
        self.assertTrue(report["evidence_context"]["is_synthetic_demo"])
        pipeline_by_name = {stage["name"]: stage for stage in report["model_pipeline"]}
        self.assertEqual(pipeline_by_name["Xception checkpoint"]["status"], "skipped_demo")
        self.assertEqual(pipeline_by_name["Xception inference"]["status"], "not_run")
        self.assertEqual(pipeline_by_name["Original / deepfake decision"]["status"], "withheld")
        self.assertEqual(
            report["evidence_assessment"][0]["status"],
            "measured_descriptive"
            if report["face_review"]["status"] in {"face_detected", "no_face_detected"}
            else "not_configured",
        )
        self.assertEqual(report["frequency_analysis"]["status"], "measured_descriptive")

    def test_demo_filename_does_not_make_unrelated_upload_a_synthetic_demo(self):
        image_path = self.root / "sample_edited_illustration.png"
        Image.new("RGB", (8, 5), "navy").save(image_path)

        with patch("app.services.forensics.get_xception_detector", return_value=None):
            report = analyze_media(image_path, image_path.name)

        self.assertFalse(report["evidence_context"]["is_synthetic_demo"])
        self.assertEqual(report["decision_status"], "not_configured")

    def test_image_exif_values_are_returned_only_when_present(self):
        image_path = self.root / "tagged.jpg"
        exif = Image.Exif()
        exif[305] = "Unit Test Camera Software"
        exif[36867] = "2026:10:04 12:30:00"
        Image.new("RGB", (8, 5), "navy").save(image_path, exif=exif)

        report = analyze_media(image_path, image_path.name)

        self.assertEqual(report["metadata_summary"]["exif_fields"]["software"], "Unit Test Camera Software")
        self.assertEqual(report["metadata_summary"]["exif_fields"]["date_time_original"], "2026:10:04 12:30:00")

    def test_gps_coordinates_are_converted_only_from_valid_exif_values(self):
        self.assertEqual(
            gps_decimal_coordinate((40, 30, 0), "N", "latitude"),
            40.5,
        )
        self.assertEqual(
            gps_decimal_coordinate((73, 59, 0), "W", "longitude"),
            -73.9833333,
        )
        self.assertIsNone(gps_decimal_coordinate((91, 0, 0), "N", "latitude"))
        self.assertIsNone(gps_decimal_coordinate((40, 60, 0), "N", "latitude"))
        self.assertIsNone(gps_decimal_coordinate((40, 0, 0), "X", "latitude"))

    def test_animated_gif_records_frame_count_without_claiming_authenticity(self):
        gif_path = self.root / "animated.gif"
        frames = [
            Image.new("RGB", (16, 12), "red"),
            Image.new("RGB", (16, 12), "blue"),
        ]
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=100, loop=0)

        report = analyze_media(gif_path, gif_path.name)

        self.assertEqual(report["media_type"], "image")
        self.assertTrue(report["metadata_summary"]["is_animated"])
        self.assertEqual(report["metadata_summary"]["frame_count"], 2)
        self.assertEqual(report["verdict"], "Insufficient evidence")
        self.assertEqual(report["decision_status"], "not_applicable_animated_image")
        self.assertIsNone(report["authenticity_probability"])
        self.assertIsNone(report["deepfake_probability"])
        self.assertTrue(any(
            finding["signal"] == "Animated image frames present"
            for finding in report["metadata_findings"]
        ))

    def test_audio_report_includes_descriptive_signal_without_authenticity_verdict(self):
        audio_path = self.root / "tone.wav"
        sf.write(audio_path, np.sin(np.linspace(0, 8, 512, dtype=np.float32)), 16000)

        report = analyze_media(audio_path, audio_path.name)
        evidence_by_name = {item["name"]: item for item in report["evidence_assessment"]}

        self.assertEqual(report["audio_analysis"]["signal_status"], "measured_descriptive")
        self.assertEqual(report["audio_analysis"]["audio_authenticity_status"], "not_configured")
        self.assertEqual(evidence_by_name["Frequency / compression analysis"]["status"], "measured_descriptive")
        self.assertEqual(evidence_by_name["Temporal analysis"]["status"], "not_applicable")
        self.assertEqual(report["verdict"], "Insufficient evidence")
        self.assertIsNone(report["deepfake_probability"])
        self.assertTrue(any("descriptive waveform" in item["message"] for item in report["audit_log"]))

    def test_video_report_surfaces_sampled_frame_observations_without_verdict(self):
        video_path = self.root / "clip.mp4"
        video_path.write_bytes(b"\x00\x00\x00\x18ftypisom1234")
        metadata = {
            "status": "ok",
            "width": 1280,
            "height": 720,
            "decoder_reported_fps": 25.0,
            "container_reported_frame_count": 250,
            "duration_estimate_seconds": 10.0,
        }
        video_review = {
            "status": "measured_descriptive",
            "sampled_frame_count": 1,
            "frames": [{
                "index": 0,
                "timestamp_seconds": 0.0,
                "face_review": {"status": "no_face_detected", "detected_face_count": 0},
            }],
            "interpretation": "Sparse keyframe review only.",
        }

        with (
            patch("app.services.forensics.get_basic_video_metadata", return_value=metadata),
            patch("app.services.forensics.get_xception_detector", return_value=None),
            patch("app.services.forensics.analyze_video", return_value=video_review),
        ):
            report = analyze_media(video_path, video_path.name)

        evidence_by_name = {item["name"]: item for item in report["evidence_assessment"]}
        self.assertEqual(report["video_analysis"]["sampled_frame_count"], 1)
        self.assertEqual(evidence_by_name["Temporal analysis"]["status"], "measured_descriptive")
        self.assertEqual(evidence_by_name["Spatial / face analysis"]["status"], "measured_descriptive")
        self.assertEqual(report["verdict"], "Insufficient evidence")
        self.assertIsNone(report["deepfake_probability"])
        self.assertTrue(any("keyframes" in item["message"] for item in report["audit_log"]))

    def test_video_frame_scores_do_not_become_a_clip_verdict(self):
        video_path = self.root / "clip.mp4"
        video_path.write_bytes(b"\x00\x00\x00\x18ftypisom1234")
        metadata = {
            "status": "ok",
            "width": 1280,
            "height": 720,
            "decoder_reported_fps": 25.0,
            "container_reported_frame_count": 250,
            "duration_estimate_seconds": 10.0,
        }
        frame_analysis = {
            "status": "frame_estimates_available",
            "architecture": "RamadhanZome/deepfake-xception",
            "calibration": "uncalibrated raw softmax scores",
            "analyzed_frame_count": 2,
            "sampled_frame_count": 2,
            "video_level_decision": "withheld",
            "interpretation": "Frame scores only; no clip result.",
        }
        video_review = {
            "status": "measured_descriptive",
            "sampled_frame_count": 2,
            "frames": [],
            "model_frame_analysis": frame_analysis,
            "interpretation": "Frame scores only; no clip result.",
        }

        with (
            patch("app.services.forensics.get_basic_video_metadata", return_value=metadata),
            patch("app.services.forensics.get_xception_detector", return_value=object()),
            patch("app.services.forensics.analyze_video", return_value=video_review),
        ):
            report = analyze_media(video_path, video_path.name)

        stages = {stage["number"]: stage for stage in report["model_pipeline"]}
        assert report["decision_status"] == "frame_estimates_only"
        assert report["verdict"] == "Insufficient evidence"
        assert report["deepfake_probability"] is None
        assert report["model_results"]["video_xception_frames"] == frame_analysis
        assert stages[8]["status"] == "sampled_frames_only"
        assert stages[9]["status"] == "not_validated_for_video"
        assert stages[10]["status"] == "withheld"


if __name__ == "__main__":
    unittest.main()
