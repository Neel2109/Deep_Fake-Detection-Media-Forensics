"""Tests for honest face-candidate availability reporting."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from image.face_detector import inspect_face_candidates


class FaceCandidateTests(unittest.TestCase):
    def test_unavailable_cascade_does_not_claim_zero_faces_were_found(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "evidence.png"
            Image.new("RGB", (8, 8), "white").save(image_path)
            unavailable_result = {
                "detector": "OpenCV Haar cascade",
                "status": "unavailable",
                "detected_face_count": 0,
                "selected_crop": "full_image_fallback",
                "alignment": "not_performed_landmarks_unavailable",
            }
            with patch(
                "image.face_detector.detect_largest_face_crop",
                return_value=(Image.new("RGB", (8, 8), "white"), unavailable_result),
            ):
                review = inspect_face_candidates(image_path)

        self.assertEqual(review["status"], "unavailable")
        self.assertIn("No face count", review["interpretation"])
        self.assertIn("detector is unavailable", review["interpretation"])


if __name__ == "__main__":
    unittest.main()
