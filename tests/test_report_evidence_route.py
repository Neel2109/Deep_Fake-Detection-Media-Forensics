import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.responses import FileResponse

from app.api.routes import get_report_evidence


class ReportEvidenceRouteTests(unittest.TestCase):
    def test_returns_image_evidence_from_the_upload_directory(self):
        with TemporaryDirectory() as temporary_directory:
            upload_root = Path(temporary_directory)
            image_path = upload_root / "stored-evidence.png"
            image_path.write_bytes(b"\x89PNG\r\n\x1a\n")
            report = {
                "case_id": "case-1",
                "filename": "source.png",
                "media_type": "image",
                "evidence_preview_token": "valid-test-token",
                "file_integrity": {"detected_mime_type": "image/png"},
            }
            request = SimpleNamespace(
                app=SimpleNamespace(
                    state=SimpleNamespace(
                        reports=[report],
                        evidence_files={"case-1": str(image_path)},
                        evidence_tokens={"case-1": "valid-test-token"},
                    )
                )
            )

            with patch("app.api.routes.UPLOAD_DIR", upload_root):
                response = asyncio.run(get_report_evidence("case-1", request, "valid-test-token"))

            self.assertIsInstance(response, FileResponse)
            self.assertEqual(Path(response.path), image_path)
            self.assertEqual(response.media_type, "image/png")
            self.assertIn("inline", response.headers["content-disposition"])

    def test_rejects_evidence_path_outside_the_upload_directory(self):
        with TemporaryDirectory() as temporary_directory:
            upload_root = Path(temporary_directory) / "uploads"
            upload_root.mkdir()
            outside_path = Path(temporary_directory) / "outside.png"
            outside_path.write_bytes(b"not evidence")
            request = SimpleNamespace(
                app=SimpleNamespace(
                    state=SimpleNamespace(
                        reports=[{
                            "case_id": "case-1",
                            "filename": "source.png",
                            "media_type": "image",
                            "evidence_preview_token": "valid-test-token",
                            "file_integrity": {"detected_mime_type": "image/png"},
                        }],
                        evidence_files={"case-1": str(outside_path)},
                        evidence_tokens={"case-1": "valid-test-token"},
                    )
                )
            )

            with patch("app.api.routes.UPLOAD_DIR", upload_root):
                with self.assertRaises(HTTPException) as error:
                    asyncio.run(get_report_evidence("case-1", request, "valid-test-token"))

            self.assertEqual(error.exception.status_code, 404)

    def test_does_not_return_non_image_evidence_as_an_image_preview(self):
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(
                    reports=[{
                        "case_id": "case-1",
                        "filename": "source.mp4",
                        "media_type": "video",
                        "evidence_preview_token": "valid-test-token",
                    }],
                    evidence_files={},
                    evidence_tokens={"case-1": "valid-test-token"},
                )
            )
        )
        with self.assertRaises(HTTPException) as error:
            asyncio.run(get_report_evidence("case-1", request, "valid-test-token"))

        self.assertEqual(error.exception.status_code, 415)

    def test_rejects_missing_or_invalid_evidence_preview_token(self):
        with TemporaryDirectory() as temporary_directory:
            upload_root = Path(temporary_directory)
            image_path = upload_root / "stored-evidence.png"
            image_path.write_bytes(b"\x89PNG\r\n\x1a\n")
            request = SimpleNamespace(
                app=SimpleNamespace(
                    state=SimpleNamespace(
                        reports=[{
                            "case_id": "case-1",
                            "filename": "source.png",
                            "media_type": "image",
                            "evidence_preview_token": "valid-test-token",
                            "file_integrity": {"detected_mime_type": "image/png"},
                        }],
                        evidence_files={"case-1": str(image_path)},
                        evidence_tokens={"case-1": "valid-test-token"},
                    )
                )
            )

            with patch("app.api.routes.UPLOAD_DIR", upload_root):
                for token in (None, "wrong-token"):
                    with self.subTest(token=token):
                        with self.assertRaises(HTTPException) as error:
                            asyncio.run(get_report_evidence("case-1", request, token))
                        self.assertEqual(error.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
