from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from app.api import routes
from app.api.routes import router
from forensics.image_comparison import compare_image_reports


def _report(filename, file_hash, verdict="Insufficient evidence", status="not_configured"):
    return {
        "filename": filename,
        "file_hash": file_hash,
        "verdict": verdict,
        "decision_status": status,
        "authenticity_probability": None,
        "deepfake_probability": None,
        "decision_basis": "Detector not configured.",
        "metadata_summary": {"width": 4, "height": 4},
        "face_review": {
            "status": "no_face_detected",
            "detected_face_count": 0,
        },
    }


def test_pair_comparison_never_calls_unconfigured_result_original_or_fake(tmp_path):
    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    Image.new("RGB", (4, 4), "black").save(image_a)
    Image.new("RGB", (4, 4), "white").save(image_b)

    result = compare_image_reports(
        image_a,
        image_b,
        _report("a.png", "a" * 64),
        _report("b.png", "b" * 64),
    )

    assert result["status"] == "inconclusive"
    assert "withheld" in result["finding"]
    assert result["images"]["image_a"]["face_observation"]["candidate_count"] == 0
    assert result["pixel_difference"]["changed_pixel_ratio"] == 1.0
    assert any("do not identify people" in item for item in result["limitations"])


def test_pair_comparison_qualifies_opposed_model_classes(tmp_path):
    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    Image.new("RGB", (4, 4), "black").save(image_a)
    Image.new("RGB", (4, 4), "white").save(image_b)

    result = compare_image_reports(
        image_a,
        image_b,
        _report("a.png", "a" * 64, "Likely original", "classified"),
        _report("b.png", "b" * 64, "Likely deepfake", "classified"),
    )

    assert result["status"] == "model_supported_contrast"
    assert result["likely_manipulated_image"] == "image_b"
    assert "not confirmation or proof" in result["finding"]


def test_compare_images_endpoint_validates_images_and_returns_report(tmp_path, monkeypatch):
    monkeypatch.setattr(routes, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr("app.services.forensics.get_xception_detector", lambda _: None)
    app = FastAPI()
    app.include_router(router)

    from io import BytesIO

    def image_bytes(color):
        stream = BytesIO()
        Image.new("RGB", (16, 12), color).save(stream, format="PNG")
        return stream.getvalue()

    with TestClient(app) as client:
        response = client.post(
            "/api/compare-images",
            files={
                "image_a": ("original.png", image_bytes("navy"), "image/png"),
                "image_b": ("edited.png", image_bytes("orange"), "image/png"),
            },
        )

    assert response.status_code == 200
    report = response.json()
    assert report["report_type"] == "two_image_comparison"
    assert report["status"] == "inconclusive"
    assert report["images"]["image_a"]["model_assessment"]["verdict"] == "Insufficient evidence"
    assert report["pixel_difference"]["status"] == "measured"
    assert not list(tmp_path.glob("comparison-*"))


def test_compare_images_endpoint_rejects_media_with_mismatched_extension(tmp_path, monkeypatch):
    monkeypatch.setattr(routes, "UPLOAD_DIR", tmp_path)
    app = FastAPI()
    app.include_router(router)

    from io import BytesIO

    stream = BytesIO()
    Image.new("RGB", (8, 8), "navy").save(stream, format="PNG")
    valid_png = stream.getvalue()
    with TestClient(app) as client:
        response = client.post(
            "/api/compare-images",
            files={
                "image_a": ("first.png", valid_png, "image/png"),
                "image_b": ("second.jpg", valid_png, "image/jpeg"),
            },
        )

    assert response.status_code == 415
    assert "extension does not match" in response.json()["detail"]
