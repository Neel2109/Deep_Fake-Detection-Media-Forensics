import csv
from io import StringIO
import hashlib

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from app.api import routes
from app.api.routes import _public_report, router, save_analyst_review
from app.schemas import AnalystReview
from forensics.chain_of_custody import seal_audit_log


def test_public_report_omits_internal_evidence_path():
    report = {"case_id": "case-1", "file_path": "C:\\private\\uploads\\evidence.png"}

    assert _public_report(report) == {
        "case_id": "case-1",
        "chain_of_custody": {"verification_status": "not_sealed"},
    }


def test_audit_csv_escapes_spreadsheet_formulas():
    app = FastAPI()
    app.include_router(router)
    app.state.reports = [{
        "case_id": "=1+1",
        "filename": "+cmd",
        "media_type": "image",
        "audit_log": [{"timestamp": "2026-01-01", "message": "@SUM(A1:A2)"}],
    }]

    with TestClient(app) as client:
        response = client.get("/api/reports/audit.csv")

    assert response.status_code == 200
    rows = list(csv.reader(StringIO(response.text)))
    assert rows[1] == ["'=1+1", "'+cmd", "image", "2026-01-01", "", "'@SUM(A1:A2)", "", ""]


def test_reanalysis_refreshes_pipeline_after_verifying_stored_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr("app.services.forensics.get_xception_detector", lambda _: None)
    upload_root = tmp_path / "uploads"
    upload_root.mkdir()
    evidence = upload_root / "evidence.png"
    Image.new("RGB", (24, 16), "navy").save(evidence)
    report = {
        "case_id": "case-1",
        "filename": "evidence.png",
        "file_hash": hashlib.sha256(evidence.read_bytes()).hexdigest(),
        "file_path": str(evidence),
        "media_type": "image",
        "created_at": "2026-01-01T00:00:00+00:00",
        "audit_log": [{"timestamp": "2026-01-01", "message": "Original upload received."}],
    }
    seal_audit_log(report)
    case = {"case_id": "case-1", "title": "Evidence review", "created_at": report["created_at"]}
    app = FastAPI()
    app.include_router(router)
    app.state.reports = [report]
    app.state.cases = [case]
    app.state.evidence_files = {}
    app.state.evidence_tokens = {}
    monkeypatch.setattr(routes, "UPLOAD_DIR", upload_root)
    monkeypatch.setattr(routes, "save_analysis", lambda updated, saved_case: None)

    with TestClient(app) as client:
        response = client.post("/api/reports/case-1/reanalyze")

    assert response.status_code == 200
    refreshed = response.json()
    assert refreshed["file_hash"] == report["file_hash"]
    assert len(refreshed["model_pipeline"]) == 14
    assert refreshed["decision_status"] == "not_configured"
    assert "file_path" not in refreshed
    assert app.state.reports[0]["model_pipeline"] == refreshed["model_pipeline"]
    assert refreshed["chain_of_custody"]["verification_status"] == "verified"
    assert refreshed["audit_log"][-1]["event_type"] == "evidence_reanalyzed"


def test_reanalysis_refuses_modified_stored_evidence(tmp_path, monkeypatch):
    upload_root = tmp_path / "uploads"
    upload_root.mkdir()
    evidence = upload_root / "evidence.png"
    evidence.write_bytes(b"original evidence")
    report = {
        "case_id": "case-1",
        "filename": "evidence.png",
        "file_hash": hashlib.sha256(evidence.read_bytes()).hexdigest(),
        "file_path": str(evidence),
    }
    evidence.write_bytes(b"changed evidence")
    app = FastAPI()
    app.include_router(router)
    app.state.reports = [report]
    app.state.cases = []
    monkeypatch.setattr(routes, "UPLOAD_DIR", upload_root)

    with TestClient(app) as client:
        response = client.post("/api/reports/case-1/reanalyze")

    assert response.status_code == 409
    assert "SHA-256" in response.json()["detail"]


def test_reanalysis_refuses_a_tampered_report_audit_chain(tmp_path, monkeypatch):
    upload_root = tmp_path / "uploads"
    upload_root.mkdir()
    evidence = upload_root / "evidence.png"
    Image.new("RGB", (24, 16), "navy").save(evidence)
    report = {
        "case_id": "case-1",
        "filename": "evidence.png",
        "file_hash": hashlib.sha256(evidence.read_bytes()).hexdigest(),
        "file_path": str(evidence),
        "media_type": "image",
        "created_at": "2026-01-01T00:00:00+00:00",
        "audit_log": [{"timestamp": "2026-01-01", "message": "Received."}],
    }
    seal_audit_log(report)
    report["audit_log"][0]["message"] = "Edited."
    app = FastAPI()
    app.include_router(router)
    app.state.reports = [report]
    app.state.cases = []
    monkeypatch.setattr(routes, "UPLOAD_DIR", upload_root)

    with TestClient(app) as client:
        response = client.post("/api/reports/case-1/reanalyze")

    assert response.status_code == 409
    assert "audit chain" in response.json()["detail"]


def test_public_report_marks_tampered_audit_chain():
    report = {
        "case_id": "case-1",
        "file_hash": "a" * 64,
        "audit_log": [{"timestamp": "2026-01-01", "message": "Received."}],
    }
    seal_audit_log(report)
    report["audit_log"][0]["message"] = "Edited."

    assert _public_report(report)["chain_of_custody"]["verification_status"] == "verification_failed"


def test_analyst_review_is_appended_to_the_verified_chain(monkeypatch):
    report = {
        "case_id": "case-1",
        "file_hash": "a" * 64,
        "created_at": "2026-01-01T00:00:00+00:00",
        "audit_log": [{"timestamp": "2026-01-01", "message": "Received."}],
    }
    seal_audit_log(report)
    app = FastAPI()
    app.include_router(router)
    app.state.reports = [report]
    app.state.cases = []
    monkeypatch.setattr(routes, "save_report", lambda updated: None)

    with TestClient(app) as client:
        response = client.patch(
            "/api/reports/case-1/review",
            json={
                "outcome": "inconclusive",
                "rationale": "The available evidence is insufficient for a conclusion.",
                "reviewer": "Investigator",
            },
        )

    assert response.status_code == 200
    assert report["audit_log"][-1]["event_type"] == "analyst_review_recorded"
    assert report["audit_log"][-1]["details"]["outcome"] == "inconclusive"
    assert _public_report(report)["chain_of_custody"]["verification_status"] == "verified"
