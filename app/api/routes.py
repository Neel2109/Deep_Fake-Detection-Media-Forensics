import hmac
import csv
from io import StringIO
import secrets
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
import sqlite3

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from PIL import UnidentifiedImageError

from app.config import MAX_UPLOAD_BYTES, UPLOAD_DIR, ensure_storage_dirs
from app.schemas import AnalystReview, CaseCreate, UserLogin, UserRegister
from app.services.forensics import analyze_media, detect_media_type, inspect_file_signature, sha256_file
from app.services.xception_detector import XceptionModelError
from database.connection import save_analysis, save_case, save_report
from forensics.chain_of_custody import append_audit_event, verify_audit_chain
from forensics.image_comparison import compare_image_reports

router = APIRouter(prefix="/api")


def _public_report(report: dict) -> dict:
    public_report = {key: value for key, value in report.items() if key != "file_path"}
    chain = public_report.get("chain_of_custody")
    if chain:
        public_report["chain_of_custody"] = {
            **chain,
            "verification_status": "verified" if verify_audit_chain(report) else "verification_failed",
        }
    else:
        public_report["chain_of_custody"] = {"verification_status": "not_sealed"}
    return public_report


@router.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "DeepTrace AI", "timestamp": datetime.now(timezone.utc).isoformat()}


@router.post("/auth/register")
async def register_user(payload: UserRegister) -> dict:
    return {
        "user_id": f"user-{uuid.uuid4().hex[:8]}",
        "email": payload.email,
        "full_name": payload.full_name,
        "role": "analyst",
        "status": "registered",
    }


@router.post("/auth/login")
async def login_user(payload: UserLogin) -> dict:
    if payload.email and payload.password:
        return {
            "access_token": f"demo-token-{uuid.uuid4().hex[:12]}",
            "token_type": "bearer",
            "user": {"email": payload.email, "role": "analyst"},
        }
    raise HTTPException(status_code=401, detail="Invalid credentials")


@router.get("/cases")
async def list_cases(request: Request) -> dict:
    cases = getattr(request.app.state, "cases", [])
    return {"items": cases}


@router.post("/cases")
async def create_case(payload: CaseCreate, request: Request) -> dict:
    case_id = f"case-{uuid.uuid4().hex[:8]}"
    case = {
        "case_id": case_id,
        "title": payload.title,
        "category": payload.category,
        "investigator": payload.investigator,
        "description": payload.description,
        "priority": payload.priority,
        "status": "new",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    save_case(case)
    request.app.state.cases.append(case)
    return case


@router.post("/compare-images")
async def compare_images(image_a: UploadFile = File(...), image_b: UploadFile = File(...)) -> dict:
    ensure_storage_dirs()
    uploads = (image_a, image_b)
    try:
        with tempfile.TemporaryDirectory(prefix="comparison-", dir=UPLOAD_DIR) as temp_directory:
            image_paths: list[Path] = []
            image_reports: list[dict] = []
            for index, upload in enumerate(uploads):
                if not upload.filename:
                    raise HTTPException(status_code=400, detail=f"Image {index + 1} needs a filename")
                filename = Path(upload.filename).name
                if detect_media_type(filename) != "image":
                    raise HTTPException(status_code=415, detail=f"Image {index + 1} must be a supported image file")

                image_path = Path(temp_directory) / f"image-{index + 1}{Path(filename).suffix.lower()}"
                total_size = 0
                with image_path.open("wb") as handle:
                    while chunk := await upload.read(1024 * 1024):
                        total_size += len(chunk)
                        if total_size > MAX_UPLOAD_BYTES:
                            raise HTTPException(
                                status_code=413,
                                detail=f"Each image must be no larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB",
                            )
                        handle.write(chunk)
                if total_size == 0:
                    raise HTTPException(status_code=400, detail=f"Image {index + 1} is empty")

                signature = inspect_file_signature(image_path, filename)
                if not signature["signature_recognized"] or not signature["detected_mime_type"].startswith("image/"):
                    raise HTTPException(status_code=415, detail=f"Image {index + 1} has an unsupported file signature")
                if not signature["extension_matches_signature"]:
                    raise HTTPException(
                        status_code=415,
                        detail=f"Image {index + 1} extension does not match its detected content type",
                    )

                report = analyze_media(image_path, filename, case_id=f"comparison-image-{index + 1}")
                if report["metadata_summary"].get("is_animated"):
                    raise HTTPException(status_code=415, detail="Animated images are not supported for pair comparison")
                image_paths.append(image_path)
                image_reports.append(report)

            return compare_image_reports(
                image_paths[0],
                image_paths[1],
                image_reports[0],
                image_reports[1],
            )
    except XceptionModelError as error:
        raise HTTPException(
            status_code=503,
            detail=f"The configured image detector could not analyze the comparison: {error}",
        ) from error
    except UnidentifiedImageError as error:
        raise HTTPException(status_code=400, detail="An uploaded image is invalid or corrupted") from error
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=400, detail=f"Could not compare uploaded images: {error}") from error
    finally:
        for upload in uploads:
            await upload.close()


@router.post("/analyze")
async def analyze_upload(
    request: Request,
    file: UploadFile = File(...),
    case_title: str = Form("Case title"),
    investigator: str = Form("Analyst"),
    case_id: str = Form("case-unknown"),
):
    ensure_storage_dirs()
    if not file.filename:
        raise HTTPException(status_code=400, detail="A file is required for analysis")

    original_name = Path(file.filename).name
    if detect_media_type(original_name) == "unsupported":
        raise HTTPException(status_code=415, detail="Unsupported media extension")
    file_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{original_name}"
    total_size = 0
    try:
        with file_path.open("wb") as handle:
            while chunk := await file.read(1024 * 1024):
                total_size += len(chunk)
                if total_size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Upload exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB size limit",
                    )
                handle.write(chunk)

        signature = inspect_file_signature(file_path, original_name)
        if not signature["signature_recognized"]:
            raise HTTPException(status_code=415, detail="File signature is not a supported image, video, or audio format")
        if not signature["extension_matches_signature"]:
            raise HTTPException(
                status_code=415,
                detail=f"File extension does not match detected content type ({signature['detected_mime_type']})",
            )
        report = analyze_media(file_path, original_name, case_id)
    except XceptionModelError as error:
        file_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=503,
            detail=f"Configured Xception detector could not analyze this media: {error}",
        ) from error
    except UnidentifiedImageError as error:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Image data is invalid or corrupted") from error
    except (OSError, ValueError) as error:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Could not inspect uploaded media: {error}") from error
    except HTTPException:
        file_path.unlink(missing_ok=True)
        raise
    finally:
        await file.close()

    report["case_title"] = case_title
    report["investigator"] = investigator
    report["file_path"] = str(file_path)
    append_audit_event(
        report,
        event_type="case_created",
        timestamp=datetime.now(timezone.utc).isoformat(),
        message="Case details were associated with the uploaded evidence.",
        details={"case_title": case_title, "investigator": investigator},
    )
    preview_token = secrets.token_urlsafe(32) if report["media_type"] == "image" else None
    if preview_token:
        report["evidence_preview_token"] = preview_token

    case_record = {
        "case_id": case_id,
        "title": case_title,
        "category": f"{report['media_type'].title()} authenticity review",
        "investigator": investigator,
        "description": "Case created from a media analysis upload.",
        "priority": "Medium",
        "status": "awaiting_review",
        "created_at": report["created_at"],
        "evidence_count": 1,
    }
    try:
        save_analysis(report, case_record)
    except (OSError, sqlite3.Error, TypeError, ValueError) as error:
        file_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=503,
            detail=f"Could not persist the analysis record: {error}",
        ) from error
    reports = request.app.state.reports
    existing_report_index = next(
        (index for index, existing in enumerate(reports) if existing.get("case_id") == case_id),
        None,
    )
    if existing_report_index is None:
        reports.append(report)
    else:
        reports[existing_report_index] = report

    cases = request.app.state.cases
    existing_case_index = next(
        (index for index, existing in enumerate(cases) if existing.get("case_id") == case_id),
        None,
    )
    if existing_case_index is None:
        cases.append(case_record)
    else:
        cases[existing_case_index] = case_record
    if preview_token:
        request.app.state.evidence_files[case_id] = str(file_path)
        request.app.state.evidence_tokens[case_id] = preview_token
    return _public_report(report)


@router.get("/reports")
async def list_reports(request: Request) -> dict:
    return {"items": [_public_report(report) for report in getattr(request.app.state, "reports", [])]}


@router.get("/reports/audit.csv")
async def export_audit_csv(request: Request) -> StreamingResponse:
    output = StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow([
        "case_id",
        "filename",
        "media_type",
        "timestamp",
        "event_type",
        "event",
        "previous_event_sha256",
        "event_sha256",
    ])

    def safe_csv_value(value: object) -> str:
        text = str(value)
        if text.startswith(("=", "+", "-", "@", "\t", "\r")):
            return "'" + text
        return text

    for report in getattr(request.app.state, "reports", []):
        for entry in report.get("audit_log", []):
            writer.writerow([
                safe_csv_value(report.get("case_id", "")),
                safe_csv_value(report.get("filename", "")),
                safe_csv_value(report.get("media_type", "")),
                safe_csv_value(entry.get("timestamp", "")),
                safe_csv_value(entry.get("event_type", "")),
                safe_csv_value(entry.get("message", "")),
                safe_csv_value(entry.get("previous_event_sha256", "")),
                safe_csv_value(entry.get("event_sha256", "")),
            ])
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="deeptrace-audit-history.csv"'},
    )


@router.get("/reports/{report_id}")
async def get_report(report_id: str, request: Request) -> dict:
    reports = getattr(request.app.state, "reports", [])
    report = next((item for item in reports if item.get("case_id") == report_id), None)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")
    return _public_report(report)


@router.post("/reports/{report_id}/reanalyze")
async def reanalyze_stored_evidence(report_id: str, request: Request) -> dict:
    reports = getattr(request.app.state, "reports", [])
    existing = next((item for item in reports if item.get("case_id") == report_id), None)
    if existing is None:
        raise HTTPException(status_code=404, detail="Report not found")
    if existing.get("chain_of_custody") and not verify_audit_chain(existing):
        raise HTTPException(
            status_code=409,
            detail="The stored report audit chain failed verification; reanalysis was stopped.",
        )

    source_path = existing.get("file_path")
    if not source_path or not existing.get("file_hash"):
        raise HTTPException(status_code=404, detail="Stored evidence is not available for reanalysis")

    try:
        upload_root = UPLOAD_DIR.resolve()
        resolved_path = Path(source_path).resolve(strict=True)
        resolved_path.relative_to(upload_root)
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=404, detail="Stored evidence is not available for reanalysis") from error
    if not resolved_path.is_file():
        raise HTTPException(status_code=404, detail="Stored evidence is not available for reanalysis")
    if not hmac.compare_digest(sha256_file(resolved_path), str(existing["file_hash"])):
        raise HTTPException(
            status_code=409,
            detail="Stored evidence bytes no longer match the report SHA-256; reanalysis was stopped.",
        )

    try:
        updated = analyze_media(
            resolved_path,
            Path(existing.get("filename", resolved_path.name)).name,
            report_id,
        )
    except XceptionModelError as error:
        raise HTTPException(
            status_code=503,
            detail=f"Configured Xception detector could not analyze this evidence: {error}",
        ) from error
    except UnidentifiedImageError as error:
        raise HTTPException(status_code=400, detail="Stored image is invalid or corrupted") from error
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=400, detail=f"Could not reanalyze stored evidence: {error}") from error

    updated["case_title"] = existing.get("case_title", "Case title")
    updated["investigator"] = existing.get("investigator", "Analyst")
    updated["file_path"] = str(resolved_path)
    if existing.get("evidence_preview_token"):
        updated["evidence_preview_token"] = existing["evidence_preview_token"]
    if existing.get("analyst_review"):
        updated["analyst_review"] = existing["analyst_review"]
    updated["audit_log"] = list(existing.get("audit_log", []))
    if existing.get("chain_of_custody"):
        updated["chain_of_custody"] = existing["chain_of_custody"]
    else:
        updated.pop("chain_of_custody", None)
    try:
        append_audit_event(
            updated,
            event_type="evidence_reanalyzed",
            timestamp=updated["created_at"],
            message="Stored evidence reanalyzed after its current SHA-256 was verified against the original report.",
            details={"verified_evidence_sha256": existing["file_hash"]},
        )
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error

    cases = getattr(request.app.state, "cases", [])
    case_record = next((item for item in cases if item.get("case_id") == report_id), None)
    if case_record is not None:
        save_analysis(updated, case_record)
    else:
        save_report(updated)
    reports[reports.index(existing)] = updated
    return _public_report(updated)


@router.get("/reports/{report_id}/evidence")
async def get_report_evidence(
    report_id: str,
    request: Request,
    token: str | None = Query(default=None),
) -> FileResponse:
    reports = getattr(request.app.state, "reports", [])
    report = next((item for item in reports if item.get("case_id") == report_id), None)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")
    if report.get("chain_of_custody") and not verify_audit_chain(report):
        raise HTTPException(
            status_code=409,
            detail="The stored report audit chain failed verification; analyst review was not recorded.",
        )
    if report.get("media_type") != "image":
        raise HTTPException(status_code=415, detail="Evidence preview is only available for images")

    evidence_tokens = getattr(request.app.state, "evidence_tokens", {})
    expected_token = evidence_tokens.get(report_id) or report.get("evidence_preview_token")
    if not token or not expected_token or not hmac.compare_digest(token, expected_token):
        raise HTTPException(status_code=404, detail="Stored evidence is not available")
    evidence_files = getattr(request.app.state, "evidence_files", {})
    source_path = evidence_files.get(report_id) or report.get("file_path")
    if not source_path:
        raise HTTPException(status_code=404, detail="Stored evidence is not available")
    upload_root = UPLOAD_DIR.resolve()
    try:
        resolved_path = Path(source_path).resolve(strict=True)
        resolved_path.relative_to(upload_root)
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=404, detail="Stored evidence is not available") from error
    if not resolved_path.is_file():
        raise HTTPException(status_code=404, detail="Stored evidence is not available")

    media_type = report.get("file_integrity", {}).get("detected_mime_type", "image/png")
    if not media_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Stored evidence is not a recognized image")
    return FileResponse(
        resolved_path,
        media_type=media_type,
        filename=Path(report.get("filename", "evidence")).name,
        content_disposition_type="inline",
    )


@router.patch("/reports/{report_id}/review")
async def save_analyst_review(report_id: str, payload: AnalystReview, request: Request) -> dict:
    reports = getattr(request.app.state, "reports", [])
    report = next((item for item in reports if item.get("case_id") == report_id), None)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")

    reviewed_at = datetime.now(timezone.utc).isoformat()
    review = {
        "outcome": payload.outcome,
        "rationale": payload.rationale.strip(),
        "reviewer": payload.reviewer.strip(),
        "reviewed_at": reviewed_at,
        "type": "human analyst assessment",
    }
    report["analyst_review"] = review
    try:
        append_audit_event(
            report,
            event_type="analyst_review_recorded",
            timestamp=reviewed_at,
            message="A human analyst recorded an interpretation.",
            details={
                "outcome": review["outcome"],
                "reviewer": review["reviewer"],
                "rationale": review["rationale"],
                "reviewed_at": review["reviewed_at"],
            },
        )
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    for case in getattr(request.app.state, "cases", []):
        if case.get("case_id") == report_id:
            case["status"] = "reviewed"
            case["reviewed_at"] = reviewed_at
            save_case(case)
            break
    save_report(report)
    return review
