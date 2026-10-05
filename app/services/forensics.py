import hashlib
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import cv2
from PIL import Image

from app.config import ALLOWED_EXTENSIONS, XCEPTION_CHECKPOINT
from app.services.xception_detector import get_xception_detector
from forensics.chain_of_custody import seal_audit_log
from forensics.exif_analyzer import analyze_exif
from forensics.noise_analysis import analyze_image_noise
from frequency.frequency_pipeline import analyze_image_frequency
from image.face_detector import inspect_face_candidates
from video.video_pipeline import analyze_video

SYNTHETIC_DEMO_SHA256 = {
    "330be38306699d0396f7f1e1f02af3800738d3d43ef4089496792ce793b97f36",
    "559a7e5b4bd249d99be51a9a8cef8951b593b772f783fe2b2b6df8a82fab349d",
    "a1a7187eb279ecd919ac2091a10d80a137dc964b5d8420a0b365ea7aef9f7307",
    "4d207f0ecb2847bc7a1884a7a0311b8fdd49693b290a492a3d8a1a670c164de6",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def detect_media_type(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    for media_type, extensions in ALLOWED_EXTENSIONS.items():
        if suffix in extensions:
            return media_type
    return "unsupported"


def get_basic_image_metadata(path: str | Path) -> Dict[str, Any]:
    with Image.open(path) as image:
        metadata = {
            "width": image.size[0],
            "height": image.size[1],
            "mode": image.mode,
            "format": image.format or "unknown",
            "pixel_bands": list(image.getbands()),
            "band_count": len(image.getbands()),
            "is_animated": bool(getattr(image, "is_animated", False)),
            "frame_count": int(getattr(image, "n_frames", 1)),
            "has_transparency": "transparency" in image.info or "A" in image.getbands(),
        }
        metadata.update(analyze_exif(image))

        if "dpi" in image.info:
            metadata["dpi"] = [round(float(value), 3) for value in image.info["dpi"]]
        if "icc_profile" in image.info:
            metadata["icc_profile_bytes"] = len(image.info["icc_profile"])

        text_fields = {
            key: value[:500]
            for key, value in image.info.items()
            if key.lower() in {"comment", "description", "software", "creation_time", "author"}
            and isinstance(value, str)
            and value
        }
        if text_fields:
            metadata["embedded_text"] = text_fields
    with Image.open(path) as image:
        image.verify()
    return metadata


def inspect_file_signature(path: str | Path, filename: str) -> Dict[str, Any]:
    with open(path, "rb") as handle:
        header = handle.read(4096)

    extension = Path(filename).suffix.lower()
    signature_label = "Unrecognized file signature"
    if header.startswith(b"\xff\xd8\xff"):
        detected_mime = "image/jpeg"
        signature_label = "JPEG marker"
    elif header.startswith(b"\x89PNG\r\n\x1a\n"):
        detected_mime = "image/png"
        signature_label = "PNG signature"
    elif header.startswith((b"GIF87a", b"GIF89a")):
        detected_mime = "image/gif"
        signature_label = header[:6].decode("ascii")
    elif header.startswith(b"RIFF") and header[8:12] == b"WEBP":
        detected_mime = "image/webp"
        signature_label = "RIFF / WEBP"
    elif header.startswith(b"BM"):
        detected_mime = "image/bmp"
        signature_label = "BMP header"
    elif header.startswith((b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")):
        detected_mime = "image/tiff"
        signature_label = "TIFF byte-order marker"
    elif len(header) >= 8 and header[4:8] == b"ftyp":
        brand = header[8:12]
        if extension in {".m4a", ".aac"} and brand in {b"M4A ", b"M4B ", b"M4P "}:
            detected_mime = "audio/mp4"
        else:
            detected_mime = "video/quicktime" if brand == b"qt  " else "video/mp4"
        signature_label = f"ISO BMFF ftyp ({brand.decode('ascii', errors='replace').strip() or 'unknown brand'})"
    elif header.startswith(b"\x1aE\xdf\xa3"):
        if b"webm" in header.lower():
            detected_mime = "video/webm"
            signature_label = "EBML / WebM DocType"
        elif b"matroska" in header.lower():
            detected_mime = "video/x-matroska"
            signature_label = "EBML / Matroska DocType"
        else:
            detected_mime = "application/ebml"
            signature_label = "EBML header (DocType not found in initial 4 KiB)"
    elif header.startswith(b"RIFF") and header[8:12] == b"AVI ":
        detected_mime = "video/x-msvideo"
        signature_label = "RIFF / AVI"
    elif header.startswith(b"RIFF") and header[8:12] == b"WAVE":
        detected_mime = "audio/wav"
        signature_label = "RIFF / WAVE"
    elif header.startswith(b"fLaC"):
        detected_mime = "audio/flac"
        signature_label = "FLAC marker"
    elif header.startswith(b"ID3") or (
        len(header) > 1
        and header[0] == 0xFF
        and header[1] & 0xE0 == 0xE0
        and header[1] & 0x06 != 0
    ):
        detected_mime = "audio/mpeg"
        signature_label = "ID3 tag" if header.startswith(b"ID3") else "MPEG audio frame sync"
    elif len(header) > 1 and header[0] == 0xFF and header[1] & 0xF6 == 0xF0:
        detected_mime = "audio/aac"
        signature_label = "ADTS frame sync"
    else:
        detected_mime = "application/octet-stream"

    known_extensions = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
        ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp", ".tif": "image/tiff",
        ".tiff": "image/tiff", ".mp4": "video/mp4", ".mov": "video/quicktime",
        ".m4v": "video/mp4", ".mkv": "video/x-matroska",
        ".webm": "video/webm", ".avi": "video/x-msvideo",
        ".wav": "audio/wav", ".flac": "audio/flac", ".mp3": "audio/mpeg",
        ".m4a": "audio/mp4", ".aac": "audio/aac",
    }
    expected_mime = known_extensions.get(extension)
    return {
        "extension": extension or "none",
        "expected_mime_type": expected_mime or "unknown",
        "detected_mime_type": detected_mime,
        "signature": signature_label,
        "signature_recognized": detected_mime != "application/octet-stream",
        "extension_matches_signature": expected_mime == detected_mime if expected_mime else False,
        "header_hex": header[:12].hex(" "),
    }


def get_basic_video_metadata(path: str | Path) -> Dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        return {"status": "unreadable"}
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc_value = int(capture.get(cv2.CAP_PROP_FOURCC))
    codec_fourcc = "".join(
        chr((fourcc_value >> (8 * index)) & 0xFF)
        for index in range(4)
        if (fourcc_value >> (8 * index)) & 0xFF
    )
    first_frame_decoded, _ = capture.read()
    try:
        decoder_backend = capture.getBackendName()
    except cv2.error:
        decoder_backend = "unknown"
    capture.release()

    fps_value = float(fps) if math.isfinite(float(fps)) else 0.0
    duration_estimate = float(frames / fps_value) if frames > 0 and fps_value > 0 else None
    return {
        "container_reported_frame_count": frames if frames > 0 else None,
        "decoder_reported_fps": round(fps_value, 3) if fps_value > 0 else None,
        "width": width,
        "height": height,
        "duration_estimate_seconds": round(duration_estimate, 3) if duration_estimate is not None else None,
        "duration_basis": "container-reported frame count divided by decoder-reported frame rate; may be approximate",
        "codec_fourcc": codec_fourcc or None,
        "decoder_backend": decoder_backend,
        "first_frame_decoded": first_frame_decoded,
        "status": "ok" if first_frame_decoded else "partial",
    }


def get_basic_audio_metadata(path: str | Path) -> Dict[str, Any]:
    from audio.audio_loader import inspect_audio

    return inspect_audio(path)


def review_metadata(metadata: Dict[str, Any], media_type: str) -> List[Dict[str, str]]:
    findings: List[Dict[str, str]] = []
    if media_type == "image":
        if not metadata.get("has_exif"):
            findings.append({
                "signal": "EXIF metadata absent",
                "interpretation": "Provenance fields are unavailable. Metadata absence alone does not indicate manipulation.",
                "severity": "context",
            })
        software_value = (
            metadata.get("exif_fields", {}).get("software")
            or metadata.get("embedded_text", {}).get("software")
        )
        if metadata.get("is_animated"):
            findings.append({
                "signal": "Animated image frames present",
                "interpretation": f"The image contains {metadata.get('frame_count', 1)} frames. Frame-level authenticity analysis is not configured.",
                "severity": "context",
            })
        if software_value:
            findings.append({
                "signal": "Software tag present",
                "interpretation": f"Embedded software field: {software_value}. A software tag is context, not proof of editing.",
                "severity": "context",
            })
    elif media_type == "video":
        if metadata.get("status") == "unreadable":
            findings.append({
                "signal": "Video decoder could not read media",
                "interpretation": "Container/codec inspection is incomplete; obtain a valid copy before drawing a conclusion.",
                "severity": "review",
            })
        elif metadata.get("status") == "partial":
            findings.append({
                "signal": "Video container opened but first frame could not be decoded",
                "interpretation": "Some stream properties were read, but decoding is incomplete. Verify the codec and obtain a validated copy.",
                "severity": "review",
            })
        else:
            findings.append({
                "signal": "Video stream properties read",
                "interpretation": (
                    f"{metadata.get('width', 'unknown')}×{metadata.get('height', 'unknown')} pixels; "
                    f"decoder-reported {metadata.get('decoder_reported_fps', 'unknown')} FPS; "
                    f"container-reported {metadata.get('container_reported_frame_count', 'unknown')} frames."
                ),
                "severity": "context",
            })
    elif media_type == "audio":
        findings.append({
            "signal": (
                "Audio stream properties read"
                if metadata.get("metadata_status") == "available"
                else "Audio stream properties unavailable"
            ),
            "interpretation": (
                f"{metadata.get('container', 'Unknown container')}; "
                f"{metadata.get('sample_rate_hz', 'unknown')} Hz; "
                f"{metadata.get('channels', 'unknown')} channels; "
                f"{metadata.get('duration_seconds', 'unknown')} s where decodable. "
                "These are descriptive file properties, not voice-authenticity findings."
                if metadata.get("metadata_status") == "available"
                else metadata.get(
                    "metadata_limitation",
                    "Audio stream properties could not be decoded.",
                )
            ),
            "severity": (
                "context" if metadata.get("metadata_status") == "available" else "not_assessed"
            ),
        })

    if not findings:
        findings.append({
            "signal": "Basic properties extracted",
            "interpretation": "No provenance conclusion can be made from these properties alone.",
            "severity": "context",
        })
    return findings


def analyze_media(file_path: str | Path, filename: str, case_id: str = "case-unknown") -> Dict[str, Any]:
    media_type = detect_media_type(filename)
    if media_type == "unsupported":
        raise ValueError(f"Unsupported file type for {filename}")

    file_hash = sha256_file(file_path)
    file_size = Path(file_path).stat().st_size
    base_time = datetime.now(timezone.utc).isoformat()
    file_signature = inspect_file_signature(file_path, filename)

    metadata: Dict[str, Any]
    if media_type == "image":
        metadata = get_basic_image_metadata(file_path)
    elif media_type == "video":
        metadata = get_basic_video_metadata(file_path)
    else:
        metadata = get_basic_audio_metadata(file_path)

    metadata_findings = review_metadata(metadata, media_type)
    synthetic_demo = media_type == "image" and file_hash.casefold() in SYNTHETIC_DEMO_SHA256
    face_review = None
    frequency_analysis = None
    noise_analysis = None
    video_analysis = None
    audio_analysis = None
    video_detector = None
    if media_type == "image" and not metadata.get("is_animated", False):
        face_review = inspect_face_candidates(file_path)
        frequency_analysis = analyze_image_frequency(file_path)
        noise_analysis = analyze_image_noise(file_path)
    elif media_type == "video":
        video_detector = get_xception_detector(XCEPTION_CHECKPOINT)
        video_analysis = analyze_video(file_path, metadata, video_detector)
    elif media_type == "audio":
        audio_analysis = metadata

    decision = "Insufficient evidence"
    decision_basis = "No trained image, video, audio, frequency, or multimodal classifier is configured."
    model_status = "not_configured"
    is_animated_image = media_type == "image" and metadata.get("is_animated", False)
    xception_result = None
    if media_type == "image" and not is_animated_image and not synthetic_demo:
        detector = get_xception_detector(XCEPTION_CHECKPOINT)
        if detector is not None:
            xception_result = detector.predict(file_path)
            decision = xception_result["verdict"]
            model_status = xception_result["status"]
    if is_animated_image:
        model_status = "not_applicable_animated_image"
        decision_basis = (
            "The image contains multiple frames. The configured Xception detector supports static "
            "images only, and no frame-level or temporal classifier is configured."
        )
    if synthetic_demo:
        model_status = "demo_sample_not_classified"
        decision_basis = (
            "This bundled synthetic illustration is for UI demonstration only. It is not genuine "
            "media or ground truth, so model inference is intentionally skipped."
        )
    video_frame_analysis = (
        video_analysis.get("model_frame_analysis")
        if video_analysis
        else None
    )
    has_video_frame_estimates = bool(
        video_frame_analysis and video_frame_analysis.get("analyzed_frame_count", 0)
    )
    if has_video_frame_estimates:
        model_status = "frame_estimates_only"
        decision_basis = (
            "The configured image Xception model ran on sampled video frames. "
            "Its image scores are not validated for video-level decisions, so the overall "
            "original/deepfake verdict is withheld."
        )

    metadata_assessment_interpretation = (
        "Image dimensions, pixel mode and available embedded metadata were read; these properties do not prove authenticity."
        if media_type == "image"
        else "Basic properties come from the available video decoder; the reported frame count and duration are approximate."
        if media_type == "video"
        else (
            "Audio container properties and decoder-derived stream information were reviewed; "
            "these properties do not establish voice authenticity."
            if metadata.get("metadata_status") == "available"
            else "Audio container parsing or stream decoding was incomplete."
        )
    )
    evidence_assessment = [
        {
            "name": "Spatial / face analysis",
            "status": (
                "not_applicable" if media_type == "audio"
                else "measured_descriptive"
                if face_review and face_review["status"] in {"face_detected", "no_face_detected"}
                else "measured_descriptive"
                if video_analysis and any(
                    frame["face_review"]["status"] in {"face_detected", "no_face_detected"}
                    for frame in video_analysis["frames"]
                )
                else "not_configured"
            ),
            "interpretation": (
                face_review["interpretation"] if face_review
                else video_analysis["interpretation"]
                if video_analysis
                else "No face detector or trained spatial classifier is configured."
            ),
        },
        {
            "name": "Temporal analysis",
            "status": (
                "measured_descriptive" if video_analysis and video_analysis["frames"]
                else "not_configured" if media_type == "video" or is_animated_image
                else "not_applicable"
            ),
            "interpretation": (
                video_analysis["interpretation"]
                if video_analysis
                else "No validated temporal manipulation detector is configured for this video or animated image."
                if media_type == "video" or is_animated_image
                else "Temporal video/frame analysis does not apply to this still image or audio-only file."
            ),
        },
        {
            "name": "Frequency / compression analysis",
            "status": (
                "measured_descriptive"
                if frequency_analysis or (audio_analysis and audio_analysis.get("measurements"))
                else "not_configured"
            ),
            "interpretation": (
                frequency_analysis["interpretation"]
                if frequency_analysis
                else audio_analysis["interpretation"]
                if audio_analysis and audio_analysis.get("measurements")
                else "No frequency-domain measurements are available for this media type."
            ),
        },
        {
            "name": "Audio / synchronization analysis",
            "status": (
                "measured_descriptive"
                if audio_analysis and audio_analysis.get("signal_status") == "measured_descriptive"
                else "not_applicable" if media_type == "image"
                else "not_configured"
            ),
            "interpretation": (
                audio_analysis["interpretation"]
                if audio_analysis
                else "Audio authenticity and synchronization analysis are not applicable to an image."
                if media_type == "image"
                else "No synthetic-voice or audio-video synchronization model is configured."
            ),
        },
        {
            "name": "Metadata and container review",
            "status": "reviewed",
            "interpretation": metadata_assessment_interpretation,
        },
    ]

    if is_animated_image:
        evidence_assessment[0] = {
            "name": "Spatial / face analysis",
            "status": "not_applicable",
            "interpretation": "The static-image Xception model does not process animated image frames.",
        }

    summary = (
        "The file was fingerprinted and its image properties were read, but this prototype cannot determine whether it is original or a deepfake."
        if media_type == "image"
        else "The file was fingerprinted and basic video stream properties were read, but this prototype cannot determine whether it is original or manipulated."
        if media_type == "video" and metadata.get("status") == "ok"
        else "The file was fingerprinted, but video decoding was incomplete; this prototype cannot determine whether it is original or manipulated."
        if media_type == "video"
        else (
            "The file was fingerprinted and its audio container and decoded signal properties were reviewed. "
            "No synthetic-voice or authenticity classification is configured."
            if audio_analysis and audio_analysis.get("signal_status") == "measured_descriptive"
            else "The file was fingerprinted; audio stream decoding was incomplete and no voice-authenticity classification is configured."
        )
    )
    explanation = [
        (
            "Animated image frames are not classified; no frame-level or temporal detector is configured."
            if is_animated_image
            else "Original-versus-deepfake classification is unavailable because no trained detector is configured."
        ),
        "A definitive authenticity or manipulation probability was not generated.",
        "The SHA-256 value identifies these exact file bytes; without a trusted reference hash it does not prove the source or originality.",
        "Missing or editable metadata is a provenance limitation, not proof of manipulation.",
        "Obtain source material and chain-of-custody records, then review with a validated detector and qualified analyst.",
    ]
    explanation.extend(item["interpretation"] for item in metadata_findings)
    if is_animated_image:
        summary = (
            f"The file contains {metadata.get('frame_count', 1)} animated frames. Metadata was reviewed, "
            "but frame-level authenticity analysis is not configured."
        )
    if synthetic_demo:
        summary = (
            "Synthetic demonstration illustration only; this is not genuine media or a ground-truth "
            "original/deepfake example. Classifier inference was intentionally skipped."
        )
        explanation = [
            "This bundled illustration is a UI demonstration, not a real deepfake example or a ground-truth sample.",
            "Classifier inference was intentionally skipped so a synthetic illustration is not presented as a valid real/deepfake model test.",
            "The file was still fingerprinted and its available image properties were reviewed.",
            "The SHA-256 value identifies these exact bytes; without a trusted reference hash it does not prove source or originality.",
        ]
        explanation.extend(item["interpretation"] for item in metadata_findings)

    if xception_result is not None:
        deepfake_probability = xception_result["deepfake_probability"]
        authenticity_probability = xception_result["authenticity_probability"]
        is_uncalibrated = xception_result["status"] == "uncalibrated_prediction"
        if xception_result["status"] == "conformal_abstention":
            decision_basis = (
                "The locally calibrated Xception model abstained because its class-conditional "
                "conformal prediction set did not uniquely support the threshold-selected class. "
                "No original/deepfake class was assigned."
            )
        elif is_uncalibrated:
            decision_basis = (
                "The pinned public Xception checkpoint returned raw, uncalibrated softmax scores. "
                "The displayed class is its higher-scoring class, not a validated forensic conclusion. "
                "It was trained on FFHQ real faces and StyleGAN-generated fake faces; applicability to "
                "other sources and manipulation methods is unknown."
            )
        elif xception_result["model"]["thresholds"]["real_max_probability"] is None:
            decision_basis = (
                "A trained Xception model produced a temperature-scaled probability, but calibration "
                "did not support a non-overlapping real/suspicious/fake operating region; no class was assigned."
            )
        else:
            decision_basis = (
                "A trained Xception model and held-out temperature calibration were used. The result "
                "is model decision support, not proof of authenticity; inspect calibration, thresholds, "
                "face-crop details and domain-shift limitations."
            )
        face_review = xception_result["face_review"]
        face_description = (
            f"{face_review['detected_face_count']} Haar-cascade face candidate(s); "
            f"{face_review['selected_crop'].replace('_', ' ')}; "
            f"alignment {face_review['alignment'].replace('_', ' ')}"
        )
        evidence_assessment[0] = {
            "name": "Spatial / face analysis",
            "status": "analyzed",
            "interpretation": (
                (
                    f"Image-level Xception deepfake softmax score: {deepfake_probability:.4f}; "
                    f"authenticity softmax score: {authenticity_probability:.4f}. "
                    if is_uncalibrated
                    else f"Image-level Xception deepfake probability: {deepfake_probability:.4f}; "
                    f"authenticity probability: {authenticity_probability:.4f}. "
                )
                + (
                    "These are uncalibrated softmax scores from a checkpoint trained on FFHQ and "
                    "StyleGAN faces; applicability to other domains is unknown. "
                    if is_uncalibrated
                    else f"Validation AUC: {xception_result['model']['validation_auc']:.4f}. "
                )
                +
                f"{face_description}. "
                + (
                    f"Conformal prediction set at alpha={xception_result['conformal_prediction']['alpha']:.2f}: "
                    f"{', '.join(xception_result['conformal_prediction']['prediction_set']) or 'empty'}; "
                    "coverage assumes exchangeability with calibration data. "
                    if xception_result.get("conformal_prediction")
                    else ""
                )
                + (
                    "The face-candidate observation is separate from this model's full-image input."
                    if is_uncalibrated
                    else "AUC describes ranking on the recorded validation split and is not the probability "
                    "that this individual result is correct."
                )
            ),
        }
        probability_summary = (
            f"The uncalibrated Xception softmax scores favored “{decision.lower()}”."
            if is_uncalibrated
            else
            "Xception abstained because the conformal prediction set did not uniquely support its threshold-selected class."
            if xception_result["status"] == "conformal_abstention"
            else
            "A calibrated Xception probability was produced, but no class was assigned because "
            "the calibration set did not support a safe threshold gap."
            if model_status == "calibrated_probability_only"
            else f"Xception estimated a deepfake probability of {deepfake_probability:.1%}; "
            f"the calibrated operating region returned “{decision.lower()}”."
        )
        summary = (
            f"{probability_summary} The estimate applies to this static image and this checkpoint's "
            + (
                "training distribution. It is not proof, and performance may change for other "
                if is_uncalibrated
                else "training and calibration distributions. It is not proof, and performance may change for other "
            )
            +
            "sources, demographics, compression levels or manipulation methods."
        )
        explanation = [
            "A trained Xception image classifier was run on this static image; video, audio, frequency and temporal models were not run.",
            (
                f"Uncalibrated deepfake softmax score: {deepfake_probability:.6f}; "
                f"authenticity softmax score: {authenticity_probability:.6f}."
                if is_uncalibrated
                else f"Calibrated deepfake probability: {deepfake_probability:.6f}; authenticity probability: {authenticity_probability:.6f}."
            ),
            *(
                [
                    (
                        f"Class-conditional conformal prediction set at alpha="
                        f"{xception_result['conformal_prediction']['alpha']:.2f}: "
                        f"{', '.join(xception_result['conformal_prediction']['prediction_set']) or 'empty'}. "
                        "Its coverage interpretation assumes exchangeability with the calibration data "
                        "and is not guaranteed under domain shift."
                    )
                ]
                if xception_result.get("conformal_prediction")
                else []
            ),
            (
                "The model-card validation accuracy is an unverified publisher claim, not a result "
                "reproduced by this project."
                if is_uncalibrated
                else f"Model validation AUC: {xception_result['model']['validation_auc']:.6f}; this is validation-set performance, not per-file confidence."
            ),
            f"Face review: {face_description}.",
            *xception_result["limitations"],
            "The SHA-256 value identifies these exact file bytes; without a trusted reference hash it does not prove source or originality.",
            "Missing or editable metadata is a provenance limitation, not proof of manipulation.",
            "Have a qualified analyst compare the evidence with trusted source material and chain-of-custody records.",
        ]
        explanation.extend(item["interpretation"] for item in metadata_findings)

    explainability = (xception_result or {}).get("explainability") or {}
    model_pipeline = [
        {
            "number": 1,
            "name": "File validation and decode",
            "status": "completed" if file_signature["signature_recognized"] else "review_required",
            "detail": (
                f"{file_signature['detected_mime_type']} signature "
                f"{'matches' if file_signature['extension_matches_signature'] else 'does not match'} "
                "the supplied file extension; media decoder inspection completed."
            ),
        },
        {
            "number": 2,
            "name": "Evidence fingerprint",
            "status": "completed",
            "detail": f"SHA-256 calculated for {file_size:,} bytes.",
        },
        {
            "number": 3,
            "name": "Metadata and container review",
            "status": "completed" if media_type != "audio" or metadata.get("metadata_status") == "available" else "partial",
            "detail": (
                "Available decoded file properties were recorded; metadata is contextual and is not an authenticity classifier."
            ),
        },
        {
            "number": 4,
            "name": "Descriptive signal measurements",
            "status": "measured_descriptive"
            if frequency_analysis or noise_analysis or video_analysis or (audio_analysis and audio_analysis.get("measurements"))
            else "not_applicable",
            "detail": (
                "FFT/residual, sampled-frame, or waveform/FFT observations are listed in their modality panels. "
                "These measurements are not trained manipulation scores."
                if frequency_analysis or noise_analysis or video_analysis or (audio_analysis and audio_analysis.get("measurements"))
                else "No descriptive signal measurement applies to this media or could be produced."
            ),
        },
        {
            "number": 5,
            "name": "Face-candidate detection",
            "status": (
                "measured_descriptive"
                if face_review and face_review.get("status") in {"face_detected", "no_face_detected"}
                else "unavailable"
                if face_review
                else "not_applicable"
                if media_type == "audio"
                else "not_run"
            ),
            "detail": (
                face_review.get("interpretation", "Face-candidate review was attempted.")
                if face_review
                else "Per-frame candidate observations are available in video review."
                if video_analysis and video_analysis.get("frames")
                else "Face detection is not applicable to audio."
                if media_type == "audio"
                else "No face-candidate detector ran for this media."
            ),
        },
        {
            "number": 6,
            "name": "Xception checkpoint",
            "status": (
                "loaded"
                if xception_result
                else "loaded_for_frame_analysis"
                if media_type == "video" and video_detector is not None
                else "skipped_demo"
                if synthetic_demo
                else "not_applicable"
                if media_type != "image" or is_animated_image
                else "not_configured"
            ),
            "detail": (
                f"Loaded {xception_result['model']['architecture']}; "
                f"{xception_result['model']['calibration']}."
                if xception_result
                else (
                    "Inference was deliberately skipped for this bundled synthetic demonstration sample. "
                    f"Checkpoint file {'exists' if XCEPTION_CHECKPOINT.is_file() else 'is not installed'}."
                )
                if synthetic_demo
                else "No compatible trained checkpoint is installed at weights/xception_deepfake.pth."
                if media_type == "image" and not is_animated_image
                else (
                    f"Loaded {video_frame_analysis['architecture']} for sampled-frame estimates only."
                    if video_frame_analysis and video_frame_analysis.get("architecture")
                    else "The configured image checkpoint is unavailable; sampled frames remain descriptive only."
                )
                if media_type == "video"
                else "The configured Xception checkpoint supports static images only."
            ),
        },
        {
            "number": 7,
            "name": "Model-specific preprocessing",
            "status": "completed" if xception_result or has_video_frame_estimates else "not_run",
            "detail": (
                xception_result["model"].get(
                    "input_preprocessing",
                    "Largest detected face crop (or full-image fallback) resized to 299 × 299 and ImageNet-normalized.",
                )
                if xception_result
                else (
                    "Each decoded, downscaled keyframe was passed to the image checkpoint; "
                    "face-candidate review is separate from model preprocessing."
                )
                if has_video_frame_estimates
                else "Not run: Xception inference did not execute."
            ),
        },
        {
            "number": 8,
            "name": "Xception inference",
            "status": "completed" if xception_result else "sampled_frames_only" if has_video_frame_estimates else "not_run",
            "detail": (
                (
                    f"Produced raw, uncalibrated deepfake softmax score "
                    f"{xception_result['deepfake_probability']:.6f}."
                    if xception_result["status"] == "uncalibrated_prediction"
                    else f"Produced a calibrated deepfake probability of {xception_result['deepfake_probability']:.6f}."
                )
                if xception_result
                else (
                    f"Produced image-model estimates for {video_frame_analysis['analyzed_frame_count']} "
                    "sampled frame(s); no video-level probability or verdict was generated."
                )
                if has_video_frame_estimates
                else "No model logits or class probabilities were produced."
            ),
        },
        {
            "number": 9,
            "name": "Calibration and operating thresholds",
            "status": (
                "conformal_abstention"
                if xception_result and xception_result["status"] == "conformal_abstention"
                else "applied"
                if xception_result and xception_result["status"] == "classified"
                else "not_validated_for_video"
                if has_video_frame_estimates
                else "not_calibrated"
                if xception_result and xception_result["status"] == "uncalibrated_prediction"
                else "probability_only"
                if xception_result
                else "not_run"
            ),
            "detail": (
                "Held-out temperature scaling and the conformal set supported the threshold-selected class."
                if xception_result and xception_result["status"] == "classified"
                and xception_result.get("conformal_prediction")
                else "Held-out temperature scaling and checkpoint-recorded real/suspicious/fake thresholds were applied."
                if xception_result and xception_result["status"] == "classified"
                else "The conformal set did not uniquely support the threshold-selected class; the decision was withheld."
                if xception_result and xception_result["status"] == "conformal_abstention"
                else (
                    "The model's calibration applies to still images, not the sampled-frame/video decision path; "
                    "video-level calibration is unavailable."
                )
                if video_frame_analysis
                else "No calibration or operating thresholds are available; the displayed softmax scores are uncalibrated."
                if xception_result and xception_result["status"] == "uncalibrated_prediction"
                else "Probability is calibrated, but this checkpoint has no non-overlapping class thresholds."
                if xception_result
                else "No calibrated model output exists to threshold."
            ),
        },
        {
            "number": 10,
            "name": "Original / deepfake decision",
            "status": (
                "uncalibrated_prediction"
                if xception_result and xception_result["status"] == "uncalibrated_prediction"
                else "classified"
                if xception_result and xception_result["status"] == "classified"
                else "withheld"
            ),
            "detail": (
                (
                    f"The higher raw softmax class was {decision}; the prediction is uncalibrated."
                    if xception_result["status"] == "uncalibrated_prediction"
                    else f"Checkpoint thresholds produced: {decision}."
                )
                if xception_result and xception_result["status"] == "classified"
                else "The conformal prediction set did not uniquely support a class; no original/deepfake label was assigned."
                if xception_result and xception_result["status"] == "conformal_abstention"
                else f"The higher raw softmax class was {decision}; the prediction is uncalibrated."
                if xception_result
                else "No video-level class was assigned; per-frame image-model estimates are not combined into a clip verdict."
                if has_video_frame_estimates
                else "No real/deepfake class was assigned because a trained, calibrated classifier did not produce a usable decision."
            ),
        },
        {
            "number": 11,
            "name": "Grad-CAM explainability",
            "status": (
                explainability.get("status", "not_generated")
                if xception_result
                else "not_configured"
            ),
            "detail": (
                explainability.get("interpretation")
                or explainability.get("reason")
                or "The Xception model did not return an explanation artifact."
                if xception_result
                else "A compatible Xception image model did not run, so no heatmap was generated."
            ),
        },
        {
            "number": 12,
            "name": "Sampled-frame image estimates",
            "status": (
                "frame_estimates_available"
                if video_frame_analysis and video_frame_analysis.get("analyzed_frame_count")
                else "not_configured"
                if media_type == "video"
                else "not_applicable"
            ),
            "detail": (
                video_frame_analysis["interpretation"]
                if video_frame_analysis
                else "Sampled keyframes are descriptive previews; no compatible image checkpoint is configured."
            )
            if media_type == "video"
            else "This file is not a video.",
        },
        {
            "number": 13,
            "name": "Synthetic-voice classifier",
            "status": "not_configured" if media_type == "audio" else "not_applicable",
            "detail": "Waveform and FFT summaries are descriptive; no synthetic-voice classifier is configured."
            if media_type == "audio"
            else "This file has no audio-only classifier path.",
        },
        {
            "number": 14,
            "name": "Multimodal fusion",
            "status": "not_configured",
            "detail": "No trained and validated cross-modal fusion model is installed.",
        },
    ]

    report = {
        "case_id": case_id,
        "media_type": media_type,
        "filename": filename,
        "file_hash": file_hash,
        "file_size": file_size,
        "created_at": base_time,
        "verdict": decision,
        "decision_status": model_status,
        "decision_basis": decision_basis,
        "evidence_context": {
            "is_synthetic_demo": synthetic_demo,
            "notice": (
                "Synthetic illustration for interface demonstration only; not genuine media or ground truth."
                if synthetic_demo
                else None
            ),
        },
        "authenticity_probability": xception_result["authenticity_probability"] if xception_result else None,
        "deepfake_probability": xception_result["deepfake_probability"] if xception_result else None,
        "probability": xception_result["deepfake_probability"] if xception_result else None,
        "confidence": None,
        "forensic_score": None,
        "evidence_reliability": (
            "Uncalibrated softmax estimate from an FFHQ/StyleGAN-face checkpoint; no external domain validation."
            if xception_result and xception_result["status"] == "uncalibrated_prediction"
            else "Probability calibrated on a held-out split; external domain validity is not established."
            if xception_result
            else "Sampled still-image scores only; no validated video-level estimate."
            if video_frame_analysis and video_frame_analysis.get("analyzed_frame_count")
            else "Not rated — synthetic demonstration sample was not classified."
            if synthetic_demo
            else "Not rated — animated frames were not classified."
            if is_animated_image
            else "Not rated — model unavailable"
        ),
        "summary": summary,
        "manipulation_type": (
            "Image classification only; manipulation subtype and location are not determined."
            if xception_result
            else "Not assessed"
        ),
        "model_results": (
            {"xception": xception_result}
            if xception_result
            else (
                {"video_xception_frames": video_frame_analysis}
                if video_frame_analysis
                else {}
            )
        ),
        "face_review": face_review,
        "frequency_analysis": frequency_analysis,
        "noise_analysis": noise_analysis,
        "video_analysis": video_analysis,
        "audio_analysis": audio_analysis,
        "file_integrity": {
            **file_signature,
            "sha256": file_hash,
            "size_bytes": file_size,
            "status": "signature_and_extension_match" if file_signature["signature_recognized"] and file_signature["extension_matches_signature"] else "review_required",
            "content_validation": (
                "decoded"
                if media_type == "image"
                else "first_frame_decoded" if media_type == "video" and metadata.get("first_frame_decoded")
                else "partial_decode" if media_type == "video"
                else "not_performed"
            ),
            "meaning": "The signature check compares file-header bytes with the extension's expected type. It does not prove provenance or authenticity; SHA-256 identifies these exact bytes for later comparison.",
        },
        "metadata_summary": metadata,
        "metadata_findings": metadata_findings,
        "evidence_assessment": evidence_assessment,
        "pipeline_stages": [
            {
                "number": 1,
                "name": "Evidence integrity",
                "status": "completed",
                "summary": "SHA-256 fingerprint generated and binary file signature checked.",
            },
            {
                "number": 2,
                "name": "Metadata review",
                "status": (
                    "completed"
                    if media_type == "image"
                    or (media_type == "video" and metadata.get("status") == "ok")
                    or (media_type == "audio" and metadata.get("metadata_status") == "available")
                    else "partial"
                ),
                "summary": (
                    "Image dimensions, pixel mode, frame properties, and available embedded tags were read; absent tags are not proof of manipulation."
                    if media_type == "image"
                    else "Video properties come from the available decoder; reported frame counts and duration estimates may be approximate."
                    if media_type == "video"
                    else (
                        "Available audio container properties and decoder-derived stream information were reviewed; "
                        "metadata alone does not establish voice authenticity."
                        if metadata.get("metadata_status") == "available"
                        else "Audio container parsing or stream decoding was incomplete."
                    )
                ),
            },
            {
                "number": 3,
                "name": "Signal assessment",
                "status": (
                    "conformal_abstention"
                    if xception_result and xception_result["status"] == "conformal_abstention"
                    else "completed"
                    if xception_result and model_status == "classified"
                    else "uncalibrated_model_estimate"
                    if xception_result and model_status == "uncalibrated_prediction"
                    else "calibrated_probability_only"
                    if xception_result
                    else "demo_sample_not_classified"
                    if synthetic_demo
                    else "not_applicable_animated_image"
                    if is_animated_image
                    else "model_not_configured"
                ),
                "summary": (
                    (
                        f"Static-image Xception produced uncalibrated deepfake and authenticity "
                        f"softmax scores of {xception_result['deepfake_probability']:.6f} and "
                        f"{xception_result['authenticity_probability']:.6f}; no video, audio, temporal, "
                        "frequency-classification, localization or fusion model was run."
                        if model_status == "uncalibrated_prediction"
                        else f"Static-image Xception produced a calibrated deepfake probability of "
                        f"{xception_result['deepfake_probability']:.6f} and authenticity probability of "
                        f"{xception_result['authenticity_probability']:.6f}; no video, audio, temporal, "
                        "frequency-classification, localization or fusion model was run."
                    )
                    if xception_result
                    else "Synthetic UI illustration detected; classifier inference was intentionally skipped."
                    if synthetic_demo
                    else "Animated image frames were not passed to the static-image detector."
                    if is_animated_image
                    else (
                        "No trained detector is installed. Descriptive file/signal observations may be present, "
                        "but no deepfake-classifier probabilities or fusion scores were fabricated."
                    )
                    if not is_animated_image
                    else "Animated image frames were not passed to the static-image detector."
                ),
            },
            {
                "number": 4,
                "name": "Analyst interpretation",
                "status": "awaiting_review",
                "summary": "An analyst must record an independent interpretation with rationale; automated conclusion remains inconclusive.",
            },
        ],
        "model_pipeline": model_pipeline,
        "suspicious_regions": [],
        "suspicious_frames": [],
        "evidence_scores": {},
        "explanation": explanation,
        "audit_log": [
            {
                "event_type": "upload_received",
                "timestamp": base_time,
                "message": "Original upload received; SHA-256 fingerprint calculated.",
            },
            {
                "event_type": "metadata_reviewed",
                "timestamp": base_time,
                "message": "File extension, binary signature, and basic metadata reviewed.",
            },
            *(
                [
                    {
                        "event_type": "descriptive_measurements_completed",
                        "timestamp": base_time,
                        "message": (
                            "Descriptive image measurements completed: "
                            + (
                                f"Haar face candidates={face_review['detected_face_count']}; "
                                if face_review["status"] in {"face_detected", "no_face_detected"}
                                else f"Haar detector status={face_review['status']}; "
                            )
                            + "FFT spectrum and high-pass "
                            "residual recorded. These are not authenticity scores."
                        ),
                    }
                ]
                if face_review and frequency_analysis and noise_analysis
                else []
            ),
            {
                "event_type": "classifier_assessment",
                "timestamp": base_time,
                "message": (
                    "Static-image Xception inference and held-out calibration were applied."
                    if xception_result
                    else "Synthetic UI demonstration image recognized; classifier inference intentionally skipped."
                    if synthetic_demo
                    else "Animated image frame classification is not configured; authenticity verdict withheld."
                    if is_animated_image
                    else "No trained classifier configured; authenticity verdict withheld."
                ),
            },
        ],
    }
    if video_analysis:
        report["explanation"].append(video_analysis["interpretation"])
    if audio_analysis:
        report["explanation"].append(audio_analysis["interpretation"])
    if video_analysis and video_analysis.get("frames"):
        report["audit_log"].append({
            "event_type": "video_frames_reviewed",
            "timestamp": base_time,
            "message": (
                f"Decoded {video_analysis['sampled_frame_count']} bounded, evenly spaced preview keyframes; "
                "no temporal manipulation classifier was run."
            ),
        })
    if audio_analysis and audio_analysis.get("measurements"):
        report["audit_log"].append({
            "event_type": "audio_signal_reviewed",
            "timestamp": base_time,
            "message": (
                f"Recorded descriptive waveform and {audio_analysis['measurements']['spectrogram_summary']['window_count']} "
                "coarse FFT window(s); no synthetic-voice classifier was run."
            ),
        })
    seal_audit_log(report)
    return report
