"""Pairwise image measurements and qualified model-result comparison."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import uuid

import numpy as np
from PIL import Image, ImageChops, ImageOps, ImageStat


def _image_measurements(path_a: str | Path, path_b: str | Path) -> dict[str, Any]:
    with Image.open(path_a) as source_a:
        image_a = ImageOps.exif_transpose(source_a).convert("RGB")
        image_a.load()
    with Image.open(path_b) as source_b:
        image_b = ImageOps.exif_transpose(source_b).convert("RGB")
        image_b.load()

    if image_a.size != image_b.size:
        return {
            "status": "not_comparable",
            "reason": "Image dimensions differ; no resizing or registration was applied.",
            "image_a_dimensions": list(image_a.size),
            "image_b_dimensions": list(image_b.size),
        }

    original_size = image_a.size
    resized_for_bounds = max(original_size) > 512
    if resized_for_bounds:
        image_a.thumbnail((512, 512), Image.Resampling.LANCZOS)
        image_b.thumbnail((512, 512), Image.Resampling.LANCZOS)
    difference = ImageChops.difference(image_a, image_b)
    changed_pixels = int(np.any(np.asarray(difference) != 0, axis=2).sum())
    total_pixels = image_a.width * image_a.height
    return {
        "status": "measured",
        "comparison_dimensions": [image_a.width, image_a.height],
        "resized_for_bounds": resized_for_bounds,
        "mean_absolute_difference_8bit": round(sum(ImageStat.Stat(difference).mean) / 3, 4),
        "changed_pixel_ratio": round(changed_pixels / total_pixels, 6),
        "exact_pixel_match_at_comparison_size": changed_pixels == 0,
        "interpretation": (
            "Direct coordinate-wise pixel difference only. It does not align images or establish "
            "which image is original or manipulated."
        ),
    }


def _model_observation(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "verdict": report.get("verdict", "Insufficient evidence"),
        "decision_status": report.get("decision_status", "not_configured"),
        "authenticity_probability": report.get("authenticity_probability"),
        "deepfake_probability": report.get("deepfake_probability"),
        "decision_basis": report.get("decision_basis"),
    }


def compare_image_reports(
    path_a: str | Path,
    path_b: str | Path,
    report_a: dict[str, Any],
    report_b: dict[str, Any],
) -> dict[str, Any]:
    """Compare two analyzed still images without treating the pair as ground truth."""
    hash_a = report_a["file_hash"]
    hash_b = report_b["file_hash"]
    model_a = _model_observation(report_a)
    model_b = _model_observation(report_b)
    qualified_classes = {"Likely original", "Likely deepfake"}
    both_classified = (
        model_a["decision_status"] == "classified"
        and model_b["decision_status"] == "classified"
        and model_a["verdict"] in qualified_classes
        and model_b["verdict"] in qualified_classes
    )

    likely_manipulated_image = None
    if hash_a == hash_b:
        status = "identical_bytes"
        finding = "Both uploads have identical SHA-256 hashes and contain identical file bytes."
    elif both_classified and model_a["verdict"] != model_b["verdict"]:
        likely_manipulated_image = "image_a" if model_a["verdict"] == "Likely deepfake" else "image_b"
        status = "model_supported_contrast"
        finding = (
            f"The configured model assessed {likely_manipulated_image.replace('_', ' ')} as likely "
            "deepfake and the other image as likely original. This is a qualified model assessment, "
            "not confirmation or proof."
        )
    elif both_classified:
        status = "model_agreement"
        finding = (
            f"The configured model assessed both images as {model_a['verdict'].lower()}. "
            "The pair does not establish which image, if either, is an original source."
        )
    else:
        status = "inconclusive"
        finding = (
            "A comparison-level authenticity conclusion is withheld because both images did not "
            "receive a calibrated, threshold-supported model class."
        )

    def image_record(report: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
        metadata = report.get("metadata_summary", {})
        face_review = report.get("face_review") or {}
        face_status = face_review.get("status", "not_recorded")
        return {
            "filename": Path(report.get("filename", "image")).name,
            "sha256": report["file_hash"],
            "width": metadata.get("width"),
            "height": metadata.get("height"),
            "model_assessment": model,
            "face_observation": {
                "status": face_status,
                "candidate_count": (
                    face_review.get("detected_face_count")
                    if face_status != "unavailable"
                    else None
                ),
                "candidate_boxes_xyxy": face_review.get("face_candidate_boxes_xyxy", []),
                "interpretation": (
                    "Non-identifying face-candidate observation only; no name, identity, age, "
                    "gender, or other personal attributes are inferred."
                ),
            },
            "exif_summary": {
                "has_exif": metadata.get("has_exif", False),
                "fields": metadata.get("exif_fields", {}),
                "gps_coordinates": metadata.get("gps_coordinates"),
                "parse_status": metadata.get("exif_parse_status", "not_available"),
            },
        }

    return {
        "report_type": "two_image_comparison",
        "comparison_id": f"comparison-{uuid.uuid4().hex[:12]}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "finding": finding,
        "likely_manipulated_image": likely_manipulated_image,
        "images": {
            "image_a": image_record(report_a, model_a),
            "image_b": image_record(report_b, model_b),
        },
        "same_file_bytes": hash_a == hash_b,
        "pixel_difference": _image_measurements(path_a, path_b),
        "limitations": [
            "The model labels are likely assessments, not confirmation. Checkpoint domain shift and dataset limitations apply.",
            "Pixel differences can result from benign edits, resizing, color changes, or re-encoding; they do not prove manipulation.",
            "EXIF can be removed or edited and is not proof of source or authenticity.",
            "Face observations are detections only and do not identify people or establish whether a face was manipulated.",
        ],
    }


__all__ = ["compare_image_reports"]
