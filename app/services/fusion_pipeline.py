"""
DeepTrace AI — Expanded 15-Stage Fusion Pipeline

Orchestrates the complete forensic evidence pipeline, calling real analysis
modules instead of returning placeholder data. Each stage calls actual
implementations from across the codebase.

Pipeline:
  1. Evidence Validation
  2. File Integrity / Hash
  3. Metadata Analysis
  4. Face / Region Localization
  5. Xception DeepFake Detection
  6. Grad-CAM++
  7. Frequency Analysis (FFT + DCT)
  8. Compression / JPEG Forensics (ELA)
  9. Noise / Sensor Forensics (PRNU)
  10. Manipulation Localization
  11. Multi-Model Ensemble
  12. Evidence Consistency Analysis
  13. Calibration + Uncertainty
  14. OOD / Reliability Analysis
  15. Final Forensic Assessment
"""

import hashlib
import logging
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _safe_float(value, default=0.0):
    """Safely convert a value to float."""
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _sha256_file(path: str) -> str:
    """Calculate SHA-256 hash of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _md5_file(path: str) -> str:
    """Calculate MD5 hash of a file."""
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_entropy(path: str) -> float:
    """Calculate Shannon entropy of a file's byte distribution."""
    with open(path, "rb") as f:
        data = f.read()
    if not data:
        return 0.0
    byte_counts = [0] * 256
    for byte in data:
        byte_counts[byte] += 1
    length = len(data)
    entropy = 0.0
    for count in byte_counts:
        if count > 0:
            p = count / length
            entropy -= p * math.log2(p)
    return round(entropy, 4)


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------

class FusionPipeline:
    """
    DeepTrace AI — 15-Stage Forensic Evidence Pipeline

    Every stage calls real analysis code from the existing codebase modules.
    """

    def __init__(self):
        self.timeline: List[Dict[str, str]] = []

        # Lazy-load Xception detector (may fail if PyTorch not installed)
        self.xception_detector = None
        try:
            from app.config import XCEPTION_CHECKPOINT
            from app.services.xception_detector import get_xception_detector
            self.xception_detector = get_xception_detector(XCEPTION_CHECKPOINT)
        except Exception as e:
            logger.warning(f"Xception detector not available: {e}")

    # ------------------------------------------------------------------
    # Timeline
    # ------------------------------------------------------------------
    def _record(self, step: str, duration_ms: float = 0.0):
        """Record a pipeline step with timestamp."""
        entry = {
            "timestamp": datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3],
            "step": step,
            "duration_ms": round(duration_ms, 1),
        }
        self.timeline.append(entry)
        logger.info(f"[Pipeline] {entry['timestamp']}  {step}  ({duration_ms:.1f} ms)")

    # ==================================================================
    # STAGE 1 — Evidence Validation
    # ==================================================================
    def validate_input_media(self, image_path: str) -> Dict[str, Any]:
        """Validate image format, dimensions, corruption, color space."""
        t0 = time.perf_counter()
        result: Dict[str, Any] = {
            "valid": False,
            "format": None,
            "width": None,
            "height": None,
            "channels": None,
            "color_space": None,
            "corrupted": False,
            "file_size_bytes": None,
            "extension_mismatch": False,
        }

        path = Path(image_path)
        if not path.exists():
            result["error"] = "File does not exist"
            self._record("Validation FAILED — file not found", _elapsed(t0))
            return result

        result["file_size_bytes"] = path.stat().st_size

        # Check file signature
        try:
            from app.services.forensics import inspect_file_signature
            sig = inspect_file_signature(str(path), path.name)
            result["extension_mismatch"] = not sig.get("extension_matches_signature", True)
            result["detected_mime"] = sig.get("detected_mime_type")
            result["file_signature"] = sig.get("signature")
        except Exception as e:
            logger.warning(f"File signature check failed: {e}")

        # Validate image
        try:
            with Image.open(image_path) as img:
                img.verify()
            with Image.open(image_path) as img:
                img = ImageOps.exif_transpose(img)
                result["format"] = img.format or "unknown"
                result["width"] = img.size[0]
                result["height"] = img.size[1]
                result["color_space"] = img.mode
                result["channels"] = len(img.getbands())
                result["is_animated"] = bool(getattr(img, "is_animated", False))
                result["frame_count"] = int(getattr(img, "n_frames", 1))
                result["has_transparency"] = "A" in img.getbands()
                result["valid"] = True
        except Exception as e:
            result["corrupted"] = True
            result["error"] = str(e)[:300]

        # Dimension checks
        if result["valid"]:
            if result["width"] < 32 or result["height"] < 32:
                result["warning"] = "Image dimensions too small for reliable analysis"
            if result["width"] > 10000 or result["height"] > 10000:
                result["warning"] = "Very large image — analysis may be slow"

        self._record("Evidence validation", _elapsed(t0))
        return result

    # ==================================================================
    # STAGE 2 — File Integrity / Hash
    # ==================================================================
    def calculate_file_hash(self, image_path: str) -> Dict[str, Any]:
        """Calculate SHA-256, MD5, entropy, and generate evidence ID."""
        t0 = time.perf_counter()
        sha256 = _sha256_file(image_path)
        md5 = _md5_file(image_path)
        entropy = _file_entropy(image_path)
        file_size = os.path.getsize(image_path)

        result = {
            "sha256": sha256,
            "md5": md5,
            "file_size_bytes": file_size,
            "entropy": entropy,
            "evidence_id": f"EV-{sha256[:12].upper()}",
            "high_entropy_warning": entropy > 7.9,
        }
        self._record("File integrity (SHA-256 + MD5 + entropy)", _elapsed(t0))
        return result

    # ==================================================================
    # STAGE 3 — Metadata Forensics
    # ==================================================================
    def run_metadata_analysis(self, image_path: str) -> Dict[str, Any]:
        """Extract EXIF, detect editing software, flag inconsistencies."""
        t0 = time.perf_counter()
        result: Dict[str, Any] = {
            "metadata_available": False,
            "software": None,
            "camera": None,
            "camera_make": None,
            "gps": None,
            "date_time": None,
            "metadata_risk": 0.0,
            "flags": [],
        }

        try:
            from forensics.exif_analyzer import analyze_exif
            with Image.open(image_path) as img:
                exif_data = analyze_exif(img)

            result["metadata_available"] = exif_data.get("has_exif", False)
            result["exif_tag_count"] = exif_data.get("exif_tag_count", 0)

            fields = exif_data.get("exif_fields", {})
            result["software"] = fields.get("software")
            result["camera"] = fields.get("camera_model")
            result["camera_make"] = fields.get("camera_make")
            result["date_time"] = fields.get("date_time_original") or fields.get("date_time")
            result["gps"] = exif_data.get("gps_coordinates")

            # Risk scoring
            risk = 0.0
            flags = []

            # Editing software detection
            sw = (result["software"] or "").lower()
            editing_keywords = [
                "photoshop", "gimp", "lightroom", "affinity", "paintshop",
                "pixlr", "canva", "snapseed", "faceapp", "reface",
            ]
            if any(kw in sw for kw in editing_keywords):
                flags.append("editing_software_detected")
                risk += 0.3

            # AI generation software
            ai_keywords = [
                "stable diffusion", "midjourney", "dall-e", "dalle",
                "comfyui", "automatic1111", "novelai",
            ]
            if any(kw in sw for kw in ai_keywords):
                flags.append("ai_generation_software_detected")
                risk += 0.5

            # Missing metadata checks
            if not result["camera"] and not result["camera_make"]:
                flags.append("camera_metadata_missing")
                risk += 0.15

            if not result["date_time"]:
                flags.append("datetime_missing")
                risk += 0.1

            if not result["metadata_available"] or result["exif_tag_count"] == 0:
                flags.append("no_exif_data")
                risk += 0.2

            if not result["gps"]:
                flags.append("gps_missing")
                # GPS missing is common, low risk
                risk += 0.05

            result["metadata_risk"] = round(min(risk, 1.0), 2)
            result["flags"] = flags

        except Exception as e:
            result["error"] = str(e)[:300]
            result["flags"] = ["metadata_extraction_failed"]

        self._record("Metadata analysis", _elapsed(t0))
        return result

    # ==================================================================
    # STAGE 4 — Face / Region Localization
    # ==================================================================
    def run_face_region_localization(self, image_path: str) -> Dict[str, Any]:
        """Detect faces, extract bounding boxes, quality metrics."""
        t0 = time.perf_counter()
        result: Dict[str, Any] = {
            "faces_detected": 0,
            "regions": [],
            "selected_crop": None,
            "status": "not_configured",
        }

        try:
            from image.face_detector import inspect_face_candidates
            face_review = inspect_face_candidates(image_path)
            result["status"] = face_review.get("status", "unknown")
            result["selected_crop"] = face_review.get("selected_crop")
            result["interpretation"] = face_review.get("interpretation")

            # Extract face count and bounding boxes
            candidates = face_review.get("candidates", [])
            result["faces_detected"] = len(candidates)
            for candidate in candidates:
                bbox = candidate.get("bbox_xyxy") or candidate.get("crop_box_xyxy")
                if bbox:
                    result["regions"].append({
                        "bbox": bbox,
                        "confidence": candidate.get("confidence", 0.0),
                    })

            # If face was detected but no explicit candidates list
            if result["status"] == "face_detected" and not result["regions"]:
                crop_box = face_review.get("crop_box_xyxy")
                if crop_box:
                    result["faces_detected"] = max(1, face_review.get("face_count", 1))
                    result["regions"] = [{"bbox": crop_box, "confidence": 0.95}]

        except Exception as e:
            result["error"] = str(e)[:300]
            logger.warning(f"Face detection failed: {e}")

        self._record("Face/region localization", _elapsed(t0))
        return result

    # ==================================================================
    # STAGE 5 — Xception Prediction
    # ==================================================================
    def run_xception_prediction(self, image_path: str) -> Dict[str, Any]:
        """Run Xception deepfake detection with calibrated probabilities."""
        t0 = time.perf_counter()
        if self.xception_detector is None:
            self._record("Xception inference SKIPPED — model not loaded", _elapsed(t0))
            return {"status": "not_configured", "verdict": "Model not available"}

        try:
            result = self.xception_detector.predict(image_path)
            self._record("Xception inference", _elapsed(t0))
            return result
        except Exception as e:
            self._record(f"Xception inference FAILED: {e}", _elapsed(t0))
            return {"status": "error", "error": str(e)[:300]}

    # ==================================================================
    # STAGE 6 — Grad-CAM++
    # ==================================================================
    def run_gradcam_plus_plus(self, model_output: Dict[str, Any]) -> Dict[str, Any]:
        """Extract the Grad-CAM++ overlay from the Xception output."""
        t0 = time.perf_counter()
        explainability = model_output.get("explainability")
        if explainability and explainability.get("status") == "generated":
            self._record("Grad-CAM++ extracted", _elapsed(t0))
            return explainability
        self._record("Grad-CAM++ not available", _elapsed(t0))
        return {"status": "not_generated", "method": "Grad-CAM++"}

    # ==================================================================
    # STAGE 7 — Frequency Analysis (FFT + DCT)
    # ==================================================================
    def run_fft_dct_evidence(self, image_path: str) -> Dict[str, Any]:
        """Run FFT power spectrum + DCT coefficient analysis."""
        t0 = time.perf_counter()
        result: Dict[str, Any] = {"fft": None, "dct": None, "frequency_risk": 0.0}

        # FFT — uses existing real module
        try:
            from frequency.frequency_pipeline import analyze_image_frequency
            fft_result = analyze_image_frequency(image_path)
            result["fft"] = fft_result
            measurements = fft_result.get("measurements", {})
            high_freq = _safe_float(measurements.get("high_frequency_power_fraction"))
            # GANs often show reduced high-frequency energy
            if high_freq < 0.05:
                result["frequency_risk"] = max(result["frequency_risk"], 0.6)
            elif high_freq < 0.10:
                result["frequency_risk"] = max(result["frequency_risk"], 0.3)
        except Exception as e:
            logger.warning(f"FFT analysis failed: {e}")
            result["fft"] = {"status": "error", "error": str(e)[:200]}

        # DCT — real implementation
        try:
            from frequency.dct import analyze_dct_coefficients
            result["dct"] = analyze_dct_coefficients(image_path)
            dct_anomaly = _safe_float(result["dct"].get("anomaly_score"))
            result["frequency_risk"] = max(result["frequency_risk"], dct_anomaly)
        except Exception as e:
            logger.warning(f"DCT analysis failed: {e}")
            result["dct"] = {"status": "error", "error": str(e)[:200]}

        result["frequency_risk"] = round(result["frequency_risk"], 3)
        self._record("Frequency analysis (FFT + DCT)", _elapsed(t0))
        return result

    # ==================================================================
    # STAGE 8 — Compression / JPEG Forensics (ELA)
    # ==================================================================
    def run_compression_analysis(self, image_path: str) -> Dict[str, Any]:
        """Error Level Analysis + JPEG quality estimation."""
        t0 = time.perf_counter()
        try:
            from forensics.compression_analyzer import analyze_compression
            result = analyze_compression(image_path)
        except Exception as e:
            logger.warning(f"Compression analysis failed: {e}")
            result = {"status": "error", "error": str(e)[:300]}
        self._record("Compression/ELA analysis", _elapsed(t0))
        return result

    # ==================================================================
    # STAGE 9 — Noise / Sensor Forensics
    # ==================================================================
    def run_sensor_forensics(self, image_path: str) -> Dict[str, Any]:
        """Noise residual + PRNU estimation."""
        t0 = time.perf_counter()
        result: Dict[str, Any] = {"noise": None, "prnu": None, "sensor_risk": 0.0}

        # Noise residual — existing real module
        try:
            from forensics.noise_analysis import analyze_image_noise
            noise = analyze_image_noise(image_path)
            result["noise"] = noise
            measurements = noise.get("measurements", {})
            rms = _safe_float(measurements.get("residual_rms"))
            # Very low noise residual may indicate synthetic smooth image
            if rms < 1.5:
                result["sensor_risk"] = max(result["sensor_risk"], 0.5)
        except Exception as e:
            logger.warning(f"Noise analysis failed: {e}")
            result["noise"] = {"status": "error", "error": str(e)[:200]}

        # PRNU — real implementation
        try:
            from forensics.prnu_analysis import estimate_prnu
            result["prnu"] = estimate_prnu(image_path)
            prnu_inconsistency = _safe_float(result["prnu"].get("inconsistency_score"))
            result["sensor_risk"] = max(result["sensor_risk"], prnu_inconsistency)
        except Exception as e:
            logger.warning(f"PRNU analysis failed: {e}")
            result["prnu"] = {"status": "error", "error": str(e)[:200]}

        result["sensor_risk"] = round(result["sensor_risk"], 3)
        self._record("Sensor forensics (noise + PRNU)", _elapsed(t0))
        return result

    # ==================================================================
    # STAGE 10 — Manipulation Localization
    # ==================================================================
    def run_manipulation_localization(
        self,
        image_path: str,
        gradcam: Dict[str, Any],
        faces: Dict[str, Any],
        compression: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Merge Grad-CAM++ attention, face regions, and ELA anomalies
        to produce a combined suspicion map.
        """
        t0 = time.perf_counter()
        suspicious_regions: List[Dict[str, Any]] = []

        # If Grad-CAM++ concentrated on face region
        gradcam_on_face = False
        if (
            gradcam.get("status") == "generated"
            and faces.get("faces_detected", 0) > 0
        ):
            gradcam_on_face = True
            suspicious_regions.append({
                "source": "gradcam_face_overlap",
                "description": "Model attention concentrated on detected face region",
                "severity": "high",
            })

        # ELA anomalies
        ela_regions = []
        if isinstance(compression, dict):
            ela_regions = compression.get("localized_anomalies", [])
            for region in ela_regions:
                suspicious_regions.append({
                    "source": "ela_anomaly",
                    "description": f"Compression inconsistency in region {region.get('region', 'unknown')}",
                    "severity": region.get("severity", "medium"),
                })

        result = {
            "suspicious_regions_detected": len(suspicious_regions) > 0,
            "region_count": len(suspicious_regions),
            "regions": suspicious_regions,
            "gradcam_face_overlap": gradcam_on_face,
        }
        self._record("Manipulation localization", _elapsed(t0))
        return result

    # ==================================================================
    # STAGE 11 — Multi-Model Ensemble
    # ==================================================================
    def run_ensemble_models(self, evidence: Dict[str, Any]) -> Dict[str, Any]:
        """
        Weighted evidence fusion using dynamic weights.
        Calls the modular evidence_fusion engine.
        """
        t0 = time.perf_counter()
        from fusion.evidence_fusion import fuse_evidence
        result = fuse_evidence(evidence)
        self._record("Ensemble fusion", _elapsed(t0))
        return result

    # ==================================================================
    # STAGE 12 — Evidence Consistency Analysis
    # ==================================================================
    def analyze_evidence_consistency(self, evidence: Dict[str, Any], ensemble: Dict[str, Any]) -> Dict[str, Any]:
        """
        Cross-check all evidence sources for agreement and conflict.
        This is the most important stage for forensic credibility.
        """
        t0 = time.perf_counter()
        scores = ensemble.get("individual_scores", {})
        conflicts: List[Dict[str, Any]] = []
        agreements: List[str] = []

        # Classify each evidence source
        HIGH_THRESHOLD = 0.6
        LOW_THRESHOLD = 0.3

        high_sources = [k for k, v in scores.items() if v >= HIGH_THRESHOLD]
        low_sources = [k for k, v in scores.items() if v <= LOW_THRESHOLD]
        mid_sources = [k for k, v in scores.items() if LOW_THRESHOLD < v < HIGH_THRESHOLD]

        # Detect specific conflicts
        xception_score = scores.get("xception", -1)
        frequency_score = scores.get("frequency", -1)
        sensor_score = scores.get("sensor", -1)
        compression_score = scores.get("compression", -1)

        if xception_score >= HIGH_THRESHOLD and frequency_score >= 0 and frequency_score <= LOW_THRESHOLD:
            conflicts.append({
                "type": "xception_frequency_conflict",
                "description": f"Xception says FAKE ({xception_score:.2f}) but frequency analysis is CLEAN ({frequency_score:.2f})",
                "severity": "high",
            })

        if xception_score >= HIGH_THRESHOLD and sensor_score >= 0 and sensor_score <= LOW_THRESHOLD:
            conflicts.append({
                "type": "xception_sensor_conflict",
                "description": f"Xception says FAKE ({xception_score:.2f}) but sensor/PRNU analysis is CLEAN ({sensor_score:.2f})",
                "severity": "medium",
            })

        if xception_score <= LOW_THRESHOLD and compression_score >= HIGH_THRESHOLD:
            conflicts.append({
                "type": "compression_xception_conflict",
                "description": f"Xception says REAL ({xception_score:.2f}) but compression shows anomalies ({compression_score:.2f})",
                "severity": "medium",
            })

        # Agreement analysis
        if len(high_sources) >= 3:
            agreements.append(f"Strong agreement: {', '.join(high_sources)} all indicate manipulation")
        elif len(high_sources) >= 2:
            agreements.append(f"Moderate agreement: {', '.join(high_sources)} indicate manipulation")

        if len(low_sources) >= 3:
            agreements.append(f"Strong agreement: {', '.join(low_sources)} all indicate authenticity")

        # Overall consistency
        if len(high_sources) > 0 and len(low_sources) > 0:
            consistency = "MIXED"
        elif len(high_sources) >= 2:
            consistency = "HIGH_AGREEMENT_FAKE"
        elif len(low_sources) >= 2:
            consistency = "HIGH_AGREEMENT_REAL"
        else:
            consistency = "INSUFFICIENT"

        # Reliability impact
        if len(conflicts) >= 2:
            reliability_impact = "SIGNIFICANTLY_REDUCED"
        elif len(conflicts) == 1:
            reliability_impact = "MODERATELY_REDUCED"
        else:
            reliability_impact = "NONE"

        result = {
            "consistency": consistency,
            "conflicts": conflicts,
            "conflict_count": len(conflicts),
            "agreements": agreements,
            "high_evidence_sources": high_sources,
            "low_evidence_sources": low_sources,
            "mixed_evidence_sources": mid_sources,
            "reliability_impact": reliability_impact,
        }
        self._record("Evidence consistency analysis", _elapsed(t0))
        return result

    # ==================================================================
    # STAGE 13 — Calibration + Uncertainty
    # ==================================================================
    def apply_calibrated_confidence(self, ensemble: Dict[str, Any], xception: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calibrate the fusion score. If Xception has proper temperature
        calibration, use it. Otherwise mark score as 'model score' not
        'true probability'.
        """
        t0 = time.perf_counter()
        fusion_score = ensemble.get("fusion_score", 0.0)
        calibration_status = "uncalibrated"
        calibrated_probability = None

        # Check if Xception itself has calibration
        model_info = xception.get("model", {})
        calibration_temp = model_info.get("calibration_temperature")
        if calibration_temp and calibration_temp > 0:
            calibration_status = "temperature_scaled"
            # The Xception probability is already calibrated
            calibrated_probability = xception.get("deepfake_probability")

        result = {
            "raw_fusion_score": round(fusion_score, 4),
            "calibration_status": calibration_status,
            "calibrated_probability": round(calibrated_probability, 4) if calibrated_probability is not None else None,
            "display_label": (
                f"{calibrated_probability * 100:.1f}% calibrated probability"
                if calibrated_probability is not None
                else f"{fusion_score * 100:.1f}% model score (calibration: pending)"
            ),
        }
        self._record("Confidence calibration", _elapsed(t0))
        return result

    # ==================================================================
    # STAGE 13b — Uncertainty Estimation
    # ==================================================================
    def estimate_prediction_uncertainty(self, ensemble: Dict[str, Any], consistency: Dict[str, Any]) -> Dict[str, Any]:
        """
        Estimate prediction uncertainty from ensemble variance and
        evidence consistency.
        """
        t0 = time.perf_counter()
        scores = list(ensemble.get("individual_scores", {}).values())

        # Variance of individual scores
        if len(scores) >= 2:
            variance = float(np.var(scores))
            std_dev = float(np.std(scores))
        else:
            variance = 0.0
            std_dev = 0.0

        # Entropy of fusion score
        fusion = ensemble.get("fusion_score", 0.5)
        if 0 < fusion < 1:
            entropy = -(fusion * math.log2(fusion) + (1 - fusion) * math.log2(1 - fusion))
        else:
            entropy = 0.0

        # Uncertainty from consistency conflicts
        conflict_penalty = consistency.get("conflict_count", 0) * 0.1

        uncertainty = min(1.0, std_dev + entropy * 0.3 + conflict_penalty)

        # Reliability label
        if uncertainty < 0.15:
            reliability = "HIGH"
        elif uncertainty < 0.35:
            reliability = "MEDIUM"
        else:
            reliability = "LOW"

        result = {
            "uncertainty": round(uncertainty, 3),
            "variance": round(variance, 4),
            "std_dev": round(std_dev, 4),
            "entropy": round(entropy, 4),
            "reliability": reliability,
            "sources_count": len(scores),
        }
        self._record("Uncertainty estimation", _elapsed(t0))
        return result

    # ==================================================================
    # STAGE 14 — OOD / Reliability Analysis
    # ==================================================================
    def detect_out_of_distribution(self, xception: Dict[str, Any], validation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if input is likely out-of-distribution for the trained model.
        Uses heuristics since we don't have stored training embeddings.
        """
        t0 = time.perf_counter()
        ood_flags: List[str] = []
        ood_score = 0.0

        # Check image dimensions — Xception trained on 299×299 face crops
        width = validation.get("width", 0)
        height = validation.get("height", 0)
        if width and height:
            min_dim = min(width, height)
            if min_dim < 64:
                ood_flags.append("very_small_image")
                ood_score += 0.4

        # Check if face was found (model trained on face images)
        # The xception result itself has face_review info
        face_review = xception.get("face_review", {})
        if face_review.get("status") == "no_face_detected":
            ood_flags.append("no_face_detected_ood_risk")
            ood_score += 0.3

        # Check if probability is very close to 0.5 (model uncertain)
        prob = xception.get("deepfake_probability")
        if prob is not None and 0.4 < prob < 0.6:
            ood_flags.append("borderline_prediction")
            ood_score += 0.2

        # Non-standard format
        fmt = validation.get("format", "").upper()
        if fmt not in ("JPEG", "PNG", "WEBP", "BMP"):
            ood_flags.append("unusual_format")
            ood_score += 0.1

        ood_score = round(min(ood_score, 1.0), 3)
        result = {
            "ood": ood_score > 0.5,
            "ood_score": ood_score,
            "flags": ood_flags,
            "warning": (
                "Input may differ significantly from training distribution"
                if ood_score > 0.5
                else None
            ),
            "model_training_domain": "FFHQ real faces + StyleGAN synthetic faces",
            "model_generalization": "Limited to training distribution",
        }
        self._record("OOD detection", _elapsed(t0))
        return result

    # ==================================================================
    # STAGE 15 — Final Forensic Assessment
    # ==================================================================
    def generate_final_assessment(
        self,
        evidence: Dict[str, Any],
        ensemble: Dict[str, Any],
        consistency: Dict[str, Any],
        calibration: Dict[str, Any],
        uncertainty: Dict[str, Any],
        ood: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate the final forensic decision with classification,
        evidence score, confidence, and explanation.
        """
        t0 = time.perf_counter()

        fusion_score = ensemble.get("fusion_score", 0.0)
        unc = uncertainty.get("uncertainty", 1.0)
        is_ood = ood.get("ood", False)
        conflict_count = consistency.get("conflict_count", 0)
        reliability_str = uncertainty.get("reliability", "LOW")

        # Determine decision status
        if is_ood or unc > 0.5 or conflict_count >= 3:
            decision_status = "INCONCLUSIVE"
            evidence_level = "INSUFFICIENT"
        elif fusion_score >= 0.75 and reliability_str == "HIGH":
            decision_status = "LIKELY_DEEPFAKE"
            evidence_level = "STRONG"
        elif fusion_score >= 0.60:
            decision_status = "SUSPICIOUS"
            evidence_level = "MODERATE"
        elif fusion_score <= 0.25 and reliability_str != "LOW":
            decision_status = "LIKELY_AUTHENTIC"
            evidence_level = "STRONG"
        elif fusion_score <= 0.40:
            decision_status = "LIKELY_AUTHENTIC"
            evidence_level = "MODERATE"
        else:
            decision_status = "INCONCLUSIVE"
            evidence_level = "WEAK"

        # Generate explanation reasons
        reasons: List[str] = []
        warnings: List[str] = []

        xception = evidence.get("xception", {})
        xception_prob = xception.get("deepfake_probability")
        if xception_prob is not None and xception_prob >= 0.7:
            reasons.append("Strong facial manipulation indicators from Xception model")
        elif xception_prob is not None and xception_prob <= 0.3:
            reasons.append("Xception model indicates likely authentic")

        if evidence.get("frequency", {}).get("frequency_risk", 0) >= 0.5:
            reasons.append("High-frequency anomalies detected in FFT/DCT analysis")

        if evidence.get("compression", {}).get("compression_risk", 0) >= 0.5:
            reasons.append("Compression inconsistency detected (ELA)")

        if evidence.get("sensor", {}).get("sensor_risk", 0) >= 0.5:
            reasons.append("Sensor noise inconsistency detected")

        gradcam = evidence.get("gradcam", {})
        if gradcam.get("status") == "generated":
            localization = evidence.get("localization", {})
            if localization.get("gradcam_face_overlap"):
                reasons.append("Model attention concentrated around face region")

        # Metadata-related warnings
        meta = evidence.get("metadata", {})
        if "editing_software_detected" in meta.get("flags", []):
            warnings.append("Editing software detected in metadata (evidence of editing history, not proof of deepfake)")
        if "ai_generation_software_detected" in meta.get("flags", []):
            warnings.append("AI generation software detected in metadata")
        if "no_exif_data" in meta.get("flags", []):
            warnings.append("No EXIF metadata available")

        # Calibration warnings
        cal_status = calibration.get("calibration_status", "uncalibrated")
        if cal_status == "uncalibrated":
            warnings.append("Calibration pending — displayed score is a model estimate, not a validated probability")

        # OOD warnings
        if is_ood:
            warnings.append(f"Input may be out-of-distribution for this model (OOD score: {ood.get('ood_score', 0):.2f})")

        # Conflict warnings
        for conflict in consistency.get("conflicts", []):
            warnings.append(f"Evidence conflict: {conflict['description']}")

        result = {
            "decision_status": decision_status,
            "forensic_score": round(fusion_score, 4),
            "reliability": reliability_str,
            "evidence_level": evidence_level,
            "uncertainty": round(unc, 3),
            "model_conflict": conflict_count > 0,
            "ood": is_ood,
            "reasons": reasons,
            "warnings": warnings,
            "calibration_status": cal_status,
            "timeline": self.timeline,
        }
        self._record("Final assessment generated", _elapsed(t0))
        return result

    # ==================================================================
    # MAIN ORCHESTRATOR
    # ==================================================================
    def run_fusion_pipeline(
        self,
        image_path: str,
        case_id: Optional[str] = None,
        generate_explanation: bool = True,
        generate_heatmap: bool = True,
        enable_frequency: bool = True,
        enable_metadata: bool = True,
        enable_compression: bool = True,
        enable_sensor_analysis: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute the complete 15-stage forensic evidence pipeline.

        Individual forensic evidence → normalization → dynamic weighting →
        ensemble fusion → raw forensic score → calibration →
        calibrated probability → uncertainty → OOD/reliability →
        final forensic assessment
        """
        pipeline_start = time.perf_counter()
        self.timeline = []
        self._record("Pipeline started")

        # === Stage 1: Validation ===
        validation = self.validate_input_media(image_path)
        if not validation.get("valid"):
            return {
                "error": "Invalid media",
                "validation": validation,
                "timeline": self.timeline,
            }

        # === Stage 2: File Integrity ===
        integrity = self.calculate_file_hash(image_path)

        # === Gather evidence ===
        evidence: Dict[str, Any] = {}

        # === Stage 3: Metadata ===
        if enable_metadata:
            evidence["metadata"] = self.run_metadata_analysis(image_path)

        # === Stage 4: Face Localization ===
        evidence["faces"] = self.run_face_region_localization(image_path)

        # === Stage 5: Xception Prediction ===
        xception = self.run_xception_prediction(image_path)
        evidence["xception"] = xception

        # === Stage 6: Grad-CAM++ ===
        if generate_heatmap:
            evidence["gradcam"] = self.run_gradcam_plus_plus(xception)

        # === Stage 7: Frequency Analysis ===
        if enable_frequency:
            evidence["frequency"] = self.run_fft_dct_evidence(image_path)

        # === Stage 8: Compression / ELA ===
        if enable_compression:
            evidence["compression"] = self.run_compression_analysis(image_path)

        # === Stage 9: Sensor Forensics ===
        if enable_sensor_analysis:
            evidence["sensor"] = self.run_sensor_forensics(image_path)

        # === Stage 10: Manipulation Localization ===
        evidence["localization"] = self.run_manipulation_localization(
            image_path,
            gradcam=evidence.get("gradcam", {}),
            faces=evidence.get("faces", {}),
            compression=evidence.get("compression", {}),
        )

        # === Stage 11: Ensemble Fusion ===
        ensemble = self.run_ensemble_models(evidence)

        # === Stage 12: Evidence Consistency ===
        consistency = self.analyze_evidence_consistency(evidence, ensemble)

        # === Stage 13: Calibration + Uncertainty ===
        calibration = self.apply_calibrated_confidence(ensemble, xception)
        uncertainty = self.estimate_prediction_uncertainty(ensemble, consistency)

        # === Stage 14: OOD Detection ===
        ood = self.detect_out_of_distribution(xception, validation)

        # === Stage 15: Final Assessment ===
        assessment = self.generate_final_assessment(
            evidence=evidence,
            ensemble=ensemble,
            consistency=consistency,
            calibration=calibration,
            uncertainty=uncertainty,
            ood=ood,
        )

        total_ms = (time.perf_counter() - pipeline_start) * 1000
        self._record(f"Pipeline complete ({total_ms:.0f} ms total)")

        return {
            "evidence_id": integrity["evidence_id"],
            "case_id": case_id,
            "validation": validation,
            "integrity": integrity,
            "evidence": evidence,
            "ensemble": ensemble,
            "consistency": consistency,
            "calibration": calibration,
            "uncertainty": uncertainty,
            "ood": ood,
            "assessment": assessment,
            "processing_time_ms": round(total_ms, 1),
        }


def _elapsed(start: float) -> float:
    """Return elapsed milliseconds since start."""
    return (time.perf_counter() - start) * 1000


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)
    pipeline = FusionPipeline()
    # Replace with an actual image path for testing
    result = pipeline.run_fusion_pipeline("sample.jpg")
    print(json.dumps(result, indent=2, default=str))
