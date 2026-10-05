"""
Evidence Fusion Engine.
Fuses diverse forensic signals into a single score using dynamic weighting.
"""

from typing import Any, Dict

IMPLEMENTATION_STATUS = "implemented"
FEATURE_NAME = "evidence_fusion"

def fuse_evidence(evidence: Dict[str, Any], weight_config: Dict[str, float] = None) -> Dict[str, Any]:
    """
    Fuses multiple evidence scores using dynamic weights based on availability.
    """
    if weight_config is None:
        weight_config = {
            "xception":    0.35,
            "frequency":   0.15,
            "compression": 0.15,
            "sensor":      0.10,
            "metadata":    0.10,
            "localization": 0.10,
            "faces":       0.05,
        }

    scores = {}
    active_weights = {}

    # Helper to extract safe float
    def _safe_float(v):
        try: return float(v)
        except: return 0.0

    # Extract available scores
    xception = evidence.get("xception", {})
    if xception.get("status") in ("classified", "calibrated_probability_only"):
        scores["xception"] = _safe_float(xception.get("deepfake_probability"))
        active_weights["xception"] = weight_config["xception"]

    frequency = evidence.get("frequency", {})
    freq_risk = _safe_float(frequency.get("frequency_risk"))
    if freq_risk > 0:
        scores["frequency"] = freq_risk
        active_weights["frequency"] = weight_config["frequency"]

    compression = evidence.get("compression", {})
    comp_risk = _safe_float(compression.get("compression_risk"))
    if comp_risk > 0:
        scores["compression"] = comp_risk
        active_weights["compression"] = weight_config["compression"]

    sensor = evidence.get("sensor", {})
    sensor_risk = _safe_float(sensor.get("sensor_risk"))
    if sensor_risk > 0:
        scores["sensor"] = sensor_risk
        active_weights["sensor"] = weight_config["sensor"]

    metadata = evidence.get("metadata", {})
    meta_risk = _safe_float(metadata.get("metadata_risk"))
    if meta_risk > 0:
        scores["metadata"] = meta_risk
        active_weights["metadata"] = weight_config["metadata"]

    localization = evidence.get("localization", {})
    if localization.get("suspicious_regions_detected"):
        region_count = localization.get("region_count", 0)
        scores["localization"] = min(1.0, region_count * 0.3)
        active_weights["localization"] = weight_config["localization"]

    # Calculate weighted fusion score
    if active_weights:
        total_weight = sum(active_weights.values())
        normalized = {k: v / total_weight for k, v in active_weights.items()}
        fusion_score = sum(scores[k] * normalized[k] for k in scores)
    else:
        fusion_score = 0.0
        normalized = {}

    return {
        "fusion_score": round(fusion_score, 4),
        "individual_scores": scores,
        "weights_used": {k: round(v, 3) for k, v in normalized.items()},
        "evidence_sources_available": len(scores),
        "evidence_sources_total": len(weight_config),
    }
