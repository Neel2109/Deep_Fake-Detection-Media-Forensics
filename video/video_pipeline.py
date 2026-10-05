"""Video metadata and keyframe review without temporal authenticity claims."""

from pathlib import Path
from typing import Any

from video.frame_extractor import extract_keyframes


def analyze_video(path: str | Path, metadata: dict, detector: Any | None = None) -> dict:
    keyframe_result = extract_keyframes(path, include_images=detector is not None)
    frame_results = []
    if detector is not None:
        for frame in keyframe_result["frames"]:
            image = frame.pop("_image")
            predict_frame = getattr(detector, "predict_frame", detector.predict_image)
            prediction = predict_frame(image)
            frame["xception_estimate"] = {
                "status": prediction["status"],
                "decision": prediction["decision"],
                "deepfake_score": prediction["deepfake_probability"],
                "authenticity_score": prediction["authenticity_probability"],
                "calibration": prediction["model"]["calibration"],
                "architecture": prediction["model"]["architecture"],
            }
            frame_results.append(frame["xception_estimate"])

    model_frame_analysis = None
    if detector is not None:
        model_frame_analysis = {
            "status": "frame_estimates_available" if frame_results else "no_frames_analyzed",
            "architecture": frame_results[0]["architecture"] if frame_results else None,
            "calibration": frame_results[0]["calibration"] if frame_results else None,
            "analyzed_frame_count": len(frame_results),
            "sampled_frame_count": keyframe_result.get("sampled_frame_count", 0),
            "video_level_decision": "withheld",
            "interpretation": (
                (
                    "Per-frame image-model scores are shown for sampled frames only. "
                    "They are not calibrated for video-level decisions, do not measure temporal consistency, "
                    "and must not be averaged into a video authenticity probability."
                    if frame_results
                    else "No usable video frames were decoded, so the image checkpoint produced no frame estimates."
                )
            ),
        }
    return {
        **keyframe_result,
        "model_frame_analysis": model_frame_analysis,
        "decoder_reported_fps": metadata.get("decoder_reported_fps"),
        "container_reported_frame_count": metadata.get("container_reported_frame_count"),
        "duration_estimate_seconds": metadata.get("duration_estimate_seconds"),
        "interpretation": (
            f"{keyframe_result['interpretation']} "
            "Frame count and duration originate from decoder/container reports and may be approximate."
            + (
                " " + model_frame_analysis["interpretation"]
                if model_frame_analysis
                else ""
            )
        ),
    }
