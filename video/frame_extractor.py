"""Bounded, evenly spaced video keyframe extraction for analyst review."""

import base64
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from image.face_detector import inspect_face_image

MAX_KEYFRAMES = 8
MAX_FRAME_SIDE = 480
JPEG_QUALITY = 78


def extract_keyframes(
    path: str | Path,
    *,
    max_frames: int = MAX_KEYFRAMES,
    max_side: int = MAX_FRAME_SIDE,
    include_images: bool = False,
) -> dict[str, Any]:
    if not 1 <= max_frames <= MAX_KEYFRAMES:
        raise ValueError(f"max_frames must be between 1 and {MAX_KEYFRAMES}")
    if max_side < 32:
        raise ValueError("max_side must be at least 32 pixels")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        return {
            "status": "unavailable",
            "frames": [],
            "interpretation": "The video decoder could not open this file; no keyframes were extracted.",
        }

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    duration = frame_count / fps if frame_count > 0 and math.isfinite(fps) and fps > 0 else None
    sample_count = min(max_frames, frame_count) if frame_count > 0 else 1
    if duration is not None and sample_count > 1:
        timestamps = np.linspace(0.0, max(0.0, duration - 0.001), sample_count)
    else:
        timestamps = np.array([0.0])

    keyframes = []
    seen_positions: set[int] = set()
    for timestamp in timestamps:
        position = max(0, int(round(float(timestamp) * fps))) if fps > 0 else 0
        if position in seen_positions:
            continue
        seen_positions.add(position)
        capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp) * 1000.0)
        decoded, frame = capture.read()
        if not decoded or frame is None or frame.size == 0:
            continue

        height, width = frame.shape[:2]
        scale = min(1.0, max_side / max(width, height))
        if scale < 1.0:
            frame = cv2.resize(
                frame,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        encoded, buffer = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
        )
        if not encoded:
            continue
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_review = inspect_face_image(Image.fromarray(rgb_frame))
        keyframe = {
            "index": len(keyframes),
            "timestamp_seconds": round(float(timestamp), 3),
            "frame_position": position,
            "width": int(frame.shape[1]),
            "height": int(frame.shape[0]),
            "preview_data_url": "data:image/jpeg;base64,"
            + base64.b64encode(buffer.tobytes()).decode("ascii"),
            "face_review": face_review,
        }
        if include_images:
            keyframe["_image"] = Image.fromarray(rgb_frame.copy())
        keyframes.append(keyframe)

    capture.release()
    return {
        "status": "measured_descriptive" if keyframes else "partial",
        "sampled_frame_count": len(keyframes),
        "requested_frame_count": sample_count,
        "sampling_strategy": "evenly spaced across decoder-reported duration",
        "max_frame_side": max_side,
        "frames": keyframes,
        "interpretation": (
            "These are representative decoded keyframes with descriptive face-candidate checks. "
            "Sampling can miss events between frames and does not detect deepfakes or temporal manipulation."
        ),
    }
