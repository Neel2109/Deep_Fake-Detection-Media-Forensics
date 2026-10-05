"""Xception inference contract and factory for the optional trained checkpoint."""

from importlib import import_module

from app.services.xception_detector import (
    ARCHITECTURE,
    IMAGE_SIZE,
    LABEL_MAPPING,
    NORMALIZATION_MEAN,
    NORMALIZATION_STD,
    XceptionModelError,
    validate_checkpoint_contract,
)


def create_model():
    try:
        timm = import_module("timm")
    except ImportError as error:
        raise XceptionModelError(
            "Xception dependencies are unavailable; install requirements-ml.txt."
        ) from error
    return timm.create_model("legacy_xception", pretrained=False, num_classes=1)


__all__ = [
    "ARCHITECTURE",
    "IMAGE_SIZE",
    "LABEL_MAPPING",
    "NORMALIZATION_MEAN",
    "NORMALIZATION_STD",
    "XceptionModelError",
    "create_model",
    "validate_checkpoint_contract",
]
