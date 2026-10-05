"""Validated checkpoint loader exposed under the requested inference package."""

from app.config import XCEPTION_CHECKPOINT
from app.services.xception_detector import get_xception_detector

__all__ = ["XCEPTION_CHECKPOINT", "get_xception_detector"]
