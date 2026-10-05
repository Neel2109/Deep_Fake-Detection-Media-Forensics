"""Analysis orchestration backed by the current multimodal intake service."""

from app.services.forensics import analyze_media

__all__ = ["analyze_media"]
