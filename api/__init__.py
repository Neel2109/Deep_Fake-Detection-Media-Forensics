"""Modular API package; live demo endpoints remain in app.api.routes."""

from app.api.routes import router as live_demo_router

__all__ = ["live_demo_router"]
