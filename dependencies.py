"""Dependency helpers shared by future modular API routers."""

from fastapi import Request


def get_application_state(request: Request):
    """Return the existing application state used for this demo session."""
    return request.app.state
