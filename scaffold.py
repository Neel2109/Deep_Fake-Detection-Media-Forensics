"""Shared status marker and exception for intentionally unimplemented modules."""

if __package__:
    from .core.exceptions import FeatureNotConfigured
else:
    from core.exceptions import FeatureNotConfigured


def require_implementation(feature: str) -> None:
    """Fail explicitly rather than returning fabricated forensic results."""
    raise FeatureNotConfigured(
        f"{feature} is scaffolded but not configured with a validated implementation."
    )
