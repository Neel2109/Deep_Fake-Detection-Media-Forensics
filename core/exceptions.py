"""Application exceptions that keep unavailable capabilities explicit."""


class DeepTraceError(RuntimeError):
    """Base error for application-level failures."""


class FeatureNotConfigured(DeepTraceError):
    """Raised when a feature has no implementation or validated model."""


__all__ = ["DeepTraceError", "FeatureNotConfigured"]
