"""Compatibility forwarding module for services package."""

from app.services.normalizer import (
    MultilingualNormalizer,
    normalize_text,
)
from app.schemas.normalization import (
    NormalizationRequest,
    NormalizationResult,
)

__all__ = [
    "MultilingualNormalizer",
    "normalize_text",
    "NormalizationRequest",
    "NormalizationResult",
]
