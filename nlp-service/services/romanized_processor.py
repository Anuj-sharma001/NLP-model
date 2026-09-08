"""Compatibility forwarding module for services package."""

from app.services.romanized_processor import (
    BaseRomanizedConverter,
    HindiRomanizedConverter,
    RomanizedProcessor,
    process_romanized_text,
)
from app.schemas.romanized import (
    RomanizedProcessRequest,
    RomanizedProcessResult,
)

__all__ = [
    "BaseRomanizedConverter",
    "HindiRomanizedConverter",
    "RomanizedProcessor",
    "process_romanized_text",
    "RomanizedProcessRequest",
    "RomanizedProcessResult",
]
