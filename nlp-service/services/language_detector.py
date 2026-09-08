"""Compatibility forwarding module for services package."""

from app.services.language_detector import (
    LANGUAGE_METADATA,
    SCRIPT_METADATA,
    calculate_latin_ratio,
    detect_language,
    parse_indiclid_label,
)

__all__ = [
    "detect_language",
    "calculate_latin_ratio",
    "parse_indiclid_label",
    "LANGUAGE_METADATA",
    "SCRIPT_METADATA",
]
