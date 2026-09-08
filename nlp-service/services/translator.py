"""Compatibility forwarding module for services package."""

from app.services.translator import (
    translate,
    translate_from_english,
    translate_to_english,
)

__all__ = [
    "translate_to_english",
    "translate_from_english",
    "translate",
]
