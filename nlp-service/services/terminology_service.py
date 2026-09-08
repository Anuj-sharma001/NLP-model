"""Compatibility forwarding module for services package."""

from app.services.terminology_service import (
    AyurvedicTerm,
    AyurvedicTermProvenance,
    ProtectedTextResult,
    TermMatch,
    TerminologyService,
)

__all__ = [
    "TerminologyService",
    "AyurvedicTerm",
    "AyurvedicTermProvenance",
    "TermMatch",
    "ProtectedTextResult",
]
