"""Pydantic schemas for the Romanized Indian Language Processing Layer."""

from typing import List, Optional
from pydantic import BaseModel, Field


class RomanizedProcessRequest(BaseModel):
    """Request payload for processing Romanized Indian-language input."""

    text: str = Field(
        ...,
        min_length=1,
        description="Raw input text, potentially in Romanized Indian language or English",
        json_schema_extra={"example": "mujhe pet me dard ho raha hai aur pitta vikriti lagti hai"},
    )
    target_language: Optional[str] = Field(
        default=None,
        description="Optional language code hint (e.g. 'hi', 'ta', 'te'). If None, automatically detected via IndicLID.",
    )
    min_conversion_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold required to perform script conversion to native script",
    )
    preserve_ayurvedic_terms: bool = Field(
        default=True,
        description="Whether to identify, shield, and canonicalize Ayurvedic technical terms",
    )
    convert_script: bool = Field(
        default=True,
        description="Whether to attempt transliteration/conversion to native Indic script if confidence is met",
    )


class RomanizedProcessResult(BaseModel):
    """Result model containing detection, conversion, and audit metadata."""

    original_text: str = Field(
        ...,
        description="The exact raw input text kept unmodified for auditing and debugging",
    )
    detected_language: str = Field(
        ...,
        description="ISO/SAKYTI language code (e.g. 'hi', 'en', 'ta')",
    )
    detected_language_name: str = Field(
        ...,
        description="Human-readable language name (e.g. 'Hindi', 'English')",
    )
    script: str = Field(
        ...,
        description="Detected script (e.g. 'Latin', 'Devanagari')",
    )
    is_romanized: bool = Field(
        ...,
        description="Whether the text is classified as a Romanized Indian language",
    )
    confidence: float = Field(
        ...,
        description="Language detection confidence score from IndicLID",
    )
    was_converted: bool = Field(
        ...,
        description="Whether script conversion was successfully performed",
    )
    converted_text: Optional[str] = Field(
        default=None,
        description="Text converted to native script (e.g., Devanagari for Hindi), or None if unconverted",
    )
    ayurvedic_terms_detected: List[str] = Field(
        default_factory=list,
        description="Canonical names of Ayurvedic terms found and preserved in the input",
    )
    processing_status: str = Field(
        ...,
        description="Status description (e.g. 'converted_to_devanagari', 'bypassed_english', 'insufficient_confidence', 'unsupported_language')",
    )
    execution_time_ms: float = Field(
        default=0.0,
        description="Processing latency in milliseconds",
    )
