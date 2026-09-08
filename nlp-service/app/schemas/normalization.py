"""Pydantic schemas for the SAKYTI multilingual text normalization service."""

from typing import List, Optional
from pydantic import BaseModel, Field


class NormalizationRequest(BaseModel):
    """Request payload for text normalization."""

    text: str = Field(
        ...,
        min_length=1,
        description="Raw input text to normalize",
        json_schema_extra={"example": "मरीज   को  bahuuut  pitta   vikriti  hai!!!! &amp; bukhar bhi   ।"},
    )
    language: Optional[str] = Field(
        default=None,
        description="Optional language code (e.g. 'hi', 'ta', 'sa', 'en') for language-aware normalization rules",
    )
    is_romanized: Optional[bool] = Field(
        default=None,
        description="Whether the input is Romanized Indic text (e.g., Hinglish). If None, auto-detected or defaults to False.",
    )
    preserve_ayurvedic_terms: bool = Field(
        default=True,
        description="Whether to shield canonical Ayurvedic medical terms from aggressive phonetic reduction or alteration",
    )
    remove_html: bool = Field(
        default=True,
        description="Whether to strip HTML tags and decode HTML entities",
    )


class NormalizationResult(BaseModel):
    """Response model containing normalized text and detection metadata."""

    original_text: str = Field(
        ...,
        description="The original input text before normalization",
    )
    normalized_text: str = Field(
        ...,
        description="The resulting normalized text",
    )
    detected_changes: List[str] = Field(
        default_factory=list,
        description="List of transformation categories/rules that modified the text",
    )
    language: Optional[str] = Field(
        default=None,
        description="Language code applied during normalization",
    )
    is_romanized: Optional[bool] = Field(
        default=None,
        description="Whether Romanized Indic normalization was applied",
    )
    execution_time_ms: float = Field(
        default=0.0,
        description="Processing latency in milliseconds",
    )
