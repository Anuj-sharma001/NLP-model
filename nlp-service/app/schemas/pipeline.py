"""Pydantic schemas for the SAKYTI Multilingual NLP Pipeline."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class PipelineQueryRequest(BaseModel):
    """Request payload for the end-to-end multilingual NLP pipeline."""

    text: str = Field(
        ...,
        min_length=1,
        description="Raw query text in any supported Indic language, Romanized Indian language, or English",
        json_schema_extra={"example": "mujhe pet me dard hai aur pitta vikriti lagti hai"},
    )
    target_language: Optional[str] = Field(
        default=None,
        description="Desired target language override (defaults to 'en')",
    )
    min_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for language identification and script conversion",
    )


class PipelineQueryResult(BaseModel):
    """Structured intermediate representation returned by the SAKYTI NLP pipeline."""

    original_text: str = Field(
        ...,
        description="Exact unmodified raw input text",
    )
    normalized_text: str = Field(
        ...,
        description="Text after Unicode NFC, whitespace, and safe punctuation normalization",
    )
    detected_language: str = Field(
        ...,
        description="ISO/SAKYTI code of detected language (e.g. 'hi', 'en', 'ta')",
    )
    confidence: float = Field(
        ...,
        description="Language detection confidence score from IndicLID",
    )
    script: str = Field(
        ...,
        description="Detected script (e.g. 'Devanagari', 'Latin', 'Tamil')",
    )
    is_romanized: bool = Field(
        ...,
        description="Whether input text was identified as a Romanized Indian language",
    )
    terminology: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of detected Ayurvedic technical terms with metadata, categories, and offsets",
    )
    english_text: Optional[str] = Field(
        default=None,
        description="Translated or direct English text with preserved Ayurvedic terminology",
    )
    target_language: Optional[str] = Field(
        default="en",
        description="Target language of the intermediate query (default: 'en')",
    )
    processing_status: str = Field(
        ...,
        description="Pipeline execution status description",
    )
    execution_time_ms: float = Field(
        default=0.0,
        description="Total pipeline processing latency in milliseconds",
    )
