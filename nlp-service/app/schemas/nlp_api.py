"""Pydantic schemas for the SAKYTI Multilingual NLP API.

Provides structured request and response models, error representations,
and model metadata for:
- POST /api/v1/nlp/analyze
- POST /api/v1/nlp/translate
- POST /api/v1/nlp/detect-language
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# =============================================================================
# 1. Structured Error Schemas
# =============================================================================

class ErrorDetail(BaseModel):
    """Detailed information for structured error payloads."""

    code: str = Field(
        ...,
        description="Application-specific error code (e.g. VALIDATION_ERROR, UNSUPPORTED_LANGUAGE, MODEL_NOT_LOADED)",
        examples=["VALIDATION_ERROR"],
    )
    message: str = Field(
        ...,
        description="Human-readable description of the error",
        examples=["Input validation failed."],
    )
    details: Optional[Any] = Field(
        default=None,
        description="Optional detailed diagnostic information or validation error list",
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Correlation ID of the HTTP request for tracing",
        examples=["c9b1f2a3-8d4e-4b6a-9f1c-7e5d2a8b3c1d"],
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the error occurred in UTC ISO format",
    )


class ErrorResponse(BaseModel):
    """Unified structured error response envelope."""

    error: ErrorDetail = Field(
        ...,
        description="Structured error details",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Input validation failed.",
                    "details": [
                        {
                            "loc": ["body", "text"],
                            "msg": "Field required",
                            "type": "missing",
                        }
                    ],
                    "request_id": "c9b1f2a3-8d4e-4b6a-9f1c-7e5d2a8b3c1d",
                    "timestamp": "2026-09-08T01:15:00.000000Z",
                }
            }
        }
    }


# =============================================================================
# 2. Model Metadata Schemas
# =============================================================================

class ModelInfo(BaseModel):
    """Diagnostic and versioning information for an individual NLP model."""

    name: str = Field(..., description="Canonical model family name", examples=["IndicLID", "IndicTrans2"])
    version: str = Field(..., description="Model release version or checkpoint identifier", examples=["v1.0-FTR", "rotary-indictrans2-indic-en-dist-200M"])
    status: str = Field(..., description="Operational readiness state", examples=["loaded", "ready", "degraded", "not_loaded"])
    device: Optional[str] = Field(default=None, description="Compute device execution backend", examples=["cuda", "cpu"])


# =============================================================================
# 3. POST /api/v1/nlp/analyze Schemas
# =============================================================================

class NLPAnalyzeRequest(BaseModel):
    """Request payload for full multilingual linguistic and Ayurvedic analysis."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Input text in any supported Indic language, Romanized Indian language, or English",
        examples=["mujhe pet me dard hai aur pitta vikriti lagti hai"],
    )
    target_language: Optional[str] = Field(
        default="en",
        description="Target language for technical translation (defaults to English 'en')",
        examples=["en"],
    )
    min_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for language identification and script conversion",
        examples=[0.5],
    )
    preserve_ayurvedic_terms: bool = Field(
        default=True,
        description="Whether to guard and canonicalize Ayurvedic technical terms from corruption or literal translation",
    )


class NLPAnalyzeResponse(BaseModel):
    """Comprehensive analysis result with linguistic IR and model versioning."""

    original_text: str = Field(..., description="Exact raw input text preserved for auditability")
    normalized_text: str = Field(..., description="Normalized text after Unicode NFC, whitespace, and safe punctuation processing")
    detected_language: str = Field(..., description="ISO/SAKYTI code of detected language (e.g. 'hi', 'en', 'ta')")
    confidence: float = Field(..., description="Language detection confidence score (0.0 - 1.0)")
    script: str = Field(..., description="Detected script name (e.g. 'Devanagari', 'Latin', 'Tamil')")
    is_romanized: bool = Field(..., description="Whether input is identified as a Romanized Indian language")
    terminology: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Extracted canonical Ayurvedic clinical terms with categories and offsets",
    )
    english_text: Optional[str] = Field(
        default=None,
        description="Clinical English translation with preserved technical Ayurvedic terms",
    )
    target_language: Optional[str] = Field(
        default="en",
        description="Target language for translation",
    )
    processing_status: str = Field(
        ...,
        description="Stage-by-stage pipeline execution transition audit log",
        examples=["normalized | romanized_converted_to_devanagari | translated_from_hi"],
    )
    execution_time_ms: float = Field(
        ...,
        description="Total execution time in milliseconds",
        examples=[45.2],
    )
    models: Dict[str, ModelInfo] = Field(
        default_factory=dict,
        description="Operational health and version information for models utilized in this analysis",
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Correlation request identifier",
        examples=["a1b2c3d4-e5f6-7890-abcd-ef1234567890"],
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Processing completion timestamp in UTC",
    )


# =============================================================================
# 4. POST /api/v1/nlp/translate Schemas
# =============================================================================

class NLPTranslateRequest(BaseModel):
    """Request payload for Indic ↔ English translation."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Source text to translate",
        examples=["वात दोष के कारण शरीर में दर्द है"],
    )
    target_language: str = Field(
        ...,
        description="Target language code (e.g. 'en', 'hi', 'ta', 'te', 'bn', 'mr', 'gu', 'kn', 'ml', 'pa', 'or', 'sa')",
        examples=["en"],
    )
    source_language: Optional[str] = Field(
        default=None,
        description="Optional source language code. If omitted, language identification is automatically performed",
        examples=["hi"],
    )
    preserve_ayurvedic_terms: bool = Field(
        default=True,
        description="Whether to preserve technical Ayurvedic terms from literal translation",
    )


class NLPTranslateResponse(BaseModel):
    """Translation response payload with model health and version metadata."""

    translated_text: str = Field(..., description="Translated output text")
    source_language: str = Field(..., description="Resolved source language tag (canonical IndicTrans2 format)")
    target_language: str = Field(..., description="Resolved target language tag (canonical IndicTrans2 format)")
    model_name: str = Field(..., description="Name of translation model used", examples=["IndicTrans2"])
    model_version: str = Field(..., description="Model version or checkpoint identifier", examples=["rotary-indictrans2-indic-en-dist-200M"])
    model_status: str = Field(default="ready", description="Model operational state", examples=["ready", "loaded"])
    device: Optional[str] = Field(default=None, description="Device used for inference", examples=["cuda", "cpu"])
    execution_time_ms: float = Field(..., description="Translation execution latency in milliseconds")
    request_id: Optional[str] = Field(default=None, description="Correlation request ID")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of translation completion in UTC",
    )


# =============================================================================
# 5. POST /api/v1/nlp/detect-language Schemas
# =============================================================================

class NLPDetectLanguageRequest(BaseModel):
    """Request payload for language and script identification."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Input text content to classify language, script, and Romanization status for",
        examples=["मुझे पेट में दर्द है"],
    )
    confidence_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional confidence threshold override (0.0 to 1.0)",
    )


class NLPDetectLanguageResponse(BaseModel):
    """Language identification result with model version and operational status."""

    language_code: str = Field(..., description="ISO/SAKYTI language code (e.g. 'hin', 'tam', 'eng')")
    language_name: str = Field(..., description="Full English language name (e.g. 'Hindi', 'Tamil', 'English')")
    script: str = Field(..., description="Detected script (e.g. 'Devanagari', 'Tamil', 'Latin')")
    confidence: float = Field(..., description="Identification confidence score (0.0 to 1.0)")
    is_romanized: bool = Field(..., description="Whether Indic language is expressed in Roman/Latin script")
    raw_model_result: Dict[str, Any] = Field(default_factory=dict, description="Diagnostic classifier predictions")
    model_name: str = Field(default="IndicLID", description="Identifier of the model used", examples=["IndicLID"])
    model_version: str = Field(default="v1.0", description="Model release version", examples=["v1.0-FTN/FTR"])
    model_status: str = Field(default="ready", description="Model operational state", examples=["ready", "loaded"])
    execution_time_ms: Optional[float] = Field(default=None, description="Detection latency in milliseconds")
    request_id: Optional[str] = Field(default=None, description="Correlation request ID")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Classification timestamp in UTC",
    )
