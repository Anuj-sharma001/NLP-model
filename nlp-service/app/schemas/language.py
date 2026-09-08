"""Language identification schemas."""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class LanguageDetectionResult(BaseModel):
    """Schema representing the output of language and script identification."""

    language_code: str = Field(
        ...,
        description="ISO language code (e.g. 'hin', 'tam', 'tel', 'ben', 'mar', 'eng', 'unknown')",
    )
    language_name: str = Field(
        ...,
        description="Full English name of identified language (e.g. 'Hindi', 'Tamil', 'English')",
    )
    script: str = Field(
        ...,
        description="Name of detected script (e.g. 'Devanagari', 'Tamil', 'Telugu', 'Bengali', 'Latin')",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Prediction confidence score between 0.0 and 1.0",
    )
    is_romanized: bool = Field(
        ...,
        description="True if an Indic language is written in Roman/Latin script, False for native script or native English",
    )
    raw_model_result: Dict[str, Any] = Field(
        default_factory=dict,
        description="Raw output metadata from the underlying IndicLID model",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "language_code": "hin",
                "language_name": "Hindi",
                "script": "Devanagari",
                "confidence": 0.985,
                "is_romanized": False,
                "raw_model_result": {
                    "raw_label": "__label__hin_Deva",
                    "model_used": "IndicLID-FTN",
                    "model_confidence": 0.985,
                },
            }
        }
    }
