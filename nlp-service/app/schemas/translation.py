from typing import List, Optional
from pydantic import BaseModel, Field


class TranslationRequest(BaseModel):
    """Schema for text translation requests."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Source text to be translated.",
        examples=["मुझे पेट में दर्द है"],
    )
    source_language: Optional[str] = Field(
        default=None,
        description="Source language code (ISO 639-1, ISO 639-3, or IndicTrans2 tag). If not provided, language detection will be used.",
        examples=["hin_Deva", "hi", "hin"],
    )
    target_language: str = Field(
        ...,
        description="Target language code (ISO 639-1, ISO 639-3, or IndicTrans2 tag).",
        examples=["eng_Latn", "en", "hin_Deva", "hi"],
    )
    preserve_ayurvedic_terms: bool = Field(
        default=True,
        description="Whether to preserve canonical Ayurvedic technical terminology from literal translation.",
    )


class TranslationResponse(BaseModel):
    """Schema for text translation responses."""

    translated_text: str = Field(
        ...,
        description="Translated text output.",
        examples=["I have a stomach ache"],
    )
    source_language: str = Field(
        ...,
        description="Resolved source language code (canonical IndicTrans2 tag).",
        examples=["hin_Deva"],
    )
    target_language: str = Field(
        ...,
        description="Resolved target language code (canonical IndicTrans2 tag).",
        examples=["eng_Latn"],
    )
    model_name: str = Field(
        ...,
        description="Name of the translation model used.",
        examples=["IndicTrans2"],
    )
    model_version: str = Field(
        ...,
        description="Version or HuggingFace checkpoint ID of the translation model.",
        examples=["rotary-indictrans2-indic-en-dist-200M"],
    )
    execution_time_ms: float = Field(
        ...,
        description="Inference execution time in milliseconds.",
        examples=[45.2],
    )
    preserved_terms: Optional[List[str]] = Field(
        default=None,
        description="Canonical Ayurvedic terms identified and preserved in this translation.",
        examples=[["Vata", "Triphala"]],
    )
