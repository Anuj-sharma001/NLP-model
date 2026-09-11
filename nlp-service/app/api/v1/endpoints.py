"""API v1 Endpoints for SAKYTI NLP Service."""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.schemas.language import LanguageDetectionResult
from app.schemas.normalization import NormalizationRequest, NormalizationResult
from app.schemas.pipeline import PipelineQueryRequest, PipelineQueryResult
from app.schemas.romanized import RomanizedProcessRequest, RomanizedProcessResult
from app.schemas.translation import TranslationRequest, TranslationResponse
from app.services.language_detector import detect_language

router = APIRouter(tags=["SAKYTI NLP"])


class LanguageDetectionRequest(BaseModel):
    """Request schema for language identification."""

    text: str = Field(
        ...,
        min_length=1,
        description="Text content to classify language and script for",
        json_schema_extra={"example": "मुझे पेट में दर्द है"},
    )
    confidence_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional minimum confidence threshold override",
    )


@router.post(
    "/detect-language",
    response_model=LanguageDetectionResult,
    summary="Detect language and script",
    status_code=status.HTTP_200_OK,
)
async def api_detect_language(
    payload: LanguageDetectionRequest,
    settings: Settings = Depends(get_settings),
) -> LanguageDetectionResult:
    """Identify the language, script, romanization status, and confidence for given text."""
    try:
        return detect_language(
            text=payload.text,
            confidence_threshold=payload.confidence_threshold,
            settings=settings,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Language detection failed: {exc}",
        )


@router.post(
    "/translate",
    response_model=TranslationResponse,
    summary="Translate text between Indic languages and English",
    status_code=status.HTTP_200_OK,
)
async def api_translate(
    payload: TranslationRequest,
) -> TranslationResponse:
    """
    Translate text using AI4Bharat IndicTrans2.
    Supports Indic -> English and English -> Indic for supported Indian languages.
    """
    from app.models.indictrans_model import (
        TranslationError,
        TranslationInputTooLongError,
        TranslationModelNotLoadedError,
        UnsupportedLanguageError,
    )
    from app.services.translator import translate

    try:
        return translate(
            text=payload.text,
            target_language=payload.target_language,
            source_language=payload.source_language,
            preserve_ayurvedic_terms=payload.preserve_ayurvedic_terms,
        )
    except (ValueError, UnsupportedLanguageError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except TranslationInputTooLongError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=str(exc),
        )
    except TranslationModelNotLoadedError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )
    except TranslationError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )


# =========================================================================
# Terminology Layer Endpoints
# =========================================================================

class TerminologyIdentifyRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Input text to scan for Ayurvedic terms.")


class TerminologyCanonicalizeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Input text containing colloquial or non-canonical Ayurvedic terms.")
    target_script: str = Field(default="latin", description="'latin' or 'devanagari'")


@router.get(
    "/terminology/terms",
    summary="List all canonical Ayurvedic terms",
    tags=["Ayurveda Terminology"],
    status_code=status.HTTP_200_OK,
)
async def api_list_terms(category: Optional[str] = None):
    """Retrieve catalog of canonical Ayurvedic terminology, optionally filtered by category."""
    from app.services.terminology_service import TerminologyService

    service = TerminologyService.get_instance()
    terms = service.list_terms(category=category)
    return {
        "count": len(terms),
        "terms": [t.model_dump() for t in terms],
    }


@router.get(
    "/terminology/terms/{term_name}",
    summary="Lookup Ayurvedic term by canonical name or alias",
    tags=["Ayurveda Terminology"],
    status_code=status.HTTP_200_OK,
)
async def api_get_term(term_name: str):
    """Retrieve term record by canonical name, Devanagari form, or Romanized transliteration."""
    from app.services.terminology_service import TerminologyService

    service = TerminologyService.get_instance()
    term = service.get_term(term_name)
    if not term:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ayurvedic term '{term_name}' not found in terminology database.",
        )
    return term.model_dump()


@router.post(
    "/terminology/identify",
    summary="Identify Ayurvedic terms in free-form text",
    tags=["Ayurveda Terminology"],
    status_code=status.HTTP_200_OK,
)
async def api_identify_terms(payload: TerminologyIdentifyRequest):
    """Identify and locate all known Ayurvedic technical terms within arbitrary text."""
    from app.services.terminology_service import TerminologyService

    service = TerminologyService.get_instance()
    matches = service.identify_terms(payload.text)
    return {
        "count": len(matches),
        "matches": [m.model_dump() for m in matches],
    }


@router.post(
    "/terminology/canonicalize",
    summary="Canonicalize Ayurvedic terms in text",
    tags=["Ayurveda Terminology"],
    status_code=status.HTTP_200_OK,
)
async def api_canonicalize_terms(payload: TerminologyCanonicalizeRequest):
    """Replace non-standard or phonetic variants with canonical forms."""
    from app.services.terminology_service import TerminologyService

    service = TerminologyService.get_instance()
    canonicalized = service.canonicalize_text(
        text=payload.text,
        target_script="devanagari" if payload.target_script.lower() == "devanagari" else "latin",
    )
    return {
        "original_text": payload.text,
        "canonicalized_text": canonicalized,
    }


# =========================================================================
# Multilingual Normalization Endpoint
# =========================================================================

@router.post(
    "/normalize",
    response_model=NormalizationResult,
    summary="Normalize multilingual text",
    tags=["Text Normalization"],
    status_code=status.HTTP_200_OK,
)
async def api_normalize_text(payload: NormalizationRequest) -> NormalizationResult:
    """
    Multilingual text normalization handling Unicode NFC, whitespace standardization,
    repeated punctuation collapsing, quote/dash canonicalization, HTML/control noise removal,
    language-aware Indic normalizer hooks, and Romanized Indian-language input,
    with strict preservation of Ayurvedic technical terminology.
    """
    from app.services.normalizer import normalize_text

    try:
        return normalize_text(
            text=payload.text,
            language=payload.language,
            is_romanized=payload.is_romanized,
            preserve_ayurvedic_terms=payload.preserve_ayurvedic_terms,
            remove_html=payload.remove_html,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Text normalization failed: {exc}",
        )


# =========================================================================
# Romanized Indian Language Processing Endpoint
# =========================================================================

@router.post(
    "/process-romanized",
    response_model=RomanizedProcessResult,
    summary="Process Romanized Indian language input",
    tags=["Romanized Input Processing"],
    status_code=status.HTTP_200_OK,
)
async def api_process_romanized(payload: RomanizedProcessRequest) -> RomanizedProcessResult:
    """
    Process Romanized Indian-language input:
    - Distinguishes Romanized Indic from English without assuming Latin script is English
    - Preserves and canonicalizes Ayurvedic technical terms (Vata, Pitta, Kapha, etc.)
    - Gated script conversion: only converts if confidence threshold is met
    - Retains original text for audit and debugging
    """
    from app.services.romanized_processor import process_romanized_text

    try:
        return process_romanized_text(
            text=payload.text,
            target_language=payload.target_language,
            min_conversion_confidence=payload.min_conversion_confidence,
            preserve_ayurvedic_terms=payload.preserve_ayurvedic_terms,
            convert_script=payload.convert_script,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Romanized processing failed: {exc}",
        )


# =========================================================================
# SAKYTI Multilingual NLP Pipeline Endpoint
# =========================================================================

@router.post(
    "/pipeline/process",
    response_model=PipelineQueryResult,
    summary="Process query through end-to-end SAKYTI NLP pipeline",
    tags=["Multilingual Pipeline"],
    status_code=status.HTTP_200_OK,
)
async def api_pipeline_process(payload: PipelineQueryRequest) -> PipelineQueryResult:
    """
    Execute full SAKYTI Multilingual NLP Pipeline:
    1. Validate input
    2. Normalize text (Unicode NFC, whitespace, punctuation, danda preservation)
    3. Identify language & script (IndicLID)
    4. Detect romanization & convert to native script if applicable
    5. Extract Ayurvedic technical terms (Vata, Pitta, Kapha, etc.)
    6. Translate to English with term preservation (IndicTrans2)
    7. Generate structured intermediate representation
    """
    from app.services.pipeline import process_query

    try:
        return process_query(
            text=payload.text,
            target_language=payload.target_language,
            min_confidence=payload.min_confidence,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline execution failed: {exc}",
        )
