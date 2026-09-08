"""Dedicated Multilingual NLP API router for SAKYTI.

Exposes:
- POST /api/v1/nlp/analyze
- POST /api/v1/nlp/translate
- POST /api/v1/nlp/detect-language

Features:
- Strict Pydantic validation
- Structured error handling with standardized schema
- Request IDs and correlation tracing
- Structured logging with execution timings
- Model health status and version metadata
- Comprehensive OpenAPI documentation
"""

import time
from typing import Any, Dict
from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.config import Settings, get_settings
from app.models.indiclid_model import IndicLIDModelManager
from app.models.indictrans_model import (
    IndicTransModelManager,
    TranslationError,
    TranslationInputTooLongError,
    TranslationModelNotLoadedError,
    UnsupportedLanguageError,
)
from app.schemas.nlp_api import (
    ErrorResponse,
    ModelInfo,
    NLPAnalyzeRequest,
    NLPAnalyzeResponse,
    NLPDetectLanguageRequest,
    NLPDetectLanguageResponse,
    NLPTranslateRequest,
    NLPTranslateResponse,
)
from app.services.language_detector import detect_language
from app.services.pipeline import SakytiPipeline
from app.services.translator import translate
from app.utils.logger import get_logger
from app.utils.request_id import get_current_request_id

logger = get_logger("sakyti-nlp.api.nlp")

router = APIRouter(
    prefix="/nlp",
    tags=["Multilingual NLP"],
    responses={
        status.HTTP_400_BAD_REQUEST: {
            "model": ErrorResponse,
            "description": "Bad Request / Unsupported Language / Invalid Parameter",
        },
        status.HTTP_413_REQUEST_ENTITY_TOO_LARGE: {
            "model": ErrorResponse,
            "description": "Payload Exceeds Maximum Length Limit",
        },
        status.HTTP_422_UNPROCESSABLE_ENTITY: {
            "model": ErrorResponse,
            "description": "Validation Error in Request Payload",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ErrorResponse,
            "description": "Internal Server Error",
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ErrorResponse,
            "description": "Model Not Loaded or Temporarily Unavailable",
        },
    },
)


def _get_active_models_metadata() -> Dict[str, ModelInfo]:
    """Retrieve current operational status and versioning of loaded NLP models."""
    settings = get_settings()
    lid_manager = IndicLIDModelManager.get_instance()
    lid_status = lid_manager.get_status()
    is_lid_loaded = any(v == "loaded" for v in lid_status.values())

    trans_manager = IndicTransModelManager.get_instance()
    trans_status = "loaded" if trans_manager.is_loaded else ("error" if trans_manager._load_error else "not_loaded")

    return {
        "indic_lid": ModelInfo(
            name="IndicLID",
            version="v1.0 (FTN/FTR FastText)",
            status="loaded" if is_lid_loaded else "not_loaded",
            device="cpu",
        ),
        "indic_trans2": ModelInfo(
            name="IndicTrans2",
            version=settings.indictrans2_indic_en_model,
            status=trans_status,
            device=str(trans_manager.device) if trans_manager.device else "cpu",
        ),
    }


# =============================================================================
# 1. POST /api/v1/nlp/analyze
# =============================================================================

@router.post(
    "/analyze",
    response_model=NLPAnalyzeResponse,
    summary="End-to-end Multilingual & Ayurvedic Text Analysis",
    description="""
Execute the master SAKYTI multilingual NLP pipeline on user input:
1. **Input Validation**: Check string type, non-emptiness, and character bounds.
2. **Unicode Normalization**: Canonical NFC normalization, whitespace collapsing, safe punctuation normalization, and danda preservation.
3. **Language & Script Detection**: AI4Bharat IndicLID language identification across Indic languages, Latin scripts, and confidence estimation.
4. **Romanization Handling**: Automatic detection of Romanized Indic inputs (e.g. Hinglish) and confidence-gated native Devanagari conversion.
5. **Ayurvedic Terminology Extraction**: Identifies and extracts canonical clinical terms (*Vata*, *Pitta*, *Kapha*, *Triphala*, *Ashwagandha*, *Prakriti*, *Vikriti*, *Agni*, etc.) with categories and span offsets.
6. **Technical Translation**: Preserves clinical terms from literal corruption and translates Indic concepts to English using IndicTrans2.
7. **Intermediate Representation**: Returns a structured JSON intermediate representation with model health and version metadata.
    """,
    status_code=status.HTTP_200_OK,
    response_description="Structured intermediate representation with terminology, normalized text, translation, and model versions.",
)
async def nlp_analyze(
    payload: NLPAnalyzeRequest,
    request: Request,
) -> NLPAnalyzeResponse:
    """Analyze input text through the 7-stage SAKYTI multilingual NLP pipeline."""
    req_id = getattr(request.state, "request_id", None) or get_current_request_id()
    logger.info(
        "Received NLP analyze request",
        extra={
            "request_id": req_id,
            "text_length": len(payload.text),
            "target_language": payload.target_language,
            "min_confidence": payload.min_confidence,
        },
    )

    try:
        pipeline = SakytiPipeline.get_instance()
        result = pipeline.process_query(
            text=payload.text,
            target_language=payload.target_language or "en",
            min_confidence=payload.min_confidence,
        )

        models_meta = _get_active_models_metadata()

        response_data = NLPAnalyzeResponse(
            original_text=result.original_text,
            normalized_text=result.normalized_text,
            detected_language=result.detected_language,
            confidence=result.confidence,
            script=result.script,
            is_romanized=result.is_romanized,
            terminology=result.terminology,
            english_text=result.english_text,
            target_language=result.target_language,
            processing_status=result.processing_status,
            execution_time_ms=result.execution_time_ms,
            models=models_meta,
            request_id=req_id,
        )

        logger.info(
            "NLP analyze completed successfully",
            extra={
                "request_id": req_id,
                "detected_language": result.detected_language,
                "terms_found": len(result.terminology),
                "execution_time_ms": result.execution_time_ms,
            },
        )
        return response_data

    except ValueError as val_err:
        logger.warning(f"Validation error in NLP analyze [{req_id}]: {val_err}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )
    except Exception as exc:
        logger.error(f"Unexpected error in NLP analyze [{req_id}]: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"NLP analysis failed: {exc}",
        )


# =============================================================================
# 2. POST /api/v1/nlp/translate
# =============================================================================

@router.post(
    "/translate",
    response_model=NLPTranslateResponse,
    summary="Direct Indic ↔ English Translation with Terminology Preservation",
    description="""
Translate text between supported Indic languages and English using official AI4Bharat IndicTrans2 models:
- Supports Indic -> English and English -> Indic for 10 initial Indic languages + Sanskrit.
- Preserves technical Ayurvedic terminology (*Vata*, *Pitta*, *Kapha*, *Prakriti*, *Agni*, *Ama*, *Triphala*, etc.) from improper translation when `preserve_ayurvedic_terms` is enabled.
- Automatically resolves source language via IndicLID if `source_language` is omitted.
- Returns comprehensive model metadata, version tags, compute device, and latency metrics.
    """,
    status_code=status.HTTP_200_OK,
    response_description="Translated text with technical term preservation and model diagnostic metadata.",
)
async def nlp_translate(
    payload: NLPTranslateRequest,
    request: Request,
) -> NLPTranslateResponse:
    """Translate clinical or conversational text between Indic languages and English."""
    req_id = getattr(request.state, "request_id", None) or get_current_request_id()
    logger.info(
        "Received NLP translate request",
        extra={
            "request_id": req_id,
            "text_length": len(payload.text),
            "source_language": payload.source_language,
            "target_language": payload.target_language,
            "preserve_terms": payload.preserve_ayurvedic_terms,
        },
    )

    try:
        res = translate(
            text=payload.text,
            target_language=payload.target_language,
            source_language=payload.source_language,
            preserve_ayurvedic_terms=payload.preserve_ayurvedic_terms,
        )

        trans_manager = IndicTransModelManager.get_instance()
        device_str = str(trans_manager.device) if trans_manager.device else "cpu"

        response_data = NLPTranslateResponse(
            translated_text=res.translated_text,
            source_language=res.source_language,
            target_language=res.target_language,
            model_name=res.model_name,
            model_version=res.model_version,
            model_status="ready" if trans_manager.is_loaded else "degraded",
            device=device_str,
            execution_time_ms=res.execution_time_ms,
            request_id=req_id,
        )

        logger.info(
            "NLP translate completed successfully",
            extra={
                "request_id": req_id,
                "source_language": res.source_language,
                "target_language": res.target_language,
                "execution_time_ms": res.execution_time_ms,
            },
        )
        return response_data

    except (ValueError, UnsupportedLanguageError) as exc:
        logger.warning(f"Unsupported language or value error in NLP translate [{req_id}]: {exc}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except TranslationInputTooLongError as exc:
        logger.warning(f"Input too long in NLP translate [{req_id}]: {exc}")
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=str(exc),
        )
    except TranslationModelNotLoadedError as exc:
        logger.error(f"Translation model not loaded [{req_id}]: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(f"Unexpected error in NLP translate [{req_id}]: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Translation failed: {exc}",
        )


# =============================================================================
# 3. POST /api/v1/nlp/detect-language
# =============================================================================

@router.post(
    "/detect-language",
    response_model=NLPDetectLanguageResponse,
    summary="Language, Script, and Romanization Identification",
    description="""
Identify the language, script, and Romanization status of input text using official AI4Bharat IndicLID:
- Classifies 22 Indic languages + English.
- Identifies native Indic scripts (Devanagari, Tamil, Bengali, Telugu, etc.) vs Latin/Roman script.
- Distinguishes Romanized Indic languages (e.g. Hinglish) from authentic English.
- Evaluates confidence scores against optional confidence threshold.
- Returns model version, health state, and classification latency.
    """,
    status_code=status.HTTP_200_OK,
    response_description="Identified language, script, romanization status, confidence, and model diagnostics.",
)
async def nlp_detect_language(
    payload: NLPDetectLanguageRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
) -> NLPDetectLanguageResponse:
    """Classify language and script using AI4Bharat IndicLID models."""
    req_id = getattr(request.state, "request_id", None) or get_current_request_id()
    logger.info(
        "Received NLP detect-language request",
        extra={
            "request_id": req_id,
            "text_length": len(payload.text),
            "confidence_threshold": payload.confidence_threshold,
        },
    )

    t0 = time.perf_counter()
    try:
        detection = detect_language(
            text=payload.text,
            confidence_threshold=payload.confidence_threshold,
            settings=settings,
        )
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

        lid_manager = IndicLIDModelManager.get_instance()
        is_ready = lid_manager.is_loaded

        response_data = NLPDetectLanguageResponse(
            language_code=detection.language_code,
            language_name=detection.language_name,
            script=detection.script,
            confidence=detection.confidence,
            is_romanized=detection.is_romanized,
            raw_model_result=detection.raw_model_result,
            model_name="IndicLID",
            model_version="v1.0 (FTN/FTR)",
            model_status="ready" if is_ready else "not_loaded",
            execution_time_ms=elapsed_ms,
            request_id=req_id,
        )

        logger.info(
            "NLP detect-language completed successfully",
            extra={
                "request_id": req_id,
                "language_code": detection.language_code,
                "script": detection.script,
                "confidence": detection.confidence,
                "is_romanized": detection.is_romanized,
                "execution_time_ms": elapsed_ms,
            },
        )
        return response_data

    except Exception as exc:
        logger.error(f"Error in NLP detect-language [{req_id}]: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Language detection failed: {exc}",
        )
