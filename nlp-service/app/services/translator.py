"""
Translation service module for SAKYTI multilingual NLP.
Exposes translate_to_english, translate_from_english, and high-level translate functions.
"""

from typing import Optional
from app.models.indictrans_model import (
    IndicTransModelManager,
    TranslationError,
    TranslationInputTooLongError,
    TranslationModelNotLoadedError,
    UnsupportedLanguageError,
    normalize_language_code,
)
from app.schemas.translation import TranslationResponse
from app.services.language_detector import detect_language
from app.utils.logger import get_logger

logger = get_logger("translator_service")


def translate_to_english(
    text: str,
    source_language: str,
    max_length: Optional[int] = None,
    num_beams: Optional[int] = None,
    preserve_ayurvedic_terms: bool = True,
) -> str:
    """
    Translate Indic language text to English, preserving canonical Ayurvedic terms.

    Args:
        text: Source Indic text to translate.
        source_language: Language code or tag (e.g., 'hi', 'hin', 'hin_Deva', 'ta', 'tam_Taml').
        max_length: Optional override for max input tokens.
        num_beams: Optional override for beam search.
        preserve_ayurvedic_terms: Whether to preserve Ayurvedic technical terms.

    Returns:
        Translated English string with preserved terminology.
    """
    if not text or not text.strip():
        raise ValueError("Input text cannot be empty or whitespace.")

    src_tag = normalize_language_code(source_language)
    if src_tag == "eng_Latn":
        raise UnsupportedLanguageError(
            "Source language for translate_to_english cannot be English (eng_Latn)."
        )

    # Protect Ayurvedic technical terms if requested
    text_to_translate = text
    placeholders = {}
    if preserve_ayurvedic_terms:
        from app.services.terminology_service import TerminologyService

        term_svc = TerminologyService.get_instance()
        prot = term_svc.protect_terms_for_translation(text)
        text_to_translate = prot.protected_text
        placeholders = prot.placeholders

    manager = IndicTransModelManager.get_instance()
    translated_text, _, _, _ = manager.translate(
        text=text_to_translate,
        source_language=src_tag,
        target_language="eng_Latn",
        max_length=max_length,
        num_beams=num_beams,
    )

    if preserve_ayurvedic_terms and placeholders:
        from app.services.terminology_service import TerminologyService

        return TerminologyService.get_instance().restore_terms_after_translation(
            translated_text, placeholders, target_language="eng_Latn"
        )

    return translated_text


def translate_from_english(
    text: str,
    target_language: str,
    max_length: Optional[int] = None,
    num_beams: Optional[int] = None,
    preserve_ayurvedic_terms: bool = True,
) -> str:
    """
    Translate English text to an Indic target language, preserving canonical Ayurvedic terms.

    Args:
        text: Source English text to translate.
        target_language: Target language code or tag (e.g., 'hi', 'hin', 'hin_Deva', 'ta', 'tam_Taml').
        max_length: Optional override for max input tokens.
        num_beams: Optional override for beam search.
        preserve_ayurvedic_terms: Whether to preserve Ayurvedic technical terms.

    Returns:
        Translated Indic language string.
    """
    if not text or not text.strip():
        raise ValueError("Input text cannot be empty or whitespace.")

    tgt_tag = normalize_language_code(target_language)
    if tgt_tag == "eng_Latn":
        raise UnsupportedLanguageError(
            "Target language for translate_from_english cannot be English (eng_Latn)."
        )

    # Protect Ayurvedic technical terms if requested
    text_to_translate = text
    placeholders = {}
    if preserve_ayurvedic_terms:
        from app.services.terminology_service import TerminologyService

        term_svc = TerminologyService.get_instance()
        prot = term_svc.protect_terms_for_translation(text)
        text_to_translate = prot.protected_text
        placeholders = prot.placeholders

    manager = IndicTransModelManager.get_instance()
    translated_text, _, _, _ = manager.translate(
        text=text_to_translate,
        source_language="eng_Latn",
        target_language=tgt_tag,
        max_length=max_length,
        num_beams=num_beams,
    )

    if preserve_ayurvedic_terms and placeholders:
        from app.services.terminology_service import TerminologyService

        return TerminologyService.get_instance().restore_terms_after_translation(
            translated_text, placeholders, target_language=tgt_tag
        )

    return translated_text


def translate(
    text: str,
    target_language: str,
    source_language: Optional[str] = None,
    max_length: Optional[int] = None,
    num_beams: Optional[int] = None,
    preserve_ayurvedic_terms: bool = True,
) -> TranslationResponse:
    """
    High-level translation function returning a structured TranslationResponse schema.
    If source_language is omitted, language detection is automatically performed.
    """
    if not text or not text.strip():
        raise ValueError("Input text cannot be empty or whitespace.")

    resolved_src: str
    if source_language:
        resolved_src = normalize_language_code(source_language)
    else:
        # Detect language automatically
        detection_result = detect_language(text)
        resolved_src = normalize_language_code(detection_result.language_code)
        logger.info(
            "Auto-detected source language for translation",
            extra={
                "detected_code": detection_result.language_code,
                "resolved_tag": resolved_src,
                "confidence": detection_result.confidence,
            },
        )

    resolved_tgt = normalize_language_code(target_language)

    # Protect Ayurvedic technical terms
    text_to_translate = text
    placeholders = {}
    preserved_terms: List[str] = []
    if preserve_ayurvedic_terms:
        from app.services.terminology_service import TerminologyService

        term_svc = TerminologyService.get_instance()
        prot = term_svc.protect_terms_for_translation(text)
        text_to_translate = prot.protected_text
        placeholders = prot.placeholders
        preserved_terms = [t.canonical_term for t in placeholders.values()]

    manager = IndicTransModelManager.get_instance()
    translated_text, model_name, model_version, elapsed_ms = manager.translate(
        text=text_to_translate,
        source_language=resolved_src,
        target_language=resolved_tgt,
        max_length=max_length,
        num_beams=num_beams,
    )

    if preserve_ayurvedic_terms and placeholders:
        from app.services.terminology_service import TerminologyService

        translated_text = TerminologyService.get_instance().restore_terms_after_translation(
            translated_text, placeholders, target_language=resolved_tgt
        )

    return TranslationResponse(
        translated_text=translated_text,
        source_language=resolved_src,
        target_language=resolved_tgt,
        model_name=model_name,
        model_version=model_version,
        execution_time_ms=round(elapsed_ms, 2),
        preserved_terms=preserved_terms or None,
    )
