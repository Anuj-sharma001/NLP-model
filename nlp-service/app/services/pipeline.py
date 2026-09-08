"""
SAKYTI Multilingual NLP Pipeline Service.

Coordinates the end-to-end query processing workflow:
1. Input validation
2. Text normalization
3. Language identification (IndicLID)
4. Romanization detection and handling
5. Ayurvedic terminology extraction
6. English translation (IndicTrans2)
7. Structured intermediate representation generation
"""

import time
from typing import Any, Dict, List, Optional, Tuple

from app.schemas.language import LanguageDetectionResult
from app.schemas.normalization import NormalizationResult
from app.schemas.pipeline import PipelineQueryResult
from app.schemas.romanized import RomanizedProcessResult
from app.schemas.terminology import TermMatch
from app.services.language_detector import detect_language
from app.services.normalizer import MultilingualNormalizer
from app.services.romanized_processor import RomanizedProcessor
from app.services.terminology_service import TerminologyService
from app.services.translator import translate_to_english
from app.utils.logger import get_logger

logger = get_logger("pipeline")


class SakytiPipeline:
    """
    Master pipeline orchestrating all SAKYTI linguistic components.
    Provides independent stage methods for robust stage-by-stage testing
    and full query processing into a structured intermediate representation.
    """

    _instance: Optional["SakytiPipeline"] = None

    def __init__(self) -> None:
        self.normalizer = MultilingualNormalizer.get_instance()
        self.romanized_processor = RomanizedProcessor.get_instance()
        self.terminology_service = TerminologyService.get_instance()

    @classmethod
    def get_instance(cls) -> "SakytiPipeline":
        """Get or create singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # =====================================================================
    # Stage 1: Input Validation
    # =====================================================================

    def validate_input(self, text: Any, max_length: int = 2000) -> str:
        """
        Stage 1: Validate input query text.
        Ensures text is a non-empty string and does not exceed max length.
        """
        if not isinstance(text, str):
            raise ValueError(f"Input query must be a string, got {type(text).__name__}")

        cleaned = text.strip()
        if not cleaned:
            raise ValueError("Input query cannot be empty or whitespace-only")

        if len(cleaned) > max_length:
            raise ValueError(
                f"Input query length ({len(cleaned)}) exceeds maximum allowed length of {max_length} characters"
            )

        return cleaned

    # =====================================================================
    # Stage 2: Text Normalization
    # =====================================================================

    def normalize_stage(
        self,
        text: str,
        language: Optional[str] = None,
    ) -> NormalizationResult:
        """
        Stage 2: Multilingual text normalization.
        Applies Unicode NFC, whitespace standardization, quote/dash canonicalization,
        and safe punctuation handling (preserving Devanagari danda and Ayurvedic terms).
        """
        return self.normalizer.normalize(
            text=text,
            language=language,
            preserve_ayurvedic_terms=True,
            remove_html=True,
        )

    # =====================================================================
    # Stage 3: Language Identification
    # =====================================================================

    def detect_language_stage(
        self,
        text: str,
        confidence_threshold: Optional[float] = None,
    ) -> LanguageDetectionResult:
        """
        Stage 3: Language and script identification using AI4Bharat IndicLID.
        """
        return detect_language(text, confidence_threshold=confidence_threshold)

    # =====================================================================
    # Stage 4: Romanization Detection & Handling
    # =====================================================================

    def detect_romanization_stage(
        self,
        text: str,
        detection: Optional[LanguageDetectionResult] = None,
        target_language: Optional[str] = None,
        min_conversion_confidence: float = 0.5,
    ) -> RomanizedProcessResult:
        """
        Stage 4: Detect and process Romanized Indian languages vs English.
        Converts Hinglish/Romanized Indic to native script if confidence threshold is met.
        """
        target_lang = target_language or (detection.language_code if detection else None)
        return self.romanized_processor.process(
            text=text,
            target_language=target_lang,
            min_conversion_confidence=min_conversion_confidence,
            preserve_ayurvedic_terms=True,
            convert_script=True,
        )

    # =====================================================================
    # Stage 5: Ayurvedic Terminology Extraction
    # =====================================================================

    def extract_terminology_stage(self, text: str) -> List[Dict[str, Any]]:
        """
        Stage 5: Identify and extract canonical Ayurvedic technical terms.
        Returns serializable list of term metadata with categories and spans.
        """
        matches: List[TermMatch] = self.terminology_service.identify_terms(text)
        serialized = []
        for m in matches:
            serialized.append({
                "canonical_term": m.term.canonical_term,
                "sanskrit_form": m.term.sanskrit_form,
                "category": m.term.category,
                "matched_text": m.matched_text,
                "start": m.start,
                "end": m.end,
                "do_not_translate": m.term.do_not_translate,
            })
        return serialized

    # =====================================================================
    # Stage 6: English Translation
    # =====================================================================

    def translate_stage(
        self,
        text: str,
        source_language: str,
        is_english: bool,
    ) -> Tuple[Optional[str], str]:
        """
        Stage 6: Translate Indic text to English when required via IndicTrans2,
        with strict preservation of Ayurvedic technical terms.
        If input is already English, passes through without translation.
        """
        if is_english:
            return text, "direct_english"

        # Supported Indic languages in IndicTrans2
        supported_indic = {
            "hi", "ta", "te", "bn", "mr", "gu", "kn", "ml", "pa", "or", "sa",
            "hin", "tam", "tel", "ben", "mar", "guj", "kan", "mal", "pan", "ory", "san"
        }

        clean_src = source_language.lower().strip()
        if clean_src not in supported_indic:
            return text, f"translation_unsupported_for_{clean_src}"

        try:
            from app.models.indictrans_model import IndicTransModelManager
            manager = IndicTransModelManager.get_instance()
            if not manager.is_loaded:
                logger.info("IndicTransModelManager not pre-loaded. Initializing now...")
                manager.initialize()

            translated_text = translate_to_english(
                text=text,
                source_language=clean_src,
            )
            return translated_text, f"translated_from_{clean_src}"
        except Exception as exc:
            logger.error(f"Translation failed in pipeline for '{clean_src}': {exc}", exc_info=True)
            return text, f"translation_failed: {exc}"

    # =====================================================================
    # Master Pipeline Orchestration
    # =====================================================================

    def process_query(
        self,
        text: str,
        target_language: Optional[str] = "en",
        min_confidence: float = 0.5,
    ) -> PipelineQueryResult:
        """
        Execute full SAKYTI Multilingual NLP Pipeline:
        1. Validate input
        2. Normalize text
        3. Detect language & script
        4. Detect romanization & convert if applicable
        5. Extract Ayurvedic technical terms
        6. Translate to English when required
        7. Return structured intermediate representation
        """
        start_time = time.perf_counter()
        original_text = text

        # 1. Validate input
        validated_text = self.validate_input(text)

        # 2. Normalize text
        norm_result = self.normalize_stage(validated_text)
        normalized_text = norm_result.normalized_text

        # 3. Detect language
        lang_detection = self.detect_language_stage(normalized_text, confidence_threshold=min_confidence)
        det_lang = lang_detection.language_code
        det_name = lang_detection.language_name
        det_script = lang_detection.script
        det_confidence = lang_detection.confidence

        is_english = det_lang in ["en", "eng"] or (det_script == "Latin" and not lang_detection.is_romanized)

        # 4. Detect romanization & process
        roman_result = self.detect_romanization_stage(
            text=normalized_text,
            detection=lang_detection,
            min_conversion_confidence=min_confidence,
        )
        is_romanized = roman_result.is_romanized

        # Choose the best text representation for downstream translation
        # If Hinglish was converted to Devanagari, use the converted text for translation!
        text_for_translation = normalized_text
        source_lang_for_translation = det_lang

        if roman_result.was_converted and roman_result.converted_text:
            text_for_translation = roman_result.converted_text
            # Converted Hindi is in Devanagari
            source_lang_for_translation = "hi"

        # 5. Extract Ayurvedic terminology (scanned across original, normalized, and converted)
        terms_original = self.extract_terminology_stage(original_text)
        terms_norm = self.extract_terminology_stage(normalized_text)
        terms_converted = (
            self.extract_terminology_stage(roman_result.converted_text)
            if roman_result.converted_text
            else []
        )

        # Merge unique terms by canonical_term
        seen_canonical = set()
        merged_terminology = []
        for t in terms_original + terms_norm + terms_converted:
            c = t["canonical_term"]
            if c not in seen_canonical:
                seen_canonical.add(c)
                merged_terminology.append(t)

        # 6. Translate to English when required
        target_lang = target_language or "en"
        english_text, trans_status = self.translate_stage(
            text=text_for_translation,
            source_language=source_lang_for_translation,
            is_english=is_english,
        )

        # 7. Processing status summary
        status_parts = []
        if norm_result.detected_changes:
            status_parts.append("normalized")
        if is_romanized:
            status_parts.append(f"romanized_{roman_result.processing_status}")
        status_parts.append(trans_status)
        final_status = " | ".join(status_parts) if status_parts else "completed"

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return PipelineQueryResult(
            original_text=original_text,
            normalized_text=normalized_text,
            detected_language=det_lang,
            confidence=round(det_confidence, 4),
            script=det_script,
            is_romanized=is_romanized,
            terminology=merged_terminology,
            english_text=english_text,
            target_language=target_lang,
            processing_status=final_status,
            execution_time_ms=round(elapsed_ms, 2),
        )


# Module-level convenience function
def process_query(
    text: str,
    target_language: Optional[str] = "en",
    min_confidence: float = 0.5,
) -> PipelineQueryResult:
    """Execute SAKYTI Multilingual NLP Pipeline on a query string."""
    return SakytiPipeline.get_instance().process_query(
        text=text,
        target_language=target_language,
        min_confidence=min_confidence,
    )
