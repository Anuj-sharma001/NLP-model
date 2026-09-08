"""Language Identification Service using AI4Bharat IndicLID.

This service detects languages and scripts for native Indic scripts, Romanized Indic text,
and English, while strictly separating inference logic from model lifecycle management.
"""

import re
from typing import Any, Dict, Optional, Tuple

from app.config import Settings, get_settings
from app.models.indiclid_model import IndicLIDModelManager
from app.schemas.language import LanguageDetectionResult
from app.utils.logger import get_logger

logger = get_logger("sakyti-nlp.services.language_detector")

# Comprehensive mapping of IndicLID language codes to canonical language names
LANGUAGE_METADATA: Dict[str, Tuple[str, str]] = {
    "asm": ("Assamese", "as"),
    "ben": ("Bengali", "bn"),
    "brx": ("Bodo", "brx"),
    "doi": ("Dogri", "doi"),
    "eng": ("English", "en"),
    "guj": ("Gujarati", "gu"),
    "hin": ("Hindi", "hi"),
    "kan": ("Kannada", "kn"),
    "kas": ("Kashmiri", "ks"),
    "kok": ("Konkani", "kok"),
    "mai": ("Maithili", "mai"),
    "mal": ("Malayalam", "ml"),
    "mar": ("Marathi", "mr"),
    "mni": ("Manipuri", "mni"),
    "nep": ("Nepali", "ne"),
    "ori": ("Odia", "or"),
    "pan": ("Punjabi", "pa"),
    "san": ("Sanskrit", "sa"),
    "sat": ("Santali", "sat"),
    "snd": ("Sindhi", "sd"),
    "tam": ("Tamil", "ta"),
    "tel": ("Telugu", "te"),
    "urd": ("Urdu", "ur"),
    "other": ("Other", "und"),
}

# Mapping of script codes to human-readable names
SCRIPT_METADATA: Dict[str, str] = {
    "Deva": "Devanagari",
    "Tamil": "Tamil",
    "Telu": "Telugu",
    "Beng": "Bengali",
    "Gujr": "Gujarati",
    "Guru": "Gurmukhi",
    "Knda": "Kannada",
    "Mlym": "Malayalam",
    "Orya": "Odia",
    "Latn": "Latin",
    "Arab": "Perso-Arabic",
    "Olch": "Ol Chiki",
    "Meti": "Meetei Mayek",
}

# Regex to detect Latin characters
LATIN_CHAR_REGEX = re.compile(r"[a-zA-Z]")
# Regex to detect symbols, whitespace, numbers
NON_LETTER_REGEX = re.compile(r"[\s\d@_!#$%^&*()<>?/\\|}{~:;,\.\[\]\"\'\-+=`]")


def calculate_latin_ratio(text: str) -> float:
    """Calculate the ratio of Latin letters to total alphabetic characters.

    Replicates AI4Bharat IndicLID character-level script routing logic.
    """
    total_letters = len(re.findall(r"\w", text, flags=re.UNICODE))
    if total_letters == 0:
        return 0.0

    latin_chars = len(LATIN_CHAR_REGEX.findall(text))
    return latin_chars / total_letters


def parse_indiclid_label(raw_label: str) -> Tuple[str, str]:
    """Parse IndicLID model output label (e.g. '__label__hin_Deva' -> ('hin', 'Deva'))."""
    clean_label = raw_label.replace("__label__", "")
    if "_" in clean_label:
        parts = clean_label.split("_", 1)
        return parts[0], parts[1]
    return clean_label, "Unknown"


def detect_language(
    text: str,
    confidence_threshold: Optional[float] = None,
    settings: Optional[Settings] = None,
) -> LanguageDetectionResult:
    """Identify the language and script of the provided text using AI4Bharat IndicLID.

    Args:
        text: The input text to classify.
        confidence_threshold: Minimum confidence score required to accept the prediction.
            If None, uses `settings.language_detection_confidence_threshold`.
        settings: Application settings instance (defaults to cached singleton).

    Returns:
        LanguageDetectionResult containing language code, name, script, confidence,
        romanization flag, and raw model metadata.
    """
    if settings is None:
        settings = get_settings()

    threshold = (
        confidence_threshold
        if confidence_threshold is not None
        else settings.language_detection_confidence_threshold
    )

    # 1. Input Validation: Safe handling of None, non-string, empty, or whitespace-only inputs
    if not isinstance(text, str):
        logger.warning(f"Invalid input type: {type(text)}")
        return LanguageDetectionResult(
            language_code="unknown",
            language_name="Unknown",
            script="Unknown",
            confidence=0.0,
            is_romanized=False,
            raw_model_result={"error": "invalid_input_type", "type": str(type(text))},
        )

    cleaned_text = text.strip()
    if not cleaned_text:
        return LanguageDetectionResult(
            language_code="unknown",
            language_name="Unknown",
            script="Unknown",
            confidence=0.0,
            is_romanized=False,
            raw_model_result={"error": "empty_input"},
        )

    # 2. Check for minimal linguistic content
    alphabetic_chars = re.findall(r"[^\W\d_]", cleaned_text, flags=re.UNICODE)
    if len(alphabetic_chars) == 0:
        return LanguageDetectionResult(
            language_code="unknown",
            language_name="Unknown",
            script="Unknown",
            confidence=0.0,
            is_romanized=False,
            raw_model_result={"error": "no_alphabetic_characters", "text": cleaned_text},
        )

    # Very short single-character noise check
    if len(alphabetic_chars) == 1 and len(cleaned_text) < 2:
        return LanguageDetectionResult(
            language_code="unknown",
            language_name="Unknown",
            script="Unknown",
            confidence=0.0,
            is_romanized=False,
            raw_model_result={"error": "insufficient_text_length", "text": cleaned_text},
        )

    # 3. Model Inference via IndicLIDModelManager (loaded once at startup)
    model_manager = IndicLIDModelManager.get_instance()
    if not model_manager.is_loaded:
        logger.warning("IndicLIDModelManager not pre-loaded. Initializing now...")
        model_manager.load_models(settings=settings)

    # Compute Latin script ratio to route between IndicLID-FTR and IndicLID-FTN
    latin_ratio = calculate_latin_ratio(cleaned_text)
    is_primarily_latin = latin_ratio > settings.language_detection_roman_threshold

    # Clean newlines for FastText prediction
    single_line_text = cleaned_text.replace("\n", " ").strip()

    try:
        if is_primarily_latin:
            model_name = "IndicLID-FTR"
            labels, probs = model_manager.predict_roman(single_line_text, k=3)
        else:
            model_name = "IndicLID-FTN"
            labels, probs = model_manager.predict_native(single_line_text, k=3)

        raw_label = labels[0]
        raw_prob = min(max(probs[0], 0.0), 1.0)
    except Exception as exc:
        logger.error(f"Inference failure for text '{cleaned_text[:50]}...': {exc}", exc_info=True)
        return LanguageDetectionResult(
            language_code="unknown",
            language_name="Unknown",
            script="Unknown",
            confidence=0.0,
            is_romanized=False,
            raw_model_result={"error": f"inference_exception: {exc}"},
        )

    # 4. Parse IndicLID Tag
    lang_tag, script_tag = parse_indiclid_label(raw_label)

    # Resolve Human-readable Metadata
    lang_info = LANGUAGE_METADATA.get(lang_tag, (lang_tag.capitalize(), lang_tag))
    lang_name = lang_info[0]
    script_name = SCRIPT_METADATA.get(script_tag, script_tag)

    # Check Romanization: Indic language in Latin script
    is_romanized = (script_tag == "Latn") and (lang_tag != "eng") and (lang_tag != "other")

    raw_model_result: Dict[str, Any] = {
        "model_used": model_name,
        "raw_label": raw_label,
        "model_confidence": round(raw_prob, 4),
        "latin_ratio": round(latin_ratio, 3),
        "top_predictions": [
            {"label": lbl, "confidence": round(pr, 4)}
            for lbl, pr in zip(labels, probs)
        ],
    }

    # 5. Apply Configurable Confidence Threshold
    if raw_prob < threshold:
        logger.info(
            f"Prediction below threshold ({raw_prob:.3f} < {threshold:.3f}) for: '{cleaned_text[:30]}'",
            extra={"raw_label": raw_label, "confidence": raw_prob},
        )
        return LanguageDetectionResult(
            language_code="unknown",
            language_name="Unknown",
            script=script_name if script_name != "Unknown" else "Unknown",
            confidence=round(raw_prob, 4),
            is_romanized=is_romanized,
            raw_model_result={
                **raw_model_result,
                "threshold_applied": threshold,
                "below_threshold": True,
                "candidate_language_code": lang_tag,
                "candidate_language_name": lang_name,
            },
        )

    return LanguageDetectionResult(
        language_code=lang_tag,
        language_name=lang_name,
        script=script_name,
        confidence=round(raw_prob, 4),
        is_romanized=is_romanized,
        raw_model_result=raw_model_result,
    )
