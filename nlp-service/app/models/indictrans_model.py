"""
IndicTrans2 translation model lifecycle manager and inference engine.
Implements decoupled model loading, device selection, input validation,
and bidirectional Indic <-> English translation using AI4Bharat IndicTrans2.
"""

import time
from typing import Any, Dict, List, Optional, Tuple
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from app.config import Settings, get_settings
from app.utils.indic_processor import IndicProcessor
from app.utils.logger import get_logger

logger = get_logger("indictrans_model")


class TranslationError(Exception):
    """Base exception for translation errors."""
    pass


class TranslationModelNotLoadedError(TranslationError):
    """Raised when translation is requested before models are loaded."""
    pass


class UnsupportedLanguageError(TranslationError):
    """Raised when an unsupported language tag or code is provided."""
    pass


class TranslationInputTooLongError(TranslationError):
    """Raised when input text exceeds maximum allowed character length."""
    pass


# Canonical mapping for initial supported languages: code / alias -> FLORES-200 IndicTrans2 tag
LANGUAGE_TAG_MAP: Dict[str, str] = {
    # English
    "en": "eng_Latn",
    "eng": "eng_Latn",
    "english": "eng_Latn",
    "eng_latn": "eng_Latn",
    "eng_Latn": "eng_Latn",

    # Hindi
    "hi": "hin_Deva",
    "hin": "hin_Deva",
    "hindi": "hin_Deva",
    "hin_deva": "hin_Deva",
    "hin_Deva": "hin_Deva",

    # Tamil
    "ta": "tam_Taml",
    "tam": "tam_Taml",
    "tamil": "tam_Taml",
    "tam_taml": "tam_Taml",
    "tam_Taml": "tam_Taml",

    # Telugu
    "te": "tel_Telu",
    "tel": "tel_Telu",
    "telugu": "tel_Telu",
    "tel_telu": "tel_Telu",
    "tel_Telu": "tel_Telu",

    # Bengali
    "bn": "ben_Beng",
    "ben": "ben_Beng",
    "bengali": "ben_Beng",
    "ben_beng": "ben_Beng",
    "ben_Beng": "ben_Beng",

    # Marathi
    "mr": "mar_Deva",
    "mar": "mar_Deva",
    "marathi": "mar_Deva",
    "mar_deva": "mar_Deva",
    "mar_Deva": "mar_Deva",

    # Gujarati
    "gu": "guj_Gujr",
    "guj": "guj_Gujr",
    "gujarati": "guj_Gujr",
    "guj_gujr": "guj_Gujr",
    "guj_Gujr": "guj_Gujr",

    # Kannada
    "kn": "kan_Knda",
    "kan": "kan_Knda",
    "kannada": "kan_Knda",
    "kan_knda": "kan_Knda",
    "kan_Knda": "kan_Knda",

    # Malayalam
    "ml": "mal_Mlym",
    "mal": "mal_Mlym",
    "malayalam": "mal_Mlym",
    "mal_mlym": "mal_Mlym",
    "mal_Mlym": "mal_Mlym",

    # Punjabi
    "pa": "pan_Guru",
    "pan": "pan_Guru",
    "punjabi": "pan_Guru",
    "pan_guru": "pan_Guru",
    "pan_Guru": "pan_Guru",

    # Odia
    "or": "ory_Orya",
    "ory": "ory_Orya",
    "odia": "ory_Orya",
    "oriya": "ory_Orya",
    "ory_orya": "ory_Orya",
    "ory_Orya": "ory_Orya",

    # Sanskrit (Ayurveda knowledge language)
    "sa": "san_Deva",
    "san": "san_Deva",
    "sanskrit": "san_Deva",
    "san_deva": "san_Deva",
    "san_Deva": "san_Deva",
}

INDIC_LANGUAGES = {
    "hin_Deva",
    "tam_Taml",
    "tel_Telu",
    "ben_Beng",
    "mar_Deva",
    "guj_Gujr",
    "kan_Knda",
    "mal_Mlym",
    "pan_Guru",
    "ory_Orya",
    "san_Deva",
}


def normalize_language_code(code: str) -> str:
    """
    Resolve any ISO 639-1, ISO 639-3, or alias code into the canonical IndicTrans2 tag.
    Raises UnsupportedLanguageError if the language is not recognized.
    """
    if not code or not isinstance(code, str):
        raise UnsupportedLanguageError(f"Invalid language code: '{code}'. Expected non-empty string.")

    cleaned = code.strip().lower()
    if cleaned in LANGUAGE_TAG_MAP:
        return LANGUAGE_TAG_MAP[cleaned]

    # Check exact case match
    if code in LANGUAGE_TAG_MAP:
        return LANGUAGE_TAG_MAP[code]

    raise UnsupportedLanguageError(
        f"Language '{code}' is not supported. Supported languages: Hindi (hi), Tamil (ta), "
        f"Telugu (te), Bengali (bn), Marathi (mr), Gujarati (gu), Kannada (kn), Malayalam (ml), "
        f"Punjabi (pa), Odia (or), Sanskrit (sa), and English (en)."
    )


class IndicTransModelManager:
    """
    Singleton manager for loading, maintaining, and running IndicTrans2 models.
    Supports both Indic -> English and English -> Indic translation directions.
    """

    _instance: Optional["IndicTransModelManager"] = None

    def __init__(self) -> None:
        self.device: torch.device = torch.device("cpu")
        self.processor: Optional[IndicProcessor] = None

        self.indic_en_model: Optional[Any] = None
        self.indic_en_tokenizer: Optional[Any] = None
        self.indic_en_model_name: str = ""

        self.en_indic_model: Optional[Any] = None
        self.en_indic_tokenizer: Optional[Any] = None
        self.en_indic_model_name: str = ""

        self.is_loaded: bool = False
        self._load_error: Optional[str] = None
        self._settings: Optional[Settings] = None

    @classmethod
    def get_instance(cls) -> "IndicTransModelManager":
        """Get or create singleton manager instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def initialize(self, settings: Optional[Settings] = None) -> None:
        """
        Load translation models and tokenizers into memory.
        Executed once during application startup.
        """
        if self.is_loaded:
            logger.info("IndicTrans2 models are already initialized and loaded.")
            return

        self._settings = settings or get_settings()

        # Hardware device selection (CUDA if available, else CPU fallback)
        if torch.cuda.is_available() and self._settings.device != "cpu":
            self.device = torch.device("cuda")
            logger.info("Using CUDA acceleration for translation", extra={"device": str(self.device)})
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU for translation inference", extra={"device": str(self.device)})

        # Initialize IndicProcessor
        self.processor = IndicProcessor(inference=True)

        self.indic_en_model_name = self._settings.indictrans2_indic_en_model
        self.en_indic_model_name = self._settings.indictrans2_en_indic_model

        try:
            # 1. Load Indic -> English Tokenizer & Model
            logger.info(
                "Loading Indic -> English translation model",
                extra={
                    "model": self.indic_en_model_name,
                    "device": str(self.device),
                },
            )
            start_time = time.perf_counter()
            self.indic_en_tokenizer = AutoTokenizer.from_pretrained(
                self.indic_en_model_name,
                trust_remote_code=True,
            )
            self.indic_en_model = AutoModelForSeq2SeqLM.from_pretrained(
                self.indic_en_model_name,
                trust_remote_code=True,
            )
            self.indic_en_model.to(self.device)
            self.indic_en_model.eval()
            elapsed_indic_en = (time.perf_counter() - start_time) * 1000
            logger.info(
                "Loaded Indic -> English translation model successfully",
                extra={
                    "model": self.indic_en_model_name,
                    "elapsed_ms": round(elapsed_indic_en, 2),
                },
            )

            # 2. Load English -> Indic Tokenizer & Model
            logger.info(
                "Loading English -> Indic translation model",
                extra={
                    "model": self.en_indic_model_name,
                    "device": str(self.device),
                },
            )
            start_time = time.perf_counter()
            self.en_indic_tokenizer = AutoTokenizer.from_pretrained(
                self.en_indic_model_name,
                trust_remote_code=True,
            )
            self.en_indic_model = AutoModelForSeq2SeqLM.from_pretrained(
                self.en_indic_model_name,
                trust_remote_code=True,
            )
            self.en_indic_model.to(self.device)
            self.en_indic_model.eval()
            elapsed_en_indic = (time.perf_counter() - start_time) * 1000
            logger.info(
                "Loaded English -> Indic translation model successfully",
                extra={
                    "model": self.en_indic_model_name,
                    "elapsed_ms": round(elapsed_en_indic, 2),
                },
            )

            self.is_loaded = True
            self._load_error = None

        except Exception as e:
            self.is_loaded = False
            self._load_error = str(e)
            logger.error(
                f"Failed to initialize IndicTrans2 models: {e}",
                extra={
                    "indic_en_model": self.indic_en_model_name,
                    "en_indic_model": self.en_indic_model_name,
                },
                exc_info=True,
            )
            raise TranslationModelNotLoadedError(
                f"Failed to load IndicTrans2 models: {e}"
            ) from e

    def get_model_info(self) -> Dict[str, Any]:
        """Return diagnostic and version information about translation models."""
        return {
            "loaded": self.is_loaded,
            "device": str(self.device),
            "indic_en_model": self.indic_en_model_name,
            "en_indic_model": self.en_indic_model_name,
            "supported_indic_languages": sorted(list(INDIC_LANGUAGES)),
            "load_error": self._load_error,
        }

    def _select_model_and_tokenizer(
        self, src_lang: str, tgt_lang: str
    ) -> Tuple[Any, Any, str]:
        """
        Select appropriate model (indic_en vs en_indic) based on language direction.
        Never falls back to another model.
        """
        if not self.is_loaded:
            raise TranslationModelNotLoadedError(
                "Translation models are not loaded. Ensure service startup initialized models."
            )

        if src_lang in INDIC_LANGUAGES and tgt_lang == "eng_Latn":
            return self.indic_en_model, self.indic_en_tokenizer, self.indic_en_model_name
        elif src_lang == "eng_Latn" and tgt_lang in INDIC_LANGUAGES:
            return self.en_indic_model, self.en_indic_tokenizer, self.en_indic_model_name
        elif src_lang == tgt_lang:
            raise UnsupportedLanguageError(
                f"Source and target languages are identical: '{src_lang}'. Direct copy or distinct languages expected."
            )
        else:
            raise UnsupportedLanguageError(
                f"Translation from '{src_lang}' to '{tgt_lang}' is not supported in this phase. "
                f"Only Indic -> English and English -> Indic translations are supported."
            )

    def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
        max_length: Optional[int] = None,
        num_beams: Optional[int] = None,
    ) -> Tuple[str, str, str, float]:
        """
        Translate text from source_language to target_language.

        Returns:
            Tuple of (translated_text, model_name, model_version, execution_time_ms)
        """
        if not text or not text.strip():
            raise ValueError("Translation input text cannot be empty or whitespace.")

        cleaned_text = text.strip()

        # Input character protection
        if len(cleaned_text) > 5000:
            raise TranslationInputTooLongError(
                f"Input text length ({len(cleaned_text)} characters) exceeds maximum allowed limit of 5000 characters."
            )

        # Normalize and validate language tags
        src_tag = normalize_language_code(source_language)
        tgt_tag = normalize_language_code(target_language)

        # Select model and tokenizer
        model, tokenizer, model_id = self._select_model_and_tokenizer(src_tag, tgt_tag)

        settings = self._settings or get_settings()
        max_in_len = max_length or settings.indictrans2_max_input_length
        max_out_len = settings.indictrans2_max_output_length
        beams = num_beams or settings.indictrans2_num_beams

        start_time = time.perf_counter()

        try:
            # 1. Preprocess using IndicProcessor
            preprocessed_batch = self.processor.preprocess_batch(
                [cleaned_text],
                src_lang=src_tag,
                tgt_lang=tgt_tag,
            )

            # 2. Tokenize
            inputs = tokenizer(
                preprocessed_batch,
                padding="longest",
                truncation=True,
                max_length=max_in_len,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 3. Model Generation
            with torch.no_grad():
                generated_tokens = model.generate(
                    **inputs,
                    max_new_tokens=max_out_len,
                    num_beams=beams,
                    num_return_sequences=1,
                    use_cache=False,
                    early_stopping=True,
                )

            # 4. Decode tokens
            decoded = tokenizer.batch_decode(
                generated_tokens.detach().cpu().tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            # 5. Postprocess using IndicProcessor
            postprocessed_batch = self.processor.postprocess_batch(
                decoded,
                lang=tgt_tag,
            )

            translated_text = postprocessed_batch[0].strip()
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            logger.info(
                "Translation completed successfully",
                extra={
                    "src_lang": src_tag,
                    "tgt_lang": tgt_tag,
                    "input_length": len(cleaned_text),
                    "output_length": len(translated_text),
                    "elapsed_ms": round(elapsed_ms, 2),
                    "model": model_id,
                },
            )

            return translated_text, "IndicTrans2", model_id, elapsed_ms

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                f"Translation failed during execution: {e}",
                extra={
                    "src_lang": src_tag,
                    "tgt_lang": tgt_tag,
                    "elapsed_ms": round(elapsed_ms, 2),
                },
                exc_info=True,
            )
            if isinstance(e, TranslationError):
                raise
            raise TranslationError(f"Translation inference error: {e}") from e
