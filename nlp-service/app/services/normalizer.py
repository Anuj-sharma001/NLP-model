"""
Multilingual Text Normalizer for SAKYTI NLP Service.

Provides comprehensive, production-grade text normalization for Indic languages,
Romanized Indic scripts, and English, with strict Ayurvedic terminology preservation.
"""

import html
from itertools import count
import time
from typing import Dict, List, Optional, Set, Tuple
import unicodedata
import regex as re

from app.schemas.normalization import NormalizationRequest, NormalizationResult
from app.utils.logger import get_logger

logger = get_logger("normalizer")

# Non-printing / invisible characters to remove (excluding ZWJ \u200D and ZWNJ \u200C)
INVISIBLE_CHARS_PATTERN = re.compile(
    r"[\uFEFF\u200B\u00AD\u2060\u200E\u200F\u202A-\u202E]"
)

# ASCII control characters to strip (keeping \t and \n)
CONTROL_CHARS_PATTERN = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"
)

# HTML tags regex
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")

# Exotic whitespace characters mapping
EXOTIC_WHITESPACE = {
    "\u00A0": " ",  # No-break space
    "\u202F": " ",  # Narrow no-break space
    "\u2000": " ",  # En quad
    "\u2001": " ",  # Em quad
    "\u2002": " ",  # En space
    "\u2003": " ",  # Em space
    "\u2004": " ",  # Three-per-em space
    "\u2005": " ",  # Four-per-em space
    "\u2006": " ",  # Six-per-em space
    "\u2007": " ",  # Figure space
    "\u2008": " ",  # Punctuation space
    "\u2009": " ",  # Thin space
    "\u200A": " ",  # Hair space
    "\u205F": " ",  # Medium mathematical space
    "\u3000": " ",  # Ideographic space
}
EXOTIC_WHITESPACE_PATTERN = re.compile(
    "[" + "".join(re.escape(k) for k in EXOTIC_WHITESPACE.keys()) + "]"
)

# Smart quotes
SMART_SINGLE_QUOTES = re.compile(r"[\u2018\u2019\u201A\u201B\u2032\u2035]")
SMART_DOUBLE_QUOTES = re.compile(r"[\u201C\u201D\u201E\u201F\u2033\u2036]")

# Typographical dashes
DASHES_PATTERN = re.compile(r"[\u2012\u2013\u2014\u2015]")

# Repeated punctuation
EXCLAMATION_REPEATS = re.compile(r"!{2,}")
QUESTION_REPEATS = re.compile(r"\?{2,}")
INTERROBANG_1 = re.compile(r"!+\?+[!?]*")
INTERROBANG_2 = re.compile(r"\?+!+[!?]*")
PERIOD_REPEATS = re.compile(r"\.{4,}")  # Leave 3 dots (...) intact as valid ellipsis
COMMA_REPEATS = re.compile(r",{2,}")
SEMICOLON_REPEATS = re.compile(r";{2,}")
COLON_REPEATS = re.compile(r":{2,}")

# Whitespace before western punctuation (safe: excludes Devanagari danda \u0964, \u0965)
SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([,;:\.!?])")

# Excessive horizontal spaces
MULTI_SPACE_PATTERN = re.compile(r"[ \t]+")
# Excessive newlines
MULTI_NEWLINE_PATTERN = re.compile(r"\n{3,}")

# Supported Indic languages for IndicNormalizerFactory
SUPPORTED_INDIC_LANGS = {
    "hi", "ta", "te", "bn", "mr", "gu", "kn", "ml", "pa", "or", "sa"
}


class MultilingualNormalizer:
    """
    Multilingual Text Normalization engine for SAKYTI.
    Handles Unicode, whitespace, punctuation, common noise, Indic script normalization,
    and Romanized Indian-language input while safeguarding Ayurvedic terminology.
    """

    _instance: Optional["MultilingualNormalizer"] = None

    def __init__(self) -> None:
        self._indic_normalizers: Dict[str, object] = {}
        self._init_indic_normalizers()

    @classmethod
    def get_instance(cls) -> "MultilingualNormalizer":
        """Get or create singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _init_indic_normalizers(self) -> None:
        """Lazily initialize or preload IndicNormalizerFactory instances."""
        try:
            from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
            factory = IndicNormalizerFactory()
            for lang in SUPPORTED_INDIC_LANGS:
                try:
                    self._indic_normalizers[lang] = factory.get_normalizer(lang)
                except Exception as exc:
                    logger.warning(f"Could not load IndicNormalizer for {lang}: {exc}")
        except ImportError:
            logger.warning("indicnlp library not installed. IndicNormalizer hooks will be disabled.")

    def normalize(
        self,
        text: str,
        language: Optional[str] = None,
        is_romanized: Optional[bool] = None,
        preserve_ayurvedic_terms: bool = True,
        remove_html: bool = True,
    ) -> NormalizationResult:
        """
        Normalize input text through sequential normalization stages.

        Returns:
            NormalizationResult containing original text, normalized text,
            and a list of all detected modification categories.
        """
        start_time = time.perf_counter()
        original_text = text
        changes: List[str] = []

        if not text:
            return NormalizationResult(
                original_text=original_text,
                normalized_text="",
                detected_changes=[],
                language=language,
                is_romanized=is_romanized,
                execution_time_ms=0.0,
            )

        current = text

        # 1. HTML entities & tags
        if remove_html:
            tag_stripped = HTML_TAG_PATTERN.sub(" ", current)
            if tag_stripped != current:
                changes.append("html_tags_removed")
                current = tag_stripped

            unescaped = html.unescape(current)
            if unescaped != current:
                changes.append("html_entities_unescaped")
                current = unescaped

        # 2. Control characters removal (excluding \t, \n)
        no_control = CONTROL_CHARS_PATTERN.sub("", current)
        if no_control != current:
            changes.append("control_characters_removed")
            current = no_control

        # 3. Unicode normalization: NFC (Canonical Decomposition + Canonical Composition)
        # NFC is critical for IAST diacritics (e.g. ā, ī, ū, ṛ, ś, ṣ) and Indic Unicode
        nfc_text = unicodedata.normalize("NFC", current)
        if nfc_text != current:
            changes.append("unicode_nfc_normalized")
            current = nfc_text

        # 4. Invisible / non-printing zero-width sanitization
        # Preserves ZWJ (\u200D) and ZWNJ (\u200C) for Indic ligatures and conjuncts!
        sanitized_invisibles = INVISIBLE_CHARS_PATTERN.sub("", current)
        if sanitized_invisibles != current:
            changes.append("invisible_characters_sanitized")
            current = sanitized_invisibles

        # 5. Shield Ayurvedic terms if requested
        term_placeholders: Dict[str, str] = {}
        if preserve_ayurvedic_terms:
            current, term_placeholders = self._shield_ayurvedic_terms(current)

        # 6. Language-aware Indic normalization hook (Devanagari, Tamil, etc.)
        norm_lang = (language or "").lower().strip()
        if norm_lang in self._indic_normalizers and not is_romanized:
            normalizer = self._indic_normalizers[norm_lang]
            try:
                indic_normalized = normalizer.normalize(current)
                if indic_normalized != current:
                    changes.append(f"indic_script_normalized_{norm_lang}")
                    current = indic_normalized
            except Exception as exc:
                logger.error(f"Error during indic script normalization for {norm_lang}: {exc}")

        # 7. Romanized Indian-language input hooks (e.g., phonetic elongation)
        should_apply_romanized = is_romanized if is_romanized is not None else bool(
            re.search(r"(?i)([a-zA-Z])\1{2,}", current)
        )

        if should_apply_romanized:
            elong_reduced = self._reduce_romanized_elongation(current)
            if elong_reduced != current:
                changes.append("romanized_elongation_reduced")
                current = elong_reduced

        # 8. Punctuation normalization
        punct_cleaned = self._normalize_punctuation(current, changes)
        current = punct_cleaned

        # 9. Restore shielded Ayurvedic terms
        if term_placeholders:
            current = self._restore_ayurvedic_terms(current, term_placeholders)
            changes.append("ayurvedic_terms_preserved")

        # 10. Whitespace normalization
        ws_cleaned = self._normalize_whitespace(current)
        if ws_cleaned != current:
            changes.append("whitespace_normalized")
            current = ws_cleaned

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return NormalizationResult(
            original_text=original_text,
            normalized_text=current,
            detected_changes=changes,
            language=language,
            is_romanized=is_romanized,
            execution_time_ms=round(elapsed_ms, 2),
        )

    def _shield_ayurvedic_terms(self, text: str) -> Tuple[str, Dict[str, str]]:
        """
        Identify known Ayurvedic medical terms using TerminologyService
        and shield them with unique placeholders to prevent distortion.
        """
        try:
            from app.services.terminology_service import TerminologyService
            service = TerminologyService.get_instance()
            matches = service.identify_terms(text)
            if not matches:
                return text, {}

            placeholders: Dict[str, str] = {}
            # Sort matches by start position, longer matches first on ties
            sorted_matches = sorted(matches, key=lambda m: (m.start, -(m.end - m.start)))

            # Deduplicate overlapping matches
            non_overlapping = []
            last_end = -1
            for m in sorted_matches:
                if m.start >= last_end:
                    non_overlapping.append(m)
                    last_end = m.end

            # Replace from right to left using indices
            text_chars = list(text)
            for idx, match in enumerate(reversed(non_overlapping)):
                tag = f"§§SAKYTI_AYUR_TERM_{idx}§§"
                matched_str = text[match.start:match.end]
                placeholders[tag] = matched_str
                text_chars[match.start:match.end] = list(tag)

            return "".join(text_chars), placeholders
        except Exception as exc:
            logger.warning(f"Failed to shield Ayurvedic terms during normalization: {exc}")
            return text, {}

    def _restore_ayurvedic_terms(self, text: str, placeholders: Dict[str, str]) -> str:
        """Restore shielded Ayurvedic terms from placeholders."""
        for tag, original_val in placeholders.items():
            text = text.replace(tag, original_val)
        return text

    def _reduce_romanized_elongation(self, text: str) -> str:
        """
        Reduce phonetic elongation in Romanized Indic text.
        e.g., 'bahuuuut' -> 'bahut', 'kaaaaafi' -> 'kaafi', 'achhaaaaa' -> 'achha'.
        - Vowels (a, e, o): 3 or more -> 2 (preserving standard Hinglish 'aa', 'ee', 'oo')
        - Other vowels (i, u): 3 or more -> 1
        - Consonants: 3 or more -> 1 (e.g. 'plzzzzz' -> 'plz', 'sachhhhh' -> 'sach')
        """
        # Reduce 3+ consecutive 'a', 'e', 'o' to 2 (e.g., 'kaaaaafi' -> 'kaafi', 'heeeelo' -> 'heelo')
        text = re.sub(r"(?i)([aeo])\1{2,}", r"\1\1", text)
        # Reduce 3+ consecutive 'i', 'u' to 1 (e.g., 'bahuuuut' -> 'bahut', 'siiiiir' -> 'sir')
        text = re.sub(r"(?i)([iu])\1{2,}", r"\1", text)
        # Reduce 3+ identical consonants to 1 (e.g., 'plzzzzz' -> 'plz', 'achhaaaaa' -> 'achha')
        text = re.sub(r"(?i)([b-df-hj-np-tv-z])\1{2,}", r"\1", text)
        return text

    def _normalize_punctuation(self, text: str, changes: List[str]) -> str:
        """
        Normalize smart quotes, typographical dashes, repeated punctuation,
        and punctuation spacing, while strictly preserving Devanagari danda (। and ॥).
        """
        initial = text

        # 1. Smart quotes
        text = SMART_SINGLE_QUOTES.sub("'", text)
        text = SMART_DOUBLE_QUOTES.sub('"', text)
        if text != initial:
            changes.append("quotes_normalized")
            initial = text

        # 2. Typographical dashes (em-dash, en-dash) -> hyphen
        text = DASHES_PATTERN.sub("-", text)
        if text != initial:
            changes.append("dashes_normalized")
            initial = text

        # 3. Repeated punctuation
        # Interrobangs: !?!! or ?!?? -> ?!
        text = INTERROBANG_1.sub("?!", text)
        text = INTERROBANG_2.sub("?!", text)
        # Exclamations: !!!! -> !
        text = EXCLAMATION_REPEATS.sub("!", text)
        # Questions: ???? -> ?
        text = QUESTION_REPEATS.sub("?", text)
        # 4+ periods -> ... (preserving standard ellipsis)
        text = PERIOD_REPEATS.sub("...", text)
        # Commas, semicolons, colons
        text = COMMA_REPEATS.sub(",", text)
        text = SEMICOLON_REPEATS.sub(";", text)
        text = COLON_REPEATS.sub(":", text)

        if text != initial:
            changes.append("repeated_punctuation_collapsed")
            initial = text

        # 4. Safe punctuation spacing: remove redundant space before western punctuation
        text = SPACE_BEFORE_PUNCTUATION.sub(r"\1", text)

        # Ensure safe spacing around Devanagari danda (।, ॥)
        # If there are multiple spaces before danda, collapse to one space: "नमस्ते   ।" -> "नमस्ते ।"
        text = re.sub(r"[ \t]+([।॥])", r" \1", text)

        if text != initial:
            changes.append("punctuation_spacing_normalized")

        return text

    def _normalize_whitespace(self, text: str) -> str:
        """
        Normalize exotic whitespaces (NBSP etc.), line breaks,
        consecutive spaces, and trailing whitespace.
        """
        # Map exotic spaces (NBSP, narrow NBSP, quad/em/en spaces)
        text = EXOTIC_WHITESPACE_PATTERN.sub(" ", text)

        # Standardize line endings (\r\n -> \n, \r -> \n)
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Collapse horizontal spaces and tabs to single ASCII space
        text = MULTI_SPACE_PATTERN.sub(" ", text)

        # Remove trailing spaces on each line
        text = re.sub(r" [ \t]*\n", "\n", text)
        text = re.sub(r"\n[ \t]* ", "\n", text)

        # Collapse 3 or more consecutive newlines to 2
        text = MULTI_NEWLINE_PATTERN.sub("\n\n", text)

        # Strip outer whitespace
        text = text.strip()

        return text


# Module-level convenience functions
def normalize_text(
    text: str,
    language: Optional[str] = None,
    is_romanized: Optional[bool] = None,
    preserve_ayurvedic_terms: bool = True,
    remove_html: bool = True,
) -> NormalizationResult:
    """Normalize text using the singleton MultilingualNormalizer instance."""
    return MultilingualNormalizer.get_instance().normalize(
        text=text,
        language=language,
        is_romanized=is_romanized,
        preserve_ayurvedic_terms=preserve_ayurvedic_terms,
        remove_html=remove_html,
    )
