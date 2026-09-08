"""
Romanized Indian Language Processing Layer for SAKYTI Multilingual NLP Service.

Handles:
1. Detecting whether input is Romanized Indic text vs English using AI4Bharat IndicLID.
2. Shielding and preserving Ayurvedic technical terminology with canonical native forms.
3. Rejecting blind English assumptions for Latin script.
4. Confidence-gated conversion: script conversion only occurs when confidence is sufficient.
5. Preserving exact original text for auditing and debugging.
6. Extensible architecture allowing easy addition of future Indian language converters.
"""

from abc import ABC, abstractmethod
import time
from typing import Dict, List, Optional, Set, Tuple
import regex as re

from app.schemas.romanized import RomanizedProcessRequest, RomanizedProcessResult
from app.services.language_detector import detect_language
from app.utils.logger import get_logger

logger = get_logger("romanized_processor")


# =========================================================================
# 1. Base Converter Interface & Extensible Registry
# =========================================================================

class BaseRomanizedConverter(ABC):
    """Abstract base class for language-specific Romanized script converters."""

    @property
    @abstractmethod
    def language_code(self) -> str:
        """ISO/SAKYTI language code this converter handles (e.g. 'hi', 'ta', 'te')."""
        ...

    @property
    @abstractmethod
    def target_script_name(self) -> str:
        """Human-readable target script name (e.g. 'devanagari', 'tamil')."""
        ...

    @abstractmethod
    def convert(self, text: str, shielded_placeholders: Dict[str, str]) -> str:
        """
        Convert Romanized text to target native script, respecting shielded placeholders.
        """
        ...


# =========================================================================
# 2. Hindi Romanized Converter (Hinglish -> Devanagari)
# =========================================================================

class HindiRomanizedConverter(BaseRomanizedConverter):
    """
    High-precision Romanized Hindi (Hinglish) to Devanagari converter.
    Combines an extensive colloquial dictionary for conversational/clinical words
    with a rule-based phonetic transliteration engine for arbitrary words.
    """

    @property
    def language_code(self) -> str:
        return "hi"

    @property
    def target_script_name(self) -> str:
        return "devanagari"

    def __init__(self) -> None:
        self._dictionary: Dict[str, str] = self._build_hinglish_dictionary()

    def _build_hinglish_dictionary(self) -> Dict[str, str]:
        """Build dictionary of high-frequency conversational and clinical Hinglish words."""
        return {
            # Pronouns & Demonstratives
            "mujhe": "मुझे",
            "mujhko": "मुझको",
            "mera": "मेरा",
            "meri": "मेरी",
            "mere": "मेरे",
            "mai": "मैं",
            "main": "मैं",
            "hum": "हम",
            "hume": "हमें",
            "hame": "हमें",
            "hamara": "हमारा",
            "hamare": "हमारे",
            "hamari": "हमारी",
            "tum": "तुम",
            "tumhe": "तुम्हें",
            "tumhara": "तुम्हारा",
            "aap": "आप",
            "aapka": "आपका",
            "aapke": "आपके",
            "aapki": "आपकी",
            "aapko": "आपको",
            "yeh": "यह",
            "ye": "ये",
            "voh": "वह",
            "woh": "वह",
            "vo": "वो",
            "wo": "वो",
            "uska": "उसका",
            "uske": "उसके",
            "uski": "उसकी",
            "use": "उसे",
            "unhe": "उन्हें",
            "unka": "उनका",
            "unke": "उनके",
            "unki": "उनकी",
            "is": "इस",
            "us": "उस",
            "in": "इन",
            "un": "उन",
            # Postpositions & Conjunctions
            "me": "में",
            "mein": "में",
            "se": "से",
            "ko": "को",
            "ka": "का",
            "ke": "के",
            "ki": "की",
            "par": "पर",
            "pe": "पे",
            "tak": "तक",
            "aur": "और",
            "ya": "या",
            "lekin": "लेकिन",
            "magar": "मगर",
            "parantu": "परन्तु",
            "bhi": "भी",
            "hi": "ही",
            "to": "तो",
            "toh": "तो",
            "karan": "कारण",
            "liye": "लिए",
            "bina": "बिना",
            "saath": "साथ",
            # Verbs & Auxiliaries
            "hai": "है",
            "hain": "हैं",
            "ho": "हो",
            "hoon": "हूँ",
            "hun": "हूँ",
            "tha": "था",
            "thi": "थी",
            "the": "थे",
            "raha": "रहा",
            "rahi": "रही",
            "rahe": "रहे",
            "kar": "कर",
            "karo": "करो",
            "kare": "करे",
            "karein": "करें",
            "karta": "करता",
            "karti": "करती",
            "karte": "करते",
            "hona": "होना",
            "hota": "होता",
            "hoti": "होती",
            "hote": "होते",
            "lena": "लेना",
            "lene": "लेने",
            "lo": "लो",
            "de": "दे",
            "do": "दो",
            "dena": "देना",
            "chahiye": "चाहिए",
            "sakta": "सकता",
            "sakti": "सकती",
            "sakte": "सकते",
            "lagta": "लगता",
            "lagti": "लगती",
            "lagte": "लगते",
            "aata": "आता",
            "aati": "आती",
            "aate": "आते",
            "gaya": "गया",
            "gayi": "गई",
            "gaye": "गए",
            # Healthcare & Symptoms
            "pet": "पेट",
            "dard": "दर्द",
            "gala": "गला",
            "kharab": "खराब",
            "bukhar": "बुखार",
            "sir": "सिर",
            "sar": "सर",
            "chakkar": "चक्कर",
            "khansi": "खांसी",
            "sardi": "सर्दी",
            "thakan": "थकान",
            "dawai": "दवाई",
            "dawa": "दवा",
            "ilaaj": "इलाज",
            "upchar": "उपचार",
            "aaram": "आराम",
            "shant": "शांत",
            "kam": "कम",
            "jyada": "ज्यादा",
            "zyada": "ज्यादा",
            "bahut": "बहुत",
            "kaafi": "काफ़ी",
            "theek": "ठीक",
            "thik": "ठीक",
            "acha": "अच्छा",
            "achha": "अच्छा",
            "bura": "बुरा",
            "samasya": "समस्या",
            "takleef": "तकलीफ़",
            "taklif": "तकलीफ़",
            "bimari": "बीमारी",
            "rog": "रोग",
            "mariz": "मरीज़",
            "mareez": "मरीज़",
            "doctor": "डॉक्टर",
            # Question & Negation
            "kya": "क्या",
            "kyon": "क्यों",
            "kyu": "क्यों",
            "kyun": "क्यों",
            "kaise": "कैसे",
            "kab": "कब",
            "kaha": "कहाँ",
            "kahan": "कहाँ",
            "kaun": "कौन",
            "kitna": "कितना",
            "kitni": "कितनी",
            "kitne": "कितने",
            "nahi": "नहीं",
            "nahin": "नहीं",
            "na": "न",
            "mat": "मत",
            # Time & Frequency
            "aaj": "आज",
            "kal": "कल",
            "parso": "परसों",
            "subah": "सुबह",
            "dopahar": "दोपहर",
            "shaam": "शाम",
            "sham": "शाम",
            "raat": "रात",
            "roz": "रोज़",
            "hamesha": "हमेशा",
            "kabhi": "कभी",
        }

    def convert(self, text: str, shielded_placeholders: Dict[str, str]) -> str:
        """Convert Romanized Hindi text into Devanagari."""
        tokens = re.split(r"(\s+|[^\w\s§]+)", text)
        converted_tokens = []

        for token in tokens:
            if not token:
                continue

            # Shielded Ayurvedic placeholder
            if "§§SAKYTI_AYUR_TERM_" in token:
                converted_tokens.append(token)
                continue

            # Whitespace or punctuation
            if re.match(r"^\s+$", token) or re.match(r"^[^\w\s]+$", token):
                # Map western question/period to Indic danda if desired, or keep punctuation
                converted_tokens.append(token)
                continue

            # Numeric
            if token.isdigit():
                converted_tokens.append(token)
                continue

            # Check dictionary
            lower_token = token.lower()
            if lower_token in self._dictionary:
                converted_tokens.append(self._dictionary[lower_token])
                continue

            # Phonetic transliteration for words outside dictionary
            devanagari_word = self._transliterate_phonetic(lower_token)
            converted_tokens.append(devanagari_word)

        result = "".join(converted_tokens)
        return result

    def _transliterate_phonetic(self, word: str) -> str:
        """
        Deterministic phonetic transliteration of a single Romanized Hindi word into Devanagari.
        """
        # Ordered multi-character and single-character consonant mappings
        consonants = [
            ("chh", "छ"), ("ksh", "क्ष"), ("gya", "ज्ञ"), ("tra", "त्र"),
            ("kh", "ख"), ("gh", "घ"), ("ch", "च"), ("jh", "झ"),
            ("th", "थ"), ("dh", "ध"), ("ph", "फ"), ("bh", "भ"),
            ("sh", "श"), ("zh", "ज़"), ("tr", "त्र"), ("gy", "ज्ञ"),
            ("k", "क"), ("g", "ग"), ("c", "क"), ("j", "ज"),
            ("t", "त"), ("d", "द"), ("n", "न"), ("p", "प"),
            ("f", "फ़"), ("b", "ब"), ("m", "म"), ("y", "य"),
            ("r", "र"), ("l", "ल"), ("v", "व"), ("w", "व"),
            ("s", "स"), ("h", "ह"), ("z", "ज़"), ("q", "क़"),
        ]

        # Dependent vowel matras
        vowel_matras = [
            ("aa", "ा"), ("ee", "ी"), ("oo", "ू"), ("ai", "ै"),
            ("au", "ौ"), ("a", ""), ("i", "ि"), ("u", "ु"),
            ("e", "े"), ("o", "ो"),
        ]

        # Independent vowels
        indep_vowels = [
            ("aa", "आ"), ("ee", "ई"), ("oo", "ऊ"), ("ai", "ऐ"),
            ("au", "औ"), ("a", "अ"), ("i", "इ"), ("u", "उ"),
            ("e", "ए"), ("o", "ओ"),
        ]

        out = []
        i = 0
        n = len(word)
        last_was_consonant = False

        while i < n:
            # Check nasal 'n' before consonant or at end of syllable
            if word[i] == "n" and i + 1 < n and word[i + 1] in "kgcjtdpbzs":
                out.append("ं")
                i += 1
                last_was_consonant = False
                continue

            # If previous was consonant, look for vowel matra
            if last_was_consonant:
                matched_matra = False
                for v_str, matra in vowel_matras:
                    if word.startswith(v_str, i):
                        out.append(matra)
                        i += len(v_str)
                        matched_matra = True
                        last_was_consonant = False
                        break
                if matched_matra:
                    continue

                # If no vowel follows a consonant, and another consonant follows -> add halant
                matched_next_c = False
                for c_str, _ in consonants:
                    if word.startswith(c_str, i):
                        out.append("्")
                        matched_next_c = True
                        break
                if matched_next_c:
                    last_was_consonant = False
                    continue

            # Look for consonant
            matched_c = False
            for c_str, dev in consonants:
                if word.startswith(c_str, i):
                    out.append(dev)
                    i += len(c_str)
                    matched_c = True
                    last_was_consonant = True
                    break
            if matched_c:
                continue

            # Look for independent vowel
            matched_iv = False
            for v_str, dev in indep_vowels:
                if word.startswith(v_str, i):
                    out.append(dev)
                    i += len(v_str)
                    matched_iv = True
                    last_was_consonant = False
                    break
            if matched_iv:
                continue

            # Fallback for unmapped character
            out.append(word[i])
            i += 1
            last_was_consonant = False

        return "".join(out)


# =========================================================================
# 3. Romanized Processing Engine & Singleton Service
# =========================================================================

class RomanizedProcessor:
    """
    Processing engine for Romanized Indian languages in SAKYTI.
    Orchestrates language detection, English discrimination, Ayurvedic term
    shielding/canonicalization, confidence gating, and native script conversion.
    """

    _instance: Optional["RomanizedProcessor"] = None

    def __init__(self) -> None:
        self._converters: Dict[str, BaseRomanizedConverter] = {}
        # Register initial Hindi converter
        self.register_converter(HindiRomanizedConverter())

    @classmethod
    def get_instance(cls) -> "RomanizedProcessor":
        """Get or create singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register_converter(self, converter: BaseRomanizedConverter) -> None:
        """
        Register a converter for a given Indian language.
        Enables extensibility so future Indian languages can be plugged in seamlessly.
        """
        code = converter.language_code.lower().strip()
        self._converters[code] = converter
        logger.info(f"Registered Romanized converter for language '{code}' -> {converter.target_script_name}")

    def get_supported_conversion_languages(self) -> List[str]:
        """Return list of language codes that support Romanized script conversion."""
        return list(self._converters.keys())

    def process(
        self,
        text: str,
        target_language: Optional[str] = None,
        min_conversion_confidence: float = 0.5,
        preserve_ayurvedic_terms: bool = True,
        convert_script: bool = True,
    ) -> RomanizedProcessResult:
        """
        Process potentially Romanized Indian language text.

        1. Detect language and script via IndicLID.
        2. Verify if input is Romanized Indic vs English.
        3. Shield and preserve Ayurvedic clinical terminology.
        4. Gate conversion based on confidence threshold.
        5. Convert to native script if eligible.
        6. Return complete audit metadata with unmodified original text.
        """
        start_time = time.perf_counter()
        original_text = text

        # 1. Handle empty / whitespace input
        if not text or not text.strip():
            return RomanizedProcessResult(
                original_text=original_text,
                detected_language="unknown",
                detected_language_name="Unknown",
                script="Unknown",
                is_romanized=False,
                confidence=0.0,
                was_converted=False,
                converted_text=None,
                ayurvedic_terms_detected=[],
                processing_status="empty_input",
                execution_time_ms=0.0,
            )

        # 2. Language & Script Identification via IndicLID
        detection = detect_language(text)
        det_lang = detection.language_code.lower().strip()
        det_script = detection.script
        det_confidence = detection.confidence
        det_name = detection.language_name

        # Map language codes (e.g. IndicLID 'hin' -> 'hi', 'eng' -> 'en', 'pan' -> 'pa')
        norm_lang_map = {
            "hin": "hi", "eng": "en", "tam": "ta", "tel": "te",
            "ben": "bn", "mar": "mr", "guj": "gu", "kan": "kn",
            "mal": "ml", "pan": "pa", "ori": "or", "san": "sa",
        }
        active_lang = norm_lang_map.get(det_lang, det_lang)

        # 3. Check for English (Requirement 3: Do not assume every Latin-script sentence is English)
        is_english = active_lang in ["en", "eng"] or (det_script == "Latin" and det_lang == "eng")
        is_romanized = detection.is_romanized and not is_english

        # 4. Ayurvedic Terminology Identification & Shielding (Requirement 2)
        shielded_text = text
        term_map: Dict[str, str] = {}
        ayurvedic_terms_found: List[str] = []

        if preserve_ayurvedic_terms:
            shielded_text, term_map, ayurvedic_terms_found = self._shield_ayurvedic_terms(
                text=text,
                target_lang=target_language or active_lang,
            )

        # 5. Conversion Decision Gating (Requirement 4: Do not convert unless confidence is sufficient)
        was_converted = False
        converted_text: Optional[str] = None
        processing_status = "unconverted"

        if is_english:
            processing_status = "bypassed_english"
        elif not is_romanized and det_script != "Latin":
            processing_status = "not_romanized_indic"
        elif not convert_script:
            processing_status = "conversion_disabled_by_request"
        elif det_confidence < min_conversion_confidence:
            processing_status = f"insufficient_confidence ({det_confidence:.2f} < {min_conversion_confidence:.2f})"
        else:
            # Determine conversion language
            effective_lang = (target_language or active_lang).lower().strip()
            effective_lang = norm_lang_map.get(effective_lang, effective_lang)

            # Check if language is directly supported or compatible with Devanagari converter
            converter = None
            if effective_lang in self._converters:
                converter = self._converters[effective_lang]
            elif effective_lang in ["mai", "bho", "mag", "ne", "nep", "kok"] and "hi" in self._converters:
                # Devanagari script-compatible North Indic languages fallback to Devanagari converter
                converter = self._converters["hi"]

            if converter is not None:
                try:
                    # Convert Romanized text using registered language converter
                    converted_candidate = converter.convert(
                        text=shielded_text,
                        shielded_placeholders=term_map,
                    )

                    # Restore shielded Ayurvedic terms with canonical native script forms
                    if term_map:
                        for placeholder, canonical_replacement in term_map.items():
                            converted_candidate = converted_candidate.replace(placeholder, canonical_replacement)

                    converted_text = converted_candidate
                    was_converted = True
                    processing_status = f"converted_to_{converter.target_script_name}"
                except Exception as exc:
                    logger.error(f"Error converting Romanized text for language '{effective_lang}': {exc}", exc_info=True)
                    processing_status = f"conversion_error: {exc}"
            else:
                processing_status = f"unsupported_converter_for_{effective_lang}"

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Requirement 5: Keep original text unchanged for audit/debugging
        return RomanizedProcessResult(
            original_text=original_text,
            detected_language=active_lang,
            detected_language_name=det_name,
            script=det_script,
            is_romanized=is_romanized,
            confidence=round(det_confidence, 4),
            was_converted=was_converted,
            converted_text=converted_text,
            ayurvedic_terms_detected=ayurvedic_terms_found,
            processing_status=processing_status,
            execution_time_ms=round(elapsed_ms, 2),
        )

    def _shield_ayurvedic_terms(
        self,
        text: str,
        target_lang: str,
    ) -> Tuple[str, Dict[str, str], List[str]]:
        """
        Identify known Ayurvedic medical terms and prepare placeholders mapped to
        authoritative canonical forms (Devanagari for Hindi/Sanskrit, or canonical Latin).
        """
        try:
            from app.services.terminology_service import TerminologyService
            service = TerminologyService.get_instance()
            matches = service.identify_terms(text)
            if not matches:
                return text, {}, []

            placeholders: Dict[str, str] = {}
            terms_found: List[str] = []

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
                canonical_name = match.term.canonical_term
                if canonical_name not in terms_found:
                    terms_found.append(canonical_name)

                # Canonical native script representation
                if target_lang in ["hi", "hin", "sa", "san", "mr", "mar"]:
                    canonical_form = match.term.sanskrit_form
                else:
                    canonical_form = match.term.canonical_term

                placeholders[tag] = canonical_form
                text_chars[match.start:match.end] = list(tag)

            return "".join(text_chars), placeholders, terms_found
        except Exception as exc:
            logger.warning(f"Failed to shield Ayurvedic terms in Romanized processor: {exc}")
            return text, {}, []


# Convenience module-level function
def process_romanized_text(
    text: str,
    target_language: Optional[str] = None,
    min_conversion_confidence: float = 0.5,
    preserve_ayurvedic_terms: bool = True,
    convert_script: bool = True,
) -> RomanizedProcessResult:
    """Process Romanized Indian-language input via the singleton RomanizedProcessor."""
    return RomanizedProcessor.get_instance().process(
        text=text,
        target_language=target_language,
        min_conversion_confidence=min_conversion_confidence,
        preserve_ayurvedic_terms=preserve_ayurvedic_terms,
        convert_script=convert_script,
    )
