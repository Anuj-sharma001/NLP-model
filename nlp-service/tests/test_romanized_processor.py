"""
Unit and integration tests for the Romanized Indian Language Processing Layer.
Verifies:
1. Detecting whether input is Romanized Indic vs English (no blind English assumption).
2. Preserving Ayurvedic terminology in canonical native forms.
3. Gated script conversion based on confidence thresholds.
4. Retaining exact original text for auditing/debugging.
5. Hindi Romanized (Hinglish) transliteration accuracy.
6. Extensible architecture for adding future Indian languages.
7. REST API endpoint integration (POST /api/v1/process-romanized).
"""

import pytest
from fastapi.testclient import TestClient

from app.schemas.romanized import RomanizedProcessRequest, RomanizedProcessResult
from app.services.romanized_processor import (
    BaseRomanizedConverter,
    HindiRomanizedConverter,
    RomanizedProcessor,
    process_romanized_text,
)
from services.romanized_processor import (
    RomanizedProcessor as RootRomanizedProcessor,
    process_romanized_text as root_process_romanized_text,
)


@pytest.fixture(scope="module")
def processor() -> RomanizedProcessor:
    """Fixture providing initialized RomanizedProcessor."""
    return RomanizedProcessor.get_instance()


# =========================================================================
# 1. Detection: Romanized Indian Language vs English (Requirements 1 & 3)
# =========================================================================

def test_detects_romanized_hindi_input(processor: RomanizedProcessor):
    """Verify input is detected as Romanized Indic, not plain English."""
    text = "mujhe pet me dard ho raha hai"
    res = processor.process(text)

    assert res.is_romanized is True
    assert res.script == "Latin"
    assert res.confidence > 0.0
    assert res.detected_language != "en"
    assert res.detected_language != "eng"


def test_does_not_assume_latin_is_english(processor: RomanizedProcessor):
    """
    Verify Requirement 3: System must NOT treat Latin-script sentences
    as English by default if they are Romanized Indian languages.
    """
    hinglish_text = "mera gala kharab hai aur bukhar hai"
    res = processor.process(hinglish_text)

    assert res.is_romanized is True
    assert res.detected_language != "en"
    assert res.processing_status != "bypassed_english"


def test_bypasses_true_english_sentences(processor: RomanizedProcessor):
    """Verify bona fide English sentences are recognized as English and not converted."""
    english_text = "The patient reported severe abdominal pain and fever."
    res = processor.process(english_text)

    assert res.detected_language in ["en", "eng"]
    assert res.is_romanized is False
    assert res.was_converted is False
    assert res.converted_text is None
    assert res.processing_status == "bypassed_english"
    assert res.original_text == english_text


# =========================================================================
# 2. Ayurvedic Terminology Preservation (Requirement 2)
# =========================================================================

def test_preserves_ayurvedic_terms_in_native_script(processor: RomanizedProcessor):
    """
    Verify Ayurvedic terms (Vata, Pitta, Triphala) are identified, shielded,
    and mapped to canonical Sanskrit/Devanagari forms upon conversion.
    """
    text = "kya triphala lene se vata aur pitta shant hota hai"
    res = processor.process(text, target_language="hi", preserve_ayurvedic_terms=True)

    assert res.was_converted is True
    assert res.converted_text is not None

    # Verify detected Ayurvedic terms
    assert "Triphala" in res.ayurvedic_terms_detected
    assert "Vata" in res.ayurvedic_terms_detected
    assert "Pitta" in res.ayurvedic_terms_detected

    # Verify canonical Devanagari forms are present without mangling
    assert "त्रिफला" in res.converted_text
    assert "वात" in res.converted_text
    assert "पित्त" in res.converted_text


def test_ayurvedic_terms_not_distorted_by_phonetic_rules(processor: RomanizedProcessor):
    """Verify 'Pitta' with double 't' is rendered as canonical 'पित्त', not corrupted."""
    text = "mareez ko pitta vikriti hai"
    res = processor.process(text, target_language="hi", preserve_ayurvedic_terms=True)

    assert "पित्त" in res.converted_text
    assert "विकृति" in res.converted_text


# =========================================================================
# 3. Confidence Gating (Requirement 4)
# =========================================================================

def test_does_not_convert_if_confidence_below_threshold(processor: RomanizedProcessor):
    """
    Verify Requirement 4: Do not convert text unless confidence is sufficient.
    Setting an impossibly high threshold (0.999) must bypass conversion.
    """
    text = "mujhe pet me dard hai"
    res = processor.process(text, min_conversion_confidence=0.999)

    assert res.was_converted is False
    assert res.converted_text is None
    assert "insufficient_confidence" in res.processing_status
    # Original text must remain completely intact
    assert res.original_text == text


def test_conversion_disabled_by_flag(processor: RomanizedProcessor):
    """Verify conversion can be explicitly disabled via convert_script=False."""
    text = "mujhe pet me dard hai"
    res = processor.process(text, convert_script=False)

    assert res.was_converted is False
    assert res.converted_text is None
    assert res.processing_status == "conversion_disabled_by_request"


# =========================================================================
# 4. Auditability: Original Text Kept Unchanged (Requirement 5)
# =========================================================================

def test_original_text_kept_unchanged_for_audit(processor: RomanizedProcessor):
    """
    Verify Requirement 5: The exact raw input text is always retained
    in original_text without any modification.
    """
    raw_inputs = [
        "mujhe pet me dard ho raha hai",
        "  untrimmed   text   with  spaces  ",
        "The patient is recovering well.",
        "kya triphala se vata kam hota hai?",
    ]

    for raw in raw_inputs:
        res = processor.process(raw)
        assert res.original_text == raw, f"Original text was altered for: {raw}"


# =========================================================================
# 5. Hindi Romanized Input Tests (Requirement 6)
# =========================================================================

def test_hindi_romanized_conversational_phrases(processor: RomanizedProcessor):
    """Verify typical Romanized Hindi (Hinglish) clinical sentences convert accurately."""
    cases = [
        ("mujhe pet me dard ho raha hai", "मुझे पेट में दर्द हो रहा है"),
        ("mera gala kharab hai aur bukhar hai", "मेरा गला खराब है और बुखार है"),
        ("aaj subah se sir dard hai", "आज सुबह से सिर दर्द है"),
    ]

    for roman_in, expected_dev in cases:
        res = processor.process(roman_in, target_language="hi", min_conversion_confidence=0.3)
        assert res.was_converted is True
        assert res.converted_text == expected_dev, (
            f"Expected '{expected_dev}', got '{res.converted_text}'"
        )


def test_hindi_converter_direct_unit():
    """Verify HindiRomanizedConverter unit behavior."""
    converter = HindiRomanizedConverter()
    assert converter.language_code == "hi"
    assert converter.target_script_name == "devanagari"

    out = converter.convert("mujhe bukhar aur khansi hai", {})
    assert "मुझे" in out
    assert "बुखार" in out
    assert "और" in out
    assert "खांसी" in out
    assert "है" in out


# =========================================================================
# 6. Extensibility for Future Indian Languages (Requirement 7)
# =========================================================================

class MockTamilConverter(BaseRomanizedConverter):
    """Mock converter to test extensibility architecture for additional Indian languages."""

    @property
    def language_code(self) -> str:
        return "ta_mock"

    @property
    def target_script_name(self) -> str:
        return "tamil"

    def convert(self, text: str, shielded_placeholders: dict) -> str:
        # Simple mock conversion
        return text.replace("vanakkam", "வணக்கம்")


def test_extensible_converter_registration(processor: RomanizedProcessor):
    """
    Verify Requirement 7: System allows registering additional Indian language
    converters dynamically without altering existing code.
    """
    mock_converter = MockTamilConverter()
    processor.register_converter(mock_converter)

    supported = processor.get_supported_conversion_languages()
    assert "ta_mock" in supported

    # Test processing with the new converter
    res = processor.process("vanakkam doctor", target_language="ta_mock", min_conversion_confidence=0.0)
    assert res.was_converted is True
    assert "வணக்கம்" in res.converted_text
    assert res.processing_status == "converted_to_tamil"


# =========================================================================
# 7. Edge Cases & Root Forwarding Module
# =========================================================================

def test_empty_and_whitespace_inputs(processor: RomanizedProcessor):
    """Verify empty or whitespace strings return gracefully."""
    res_empty = processor.process("")
    assert res_empty.was_converted is False
    assert res_empty.processing_status == "empty_input"
    assert res_empty.original_text == ""

    res_ws = processor.process("   \t\n   ")
    assert res_ws.was_converted is False
    assert res_ws.processing_status == "empty_input"


def test_root_services_forwarding_compatibility():
    """Verify root `from services.romanized_processor import ...` works properly."""
    res = root_process_romanized_text("mujhe pet me dard hai", target_language="hi")
    assert res.was_converted is True
    assert "मुझे" in res.converted_text


# =========================================================================
# 8. REST API Integration Endpoint (POST /api/v1/process-romanized)
# =========================================================================

def test_api_process_romanized_endpoint_success(client: TestClient):
    """Verify POST /api/v1/process-romanized returns 200 with structured result."""
    payload = {
        "text": "mujhe pet me dard ho raha hai aur vata dosha lagta hai",
        "target_language": "hi",
        "min_conversion_confidence": 0.4,
        "preserve_ayurvedic_terms": True,
        "convert_script": True,
    }
    response = client.post("/api/v1/process-romanized", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["original_text"] == payload["text"]
    assert data["was_converted"] is True
    assert "वात" in data["converted_text"]
    assert "मुझे" in data["converted_text"]
    assert "Vata" in data["ayurvedic_terms_detected"]
    assert data["is_romanized"] is True
    assert data["execution_time_ms"] > 0.0


def test_api_process_romanized_english_bypassed(client: TestClient):
    """Verify English clinical notes are bypassed via the API endpoint."""
    payload = {
        "text": "Patient has high fever and joint pain.",
        "convert_script": True,
    }
    response = client.post("/api/v1/process-romanized", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["was_converted"] is False
    assert data["converted_text"] is None
    assert data["processing_status"] == "bypassed_english"
    assert data["is_romanized"] is False


def test_api_process_romanized_validation_error_on_empty(client: TestClient):
    """Verify 422 validation error when text is empty."""
    response = client.post("/api/v1/process-romanized", json={"text": ""})
    assert response.status_code == 422
