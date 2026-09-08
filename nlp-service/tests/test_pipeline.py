"""
Unit and integration tests for the SAKYTI Multilingual NLP Pipeline.
Verifies:
1. Stage 1: Input Validation (empty, non-string, max length).
2. Stage 2: Text Normalization (Unicode NFC, whitespace, safe punctuation).
3. Stage 3: Language Identification (IndicLID).
4. Stage 4: Romanization Detection & Handling.
5. Stage 5: Ayurvedic Terminology Extraction.
6. Stage 6: English Translation with Terminology Preservation (IndicTrans2).
7. End-to-end pipeline execution for Hindi, Hinglish, and English queries.
8. Return schema compliance.
9. Root forwarding module compatibility.
10. REST API endpoint (POST /api/v1/pipeline/process).
"""

import pytest
from fastapi.testclient import TestClient

from app.schemas.pipeline import PipelineQueryRequest, PipelineQueryResult
from app.services.pipeline import SakytiPipeline, process_query
from services.pipeline import (
    SakytiPipeline as RootSakytiPipeline,
    process_query as root_process_query,
)


@pytest.fixture(scope="module")
def pipeline() -> SakytiPipeline:
    """Fixture providing initialized SakytiPipeline singleton."""
    return SakytiPipeline.get_instance()


# =========================================================================
# 1. Stage 1: Input Validation
# =========================================================================

def test_stage_1_validate_valid_input(pipeline: SakytiPipeline):
    """Verify valid input returns stripped text."""
    valid = "   mujhe pet me dard hai   "
    assert pipeline.validate_input(valid) == "mujhe pet me dard hai"


def test_stage_1_validate_empty_raises_error(pipeline: SakytiPipeline):
    """Verify empty or whitespace-only inputs raise ValueError."""
    with pytest.raises(ValueError, match="cannot be empty or whitespace-only"):
        pipeline.validate_input("")

    with pytest.raises(ValueError, match="cannot be empty or whitespace-only"):
        pipeline.validate_input("    \t\n   ")


def test_stage_1_validate_non_string_raises_error(pipeline: SakytiPipeline):
    """Verify non-string inputs raise ValueError."""
    with pytest.raises(ValueError, match="must be a string"):
        pipeline.validate_input(12345)

    with pytest.raises(ValueError, match="must be a string"):
        pipeline.validate_input(None)


def test_stage_1_validate_max_length_exceeded(pipeline: SakytiPipeline):
    """Verify inputs exceeding max length raise ValueError."""
    long_text = "a" * 2001
    with pytest.raises(ValueError, match="exceeds maximum allowed length"):
        pipeline.validate_input(long_text, max_length=2000)


# =========================================================================
# 2. Stage 2: Text Normalization
# =========================================================================

def test_stage_2_normalize(pipeline: SakytiPipeline):
    """Verify Stage 2 normalization collapses whitespace and preserves danda."""
    text = "मरीज   को  bahuuut  pitta   hai!!!!   ।  "
    res = pipeline.normalize_stage(text, language="hi")
    assert "  " not in res.normalized_text
    assert "!" in res.normalized_text
    assert "!!!!" not in res.normalized_text
    assert "।" in res.normalized_text


# =========================================================================
# 3. Stage 3: Language Identification
# =========================================================================

def test_stage_3_detect_language_hindi(pipeline: SakytiPipeline):
    """Verify Stage 3 detects native Hindi."""
    res = pipeline.detect_language_stage("मुझे पेट में बहुत दर्द है")
    assert res.language_code in ["hi", "hin"]
    assert res.script == "Devanagari"
    assert res.confidence > 0.5


def test_stage_3_detect_language_english(pipeline: SakytiPipeline):
    """Verify Stage 3 detects English."""
    res = pipeline.detect_language_stage("The patient presented with fever and fatigue.")
    assert res.language_code in ["en", "eng"]
    assert res.script == "Latin"


# =========================================================================
# 4. Stage 4: Romanization Detection & Handling
# =========================================================================

def test_stage_4_detect_romanization_hinglish(pipeline: SakytiPipeline):
    """Verify Stage 4 detects Romanized Hindi and converts to Devanagari."""
    text = "mujhe pet me dard ho raha hai"
    det = pipeline.detect_language_stage(text)
    res = pipeline.detect_romanization_stage(
        text,
        detection=det,
        target_language="hi",
        min_conversion_confidence=0.3,
    )

    assert res.is_romanized is True
    assert res.was_converted is True
    assert "मुझे" in res.converted_text


def test_stage_4_detect_romanization_bypasses_english(pipeline: SakytiPipeline):
    """Verify Stage 4 bypasses pure English."""
    text = "The doctor prescribed rest and herbal tea."
    det = pipeline.detect_language_stage(text)
    res = pipeline.detect_romanization_stage(text, detection=det)

    assert res.is_romanized is False
    assert res.was_converted is False
    assert res.converted_text is None
    assert res.processing_status == "bypassed_english"


# =========================================================================
# 5. Stage 5: Ayurvedic Terminology Extraction
# =========================================================================

def test_stage_5_extract_terminology(pipeline: SakytiPipeline):
    """Verify Stage 5 extracts canonical terms, categories, and offsets."""
    text = "Triphala helps in pacifying vata and pitta dosha."
    terms = pipeline.extract_terminology_stage(text)

    canonical_names = {t["canonical_term"] for t in terms}
    assert "Triphala" in canonical_names
    assert "Vata" in canonical_names
    assert "Pitta" in canonical_names

    # Check structure
    triphala_meta = next(t for t in terms if t["canonical_term"] == "Triphala")
    assert "Herbology" in triphala_meta["category"]
    assert triphala_meta["sanskrit_form"] == "त्रिफला"
    assert triphala_meta["do_not_translate"] is True


# =========================================================================
# 6. Stage 6: English Translation
# =========================================================================

def test_stage_6_translate_english_passthrough(pipeline: SakytiPipeline):
    """Verify Stage 6 directly passes through English text."""
    text = "Patient feels healthy today."
    eng_text, status = pipeline.translate_stage(text, source_language="en", is_english=True)
    assert eng_text == text
    assert status == "direct_english"


def test_stage_6_translate_indic_to_english(pipeline: SakytiPipeline):
    """Verify Stage 6 translates native Indic text while preserving terms."""
    text = "मुझे पेट में दर्द है"
    eng_text, status = pipeline.translate_stage(text, source_language="hi", is_english=False)
    assert eng_text is not None
    assert "pain" in eng_text.lower() or "stomach" in eng_text.lower() or "abdomen" in eng_text.lower()
    assert "translated_from_hi" in status


# =========================================================================
# 7. End-to-End Pipeline Execution
# =========================================================================

def test_end_to_end_native_hindi_query(pipeline: SakytiPipeline):
    """Verify end-to-end pipeline execution on native Hindi clinical query."""
    query = "वात दोष के कारण शरीर में दर्द है, त्रिफला का सेवन करें ।"
    res = pipeline.process_query(query)

    assert isinstance(res, PipelineQueryResult)
    assert res.original_text == query
    assert res.detected_language in ["hi", "hin"]
    assert res.is_romanized is False

    # Check Ayurvedic terms extracted
    term_names = [t["canonical_term"] for t in res.terminology]
    assert "Vata" in term_names
    assert "Triphala" in term_names

    # Check English translation with preserved terms
    assert res.english_text is not None
    assert "Vata" in res.english_text
    assert "Triphala" in res.english_text
    assert res.execution_time_ms > 0.0


def test_end_to_end_romanized_hinglish_query(pipeline: SakytiPipeline):
    """Verify end-to-end pipeline on Romanized Hinglish clinical query."""
    query = "mujhe pet me dard hai aur pitta vikriti lagti hai"
    res = pipeline.process_query(query)

    assert res.original_text == query
    assert res.is_romanized is True

    # Terms detected
    term_names = [t["canonical_term"] for t in res.terminology]
    assert "Pitta" in term_names
    assert "Vikriti" in term_names

    # English text contains preserved terms
    assert res.english_text is not None
    assert "Pitta" in res.english_text
    assert "Vikriti" in res.english_text


def test_end_to_end_direct_english_query(pipeline: SakytiPipeline):
    """Verify end-to-end pipeline on direct English query."""
    query = "The patient shows symptoms of excess Kapha and fatigue."
    res = pipeline.process_query(query)

    assert res.original_text == query
    assert res.detected_language in ["en", "eng"]
    assert res.is_romanized is False
    assert res.english_text == res.normalized_text

    # Terminology preserved
    term_names = [t["canonical_term"] for t in res.terminology]
    assert "Kapha" in term_names


def test_pipeline_return_schema_completeness(pipeline: SakytiPipeline):
    """Verify that returned dictionary matches all required fields."""
    res = pipeline.process_query("kya ashwagandha se neend aati hai?")
    data = res.model_dump()

    required_fields = [
        "original_text",
        "normalized_text",
        "detected_language",
        "confidence",
        "script",
        "is_romanized",
        "terminology",
        "english_text",
        "target_language",
        "processing_status",
    ]

    for field in required_fields:
        assert field in data, f"Required field '{field}' missing from pipeline return payload."


# =========================================================================
# 8. Root Forwarding Module Compatibility
# =========================================================================

def test_root_services_pipeline_forwarding():
    """Verify root `from services.pipeline import process_query` works."""
    res = root_process_query("pitta shanti ke liye kya karein?")
    assert res is not None
    assert "Pitta" in [t["canonical_term"] for t in res.terminology]


# =========================================================================
# 9. REST API Integration Endpoint (POST /api/v1/pipeline/process)
# =========================================================================

def test_api_pipeline_process_endpoint_success(client: TestClient):
    """Verify POST /api/v1/pipeline/process returns 200 with structured result."""
    payload = {
        "text": "mujhe pet me dard hai aur pitta vikriti lagti hai",
        "target_language": "en",
        "min_confidence": 0.4,
    }
    response = client.post("/api/v1/pipeline/process", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["original_text"] == payload["text"]
    assert "Pitta" in [t["canonical_term"] for t in data["terminology"]]
    assert data["english_text"] is not None
    assert "Pitta" in data["english_text"]
    assert data["target_language"] == "en"
    assert data["execution_time_ms"] > 0.0


def test_api_pipeline_process_validation_error(client: TestClient):
    """Verify 422 or 400 when invalid input is sent to pipeline endpoint."""
    response = client.post("/api/v1/pipeline/process", json={"text": ""})
    assert response.status_code in [400, 422]
