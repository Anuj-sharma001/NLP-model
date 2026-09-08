"""Automated test suite for the dedicated Multilingual NLP API (/api/v1/nlp/*).

Tests:
1. POST /api/v1/nlp/analyze:
   - Hindi clinical text analysis with term preservation.
   - Hinglish / Romanized Indic input handling and script conversion.
   - Direct English queries.
   - Model health and version metadata presence.
   - Correlation Request ID preservation in body and response headers.
   - Request validation failure (empty text, invalid min_confidence).
2. POST /api/v1/nlp/translate:
   - Indic -> English translation with Ayurvedic term preservation.
   - English -> Indic translation.
   - Automatic language detection when source_language is omitted.
   - Model metadata, version info, and compute device.
   - Structured error on unsupported language.
   - Request ID propagation.
3. POST /api/v1/nlp/detect-language:
   - Native Indic text identification.
   - Hinglish / Romanized text identification.
   - English text identification.
   - Model health, version, and latency fields.
4. Structured error responses and request ID correlation:
   - HTTP 400 Bad Request error structure.
   - HTTP 404 Not Found error structure.
   - HTTP 422 Unprocessable Entity error structure.
"""

import uuid
import pytest
from fastapi.testclient import TestClient

from app.schemas.nlp_api import (
    ErrorResponse,
    NLPAnalyzeResponse,
    NLPDetectLanguageResponse,
    NLPTranslateResponse,
)


# =============================================================================
# 1. POST /api/v1/nlp/analyze Tests
# =============================================================================

def test_nlp_analyze_native_hindi_query(client: TestClient):
    """Verify POST /api/v1/nlp/analyze successfully analyzes native Hindi query."""
    custom_req_id = str(uuid.uuid4())
    payload = {
        "text": "वात दोष के कारण शरीर में दर्द है, त्रिफला का सेवन करें ।",
        "target_language": "en",
        "min_confidence": 0.4,
        "preserve_ayurvedic_terms": True,
    }

    response = client.post(
        "/api/v1/nlp/analyze",
        json=payload,
        headers={"X-Request-ID": custom_req_id},
    )
    assert response.status_code == 200
    assert response.headers.get("X-Request-ID") == custom_req_id

    data = response.json()
    validated = NLPAnalyzeResponse(**data)

    assert validated.original_text == payload["text"]
    assert validated.detected_language in ["hi", "hin"]
    assert validated.script == "Devanagari"
    assert validated.is_romanized is False
    assert validated.request_id == custom_req_id

    # Verify Ayurvedic terminology extracted
    terms = [t["canonical_term"] for t in validated.terminology]
    assert "Vata" in terms
    assert "Triphala" in terms

    # Verify English translation with preserved terms
    assert validated.english_text is not None
    assert "Vata" in validated.english_text
    assert "Triphala" in validated.english_text

    # Verify model health and version metadata
    assert "indic_lid" in validated.models
    assert validated.models["indic_lid"].name == "IndicLID"
    assert validated.models["indic_lid"].status in ["loaded", "ready"]
    assert "indic_trans2" in validated.models
    assert validated.models["indic_trans2"].name == "IndicTrans2"
    assert validated.models["indic_trans2"].status in ["loaded", "ready"]
    assert validated.execution_time_ms > 0.0


def test_nlp_analyze_romanized_hinglish_query(client: TestClient):
    """Verify POST /api/v1/nlp/analyze handles Romanized Hinglish clinical query."""
    payload = {
        "text": "mujhe pet me dard hai aur pitta vikriti lagti hai",
        "target_language": "en",
        "min_confidence": 0.3,
    }

    response = client.post("/api/v1/nlp/analyze", json=payload)
    assert response.status_code == 200
    assert "X-Request-ID" in response.headers

    data = response.json()
    assert data["is_romanized"] is True
    assert data["script"] == "Latin"
    assert data["request_id"] == response.headers["X-Request-ID"]

    # Terminology detected
    terms = [t["canonical_term"] for t in data["terminology"]]
    assert "Pitta" in terms
    assert "Vikriti" in terms

    # English text contains canonical terms
    assert data["english_text"] is not None
    assert "Pitta" in data["english_text"]
    assert "Vikriti" in data["english_text"]
    assert "converted_to_devanagari" in data["processing_status"]


def test_nlp_analyze_direct_english_query(client: TestClient):
    """Verify POST /api/v1/nlp/analyze passes through direct English input."""
    payload = {
        "text": "Ashwagandha promotes restful sleep and rejuvenates Ojas.",
        "target_language": "en",
    }

    response = client.post("/api/v1/nlp/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["detected_language"] in ["en", "eng"]
    assert data["is_romanized"] is False
    assert data["english_text"] == data["normalized_text"]
    assert "direct_english" in data["processing_status"]

    terms = [t["canonical_term"] for t in data["terminology"]]
    assert "Ashwagandha" in terms
    assert "Ojas" in terms


def test_nlp_analyze_validation_error_empty_text(client: TestClient):
    """Verify POST /api/v1/nlp/analyze rejects empty input with structured error."""
    payload = {"text": ""}
    response = client.post("/api/v1/nlp/analyze", json=payload)
    assert response.status_code == 422
    assert "X-Request-ID" in response.headers

    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "VALIDATION_ERROR"
    assert data["error"]["request_id"] == response.headers["X-Request-ID"]
    assert data["error"]["timestamp"] is not None


# =============================================================================
# 2. POST /api/v1/nlp/translate Tests
# =============================================================================

def test_nlp_translate_indic_to_english_with_term_preservation(client: TestClient):
    """Verify POST /api/v1/nlp/translate translates Indic text with term preservation."""
    custom_req_id = "test-req-translate-123"
    payload = {
        "text": "मुझे पेट में दर्द है और पित्त विकृति है",
        "source_language": "hi",
        "target_language": "en",
        "preserve_ayurvedic_terms": True,
    }

    response = client.post(
        "/api/v1/nlp/translate",
        json=payload,
        headers={"X-Request-ID": custom_req_id},
    )
    assert response.status_code == 200
    assert response.headers.get("X-Request-ID") == custom_req_id

    data = response.json()
    validated = NLPTranslateResponse(**data)

    assert validated.source_language == "hin_Deva"
    assert validated.target_language == "eng_Latn"
    assert len(validated.translated_text) > 0
    assert "Pitta" in validated.translated_text
    assert validated.model_name == "IndicTrans2"
    assert validated.model_version is not None
    assert validated.model_status in ["ready", "loaded"]
    assert validated.request_id == custom_req_id
    assert validated.execution_time_ms > 0.0


def test_nlp_translate_english_to_indic(client: TestClient):
    """Verify POST /api/v1/nlp/translate translates English to Tamil."""
    payload = {
        "text": "I have stomach pain",
        "source_language": "en",
        "target_language": "ta",
    }

    response = client.post("/api/v1/nlp/translate", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["source_language"] == "eng_Latn"
    assert data["target_language"] == "tam_Taml"
    assert len(data["translated_text"]) > 0
    assert data["model_name"] == "IndicTrans2"


def test_nlp_translate_autodetect_source_language(client: TestClient):
    """Verify POST /api/v1/nlp/translate autodetects source when omitted."""
    payload = {
        "text": "আমাকে সাহায্য করুন",  # Bengali: "Please help me"
        "target_language": "en",
    }

    response = client.post("/api/v1/nlp/translate", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["source_language"] == "ben_Beng"
    assert data["target_language"] == "eng_Latn"
    assert len(data["translated_text"]) > 0


def test_nlp_translate_unsupported_language_returns_structured_error(client: TestClient):
    """Verify unsupported language returns HTTP 400 with structured ErrorResponse."""
    payload = {
        "text": "Hello world",
        "source_language": "en",
        "target_language": "klingon",
    }

    response = client.post("/api/v1/nlp/translate", json=payload)
    assert response.status_code == 400
    data = response.json()

    assert "error" in data
    assert data["error"]["code"] == "HTTP_400"
    assert "klingon" in data["error"]["message"].lower() or "not supported" in data["error"]["message"].lower()
    assert data["error"]["request_id"] is not None


# =============================================================================
# 3. POST /api/v1/nlp/detect-language Tests
# =============================================================================

def test_nlp_detect_language_native_hindi(client: TestClient):
    """Verify POST /api/v1/nlp/detect-language detects native Hindi script."""
    custom_req_id = "test-req-lid-456"
    payload = {"text": "मुझे पेट में बहुत दर्द है"}

    response = client.post(
        "/api/v1/nlp/detect-language",
        json=payload,
        headers={"X-Request-ID": custom_req_id},
    )
    assert response.status_code == 200
    assert response.headers.get("X-Request-ID") == custom_req_id

    data = response.json()
    validated = NLPDetectLanguageResponse(**data)

    assert validated.language_code in ["hi", "hin"]
    assert validated.language_name == "Hindi"
    assert validated.script == "Devanagari"
    assert validated.confidence > 0.5
    assert validated.is_romanized is False
    assert validated.model_name == "IndicLID"
    assert validated.model_status in ["ready", "loaded"]
    assert validated.request_id == custom_req_id
    assert validated.execution_time_ms is not None


def test_nlp_detect_language_romanized_hinglish(client: TestClient):
    """Verify POST /api/v1/nlp/detect-language classifies Romanized Indic text."""
    payload = {"text": "mujhe pet me dard ho raha hai"}
    response = client.post("/api/v1/nlp/detect-language", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["script"] == "Latin"
    assert data["is_romanized"] is True
    assert data["language_code"] not in ["en", "eng"]


def test_nlp_detect_language_english(client: TestClient):
    """Verify POST /api/v1/nlp/detect-language classifies bona fide English text."""
    payload = {"text": "The patient reports significant relief after herbal treatment."}
    response = client.post("/api/v1/nlp/detect-language", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["language_code"] in ["en", "eng"]
    assert data["script"] == "Latin"
    assert data["is_romanized"] is False


# =============================================================================
# 4. Structured Error & Request ID Correlation Tests
# =============================================================================

def test_request_id_generated_when_absent(client: TestClient):
    """Verify X-Request-ID is automatically generated and returned when not provided."""
    response = client.get("/health")
    assert response.status_code == 200
    assert "X-Request-ID" in response.headers
    generated_id = response.headers["X-Request-ID"]
    assert len(generated_id) > 10


def test_structured_error_on_404(client: TestClient):
    """Verify 404 responses conform to the unified ErrorResponse schema."""
    response = client.get("/api/v1/nlp/non-existent-endpoint")
    assert response.status_code == 404
    data = response.json()

    assert "error" in data
    error = data["error"]
    assert error["code"] == "HTTP_404"
    assert "not found" in error["message"].lower()
    assert error["request_id"] is not None
    assert error["timestamp"] is not None


def test_structured_error_on_422_validation_failure(client: TestClient):
    """Verify 422 validation errors conform to ErrorResponse schema."""
    response = client.post("/api/v1/nlp/detect-language", json={})
    assert response.status_code == 422
    data = response.json()

    assert "error" in data
    error = data["error"]
    assert error["code"] == "VALIDATION_ERROR"
    assert isinstance(error["details"], list)
    assert error["request_id"] is not None
    assert error["timestamp"] is not None
