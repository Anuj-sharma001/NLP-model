"""
Automated unit and integration tests for IndicTrans2 translation service and endpoints.
Covers all 10 initial Indic languages, input validation, error handling, and model lifecycle.
"""

import pytest
from fastapi.testclient import TestClient

from app.models.indictrans_model import (
    IndicTransModelManager,
    TranslationInputTooLongError,
    UnsupportedLanguageError,
    normalize_language_code,
)
from app.services.translator import (
    translate,
    translate_from_english,
    translate_to_english,
)

# 10 initial languages required by Task 4
INITIAL_LANGUAGES = [
    ("Hindi", "hi", "hin_Deva", "मुझे पेट में दर्द है"),
    ("Tamil", "ta", "tam_Taml", "எனக்கு வயிற்று வலி உள்ளது"),
    ("Telugu", "te", "tel_Telu", "నాకు కడుపు నొప్పి ఉంది"),
    ("Bengali", "bn", "ben_Beng", "আমার পেটে ব্যথা আছে"),
    ("Marathi", "mr", "mar_Deva", "माझ्या पोटात दुखत आहे"),
    ("Gujarati", "gu", "guj_Gujr", "મને પેટમાં દુખાવો થાય છે"),
    ("Kannada", "kn", "kan_Knda", "ನನಗೆ ಹೊಟ್ಟೆ ನೋವು ಇದೆ"),
    ("Malayalam", "ml", "mal_Mlym", "എനിക്ക് വയറുവേദനയുണ്ട്"),
    ("Punjabi", "pa", "pan_Guru", "ਮੇਰੇ ਢਿੱਡ ਵਿੱਚ ਦਰਦ ਹੈ"),
    ("Odia", "or", "ory_Orya", "ମୋ ପେଟରେ ଯନ୍ତ୍ରଣା ହେଉଛି"),
]


@pytest.fixture(scope="module", autouse=True)
def ensure_models_loaded(client: TestClient):
    """Ensure models are initialized before running translation tests."""
    manager = IndicTransModelManager.get_instance()
    if not manager.is_loaded:
        manager.initialize()
    return manager


# =========================================================================
# 1. Model Lifecycle & Initialization Tests
# =========================================================================

def test_model_manager_initialization():
    """Verify IndicTransModelManager singleton initialization and device selection."""
    manager = IndicTransModelManager.get_instance()
    assert manager.is_loaded is True
    assert manager.indic_en_model is not None
    assert manager.en_indic_model is not None
    assert manager.indic_en_tokenizer is not None
    assert manager.en_indic_tokenizer is not None
    assert manager.processor is not None

    info = manager.get_model_info()
    assert info["loaded"] is True
    assert "cpu" in info["device"].lower() or "cuda" in info["device"].lower()
    assert "hin_Deva" in info["supported_indic_languages"]
    assert "tam_Taml" in info["supported_indic_languages"]


def test_language_code_normalization():
    """Verify language code aliases map to canonical FLORES-200 tags."""
    assert normalize_language_code("hi") == "hin_Deva"
    assert normalize_language_code("hin") == "hin_Deva"
    assert normalize_language_code("hin_Deva") == "hin_Deva"
    assert normalize_language_code("en") == "eng_Latn"
    assert normalize_language_code("eng") == "eng_Latn"
    assert normalize_language_code("ta") == "tam_Taml"
    assert normalize_language_code("te") == "tel_Telu"
    assert normalize_language_code("bn") == "ben_Beng"
    assert normalize_language_code("mr") == "mar_Deva"
    assert normalize_language_code("gu") == "guj_Gujr"
    assert normalize_language_code("kn") == "kan_Knda"
    assert normalize_language_code("ml") == "mal_Mlym"
    assert normalize_language_code("pa") == "pan_Guru"
    assert normalize_language_code("or") == "ory_Orya"
    assert normalize_language_code("sa") == "san_Deva"

    with pytest.raises(UnsupportedLanguageError):
        normalize_language_code("xyz_invalid")

    with pytest.raises(UnsupportedLanguageError):
        normalize_language_code("fr")


# =========================================================================
# 2. Indic -> English Translation for All 10 Initial Languages
# =========================================================================

@pytest.mark.parametrize("lang_name, lang_code, flores_tag, sample_text", INITIAL_LANGUAGES)
def test_indic_to_english_all_languages(lang_name, lang_code, flores_tag, sample_text):
    """Test translation from Indic language to English for all 10 supported languages."""
    result = translate_to_english(text=sample_text, source_language=lang_code, num_beams=1)
    assert isinstance(result, str)
    assert len(result.strip()) > 0
    # The output should be Latin/English script
    assert any(c.isascii() and c.isalpha() for c in result)


def test_indic_to_english_hindi_semantic():
    """Verify semantic correctness of Hindi -> English translation."""
    result = translate_to_english(text="मुझे पेट में दर्द है", source_language="hi", num_beams=1)
    result_lower = result.lower()
    assert "pain" in result_lower or "ache" in result_lower or "hurt" in result_lower


# =========================================================================
# 3. English -> Indic Translation for All 10 Initial Languages
# =========================================================================

@pytest.mark.parametrize("lang_name, lang_code, flores_tag, sample_text", INITIAL_LANGUAGES)
def test_english_to_indic_all_languages(lang_name, lang_code, flores_tag, sample_text):
    """Test translation from English to Indic language for all 10 supported languages."""
    english_input = "I have stomach pain"
    result = translate_from_english(text=english_input, target_language=lang_code, num_beams=1)
    assert isinstance(result, str)
    assert len(result.strip()) > 0
    # Output should contain non-ASCII Indic characters
    assert any(ord(c) > 127 for c in result)


def test_english_to_hindi_semantic():
    """Verify semantic correctness of English -> Hindi translation."""
    result = translate_from_english(text="I have stomach pain", target_language="hi", num_beams=1)
    assert "पेट" in result or "दर्द" in result


# =========================================================================
# 4. Error Handling & Input Protection
# =========================================================================

def test_translate_empty_text_error():
    """Verify empty or blank text raises ValueError."""
    with pytest.raises(ValueError, match="empty or whitespace"):
        translate_to_english("", "hi")

    with pytest.raises(ValueError, match="empty or whitespace"):
        translate_to_english("   \t\n", "hi")

    with pytest.raises(ValueError, match="empty or whitespace"):
        translate_from_english("", "hi")


def test_translate_unsupported_language_error():
    """Verify unsupported language codes raise UnsupportedLanguageError and never silently fallback."""
    with pytest.raises(UnsupportedLanguageError):
        translate_to_english("Bonjour", "fr")

    with pytest.raises(UnsupportedLanguageError):
        translate_from_english("Hello", "spanish")


def test_translate_english_to_english_error():
    """Verify translate_to_english rejects English as source language."""
    with pytest.raises(UnsupportedLanguageError, match="cannot be English"):
        translate_to_english("Hello world", "en")

    with pytest.raises(UnsupportedLanguageError, match="cannot be English"):
        translate_from_english("Hello world", "en")


def test_translate_indic_to_indic_unsupported():
    """Verify direct Indic -> Indic without pivot raises UnsupportedLanguageError in this phase."""
    with pytest.raises(UnsupportedLanguageError, match="not supported in this phase"):
        translate(text="मुझे सिरदर्द है", source_language="hi", target_language="ta")


def test_translate_max_length_protection():
    """Verify input text exceeding 5000 characters is rejected."""
    long_text = "दर्द " * 1500  # > 6000 characters
    with pytest.raises(TranslationInputTooLongError, match="exceeds maximum allowed limit"):
        translate_to_english(long_text, "hi")


# =========================================================================
# 5. High-Level Service & REST API Integration Tests
# =========================================================================

def test_high_level_translate_with_autodetection():
    """Test high-level translate() with auto-detected source language."""
    response = translate(
        text="मुझे पेट में दर्द है",
        target_language="en",
        source_language=None,  # triggers IndicLID auto-detection
        num_beams=1,
    )
    assert response.source_language == "hin_Deva"
    assert response.target_language == "eng_Latn"
    assert len(response.translated_text) > 0
    assert response.model_name == "IndicTrans2"
    assert response.execution_time_ms > 0


def test_api_translate_endpoint_indic_to_english(client: TestClient):
    """Test POST /api/v1/translate for Indic -> English."""
    payload = {
        "text": "मुझे पेट में दर्द है",
        "source_language": "hi",
        "target_language": "en",
    }
    resp = client.post("/api/v1/translate", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["source_language"] == "hin_Deva"
    assert data["target_language"] == "eng_Latn"
    assert len(data["translated_text"]) > 0
    assert data["model_name"] == "IndicTrans2"
    assert data["execution_time_ms"] > 0


def test_api_translate_endpoint_english_to_indic(client: TestClient):
    """Test POST /api/v1/translate for English -> Indic."""
    payload = {
        "text": "I have stomach pain",
        "source_language": "en",
        "target_language": "ta",
    }
    resp = client.post("/api/v1/translate", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["source_language"] == "eng_Latn"
    assert data["target_language"] == "tam_Taml"
    assert len(data["translated_text"]) > 0


def test_api_translate_endpoint_autodetect(client: TestClient):
    """Test POST /api/v1/translate when source_language is omitted."""
    payload = {
        "text": "আমাকে সাহায্য করুন",  # Bengali: "Please help me"
        "target_language": "en",
    }
    resp = client.post("/api/v1/translate", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["source_language"] == "ben_Beng"
    assert data["target_language"] == "eng_Latn"
    assert len(data["translated_text"]) > 0


def test_api_translate_invalid_language(client: TestClient):
    """Test POST /api/v1/translate rejects unsupported language."""
    payload = {
        "text": "Hello world",
        "source_language": "en",
        "target_language": "klingon",
    }
    resp = client.post("/api/v1/translate", json=payload)
    assert resp.status_code == 400
    assert "not supported" in resp.json()["detail"].lower()


def test_api_translate_empty_text(client: TestClient):
    """Test POST /api/v1/translate rejects empty text."""
    payload = {
        "text": "",
        "source_language": "hi",
        "target_language": "en",
    }
    resp = client.post("/api/v1/translate", json=payload)
    assert resp.status_code == 422  # Pydantic validation


def test_health_endpoint_includes_translation_models(client: TestClient):
    """Verify /health endpoint reports IndicTrans2 model status."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "indictrans2_indic_en" in data["models"]
    assert data["models"]["indictrans2_indic_en"] == "loaded"
    assert "indictrans2_en_indic" in data["models"]
    assert data["models"]["indictrans2_en_indic"] == "loaded"
