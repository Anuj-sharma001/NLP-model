"""Automated test suite for SAKYTI IndicLID language identification."""

import pytest
from fastapi.testclient import TestClient

from app.models.indiclid_model import IndicLIDModelManager
from app.services.language_detector import detect_language


@pytest.fixture(scope="session", autouse=True)
def setup_models(test_settings):
    """Ensure IndicLID models are loaded once before running tests."""
    manager = IndicLIDModelManager.get_instance()
    manager.load_models(test_settings)
    assert manager.is_loaded


@pytest.mark.unit
def test_detect_english():
    """Test identification of English text."""
    sample = "I have severe abdominal pain, nausea, and indigestion after meals."
    result = detect_language(sample)

    assert result.language_code == "eng"
    assert result.language_name == "English"
    assert result.script == "Latin"
    assert result.is_romanized is False
    assert result.confidence >= 0.7
    assert result.raw_model_result["model_used"] == "IndicLID-FTR"


@pytest.mark.unit
def test_detect_hindi():
    """Test identification of native Hindi in Devanagari script."""
    sample = "मुझे पेट में दर्द है और बहुत कमजोरी महसूस हो रही है"
    result = detect_language(sample)

    assert result.language_code == "hin"
    assert result.language_name == "Hindi"
    assert result.script == "Devanagari"
    assert result.is_romanized is False
    assert result.confidence >= 0.7
    assert result.raw_model_result["model_used"] == "IndicLID-FTN"


@pytest.mark.unit
def test_detect_tamil():
    """Test identification of native Tamil."""
    sample = "எனக்கு கடுமையான வயிற்று வலி உள்ளது"
    result = detect_language(sample)

    assert result.language_code == "tam"
    assert result.language_name == "Tamil"
    assert result.script == "Tamil"
    assert result.is_romanized is False
    assert result.confidence >= 0.8
    assert result.raw_model_result["model_used"] == "IndicLID-FTN"


@pytest.mark.unit
def test_detect_telugu():
    """Test identification of native Telugu."""
    sample = "నాకు తీవ్రమైన కడుపు నొప్పి ఉంది మరియు వికారం ఉంది"
    result = detect_language(sample)

    assert result.language_code == "tel"
    assert result.language_name == "Telugu"
    assert result.script == "Telugu"
    assert result.is_romanized is False
    assert result.confidence >= 0.8
    assert result.raw_model_result["model_used"] == "IndicLID-FTN"


@pytest.mark.unit
def test_detect_bengali():
    """Test identification of native Bengali."""
    sample = "আমার পেটে খুব ব্যথা করছে এবং জ্বর আসছে"
    result = detect_language(sample)

    assert result.language_code == "ben"
    assert result.language_name == "Bengali"
    assert result.script == "Bengali"
    assert result.is_romanized is False
    assert result.confidence >= 0.8
    assert result.raw_model_result["model_used"] == "IndicLID-FTN"


@pytest.mark.unit
def test_detect_marathi():
    """Test identification of native Marathi in Devanagari script."""
    sample = "माझ्या पोटात खूप दुखत आहे आणि मला अस्वस्थ वाटत आहे"
    result = detect_language(sample)

    assert result.language_code == "mar"
    assert result.language_name == "Marathi"
    assert result.script == "Devanagari"
    assert result.is_romanized is False
    assert result.confidence >= 0.8
    assert result.raw_model_result["model_used"] == "IndicLID-FTN"


@pytest.mark.unit
def test_detect_romanized_hindi():
    """Test identification of Romanized Hindi in Latin script."""
    sample = "Aap kaise hain aur aapki tabiyat kaisi hai?"
    result = detect_language(sample)

    assert result.language_code == "hin"
    assert result.language_name == "Hindi"
    assert result.script == "Latin"
    assert result.is_romanized is True
    assert result.confidence >= 0.7
    assert result.raw_model_result["model_used"] == "IndicLID-FTR"


@pytest.mark.unit
@pytest.mark.parametrize(
    "invalid_input",
    [
        "",
        "   ",
        "\n\t  ",
        "??? !!! ...",
        "1234567890",
        "@#$%^&*()_+",
        "x",
    ],
)
def test_detect_unknown_and_short_or_invalid_text(invalid_input):
    """Test safe handling of empty, non-alphabetic, or very short inputs."""
    result = detect_language(invalid_input)

    assert result.language_code == "unknown"
    assert result.language_name == "Unknown"
    assert result.confidence == 0.0
    assert result.is_romanized is False


@pytest.mark.unit
def test_confidence_threshold_rejection():
    """Test that predictions falling below confidence threshold are flagged as unknown."""
    sample = "मुझे पेट में दर्द है"
    # Setting an unrealistically high threshold
    result = detect_language(sample, confidence_threshold=0.9999)

    assert result.language_code == "unknown"
    assert result.raw_model_result.get("below_threshold") is True
    assert result.raw_model_result.get("candidate_language_code") == "hin"


@pytest.mark.integration
def test_api_detect_language_endpoint(client: TestClient):
    """Test POST /api/v1/detect-language API endpoint."""
    response = client.post(
        "/api/v1/detect-language",
        json={"text": "मला पोटात दुखत आहे"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["language_code"] == "mar"
    assert data["language_name"] == "Marathi"
    assert data["script"] == "Devanagari"
    assert data["is_romanized"] is False
    assert data["confidence"] > 0.7
