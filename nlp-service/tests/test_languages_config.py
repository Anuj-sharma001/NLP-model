"""Test canonical language configuration in languages.json."""

import json
from pathlib import Path
import pytest


@pytest.fixture(scope="session")
def languages_data():
    """Load canonical languages.json."""
    config_path = Path(__file__).resolve().parent.parent / "data" / "languages.json"
    assert config_path.exists(), f"languages.json not found at {config_path}"
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.unit
def test_languages_json_structure(languages_data):
    """Verify top-level structure of languages.json."""
    assert "version" in languages_data
    assert "languages" in languages_data
    assert isinstance(languages_data["languages"], list)
    assert len(languages_data["languages"]) >= 7


@pytest.mark.unit
def test_sanskrit_special_treatment(languages_data):
    """Verify Sanskrit is treated as an Ayurvedic knowledge language rather than just a UI language."""
    sanskrit = next((lang for lang in languages_data["languages"] if lang["code"] == "sa"), None)
    assert sanskrit is not None
    assert sanskrit["is_ayurveda_knowledge_source"] is True
    assert sanskrit["is_user_facing"] is False
    assert sanskrit["indictrans2_tag"] == "san_Deva"
    assert sanskrit["indiclid_label_native"] == "san_Deva"
    assert sanskrit["special_domain_treatment"] is not None
    assert sanskrit["special_domain_treatment"]["is_primary_ayurvedic_root"] is True


@pytest.mark.unit
def test_initial_user_facing_languages(languages_data):
    """Verify initial SAKYTI user-facing languages."""
    expected_user_facing = {"en", "hi", "ta", "te", "bn", "mr"}
    actual_user_facing = {
        lang["code"] for lang in languages_data["languages"] if lang["is_user_facing"]
    }
    assert expected_user_facing.issubset(actual_user_facing)


@pytest.mark.unit
def test_verified_model_tags(languages_data):
    """Verify IndicTrans2 and IndicLID tags conform to official specifications."""
    for lang in languages_data["languages"]:
        assert lang["indictrans2_tag"] is not None
        assert "_" in lang["indictrans2_tag"]
        assert lang["indiclid_label_native"] is not None
        assert "_" in lang["indiclid_label_native"]
