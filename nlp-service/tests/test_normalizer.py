"""
Unit and integration tests for the SAKYTI Multilingual Normalizer.
Verifies Unicode NFC, whitespace, repeated punctuation, safe punctuation,
common noise, Indic language hooks, Romanized elongation reduction,
Ayurvedic terminology preservation, edge cases, and REST API integration.
"""

import pytest
from fastapi.testclient import TestClient

from app.schemas.normalization import NormalizationRequest, NormalizationResult
from app.services.normalizer import MultilingualNormalizer, normalize_text
from services.normalizer import (
    MultilingualNormalizer as RootNormalizer,
    normalize_text as root_normalize_text,
)


@pytest.fixture(scope="module")
def normalizer() -> MultilingualNormalizer:
    """Fixture providing initialized MultilingualNormalizer."""
    return MultilingualNormalizer.get_instance()


# =========================================================================
# 1. Unicode Normalization & Diacritics
# =========================================================================

def test_unicode_nfc_normalization(normalizer: MultilingualNormalizer):
    """Verify decomposed Unicode sequences are composed into canonical NFC forms."""
    # 'e' + combining acute accent -> 'é'
    decomposed = "cafe\u0301"
    res = normalizer.normalize(decomposed)
    assert res.normalized_text == "café"
    assert "unicode_nfc_normalized" in res.detected_changes


def test_iast_sanskrit_diacritics_preserved(normalizer: MultilingualNormalizer):
    """Verify precomposed IAST diacritics for Sanskrit/Ayurveda are preserved."""
    iast_text = "vāta pitta kapha agni prāṇa ojas śodhana dhātu"
    res = normalizer.normalize(iast_text)
    assert "vāta" in res.normalized_text
    assert "prāṇa" in res.normalized_text
    assert "śodhana" in res.normalized_text
    assert "dhātu" in res.normalized_text


def test_zero_width_character_handling(normalizer: MultilingualNormalizer):
    """
    Verify non-printing format characters (BOM, zero-width space) are stripped,
    while ZWJ and ZWNJ essential for Indic conjuncts are strictly preserved.
    """
    # ZWJ (\u200D), ZWNJ (\u200C), BOM (\uFEFF), Zero-width space (\u200B)
    text = "\uFEFFनमस्ते\u200B क\u200D्ष \u200C"
    res = normalizer.normalize(text)

    # BOM and ZWSP stripped
    assert "\uFEFF" not in res.normalized_text
    assert "\u200B" not in res.normalized_text
    # ZWJ and ZWNJ preserved
    assert "\u200D" in res.normalized_text
    assert "\u200C" in res.normalized_text
    assert "invisible_characters_sanitized" in res.detected_changes


# =========================================================================
# 2. Whitespace Normalization
# =========================================================================

def test_exotic_whitespace_normalization(normalizer: MultilingualNormalizer):
    """Verify NBSP, thin spaces, and ideographic spaces are converted to standard ASCII spaces."""
    # Non-breaking space \u00A0 and Em quad \u2001
    text = "Ayurveda\u00A0Treatment\u2001System"
    res = normalizer.normalize(text)
    assert res.normalized_text == "Ayurveda Treatment System"
    assert "whitespace_normalized" in res.detected_changes


def test_consecutive_spaces_and_tabs_collapsed(normalizer: MultilingualNormalizer):
    """Verify multiple spaces and horizontal tabs are collapsed to single spaces."""
    text = "Vata    \t   Pitta     Kapha"
    res = normalizer.normalize(text)
    assert res.normalized_text == "Vata Pitta Kapha"
    assert "whitespace_normalized" in res.detected_changes


def test_line_endings_and_excessive_newlines(normalizer: MultilingualNormalizer):
    """Verify CRLF standardization and collapsing of 3+ consecutive newlines to 2."""
    text = "Line 1\r\n\r\n\r\n\r\nLine 2\r\nLine 3"
    res = normalizer.normalize(text)
    assert res.normalized_text == "Line 1\n\nLine 2\nLine 3"
    assert "whitespace_normalized" in res.detected_changes


def test_outer_whitespace_stripped(normalizer: MultilingualNormalizer):
    """Verify leading and trailing spaces/newlines are stripped."""
    text = "   \n\t  Ayurvedic Medicine   \n  "
    res = normalizer.normalize(text)
    assert res.normalized_text == "Ayurvedic Medicine"


# =========================================================================
# 3. Repeated Punctuation Collapsing
# =========================================================================

def test_repeated_exclamation_and_question_marks(normalizer: MultilingualNormalizer):
    """Verify multiple exclamation/question marks are collapsed."""
    text = "Kaya chikitsha kya hai???? Yeh bahut acha hai!!!!"
    res = normalizer.normalize(text)
    assert "kya hai?" in res.normalized_text
    assert "acha hai!" in res.normalized_text
    assert "repeated_punctuation_collapsed" in res.detected_changes


def test_interrobang_normalization(normalizer: MultilingualNormalizer):
    """Verify mixed punctuation like !?? or ?!?! is normalized to ?!'."""
    text = "Kya aapko dard hai!?? Sach mein?!?"
    res = normalizer.normalize(text)
    assert "dard hai?!" in res.normalized_text
    assert "Sach mein?!" in res.normalized_text


def test_ellipsis_preservation(normalizer: MultilingualNormalizer):
    """Verify standard 3-dot ellipsis (...) is preserved, while 4+ dots collapse to 3."""
    three_dots = "Loading clinical notes..."
    res_three = normalizer.normalize(three_dots)
    assert res_three.normalized_text == "Loading clinical notes..."

    many_dots = "Please wait........"
    res_many = normalizer.normalize(many_dots)
    assert res_many.normalized_text == "Please wait..."
    assert "repeated_punctuation_collapsed" in res_many.detected_changes


# =========================================================================
# 4. Safe Punctuation & Quotes
# =========================================================================

def test_smart_quotes_normalization(normalizer: MultilingualNormalizer):
    """Verify directional/smart quotes are normalized to straight ASCII quotes."""
    text = "‘Single quotes’ and “Double quotes”"
    res = normalizer.normalize(text)
    assert res.normalized_text == "'Single quotes' and \"Double quotes\""
    assert "quotes_normalized" in res.detected_changes


def test_typographical_dashes_normalization(normalizer: MultilingualNormalizer):
    """Verify em-dashes and en-dashes are normalized to hyphens."""
    text = "Ayurveda—traditional Indian medicine – holistic health"
    res = normalizer.normalize(text)
    assert res.normalized_text == "Ayurveda-traditional Indian medicine - holistic health"
    assert "dashes_normalized" in res.detected_changes


def test_devanagari_danda_preservation(normalizer: MultilingualNormalizer):
    """Verify Devanagari danda (।) and double danda (॥) are strictly preserved as sentence bounds."""
    text = "रोगस्तु दोषवैषम्यम् । दोषसाम्यमरोगता ॥"
    res = normalizer.normalize(text)
    assert "।" in res.normalized_text
    assert "॥" in res.normalized_text
    assert not res.normalized_text.endswith(".")
    assert "रोगस्तु दोषवैषम्यम् । दोषसाम्यमरोगता ॥" == res.normalized_text


def test_space_before_punctuation_cleaned(normalizer: MultilingualNormalizer):
    """Verify redundant space before western punctuation marks is removed."""
    text = "Vata , Pitta , and Kapha ."
    res = normalizer.normalize(text)
    assert res.normalized_text == "Vata, Pitta, and Kapha."


# =========================================================================
# 5. Common Noise & HTML Sanitization
# =========================================================================

def test_html_entities_unescaping(normalizer: MultilingualNormalizer):
    """Verify HTML entities like &amp;, &lt;, &gt;, &quot;, &#39; are decoded."""
    text = "Triphala &amp; Ashwagandha &lt;ayurvedic herbs&gt; &quot;herbal&quot;"
    res = normalizer.normalize(text)
    assert res.normalized_text == "Triphala & Ashwagandha <ayurvedic herbs> \"herbal\""
    assert "html_entities_unescaped" in res.detected_changes


def test_html_tags_removal(normalizer: MultilingualNormalizer):
    """Verify raw HTML tags are cleanly stripped when remove_html is True."""
    text = "<p>Patient suffers from <b>Vata</b> imbalance.</p><br>"
    res = normalizer.normalize(text, remove_html=True)
    assert res.normalized_text == "Patient suffers from Vata imbalance."
    assert "html_tags_removed" in res.detected_changes


def test_html_tags_preserved_when_flag_false(normalizer: MultilingualNormalizer):
    """Verify HTML tags remain intact when remove_html=False."""
    text = "<p>Patient has <b>Pitta</b> fever.</p>"
    res = normalizer.normalize(text, remove_html=False)
    assert "<p>" in res.normalized_text
    assert "<b>" in res.normalized_text


def test_ascii_control_characters_removal(normalizer: MultilingualNormalizer):
    """Verify ASCII control codes (null bytes, bell, escape) are safely removed."""
    text = "Clinical\x00 Note\x07 with\x1b control\x0c chars"
    res = normalizer.normalize(text)
    assert "\x00" not in res.normalized_text
    assert "\x07" not in res.normalized_text
    assert "\x1b" not in res.normalized_text
    assert res.normalized_text == "Clinical Note with control chars"
    assert "control_characters_removed" in res.detected_changes


# =========================================================================
# 6. Language-Aware Indic Normalization Hooks
# =========================================================================

def test_indic_hook_hindi(normalizer: MultilingualNormalizer):
    """Verify IndicNormalizer hook for Hindi handles visarga normalization."""
    # ASCII colon used instead of Devanagari visarga
    text = "दु:ख"
    res = normalizer.normalize(text, language="hi")
    assert res.normalized_text == "दुःख"
    assert "indic_script_normalized_hi" in res.detected_changes


def test_indic_hook_sanskrit(normalizer: MultilingualNormalizer):
    """Verify Sanskrit Indic hook functions correctly."""
    text = "दु:ख"
    res = normalizer.normalize(text, language="sa")
    assert res.normalized_text == "दुःख"
    assert "indic_script_normalized_sa" in res.detected_changes


def test_indic_hook_bengali(normalizer: MultilingualNormalizer):
    """Verify Bengali Indic normalizer hook."""
    text = "দু:খ"
    res = normalizer.normalize(text, language="bn")
    assert res.normalized_text == "দুঃখ"
    assert "indic_script_normalized_bn" in res.detected_changes


# =========================================================================
# 7. Romanized Indian-Language Hooks
# =========================================================================

def test_romanized_elongation_reduction(normalizer: MultilingualNormalizer):
    """Verify phonetic elongation in Romanized Indic text is reduced."""
    text = "Mujhe bahuuuut dard ho raha hai plzzzzz"
    res = normalizer.normalize(text, is_romanized=True)
    assert "bahut" in res.normalized_text
    assert "plz" in res.normalized_text
    assert "romanized_elongation_reduced" in res.detected_changes


def test_romanized_elongation_preserves_standard_long_vowels(normalizer: MultilingualNormalizer):
    """Verify 3+ 'a', 'e', 'o' reduce to 2 to preserve standard romanized long vowels."""
    text = "kaaaaafi theeeek dooooor"
    res = normalizer.normalize(text, is_romanized=True)
    assert res.normalized_text == "kaafi theek door"


def test_romanized_elongation_disabled_when_flag_false(normalizer: MultilingualNormalizer):
    """Verify romanized elongation is skipped when is_romanized=False."""
    text = "bahuuuut kaaaaafi"
    res = normalizer.normalize(text, is_romanized=False)
    assert "bahuuuut" in res.normalized_text
    assert "kaaaaafi" in res.normalized_text
    assert "romanized_elongation_reduced" not in res.detected_changes


# =========================================================================
# 8. Ayurvedic Medical Terminology Preservation
# =========================================================================

def test_pitta_double_t_preserved(normalizer: MultilingualNormalizer):
    """
    CRITICAL: Verify 'Pitta' (with double 't') is never aggressively
    autocorrected or truncated to 'pita' by consonant rules.
    """
    text = "Patient presents with pitta dosha aggravation."
    res = normalizer.normalize(text, preserve_ayurvedic_terms=True)
    assert "pitta" in res.normalized_text
    assert "pita" not in res.normalized_text


def test_ayurvedic_terms_shielded_in_romanized_text(normalizer: MultilingualNormalizer):
    """Verify canonical Ayurvedic terms remain intact in code-mixed Romanized sentences."""
    text = "Mujhe bahuuuut vata aur kapha prakriti sambandhi samasya hai!!!!"
    res = normalizer.normalize(text, is_romanized=True, preserve_ayurvedic_terms=True)
    assert "bahut" in res.normalized_text
    assert "vata" in res.normalized_text
    assert "kapha" in res.normalized_text
    assert "prakriti" in res.normalized_text
    assert "ayurvedic_terms_preserved" in res.detected_changes


def test_all_14_canonical_terms_shielded(normalizer: MultilingualNormalizer):
    """Verify all 14 initial canonical Ayurvedic terms are recognized and shielded."""
    terms_14 = [
        "Vata", "Pitta", "Kapha", "Prakriti", "Vikriti", "Agni", "Ama",
        "Ojas", "Dhatu", "Mala", "Tridosha", "Triphala", "Ashwagandha", "Panchakarma"
    ]
    sentence = " ".join(terms_14)
    res = normalizer.normalize(sentence, preserve_ayurvedic_terms=True)
    for term in terms_14:
        assert term in res.normalized_text, f"Ayurvedic term '{term}' was altered during normalization."
    assert "ayurvedic_terms_preserved" in res.detected_changes


# =========================================================================
# 9. Edge Cases & Module Compatibility
# =========================================================================

def test_empty_and_whitespace_only_inputs(normalizer: MultilingualNormalizer):
    """Verify empty or whitespace-only inputs return cleanly without error."""
    res_empty = normalizer.normalize("")
    assert res_empty.normalized_text == ""
    assert res_empty.detected_changes == []

    res_ws = normalizer.normalize("   \t\n   ")
    assert res_ws.normalized_text == ""


def test_clean_input_has_empty_detected_changes(normalizer: MultilingualNormalizer):
    """Verify text that requires no normalization returns empty detected_changes."""
    text = "Healthy lifestyle maintains biological balance."
    res = normalizer.normalize(text)
    assert res.normalized_text == text
    assert res.detected_changes == []


def test_root_services_import_compatibility():
    """Verify root `from services.normalizer import ...` works as specified."""
    normalizer = RootNormalizer.get_instance()
    assert normalizer is not None

    res = root_normalize_text("Ayurveda  wellness   center  ।")
    assert res.normalized_text == "Ayurveda wellness center ।"
    assert "whitespace_normalized" in res.detected_changes


# =========================================================================
# 10. REST API Integration Endpoint (POST /api/v1/normalize)
# =========================================================================

def test_api_normalize_endpoint_success(client: TestClient):
    """Verify POST /api/v1/normalize returns 200 with structured NormalizationResult."""
    payload = {
        "text": "मरीज   को  bahuuut  pitta   vikriti  hai!!!! &amp; bukhar bhi   ।",
        "language": "hi",
        "is_romanized": True,
        "preserve_ayurvedic_terms": True,
        "remove_html": True,
    }
    response = client.post("/api/v1/normalize", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "normalized_text" in data
    assert "detected_changes" in data
    assert "execution_time_ms" in data
    assert data["original_text"] == payload["text"]
    assert "bahut" in data["normalized_text"]
    assert "pitta" in data["normalized_text"]
    assert "vikriti" in data["normalized_text"]
    assert "।" in data["normalized_text"]
    assert "html_entities_unescaped" in data["detected_changes"]


def test_api_normalize_validation_error_on_empty(client: TestClient):
    """Verify 422 Unprocessable Entity when text payload is empty."""
    response = client.post("/api/v1/normalize", json={"text": ""})
    assert response.status_code == 422
