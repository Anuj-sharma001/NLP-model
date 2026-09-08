"""
Automated unit and integration tests for the SAKYTI Ayurvedic terminology layer.
Verifies all 14 initial terms, canonical lookups, romanized/native spelling handling,
in-text identification, canonicalization, translation protection, and REST endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from app.schemas.terminology import AyurvedicTerm, AyurvedicTermProvenance
from app.services.terminology_service import TerminologyService

REQUIRED_INITIAL_TERMS = [
    "Vata",
    "Pitta",
    "Kapha",
    "Prakriti",
    "Vikriti",
    "Agni",
    "Ama",
    "Ojas",
    "Dhatu",
    "Mala",
    "Tridosha",
    "Triphala",
    "Ashwagandha",
    "Panchakarma",
]


@pytest.fixture(scope="module")
def term_service() -> TerminologyService:
    """Fixture providing initialized TerminologyService."""
    service = TerminologyService.get_instance()
    service.load_terms()
    return service


@pytest.fixture(scope="module", autouse=True)
def ensure_models_loaded(client: TestClient):
    """Ensure translation models are loaded for translation preservation tests."""
    from app.models.indictrans_model import IndicTransModelManager

    manager = IndicTransModelManager.get_instance()
    if not manager.is_loaded:
        manager.initialize()
    return manager


# =========================================================================
# 1. Database Schema & Initial 14 Terms Verification
# =========================================================================

def test_database_contains_all_initial_terms(term_service: TerminologyService):
    """Verify all 14 required initial Ayurvedic terms are present in the database."""
    loaded_terms = term_service.list_terms()
    canonical_names = {t.canonical_term for t in loaded_terms}

    for required in REQUIRED_INITIAL_TERMS:
        assert required in canonical_names, f"Required term '{required}' is missing from database."


@pytest.mark.parametrize("canonical_name", REQUIRED_INITIAL_TERMS)
def test_term_schema_completeness(term_service: TerminologyService, canonical_name: str):
    """Verify each required term has all mandatory fields, provenance, and do_not_translate=True."""
    term = term_service.get_term(canonical_name)
    assert term is not None
    assert term.canonical_term == canonical_name
    assert len(term.sanskrit_form.strip()) > 0
    assert len(term.english_canonical_form.strip()) > 0
    assert isinstance(term.hindi_forms, list) and len(term.hindi_forms) > 0
    assert isinstance(term.synonyms, list)
    assert isinstance(term.transliterations, list) and len(term.transliterations) > 0
    assert len(term.category.strip()) > 0
    assert term.notes is not None and len(term.notes) > 0
    assert term.do_not_translate is True

    # Provenance verification
    assert term.provenance is not None
    assert len(term.provenance.classical_texts) > 0
    assert len(term.provenance.standard_references) > 0
    assert term.provenance.definition is not None


# =========================================================================
# 2. Term Lookup & Spelling Resolution Tests
# =========================================================================

def test_lookup_by_canonical_name(term_service: TerminologyService):
    """Verify case-insensitive lookup by canonical English name."""
    assert term_service.get_term("Vata").canonical_term == "Vata"
    assert term_service.get_term("vata").canonical_term == "Vata"
    assert term_service.get_term("VATA").canonical_term == "Vata"
    assert term_service.get_term("Pitta").canonical_term == "Pitta"
    assert term_service.get_term("Kapha").canonical_term == "Kapha"


def test_lookup_by_sanskrit_devanagari(term_service: TerminologyService):
    """Verify lookup by native Devanagari Sanskrit representations."""
    assert term_service.get_term("वात").canonical_term == "Vata"
    assert term_service.get_term("पित्त").canonical_term == "Pitta"
    assert term_service.get_term("कफ").canonical_term == "Kapha"
    assert term_service.get_term("प्रकृति").canonical_term == "Prakriti"
    assert term_service.get_term("अग्नि").canonical_term == "Agni"
    assert term_service.get_term("त्रिफला").canonical_term == "Triphala"
    assert term_service.get_term("अश्वगन्धा").canonical_term == "Ashwagandha"
    assert term_service.get_term("पञ्चकर्म").canonical_term == "Panchakarma"


def test_lookup_by_hindi_variants(term_service: TerminologyService):
    """Verify lookup by common Hindi script forms."""
    assert term_service.get_term("पंचकर्म").canonical_term == "Panchakarma"
    assert term_service.get_term("अश्वगंधा").canonical_term == "Ashwagandha"
    assert term_service.get_term("बलगम").canonical_term == "Kapha"
    assert term_service.get_term("सप्तधातु").canonical_term == "Dhatu"


def test_lookup_by_romanized_transliterations(term_service: TerminologyService):
    """Verify lookup across colloquial and alternative romanized spellings."""
    assert term_service.get_term("vaata").canonical_term == "Vata"
    assert term_service.get_term("vatha").canonical_term == "Vata"
    assert term_service.get_term("waat").canonical_term == "Vata"
    assert term_service.get_term("pitham").canonical_term == "Pitta"
    assert term_service.get_term("shleshma").canonical_term == "Kapha"
    assert term_service.get_term("prakruti").canonical_term == "Prakriti"
    assert term_service.get_term("vikruthi").canonical_term == "Vikriti"
    assert term_service.get_term("trifala").canonical_term == "Triphala"
    assert term_service.get_term("asgandh").canonical_term == "Ashwagandha"
    assert term_service.get_term("panchkarma").canonical_term == "Panchakarma"


def test_lookup_nonexistent_returns_none(term_service: TerminologyService):
    """Verify invalid or non-Ayurvedic strings return None."""
    assert term_service.get_term("paracetamol") is None
    assert term_service.get_term("ibuprofen") is None
    assert term_service.get_term("") is None


# =========================================================================
# 3. In-Text Term Identification
# =========================================================================

def test_identify_terms_english_sentence(term_service: TerminologyService):
    """Verify extraction of multiple terms from an English medical sentence."""
    text = "The patient shows aggravated vata and depleted ojas due to poor agni."
    matches = term_service.identify_terms(text)
    matched_canonical = [m.term.canonical_term for m in matches]

    assert "Vata" in matched_canonical
    assert "Ojas" in matched_canonical
    assert "Agni" in matched_canonical


def test_identify_terms_hindi_sentence(term_service: TerminologyService):
    """Verify extraction of multiple terms from a Hindi clinical sentence."""
    text = "रोगी की प्रकृति में वात और पित्त का प्रकोप देखा गया है, त्रिफला देना चाहिए।"
    matches = term_service.identify_terms(text)
    matched_canonical = [m.term.canonical_term for m in matches]

    assert "Prakriti" in matched_canonical
    assert "Vata" in matched_canonical
    assert "Pitta" in matched_canonical
    assert "Triphala" in matched_canonical


def test_identify_greedy_non_overlapping(term_service: TerminologyService):
    """Verify compound terms like Panchakarma match as a whole unit."""
    text = "Recommend full course of Panchakarma therapy."
    matches = term_service.identify_terms(text)
    assert len(matches) == 1
    assert matches[0].term.canonical_term == "Panchakarma"
    assert matches[0].matched_text == "Panchakarma"


# =========================================================================
# 4. Text Canonicalization Tests
# =========================================================================

def test_canonicalize_text_latin(term_service: TerminologyService):
    """Verify non-standard romanized spellings are standardized to canonical forms."""
    text = "The doctor diagnosed vaata imbalance and prescribed trifala and panchkarma."
    canonicalized = term_service.canonicalize_text(text, target_script="latin")

    assert "Vata" in canonicalized
    assert "Triphala" in canonicalized
    assert "Panchakarma" in canonicalized
    assert "vaata" not in canonicalized
    assert "trifala" not in canonicalized
    assert "panchkarma" not in canonicalized


def test_canonicalize_text_devanagari(term_service: TerminologyService):
    """Verify romanized terms can be standardized directly to Devanagari script."""
    text = "Patient took ashwagandha for low ojas."
    canonicalized = term_service.canonicalize_text(text, target_script="devanagari")

    assert "अश्वगन्धा" in canonicalized
    assert "ओजस्" in canonicalized


# =========================================================================
# 5. Translation Protection & Term Preservation Tests
# =========================================================================

def test_protect_and_restore_terms(term_service: TerminologyService):
    """Verify text is protected with unique placeholders and faithfully restored."""
    sample = "Patient with high Vata should consume Triphala."
    protected_res = term_service.protect_terms_for_translation(sample)

    assert "__AYUR_" in protected_res.protected_text
    assert len(protected_res.placeholders) == 2

    # Simulate translation preserving the placeholder tokens
    simulated_translated = protected_res.protected_text

    # Restore to English
    restored_en = term_service.restore_terms_after_translation(
        simulated_translated, protected_res.placeholders, target_language="en"
    )
    assert "Vata" in restored_en
    assert "Triphala" in restored_en

    # Restore to Hindi
    restored_hi = term_service.restore_terms_after_translation(
        simulated_translated, protected_res.placeholders, target_language="hi"
    )
    assert "वात" in restored_hi
    assert "त्रिफला" in restored_hi


def test_translation_preserves_ayurvedic_terms():
    """Verify end-to-end that translation does not corrupt Ayurvedic technical terms."""
    from app.services.translator import translate_to_english

    # "वात के कारण दर्द है" should preserve "Vata" rather than translating to "air" or "wind"
    result = translate_to_english("वात के कारण दर्द है", "hi", num_beams=1, preserve_ayurvedic_terms=True)
    assert "Vata" in result


# =========================================================================
# 6. Extensibility Tests
# =========================================================================

def test_add_term_dynamically(term_service: TerminologyService):
    """Verify new Ayurvedic botanical terms can be added dynamically and immediately indexed."""
    new_term = AyurvedicTerm(
        canonical_term="Guduchi",
        sanskrit_form="गुडूची",
        iast="guḍūcī",
        hindi_forms=["गिलोय", "गुडूची"],
        english_canonical_form="Guduchi",
        synonyms=["Giloy", "Amrita", "अमृता"],
        transliterations=["guduchi", "giloy", "gudusi", "amruta"],
        category="Herbology / Dravyaguna",
        notes="Tinospora cordifolia, premier immunomodulator and Rasayana.",
        do_not_translate=True,
        provenance=AyurvedicTermProvenance(
            classical_texts=["Charaka Samhita Chikitsasthana 1.3"],
            standard_references=["Ayurvedic Pharmacopoeia of India"],
        ),
    )

    term_service.add_term(new_term)

    # Verify retrieval
    assert term_service.get_term("Guduchi") is not None
    assert term_service.get_term("giloy").canonical_term == "Guduchi"
    assert term_service.get_term("गुडूची").canonical_term == "Guduchi"

    # Verify in-text identification
    matches = term_service.identify_terms("Patient administered giloy for fever.")
    assert any(m.term.canonical_term == "Guduchi" for m in matches)


# =========================================================================
# 7. REST API Endpoints Integration Tests
# =========================================================================

def test_api_list_terms(client: TestClient):
    """Test GET /api/v1/terminology/terms returns terms list."""
    resp = client.get("/api/v1/terminology/terms")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] >= 14
    canonical_list = [t["canonical_term"] for t in data["terms"]]
    assert "Vata" in canonical_list
    assert "Triphala" in canonical_list


def test_api_list_terms_with_category_filter(client: TestClient):
    """Test GET /api/v1/terminology/terms?category=Dosha."""
    resp = client.get("/api/v1/terminology/terms?category=Dosha")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] >= 3
    for t in data["terms"]:
        assert t["category"].lower() == "dosha"


def test_api_get_term_by_name(client: TestClient):
    """Test GET /api/v1/terminology/terms/{term_name}."""
    resp = client.get("/api/v1/terminology/terms/Ashwagandha")
    assert resp.status_code == 200
    data = resp.json()
    assert data["canonical_term"] == "Ashwagandha"
    assert data["sanskrit_form"] == "अश्वगन्धा"
    assert data["do_not_translate"] is True
    assert "classical_texts" in data["provenance"]


def test_api_get_term_not_found(client: TestClient):
    """Test GET /api/v1/terminology/terms/{term_name} 404 for unknown term."""
    resp = client.get("/api/v1/terminology/terms/unknown_xyz")
    assert resp.status_code == 404


def test_api_identify_terms(client: TestClient):
    """Test POST /api/v1/terminology/identify."""
    payload = {"text": "Patient has aggravated vaat and low agni."}
    resp = client.post("/api/v1/terminology/identify", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] >= 2
    matched_terms = [m["term"]["canonical_term"] for m in data["matches"]]
    assert "Vata" in matched_terms
    assert "Agni" in matched_terms


def test_api_canonicalize_terms(client: TestClient):
    """Test POST /api/v1/terminology/canonicalize."""
    payload = {
        "text": "The practitioner recommended trifala and panchkarma for vaat imbalance.",
        "target_script": "latin",
    }
    resp = client.post("/api/v1/terminology/canonicalize", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "Triphala" in data["canonicalized_text"]
    assert "Panchakarma" in data["canonicalized_text"]
    assert "Vata" in data["canonicalized_text"]
