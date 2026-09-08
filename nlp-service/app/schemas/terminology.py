"""Pydantic schemas for Ayurveda terminology layer."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class AyurvedicTermProvenance(BaseModel):
    """Provenance and classical references for an Ayurvedic term."""

    classical_texts: List[str] = Field(
        default_factory=list,
        description="Classical foundational texts referencing this term (e.g., Charaka Samhita).",
        examples=[["Charaka Samhita Sutrasthana 1.55", "Sushruta Samhita Sutrasthana 21.5"]],
    )
    standard_references: List[str] = Field(
        default_factory=list,
        description="Authoritative contemporary benchmarks (e.g., WHO, AYUSH NAMASTE).",
        examples=[["WHO Benchmarks for Training in Ayurveda"]],
    )
    definition: Optional[str] = Field(
        default=None,
        description="Classical Sanskrit or translated definition/etymology.",
        examples=["वा गतिगन्धनयोः - तत्र वायोः वातं..."],
    )


class AyurvedicTerm(BaseModel):
    """Comprehensive representation of an Ayurvedic technical term."""

    canonical_term: str = Field(
        ...,
        description="Canonical name in standardized Latin script.",
        examples=["Vata"],
    )
    sanskrit_form: str = Field(
        ...,
        description="Primary Devanagari Sanskrit representation.",
        examples=["वात"],
    )
    iast: Optional[str] = Field(
        default=None,
        description="International Alphabet of Sanskrit Transliteration (IAST) representation.",
        examples=["vāta"],
    )
    hindi_forms: List[str] = Field(
        default_factory=list,
        description="Common Hindi script variants.",
        examples=[["वात", "बाय"]],
    )
    english_canonical_form: str = Field(
        ...,
        description="Canonical English/Latin form used in clinical documentation.",
        examples=["Vata"],
    )
    synonyms: List[str] = Field(
        default_factory=list,
        description="Synonyms in Sanskrit/Hindi or classical literature.",
        examples=[["Vayu", "Maruta", "Pavana", "वायु"]],
    )
    transliterations: List[str] = Field(
        default_factory=list,
        description="Alternative Romanized spellings and colloquial phonetic forms.",
        examples=[["vata", "vaata", "vatha", "vatham", "vaat", "waat"]],
    )
    category: str = Field(
        ...,
        description="Ayurvedic taxonomy category (e.g. Dosha, Constitution, Physiology, Herbology).",
        examples=["Dosha"],
    )
    notes: Optional[str] = Field(
        default=None,
        description="Clinical and conceptual description of the term.",
        examples=["Governs biological movement, cellular transport, and impulses."],
    )
    do_not_translate: bool = Field(
        default=True,
        description="Whether this technical term should be protected from translation into literal English words.",
    )
    provenance: Optional[AyurvedicTermProvenance] = Field(
        default=None,
        description="Classical and institutional provenance references.",
    )


class TermMatch(BaseModel):
    """Details of a detected Ayurvedic term within text."""

    term: AyurvedicTerm
    matched_text: str
    start: int
    end: int
    matched_by: str = Field(
        ...,
        description="Field that matched the input ('canonical', 'sanskrit', 'hindi', 'synonym', 'transliteration').",
    )


class ProtectedTextResult(BaseModel):
    """Result of wrapping text with translation placeholders for terminology preservation."""

    original_text: str
    protected_text: str
    placeholders: Dict[str, AyurvedicTerm] = Field(
        default_factory=dict,
        description="Mapping from placeholder token to matched AyurvedicTerm.",
    )
