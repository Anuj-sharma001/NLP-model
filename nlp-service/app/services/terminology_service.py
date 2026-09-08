"""
Ayurveda Terminology Service for SAKYTI Multilingual NLP layer.
Provides canonical terminology lookups, script/spelling normalization,
in-text term identification, and translation protection/restoration.
"""

import json
from pathlib import Path
from typing import Dict, List, Literal, Optional, Set, Tuple
import regex as re

from app.schemas.terminology import (
    AyurvedicTerm,
    AyurvedicTermProvenance,
    ProtectedTextResult,
    TermMatch,
)
from app.utils.logger import get_logger

logger = get_logger("terminology_service")


class TerminologyService:
    """
    Singleton service for managing the canonical Ayurvedic terminology database.
    Provides term search, identification in free text, and translation protection.
    """

    _instance: Optional["TerminologyService"] = None

    def __init__(self, data_path: Optional[Path] = None) -> None:
        self.data_path = data_path or self._resolve_default_path()
        self._terms_by_canonical: Dict[str, AyurvedicTerm] = {}
        self._lookup: Dict[str, Tuple[AyurvedicTerm, str]] = {}
        self._patterns: List[Tuple[re.Pattern, AyurvedicTerm, str]] = []
        self._is_loaded: bool = False
        self.load_terms()

    @classmethod
    def get_instance(cls, data_path: Optional[Path] = None) -> "TerminologyService":
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls(data_path=data_path)
        return cls._instance

    def _resolve_default_path(self) -> Path:
        """Locate the canonical ayurveda_terms.json file."""
        candidates = [
            Path("./data/ayurveda_terms.json"),
            Path(__file__).parent.parent.parent / "data" / "ayurveda_terms.json",
            Path(__file__).parent.parent / "data" / "ayurveda_terms.json",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        return Path("./data/ayurveda_terms.json")

    def load_terms(self, data_path: Optional[Path] = None) -> None:
        """Load terminology database from JSON file and rebuild indices."""
        target_path = data_path or self.data_path
        if not target_path.is_file():
            logger.warning(f"Ayurvedic terms file not found at: {target_path}")
            return

        with open(target_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        terms_list = raw_data.get("terms", [])
        self._terms_by_canonical.clear()
        self._lookup.clear()

        for item in terms_list:
            term = AyurvedicTerm(**item)
            self._terms_by_canonical[term.canonical_term.lower()] = term
            self._index_term(term)

        self._build_search_patterns()
        self._is_loaded = True
        logger.info(
            f"Loaded {len(self._terms_by_canonical)} Ayurvedic terms into terminology database",
            extra={"count": len(self._terms_by_canonical), "path": str(target_path)},
        )

    def _index_term(self, term: AyurvedicTerm) -> None:
        """Index a single term across canonical, native, and romanized forms with priority."""
        priority = {
            "canonical": 5,
            "english_canonical": 5,
            "sanskrit": 4,
            "hindi": 3,
            "iast": 3,
            "synonym": 2,
            "transliteration": 1,
        }

        def _safe_add(k: str, f: str):
            k_clean = k.strip()
            if not k_clean:
                return
            k_lower = k_clean.lower()
            for key in (k_clean, k_lower):
                if key not in self._lookup or priority.get(f, 0) > priority.get(self._lookup[key][1], 0):
                    self._lookup[key] = (term, f)

        # 1. Canonical Latin name
        _safe_add(term.canonical_term, "canonical")
        _safe_add(term.english_canonical_form, "english_canonical")

        # 2. Sanskrit Devanagari form
        _safe_add(term.sanskrit_form, "sanskrit")

        # 3. IAST if available
        if term.iast:
            _safe_add(term.iast, "iast")

        # 4. Hindi forms
        for h in term.hindi_forms:
            _safe_add(h, "hindi")

        # 5. Synonyms
        for syn in term.synonyms:
            _safe_add(syn, "synonym")

        # 6. Transliterations
        for tr in term.transliterations:
            _safe_add(tr, "transliteration")

    def _build_search_patterns(self) -> None:
        """Compile regex patterns ordered by term length descending to prioritize greedy matches."""
        sorted_keys = sorted(self._lookup.keys(), key=lambda k: len(k), reverse=True)
        self._patterns = []
        seen_keys: Set[str] = set()

        for key in sorted_keys:
            cleaned = key.strip()
            if not cleaned or cleaned.lower() in seen_keys:
                continue
            seen_keys.add(cleaned.lower())
            term, field = self._lookup[cleaned]

            # Use word boundaries for Latin text, and Unicode word boundaries for Indic text
            escaped = re.escape(cleaned)
            if all(ord(c) < 128 for c in cleaned):
                # Latin / ASCII string
                pattern = re.compile(rf"(?<![a-zA-Z0-9]){escaped}(?![a-zA-Z0-9])", re.IGNORECASE)
            else:
                # Indic script (Devanagari, etc.)
                pattern = re.compile(rf"{escaped}", re.IGNORECASE)

            self._patterns.append((pattern, term, field))

    def get_term(self, name_or_alias: str) -> Optional[AyurvedicTerm]:
        """
        Lookup an Ayurvedic term by any valid identifier:
        canonical name, Devanagari form, Hindi form, synonym, or transliteration.
        """
        if not name_or_alias or not isinstance(name_or_alias, str):
            return None

        cleaned = name_or_alias.strip()
        lower_cleaned = cleaned.lower()

        # 1. Highest priority: direct canonical name
        if lower_cleaned in self._terms_by_canonical:
            return self._terms_by_canonical[lower_cleaned]

        # 2. Direct lookup (case-sensitive)
        if cleaned in self._lookup:
            return self._lookup[cleaned][0]

        # 3. Case-insensitive lookup
        if lower_cleaned in self._lookup:
            return self._lookup[lower_cleaned][0]

        return None

    def list_terms(self, category: Optional[str] = None) -> List[AyurvedicTerm]:
        """Return all terms, optionally filtered by category."""
        all_terms = list(self._terms_by_canonical.values())
        if category:
            return [t for t in all_terms if t.category.lower() == category.lower()]
        return all_terms

    def list_categories(self) -> List[str]:
        """List distinct categories available in the database."""
        return sorted(list({t.category for t in self._terms_by_canonical.values()}))

    def search_terms(self, query: str) -> List[AyurvedicTerm]:
        """Search terms by keyword in canonical, forms, notes, or synonyms."""
        if not query or not query.strip():
            return []

        q = query.strip().lower()
        results: List[AyurvedicTerm] = []

        for term in self._terms_by_canonical.values():
            if (
                q in term.canonical_term.lower()
                or q in term.sanskrit_form.lower()
                or (term.iast and q in term.iast.lower())
                or any(q in h.lower() for h in term.hindi_forms)
                or any(q in s.lower() for s in term.synonyms)
                or any(q in t.lower() for t in term.transliterations)
                or (term.notes and q in term.notes.lower())
                or q in term.category.lower()
            ):
                results.append(term)

        return results

    def add_term(self, term: AyurvedicTerm) -> None:
        """Dynamically add an Ayurvedic term to the active in-memory database."""
        self._terms_by_canonical[term.canonical_term.lower()] = term
        self._index_term(term)
        self._build_search_patterns()
        logger.info(f"Added term '{term.canonical_term}' to terminology database")

    def identify_terms(self, text: str) -> List[TermMatch]:
        """
        Scan free-form text and extract all occurrences of known Ayurvedic terms.
        Handles both native Devanagari and Romanized phonetic variations.
        Non-overlapping: longer matches supersede sub-matches.
        """
        if not text or not text.strip():
            return []

        matches: List[Tuple[int, int, str, AyurvedicTerm, str]] = []
        occupied_spans: List[Tuple[int, int]] = []

        for pattern, term, field in self._patterns:
            for m in pattern.finditer(text):
                start, end = m.span()
                # Check overlap with already accepted spans
                if any(start < s_end and end > s_start for s_start, s_end in occupied_spans):
                    continue

                occupied_spans.append((start, end))
                matches.append((start, end, m.group(0), term, field))

        # Sort matches by start position in original text
        matches.sort(key=lambda x: x[0])

        return [
            TermMatch(
                term=term,
                matched_text=matched_text,
                start=start,
                end=end,
                matched_by=field,
            )
            for start, end, matched_text, term, field in matches
        ]

    def canonicalize_text(
        self,
        text: str,
        target_script: Literal["latin", "devanagari"] = "latin",
    ) -> str:
        """
        Replace colloquial, non-standard, or Romanized spellings of Ayurvedic terms
        with their canonical representations.
        """
        if not text or not text.strip():
            return text

        matches = self.identify_terms(text)
        if not matches:
            return text

        # Replace spans in reverse order to preserve string indices
        result = list(text)
        for m in reversed(matches):
            if target_script == "devanagari":
                replacement = m.term.sanskrit_form
            else:
                replacement = m.term.english_canonical_form

            result[m.start : m.end] = list(replacement)

        return "".join(result)

    def protect_terms_for_translation(
        self,
        text: str,
        placeholder_prefix: str = "__AYUR_",
    ) -> ProtectedTextResult:
        """
        Find terms flagged with `do_not_translate=True` and replace them with
        unique placeholder tokens before passing text to the translation model.
        """
        if not text or not text.strip():
            return ProtectedTextResult(
                original_text=text,
                protected_text=text,
                placeholders={},
            )

        matches = self.identify_terms(text)
        # Filter for terms marked as do_not_translate
        dnt_matches = [m for m in matches if m.term.do_not_translate]

        if not dnt_matches:
            return ProtectedTextResult(
                original_text=text,
                protected_text=text,
                placeholders={},
            )

        placeholders: Dict[str, AyurvedicTerm] = {}
        result_chars = list(text)

        # Replace in reverse order
        for idx, m in enumerate(reversed(dnt_matches)):
            placeholder = f"{placeholder_prefix}{len(dnt_matches) - 1 - idx}__"
            placeholders[placeholder] = m.term
            result_chars[m.start : m.end] = list(placeholder)

        return ProtectedTextResult(
            original_text=text,
            protected_text="".join(result_chars),
            placeholders=placeholders,
        )

    def restore_terms_after_translation(
        self,
        translated_text: str,
        placeholders: Dict[str, AyurvedicTerm],
        target_language: str = "eng_Latn",
    ) -> str:
        """
        Restore placeholders in the translated text with the appropriate canonical
        representation in the target language script.
        """
        if not translated_text or not placeholders:
            return translated_text

        is_english = target_language in ["en", "eng", "eng_Latn", "english"]
        is_hindi = target_language in ["hi", "hin", "hin_Deva", "hindi"]
        is_sanskrit = target_language in ["sa", "san", "san_Deva", "sanskrit"]

        restored = translated_text

        for token, term in placeholders.items():
            if is_english:
                replacement = term.english_canonical_form
            elif is_hindi or is_sanskrit:
                replacement = term.sanskrit_form
            else:
                # For other Indic languages, use Sanskrit/canonical form
                replacement = term.english_canonical_form

            # Extract index from token (e.g. __AYUR_0__ -> 0)
            idx_match = re.search(r"\d+", token)
            if idx_match:
                idx = idx_match.group(0)
                # Match token or any model spacing/underscore variations like "_ _ AYUR _ 0", "AYUR 0", "__AYUR_0__"
                pattern = re.compile(rf"[_\s]*\bAYUR\b[_\s]*{idx}[_\s]*|__AYUR_{idx}__", re.IGNORECASE)
                restored = pattern.sub(
                    lambda m: (" " if m.group(0).startswith(" ") and not restored.startswith(m.group(0)) else "")
                    + replacement
                    + (" " if m.group(0).endswith(" ") else ""),
                    restored,
                )
            else:
                restored = restored.replace(token, replacement)

        # Normalize multiple spaces
        restored = re.sub(r"\s+", " ", restored).strip()
        return restored
