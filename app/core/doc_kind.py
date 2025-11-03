"""Lightweight document kind inference from titles/signatures (Stage‑1 only).

This module provides a heuristic function `infer_doc_kind` which classifies
documents into a small, fixed set of ASCII identifiers based on Polish titles
and signatures. Intended for on-the-fly use in browse operations without changing
the Qdrant schema. The output values are stable ASCII keys suitable for API
filters; UI may map them to localized labels.

Kinds:
- resolution   → uchwała
- order        → zarządzenie
- announcement → komunikat
- notice       → obwieszczenie
- decision     → decyzja
- regulation   → regulamin
- policy       → polityka
- procedure    → procedura
- instruction  → instrukcja
- statute      → statut
- other        → fallback when no rule matches
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Optional


def _strip_diacritics(text: str) -> str:
    """Return a lowercase, accent-stripped version of the text."""
    if not isinstance(text, str):
        return ""
    normalized = unicodedata.normalize("NFD", text)
    stripped = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return stripped.lower()


def _normalize_tokens(values: Iterable[str]) -> str:
    parts = [v for v in (values or []) if isinstance(v, str) and v.strip()]
    if not parts:
        return ""
    return _strip_diacritics(" | ".join(parts))


# Order matters: first match wins. Anchor at title start.
_DOC_KIND_RULES: list[tuple[str, str]] = [
    # uchwala (including variants, abbreviations) — diacritics stripped earlier
    (r"^\s*(uchwal[ao]\b|uchw\.)", "resolution"),
    # zarzadzenie (variants, abbrev) — diacritics stripped earlier
    (r"^\s*(zarzadzen\w*\b|zarz\.)", "order"),
    # komunikat
    (r"^\s*komunikat\b", "announcement"),
    # obwieszczenie
    (r"^\s*obwieszczen\w*\b", "notice"),
    # decyzja
    (r"^\s*decyzj\w*\b", "decision"),
    # regulamin (put high in order to beat generic forms)
    (r"^\s*regulamin\w*\b", "regulation"),
    # polityka
    (r"^\s*polityk\w*\b", "policy"),
    # procedura
    (r"^\s*procedur\w*\b", "procedure"),
    # instrukcja
    (r"^\s*instrukcj\w*\b", "instruction"),
    # statut
    (r"^\s*statut\b", "statute"),
]


def infer_doc_kind(title: Optional[str] = None, signature: Optional[Iterable[str]] = None) -> str:
    """Infer a coarse document kind identifier using title only.

    Per requirement: kind detection must rely solely on titles; signature and
    entities are ignored for this purpose.
    """
    title_norm = _strip_diacritics(title or "")
    if not title_norm:
        return "other"
    for pattern, kind in _DOC_KIND_RULES:
        if re.search(pattern, title_norm):
            return kind
    if title_norm.startswith("regulamin"):
        return "regulation"
    return "other"
