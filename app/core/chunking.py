"""Token-aware chunking and section-aware segmentation."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

try:
    import spacy
    from spacy.language import Language
    from spacy.matcher import Matcher
    from spacy.tokens import Doc, Span
except ImportError:
    spacy = None  # type: ignore

from app.core.tokenizer import (
    TokenizerAdapter,
    count_tokens as _adapter_count_tokens,
    load_tokenizer,
    sliding_windows,
)
from app.settings import get_settings

_SETTINGS = get_settings()
_TOKENIZER_ADAPTER: TokenizerAdapter = load_tokenizer(getattr(_SETTINGS, "embedding_tokenizer", None))

SECTION_HIERARCHY: Tuple[str, ...] = (
    "chapter",
    "par",
    "ust",
    "pkt",
    "lit",
)

CHAPTER_KEYWORDS: Tuple[str, ...] = ("Załącznik", "REGULAMIN", "Rozdział")
_CHAPTER_KEYWORDS_LOWER = {kw.lower() for kw in CHAPTER_KEYWORDS}


def _clean_value(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    value = value.strip()
    return value or None


def _extract_number(text: str) -> Optional[str]:
    match = re.search(r"(\d+[a-z]?)", text, re.I)
    return match.group(1) if match else None


def _extract_letter(text: str) -> Optional[str]:
    match = re.search(r"([a-z])", text, re.I)
    return match.group(1).lower() if match else None


def _format_level(level: str, value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    value = value.strip()
    if not value:
        return None
    if level == "chapter":
        return value
    if level == "par":
        value = value.lstrip("§").strip()
        return f"§ {value}" if value else None
    if level == "ust":
        value = value.rstrip(".)").strip()
        return f"ust. {value}" if value else None
    if level == "pkt":
        value = value.rstrip(".)").strip()
        return f"pkt {value}" if value else None
    if level == "lit":
        value = value.rstrip(".)").strip().lower()
        return f"lit. {value}" if value else None
    return None


def _format_section_label(hierarchy: Dict[str, Optional[str]]) -> str:
    parts: List[str] = []
    for level in SECTION_HIERARCHY:
        formatted = _format_level(level, hierarchy.get(level))
        if formatted:
            parts.append(formatted)
    return " ".join(parts).strip() or "Preambuła"


def _is_chapter_token(text: str) -> bool:
    if not text:
        return False
    if not text[0].isupper():
        return False
    return text.lower() in _CHAPTER_KEYWORDS_LOWER


def _expand_span_to_line(doc: "Doc", span: "Span") -> Optional["Span"]:
    if span is None:
        return None
    start_char = span.start_char
    end_char = span.end_char
    text = doc.text
    n = len(text)
    while end_char < n and text[end_char] not in "\r\n":
        end_char += 1
    expanded = doc.char_span(start_char, end_char, alignment_mode="expand")
    return expanded


def count_tokens(text: str) -> int:
    return _adapter_count_tokens(_TOKENIZER_ADAPTER, text)


def _split_text_by_tokens(text: str, target_tokens: int, overlap_tokens: int) -> List[str]:
    return [chunk for chunk in sliding_windows(_TOKENIZER_ADAPTER, text, target_tokens, overlap_tokens) if chunk.strip()]


def chunk_text(
    text: str,
    target_tokens: Optional[int] = None,
    overlap_tokens: Optional[int] = None,
) -> List[str]:
    """Token-aware paragraph packing with overlap."""
    if target_tokens is None or overlap_tokens is None:
        try:
            from app.settings import get_settings

            s = get_settings()
            if target_tokens is None:
                target_tokens = int(s.chunk_tokens)
            if overlap_tokens is None:
                overlap_tokens = int(s.chunk_overlap)
        except Exception:
            target_tokens = 900 if target_tokens is None else target_tokens
            overlap_tokens = 150 if overlap_tokens is None else overlap_tokens
    try:
        from .parsing import split_into_paragraphs

        paras = split_into_paragraphs(text)
    except ImportError:
        paras = text.split("\n\n")

    chunks: List[str] = []
    buf = ""
    for p in paras:
        candidate = (buf + "\n\n" + p) if buf else p
        if count_tokens(candidate) <= target_tokens:
            buf = candidate
            continue
        if buf:
            chunks.append(buf)
            buf_tail = ""
            if overlap_tokens > 0:
                toks = _TOKENIZER_ADAPTER.encode(buf)
                body_overlap = max(0, overlap_tokens - _TOKENIZER_ADAPTER.extra_token_count)
                if body_overlap > 0:
                    tail_tokens = toks[-body_overlap:] if len(toks) > body_overlap else toks
                    buf_tail = _TOKENIZER_ADAPTER.decode(tail_tokens)
            new_buf = (buf_tail + "\n\n" + p) if buf_tail else p
            # Ensure the new buffer does not exceed the target; if it does,
            # proactively slice it into windows, finalize all but the last,
            # and keep the last window in the buffer for further packing.
            if count_tokens(new_buf) <= target_tokens:
                buf = new_buf
            else:
                parts = _split_text_by_tokens(new_buf, target_tokens, overlap_tokens)
                if parts:
                    chunks.extend(part for part in parts[:-1] if part.strip())
                    buf = parts[-1]
                else:
                    buf = ""
        else:
            long_parts = _split_text_by_tokens(p, target_tokens, overlap_tokens)
            chunks.extend(long_parts)
            buf = ""
    if buf:
        if count_tokens(buf) <= target_tokens:
            chunks.append(buf)
        else:
            chunks.extend(_split_text_by_tokens(buf, target_tokens, overlap_tokens))
    return chunks


_NLP_PIPELINE: Optional[Language] = None


def _get_nlp_pipeline() -> Optional[Language]:
    """Loads and caches the spaCy NLP pipeline with the custom section annotator."""
    global _NLP_PIPELINE
    if _NLP_PIPELINE:
        return _NLP_PIPELINE
    if spacy is None:
        raise ImportError(
            "spaCy is not installed. Please run `pip install spacy` and `python -m spacy download pl_core_news_sm`."
        )

    if not Span.has_extension("section_hierarchy"):
        Span.set_extension("section_hierarchy", default=None)
    if not Span.has_extension("section_label"):
        Span.set_extension("section_label", default=None)

    @Language.component("section_annotator_v2")
    def section_annotator_v2(doc: Doc) -> Doc:
        patterns = {
            "chapter": [[{"TEXT": {"REGEX": r"^(Załącznik|REGULAMIN|Rozdział)$"}}]],
            "par": [[{"TEXT": "§"}, {"IS_SPACE": True, "OP": "*"}, {"LIKE_NUM": True}]],
            "ust": [[{"IS_DIGIT": True}, {"TEXT": "."}]],
            "pkt": [[{"LIKE_NUM": True}, {"TEXT": ")"}]],
            "lit": [[{"IS_ALPHA": True, "LENGTH": 1}, {"TEXT": ")"}]],
        }

        extractors = {
            "chapter": lambda s: _clean_value(s.text),
            "par": lambda s: _extract_number(s.text),
            "ust": lambda s: _extract_number(s.text),
            "pkt": lambda s: _extract_number(s.text),
            "lit": lambda s: _extract_letter(s.text),
        }

        matcher = Matcher(doc.vocab)
        for name, p_list in patterns.items():
            matcher.add(name, p_list)

        matches: List[Tuple[str, Span]] = []
        for match_id, start, end in matcher(doc):
            span = doc[start:end]
            name = doc.vocab.strings[match_id]

            if name == "chapter":
                token_text = span[0].text if len(span) else ""
                if not _is_chapter_token(token_text):
                    continue
                expanded = _expand_span_to_line(doc, span)
                if not expanded or not expanded.text.strip():
                    continue
                span = expanded

            is_line_start = span.start == 0 or doc.text[span.start_char - 1] in "\n\r"
            if is_line_start:
                matches.append((name, span))

        matches.sort(key=lambda item: item[1].start)

        current_hierarchy = {level: None for level in SECTION_HIERARCHY}
        section_spans: List[Span] = []
        last_span_end = 0

        for name, marker_span in matches:
            if marker_span.start_char > last_span_end:
                content_span = doc.char_span(
                    last_span_end,
                    marker_span.start_char,
                    alignment_mode="expand",
                )
                if content_span and content_span.text.strip():
                    content_span._.section_hierarchy = current_hierarchy.copy()
                    section_spans.append(content_span)

            # Skip markers not defined in SECTION_HIERARCHY (e.g., "dash")
            if name not in SECTION_HIERARCHY:
                continue
            level_index = SECTION_HIERARCHY.index(name)
            value = extractors[name](marker_span)
            current_hierarchy[name] = value
            for downstream in SECTION_HIERARCHY[level_index + 1 :]:
                current_hierarchy[downstream] = None

            last_span_end = marker_span.end_char

        if last_span_end < len(doc.text):
            content_span = doc.char_span(last_span_end, len(doc.text), alignment_mode="expand")
            if content_span and content_span.text.strip():
                content_span._.section_hierarchy = current_hierarchy.copy()
                section_spans.append(content_span)

        for span in section_spans:
            hierarchy = getattr(span._, "section_hierarchy", None) or {}
            span._.section_label = _format_section_label(hierarchy)

        doc.spans["sections"] = section_spans
        return doc

    try:
        nlp = spacy.blank("pl")
        nlp.add_pipe("section_annotator_v2", last=True)
        _NLP_PIPELINE = nlp
        return _NLP_PIPELINE
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Error initializing spaCy pipeline: {exc}")
        return None


def chunk_text_by_sections(
    text: str,
    target_tokens: Optional[int] = None,
    overlap_tokens: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Split text into sections and produce raw, non-merged chunks.

    This function detects section boundaries and then applies token-aware
    packing within each section. It returns the smallest chunk units with a
    `section` label, without any merging. Merging of full sections is performed
    at search-time when shaping results for presentation.
    """

    if target_tokens is None or overlap_tokens is None:
        try:
            from app.settings import get_settings

            s = get_settings()
            if target_tokens is None:
                target_tokens = int(s.chunk_tokens)
            if overlap_tokens is None:
                overlap_tokens = int(s.chunk_overlap)
        except Exception:
            target_tokens = 900 if target_tokens is None else target_tokens
            overlap_tokens = 150 if overlap_tokens is None else overlap_tokens

    def _fallback() -> List[Dict[str, Any]]:
        return [
            {"text": part, "section": "Dokument"}
            for part in chunk_text(
                text, target_tokens=target_tokens, overlap_tokens=overlap_tokens
            )
            if part.strip()
        ]

    try:
        nlp = _get_nlp_pipeline()
    except ImportError:
        return _fallback()

    if nlp is None:
        return _fallback()

    doc = nlp(text)
    span_group = doc.spans.get("sections", []) if hasattr(doc.spans, "get") else []
    section_spans = list(span_group)
    if not section_spans:
        return _fallback()

    # Build a list of chunk dicts (raw, non-merged)
    out: List[Dict[str, Any]] = []
    for section_span in section_spans:
        body = section_span.text
        if not body or not body.strip():
            continue
        label = getattr(section_span._, "section_label", None) or "Dokument"
        parts = chunk_text(body, target_tokens=target_tokens, overlap_tokens=overlap_tokens)
        for part in parts:
            if not part.strip():
                continue
            out.append(
                {
                    "text": part,
                    "section": label,
                }
            )
    return out or _fallback()
