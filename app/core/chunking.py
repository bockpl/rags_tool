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

try:
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover - optional import
    tiktoken = None  # type: ignore

_TOKENIZER = None
if tiktoken is not None:  # pragma: no cover - runtime path
    try:
        _TOKENIZER = tiktoken.get_encoding("cl100k_base")
    except Exception:
        _TOKENIZER = None

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
    if not text:
        return 0
    if _TOKENIZER is None:
        return max(1, len(text) // 4)  # heuristic ~4 chars/token
    try:
        return len(_TOKENIZER.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def _split_text_by_tokens(text: str, target_tokens: int, overlap_tokens: int) -> List[str]:
    """Split a long text into token windows with overlap using tiktoken when possible."""
    chunks: List[str] = []
    if _TOKENIZER is not None:
        try:
            toks = _TOKENIZER.encode(text)
            if not toks:
                return []
            start = 0
            n = len(toks)
            while start < n:
                end = min(n, start + target_tokens)
                piece = _TOKENIZER.decode(toks[start:end])
                if piece.strip():
                    chunks.append(piece)
                if end >= n:
                    break
                start = end - max(0, overlap_tokens)
            return chunks
        except Exception:
            pass
    # Char-based fallback (heuristic ~4 chars/token)
    token_to_char = 4
    target_chars = target_tokens * token_to_char
    overlap_chars = max(0, overlap_tokens) * token_to_char
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + target_chars)
        piece = text[start:end]
        if piece.strip():
            chunks.append(piece)
        if end >= n:
            break
        start = end - overlap_chars
    return chunks


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
                if _TOKENIZER is not None:
                    try:
                        toks = _TOKENIZER.encode(buf)
                        tail = toks[-overlap_tokens:] if len(toks) > overlap_tokens else toks
                        buf_tail = _TOKENIZER.decode(tail)
                    except Exception:
                        buf_tail = buf[-max(1, overlap_tokens * 4) :]
                else:
                    buf_tail = buf[-max(1, overlap_tokens * 4) :]
            buf = (buf_tail + "\n\n" + p) if buf_tail else p
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
    merge_up_to: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Chunk text by detected sections while keeping payload compatible with Qdrant.

    Parameters
    ----------
    merge_up_to : Optional[str], default=None
        If provided, specifies the lowest hierarchy level (one of the values from
        ``SECTION_HIERARCHY``) at which consecutive chunks should be merged.
        For example, ``merge_up_to="par"`` will merge chunks that belong to the
        same attachment, regulamin, chapter, and paragraph, ignoring deeper levels
        such as ust, pkt, lit, etc.  When ``None`` (default) the original
        behaviour – merging only when the full section label matches – is kept.
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

    # Build a list of chunk dicts, preserving the full hierarchy for each chunk
    out: List[Dict[str, Any]] = []
    for section_span in section_spans:
        body = section_span.text
        if not body or not body.strip():
            continue
        label = getattr(section_span._, "section_label", None) or "Dokument"
        hierarchy = getattr(section_span._, "section_hierarchy", {}) or {}
        parts = chunk_text(body, target_tokens=target_tokens, overlap_tokens=overlap_tokens)
        for part in parts:
            if not part.strip():
                continue
            out.append(
                {
                    "text": part,
                    "section": label,
                    "hierarchy": hierarchy,
                }
            )

    # Merge consecutive chunks according to the requested hierarchy level
    merged: List[Dict[str, Any]] = []
    level_idx: Optional[int] = None
    if merge_up_to:
        try:
            level_idx = SECTION_HIERARCHY.index(merge_up_to)
        except ValueError:
            level_idx = None

    if level_idx is not None:
        grouped: Dict[tuple, Dict[str, Any]] = {}
        order: List[tuple] = []

        for item in out:
            hierarchy = item.get("hierarchy", {})
            cur_key = tuple(
                hierarchy.get(level) for level in SECTION_HIERARCHY[: level_idx + 1]
            )
            if cur_key not in grouped:
                order.append(cur_key)
                trimmed_hierarchy = {
                    level: hierarchy.get(level) if idx <= level_idx else None
                    for idx, level in enumerate(SECTION_HIERARCHY)
                }
                grouped[cur_key] = {
                    "section": _format_section_label(trimmed_hierarchy),
                    "texts": [item["text"]],
                }
            else:
                grouped[cur_key]["texts"].append(item["text"])

        for key in order:
            bucket = grouped[key]
            label = bucket["section"]
            partial: Optional[str] = None
            for piece in bucket["texts"]:
                normalized = piece if partial is None else partial.rstrip() + "\n\n" + piece.lstrip()
                if partial is None:
                    partial = piece
                    continue
                if count_tokens(normalized) <= target_tokens:
                    partial = normalized
                else:
                    merged.append({"text": partial, "section": label})
                    partial = piece
            if partial:
                merged.append({"text": partial, "section": label})
    else:
        prev_key: Optional[tuple] = None
        for item in out:
            cur_key = (item["section"],)
            if merged and prev_key == cur_key:
                merged[-1]["text"] = (
                    merged[-1]["text"].rstrip() + "\n\n" + item["text"].lstrip()
                )
            else:
                merged.append(
                    {
                        "text": item["text"],
                        "section": item["section"],
                    }
                )
            prev_key = cur_key

    return merged or _fallback()
