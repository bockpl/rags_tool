"""Token-aware chunking and section-aware segmentation."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from .parsing import split_into_paragraphs

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


def chunk_text(text: str, target_tokens: int = 900, overlap_tokens: int = 150) -> List[str]:
    """Token-aware paragraph packing with overlap."""
    paras = split_into_paragraphs(text)
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
                        buf_tail = buf[-max(1, overlap_tokens * 4):]
                else:
                    buf_tail = buf[-max(1, overlap_tokens * 4):]
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


# Section-aware segmentation (Polish regulations)
CHAPTER_RE = re.compile(r"^\s*Rozdział\s+([IVXLC\d]+)\b.*", re.IGNORECASE)
PARAGRAPH_RE = re.compile(r"^\s*§\s*(\d+[a-zA-Z]?)\b\s*(.*)$")
ATTACHMENT_RE = re.compile(r"^\s*Załącznik(\s+nr\s+\d+)?\b.*", re.IGNORECASE)
REGULAMIN_RE = re.compile(r"^\s*REGULAMIN\b.*", re.IGNORECASE)

# Enumeration markers within paragraphs
ENUM_NUM_RE = re.compile(r"^\s*(\d{1,3})[\.)]\s+(.*)$")
ENUM_LIT_RE = re.compile(r"^\s*(?:lit\.)?\s*([a-ząćęłńóśźż])[\)]\s+(.*)$", re.IGNORECASE)
ENUM_ROM_RE = re.compile(r"^\s*((?:[ivx]+|[IVX]+))[\)]\s+(.*)$")
ENUM_DASH_RE = re.compile(r"^\s*[•\-–—]\s+(.*)$")


def _segment_polish_sections(text: str) -> List[Tuple[str, str]]:
    lines = text.splitlines()
    n = len(lines)
    sections: List[Tuple[str, str]] = []

    current_top: Optional[str] = None
    current_label: Optional[str] = None
    buffer: List[str] = []

    def flush():
        nonlocal buffer, current_label
        if buffer:
            body = "\n".join(buffer).strip()
            if body:
                label = current_label or current_top or "Preambuła"
                sections.append((label, body))
        buffer = []

    i = 0
    while i < n:
        line = lines[i]
        if ATTACHMENT_RE.match(line or ""):
            flush()
            current_top = line.strip()
            current_label = current_top
            i += 1
            continue
        m_ch = CHAPTER_RE.match(line or "")
        if m_ch:
            flush()
            subtitle = ""
            j = i + 1
            while j < n and not (lines[j] or "").strip():
                j += 1
            if j < n:
                nxt = lines[j]
                if not (PARAGRAPH_RE.match(nxt or "") or CHAPTER_RE.match(nxt or "") or ATTACHMENT_RE.match(nxt or "")):
                    subtitle = nxt.strip()
            top = line.strip()
            if subtitle:
                top = f"{top} — {subtitle}"
            current_top = top
            current_label = current_top
            i += 1
            continue
        m_par = PARAGRAPH_RE.match(line or "")
        if m_par:
            flush()
            para_no = m_par.group(1)
            tail = (m_par.group(2) or "").strip()
            lab = f"§ {para_no}"
            if tail:
                lab = f"{lab} — {tail}"
            current_label = f"{current_top} {lab}" if current_top else lab
            i += 1
            continue
        if REGULAMIN_RE.match(line or ""):
            flush()
            current_top = "REGULAMIN"
            current_label = current_top
            i += 1
            continue
        buffer.append(line)
        i += 1
    flush()
    if not sections:
        return [("Preambuła", text)]
    return sections


def _match_enum_marker(line: str) -> Optional[Tuple[str, str, str]]:
    m = ENUM_NUM_RE.match(line)
    if m:
        return ("num", m.group(1), m.group(2))
    m = ENUM_LIT_RE.match(line)
    if m:
        return ("lit", m.group(1).lower(), m.group(2))
    m = ENUM_ROM_RE.match(line)
    if m:
        return ("rom", m.group(1).lower(), m.group(2))
    m = ENUM_DASH_RE.match(line)
    if m:
        return ("dash", "dash", m.group(1))
    return None


def _should_split(count: int, contents: List[str], min_count: int = 2, min_chars: int = 20) -> bool:
    if count < min_count:
        return False
    long_enough = sum(1 for c in contents if len((c or "").strip()) >= min_chars)
    return long_enough >= min_count


def _split_enumerations_in_paragraph(body: str) -> List[Tuple[str, str]]:
    lines = [l.rstrip() for l in body.splitlines()]
    items: List[Tuple[str, List[str]]] = []
    current: Optional[Tuple[str, List[str]]] = None
    for line in lines:
        m = _match_enum_marker(line)
        if m:
            if current is not None:
                items.append(current)
            kind, tag, rest = m
            label = f"{tag})" if kind in {"num", "lit", "rom"} else "•"
            current = (label, [rest])
        else:
            if current is None:
                current = ("", [line])
            else:
                current[1].append(line)
    if current is not None:
        items.append(current)
    contents = ["\n".join(v).strip() for _, v in items]
    if not _should_split(len(items), contents):
        return []
    results: List[Tuple[str, str]] = []
    if any(lbl for lbl, _ in items):
        for lbl, txt in zip([lbl for lbl, _ in items], contents):
            results.append((lbl, txt))
    else:
        results = []
    final = [(suffix, txt.strip()) for (suffix, txt) in results if (txt or "").strip()]
    return final


def chunk_text_by_sections(
    text: str,
    target_tokens: int = 900,
    overlap_tokens: int = 150,
) -> List[Dict[str, Any]]:
    segments = _segment_polish_sections(text)
    out: List[Dict[str, Any]] = []
    for label, body in segments:
        if "§" in label:
            enum_splits = _split_enumerations_in_paragraph(body)
            if enum_splits:
                for suffix, subtext in enum_splits:
                    full_label = f"{label} {suffix}".strip()
                    parts = chunk_text(subtext, target_tokens=target_tokens, overlap_tokens=overlap_tokens)
                    for p in parts:
                        out.append({"text": p, "section": full_label})
                continue
        parts = chunk_text(body, target_tokens=target_tokens, overlap_tokens=overlap_tokens)
        for p in parts:
            out.append({"text": p, "section": label})
    return out

