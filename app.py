"""
RAGS_tool — dwustopniowy RAG ze streszczeniami (FastAPI + Qdrant)
Version: 0.9.0
Author: Seweryn Sitarski (seweryn.sitarski@gmail.com) with support from Kat
License: MIT

ZAŁOŻENIA
- Narzędzie OpenAPI dla LLM (OpenWebUI / OpenAI Tools):
  • /about — metadane
  • /health — zdrowie serwisu
  • /collections/init — utworzenie/aktualizacja kolekcji w Qdrant
  • /ingest/scan — skan katalogu
  • /ingest/build — przetworzenie korpusu: parsowanie → streszczenia → wektory → indeks
  • /summaries/generate — streszczenia dla pojedynczych plików
  • /search/query — główne odpytywanie (dwustopniowe): streszczenia → pełny tekst (z hybrydą dense+tfidf)

- Wejście: katalog z dokumentami (TXT/MD/HTML; PDF opcjonalnie jeśli dostępny PyPDF2) + instancja Qdrant.
- Embedding i streszczenia: modele dostępne przez protokół OpenAI (np. vLLM/Ollama z endpointem OpenAI).
- Indeks Qdrant jako multivektor: "content_dense", "summary_dense" + (opcjonalnie) sparse TF‑IDF.
- Brak re-rankera cross-encoder (opcjonalny TODO) — używamy fuzji wyników i MMR.

URUCHOMIENIE
  export QDRANT_URL=http://127.0.0.1:6333
  export QDRANT_API_KEY=""
  export EMBEDDING_API_URL=http://127.0.0.1:8000/v1   # Twój endpoint embedding
  export EMBEDDING_API_KEY=sk-embed-xxx
  export EMBEDDING_MODEL="BAAI/bge-m3"               # lub inny zgodny z /v1/embeddings
  export SUMMARY_API_URL=http://127.0.0.1:8001/v1     # Twój endpoint do streszczeń
  export SUMMARY_API_KEY=sk-summary-xxx
  export SUMMARY_MODEL="gpt-4o-mini"                 # lub lokalny chat model zgodny z /v1/chat/completions
  export COLLECTION_NAME="rags_tool"
  export VECTOR_STORE_DIR=".rags_tool_store"         # gdzie zapisze się TF‑IDF, cache itp.
  export DEBUG=false                                  # ustaw na true, by zobaczyć logi ingestu

  # (alternatywnie) wpisz te wartości do pliku .env — aplikacja wczyta go automatycznie

  pip install fastapi uvicorn qdrant-client openai pydantic pydantic-settings tiktoken scikit-learn markdown2 beautifulsoup4 html2text
  # PyPDF2 (opcjonalnie): pip install pypdf2

  uvicorn app:app --host 0.0.0.0 --port 8080

OpenAPI będzie pod /openapi.json, Swagger UI pod /docs — gotowe do wpięcia w OpenWebUI Tools (OpenAPI).
"""

import json
import time
import uuid
import hashlib
import pathlib
import logging
import sys
from typing import List, Optional, Dict, Any, Tuple

from fastapi import FastAPI, Body, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, Field

# Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from qdrant_client.http.exceptions import UnexpectedResponse

# OpenAI‑compatible klienci
from openai import OpenAI

# Tekst / parsowanie
import re
import html
import html2text
import markdown2
try:
    import tiktoken  # token-aware chunking
except Exception:  # optional dependency at runtime (fallback available)
    tiktoken = None  # type: ignore

# PDF (opcjonalne)
try:
    import PyPDF2  # type: ignore
    HAS_PDF = True
except Exception:
    HAS_PDF = False

# TF‑IDF (sparse hybryda)
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

from settings import get_settings

# ──────────────────────────────────────────────────────────────────────────────
# KONFIG
# ──────────────────────────────────────────────────────────────────────────────
settings = get_settings()
VECTOR_STORE_DIR = settings.vector_store_dir
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

# Stałe
CONTENT_VECTOR_NAME = "content_dense"
SUMMARY_VECTOR_NAME = "summary_dense"
CONTENT_SPARSE_NAME = "content_sparse"
SUMMARY_SPARSE_NAME = "summary_sparse"
SPARSE_ENABLED = True  # TF‑IDF hybryda

# Logger skonfigurowany pod konsolę kontenera
logger = logging.getLogger("rags_tool")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )
    logger.addHandler(handler)
logger.setLevel(logging.DEBUG if settings.debug else logging.INFO)
logger.propagate = False

ADMIN_UI_REQUEST_HEADER = "x-admin-ui"
_admin_debug_activated = False

## Admin UI template moved to templates/admin.html

def _scan_files(base: pathlib.Path, pattern: str, recursive: bool) -> List[pathlib.Path]:
    """Scan for files by pattern with optional recursion and supported extensions filter."""
    iterator = base.rglob(pattern) if recursive else base.glob(pattern)
    return [p for p in iterator if p.is_file() and p.suffix.lower() in SUPPORTED_EXT]

def _collect_documents(file_paths: List[pathlib.Path], chunk_tokens: int, chunk_overlap: int) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    """Parse files, split into section-aware chunks, summarize and prepare corpora."""
    all_chunks: List[str] = []
    summary_corpus: List[str] = []
    doc_records: List[Dict[str, Any]] = []

    for path in file_paths:
        logger.debug("Processing document %s", path)
        raw = extract_text(path)
        # Prefer section-aware chunking for regulation-like documents
        chunk_items = chunk_text_by_sections(raw, target_tokens=chunk_tokens, overlap_tokens=chunk_overlap)
        chunks = chunk_items
        if not chunks:
            logger.debug("Document %s produced no chunks; skipping", path)
            continue
        doc_sum = llm_summary(raw[:12000])
        doc_id = sha1(str(path.resolve()))
        summary_signature = doc_sum.get("signature", [])
        summary_sparse_text = " ".join([doc_sum.get("summary", ""), " ".join(summary_signature)]).strip()
        rec = {
            "doc_id": doc_id,
            "path": str(path.resolve()),
            "chunks": chunks,
            "doc_summary": doc_sum.get("summary", ""),
            "doc_signature": summary_signature,
            "summary_sparse_text": summary_sparse_text,
        }
        doc_records.append(rec)
        all_chunks.extend([c.get("text", "") for c in chunks])
        if summary_sparse_text:
            summary_corpus.append(summary_sparse_text)
        logger.debug(
            "Document %s parsed | chunks=%d summary_len=%d",
            path,
            len(chunks),
            len(rec["doc_summary"] or ""),
        )

    return doc_records, all_chunks, summary_corpus


def _prepare_tfidf(
    all_chunks: List[str],
    summary_corpus: List[str],
    enable_sparse: bool,
    rebuild_tfidf: bool,
) -> Tuple[Optional[TfidfVectorizer], Optional[TfidfVectorizer]]:
    """Fit or load TF-IDF vectorizers for content and summaries."""
    if not enable_sparse:
        return None, None
    content_vec: Optional[TfidfVectorizer] = None
    summary_vec: Optional[TfidfVectorizer] = None
    logger.debug("Sparse mode enabled; rebuild=%s", rebuild_tfidf)
    if all_chunks:
        if rebuild_tfidf or not VECTORIZER_PATH.exists():
            content_vec = fit_vectorizer(all_chunks)
            logger.debug("Fitted new TF-IDF vectorizer over %d chunks", len(all_chunks))
        else:
            content_vec = load_vectorizer()
            if content_vec is None:
                content_vec = fit_vectorizer(all_chunks)
                logger.debug("Rebuilt TF-IDF vectorizer (cache missing) over %d chunks", len(all_chunks))
            else:
                logger.debug("Loaded existing TF-IDF vectorizer")
    else:
        logger.debug("No chunks available for TF-IDF fitting; reusing existing vectorizer if present")
        content_vec = load_vectorizer()

    if summary_corpus:
        if rebuild_tfidf or not SUMMARY_VECTORIZER_PATH.exists():
            summary_vec = fit_vectorizer(summary_corpus, path=SUMMARY_VECTORIZER_PATH)
            logger.debug("Fitted new summary TF-IDF vectorizer over %d summaries", len(summary_corpus))
        else:
            summary_vec = load_vectorizer(path=SUMMARY_VECTORIZER_PATH)
            if summary_vec is None:
                summary_vec = fit_vectorizer(summary_corpus, path=SUMMARY_VECTORIZER_PATH)
                logger.debug(
                    "Rebuilt summary TF-IDF vectorizer (cache missing) over %d summaries",
                    len(summary_corpus),
                )
            else:
                logger.debug("Loaded existing summary TF-IDF vectorizer")
    else:
        logger.debug("No summaries available for TF-IDF fitting; reusing existing summary vectorizer if present")
        summary_vec = load_vectorizer(path=SUMMARY_VECTORIZER_PATH)

    return content_vec, summary_vec


def _build_and_upsert_points(
    doc_records: List[Dict[str, Any]],
    content_vec: Optional[TfidfVectorizer],
    summary_vec: Optional[TfidfVectorizer],
    *,
    enable_sparse: bool,
    collection_name: str,
) -> int:
    """Create Qdrant points for all documents and upsert in batches."""
    points: List[qm.PointStruct] = []
    point_count = 0

    for rec in doc_records:
        doc_id = rec["doc_id"]
        path = rec["path"]
        chunks = rec["chunks"]
        doc_summary = rec["doc_summary"]
        doc_signature = rec["doc_signature"]
        summary_sparse_text = rec["summary_sparse_text"]

        summary_dense_vec = embed_text([doc_summary])[0]
        content_texts = [c.get("text", c) if isinstance(c, dict) else str(c) for c in chunks]
        content_vecs = embed_text(content_texts)

        if enable_sparse:
            # TF-IDF expects plain strings, not dict items
            sparse_chunks = tfidf_vector(content_texts, content_vec)
            if summary_vec is not None and summary_sparse_text:
                summary_sparse = tfidf_vector([summary_sparse_text], summary_vec, path=SUMMARY_VECTORIZER_PATH)[0]
            else:
                summary_sparse = ([], [])
        else:
            sparse_chunks = [([], []) for _ in chunks]
            summary_sparse = ([], [])

        for i, chunk_item in enumerate(chunks):
            if isinstance(chunk_item, dict):
                chunk_text_val = chunk_item.get("text", "")
                section_label = chunk_item.get("section")
            else:
                chunk_text_val = str(chunk_item)
                section_label = None
            pid = int(str(int(sha1(f"{doc_id}:{i}")[0:12], 16))[:12])
            payload = {
                "doc_id": doc_id,
                "path": path,
                "chunk_id": i,
                "is_active": True,
                "summary": doc_summary,
                "text": chunk_text_val,
                "signature": doc_signature,
            }
            if section_label:
                payload["section"] = section_label

            vectors: Dict[str, Any] = {
                CONTENT_VECTOR_NAME: content_vecs[i],
                SUMMARY_VECTOR_NAME: summary_dense_vec,
            }

            if SPARSE_ENABLED and enable_sparse:
                indices, values = sparse_chunks[i]
                if indices:
                    vectors[CONTENT_SPARSE_NAME] = qm.SparseVector(indices=indices, values=values)
                    payload["content_sparse_indices"] = indices
                    payload["content_sparse_values"] = values
                if summary_sparse[0]:
                    vectors[SUMMARY_SPARSE_NAME] = qm.SparseVector(indices=summary_sparse[0], values=summary_sparse[1])
                    payload["summary_sparse_indices"] = summary_sparse[0]
                    payload["summary_sparse_values"] = summary_sparse[1]

            points.append(qm.PointStruct(id=pid, vector=vectors, payload=payload))
            point_count += 1

        if len(points) >= 1024:
            qdrant.upsert(collection_name=collection_name, points=points)
            logger.debug("Upserted batch of %d points into %s", len(points), collection_name)
            points = []

    if points:
        qdrant.upsert(collection_name=collection_name, points=points)
        logger.debug("Upserted final batch of %d points into %s", len(points), collection_name)

    return point_count


ADMIN_OPERATION_SPECS: List[Dict[str, Any]] = [
    {"id": "about", "path": "/about", "method": "GET"},
    {"id": "health", "path": "/health", "method": "GET"},
    {
        "id": "collections-init",
        "path": "/collections/init",
        "method": "POST",
        "body": "{\n  \"collection_name\": \"rags_tool\",\n  \"force_dim_probe\": false\n}",
    },
    {
        "id": "ingest-scan",
        "path": "/ingest/scan",
        "method": "POST",
        "body": "{\n  \"base_dir\": \"/app/data\",\n  \"glob\": \"**/*\",\n  \"recursive\": true\n}",
    },
    {
        "id": "summaries-generate",
        "path": "/summaries/generate",
        "method": "POST",
        "body": "{\n  \"files\": [\n    \"/app/data/example.md\"\n  ]\n}",
    },
    {
        "id": "ingest-build",
        "path": "/ingest/build",
        "method": "POST",
        "body": "{\n  \"base_dir\": \"/app/data\",\n  \"glob\": \"**/*\",\n  \"recursive\": true,\n  \"reindex\": false,\n  \"chunk_tokens\": 900,\n  \"chunk_overlap\": 150,\n  \"collection_name\": \"rags_tool\",\n  \"enable_sparse\": true,\n  \"rebuild_tfidf\": true\n}",
    },
    {
        "id": "search-query",
        "path": "/search/query",
        "method": "POST",
        "body": "{\n  \"query\": \"Jak działa rags_tool?\",\n  \"top_m\": 10,\n  \"top_k\": 5,\n  \"mode\": \"auto\",\n  \"use_hybrid\": true,\n  \"dense_weight\": 0.6,\n  \"sparse_weight\": 0.4,\n  \"mmr_lambda\": 0.3,\n  \"per_doc_limit\": 2,\n  \"score_norm\": \"minmax\",\n  \"rep_alpha\": 0.6,\n  \"mmr_stage1\": true,\n  \"summary_mode\": \"first\",\n  \"merge_chunks\": true,\n  \"merge_group_budget_tokens\": 1200,\n  \"max_merged_per_group\": 1,\n  \"expand_neighbors\": 1,\n  \"result_format\": \"blocks\"\n}",
    },
]

# MMR
DEFAULT_MMR_LAMBDA = 0.3
DEFAULT_PER_DOC_LIMIT = 2
DEFAULT_SCORE_NORM = "minmax"  # minmax|zscore|none

# ──────────────────────────────────────────────────────────────────────────────
# INICJALIZACJA KLIENTÓW
# ──────────────────────────────────────────────────────────────────────────────
qdrant = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
embedding_client = OpenAI(base_url=settings.embedding_api_url, api_key=settings.embedding_api_key)
summary_client = OpenAI(base_url=settings.summary_api_url, api_key=settings.summary_api_key)

# ──────────────────────────────────────────────────────────────────────────────
# UTYLITKI
# ──────────────────────────────────────────────────────────────────────────────

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def now_ts() -> int:
    return int(time.time())


# ——— Parsowanie plików ———
SUPPORTED_EXT = {".txt", ".md", ".markdown", ".html", ".htm", ".pdf"}


def read_text_file(path: pathlib.Path) -> str:
    ext = path.suffix.lower()
    data = path.read_bytes()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("latin1", errors="ignore")
    if ext in {".md", ".markdown"}:
        # MD → tekst
        return markdown2.markdown(text)  # HTML
    return text


def html_to_text(content_html: str) -> str:
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.body_width = 0
    return h.handle(content_html)


def extract_text(path: pathlib.Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        if not HAS_PDF:
            raise HTTPException(status_code=400, detail="PDF wymaga PyPDF2 — zainstaluj lub pomiń PDF")
        txt = []
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                txt.append(page.extract_text() or "")
        return "\n".join(txt)
    elif ext in {".html", ".htm"}:
        raw = read_text_file(path)
        return html_to_text(raw)
    else:
        raw = read_text_file(path)
        # Jeśli markdown2 zwrócił HTML, zamień na tekst
        if raw.strip().lower().startswith("<") and "</" in raw:
            return html_to_text(raw)
        return raw


# ——— Chunking i streszczenia ———

def split_into_paragraphs(text: str) -> List[str]:
    # Split na podwójne nowe linie / długie akapity
    chunks = re.split(r"\n\s*\n", text)
    return [c.strip() for c in chunks if c.strip()]


# Initialize tokenizer once (token-aware chunking); fallback to heuristic if unavailable
try:
    _TOKENIZER = tiktoken.get_encoding("cl100k_base") if tiktoken is not None else None
except Exception:
    _TOKENIZER = None


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken if available, otherwise heuristic fallback.

    This keeps chunk sizes consistent with LLM limits and reduces too-small/too-large
    fragments that harm retrieval quality.
    """
    if _TOKENIZER is not None:
        try:
            return len(_TOKENIZER.encode(text))
        except Exception:
            pass
    # Fallback: ~4 chars per token
    return max(1, len(text) // 4)


def _split_text_by_tokens(text: str, target_tokens: int, overlap_tokens: int) -> List[str]:
    """Split a long text into token windows with overlap using tiktoken when possible.

    - Uses stride of (target_tokens - overlap_tokens)
    - Decodes back to text for each window
    - Falls back to character-based slicing when tokenizer is not available
    """
    chunks: List[str] = []
    if _TOKENIZER is not None:
        try:
            toks = _TOKENIZER.encode(text)
            if not toks:
                return []
            step = max(1, target_tokens - max(0, overlap_tokens))
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
            # fall through to char-based
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
    """Token-aware paragraph packing with overlap.

    - Accumulates paragraphs until token budget would be exceeded.
    - On overflow, emits current buffer and carries a token-based tail as overlap.
    - Paragraphs longer than `target_tokens` are split by tokens with overlap.
    - Falls back to character heuristic if tokenizer is unavailable.
    """
    paras = split_into_paragraphs(text)
    chunks: List[str] = []
    buf = ""

    for p in paras:
        candidate = (buf + "\n\n" + p) if buf else p
        if count_tokens(candidate) <= target_tokens:
            buf = candidate
            continue

        # Flush current buffer (if any), then start new one with token-overlap tail
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
                        # heuristic fallback
                        buf_tail = buf[-max(1, overlap_tokens * 4):]
                else:
                    buf_tail = buf[-max(1, overlap_tokens * 4):]
            buf = (buf_tail + "\n\n" + p) if buf_tail else p
        else:
            # Single paragraph too long: split by tokens with overlap
            long_parts = _split_text_by_tokens(p, target_tokens, overlap_tokens)
            chunks.extend(long_parts)
            buf = ""

    # Finalize remaining buffer
    if buf:
        if count_tokens(buf) <= target_tokens:
            chunks.append(buf)
        else:
            chunks.extend(_split_text_by_tokens(buf, target_tokens, overlap_tokens))

    return chunks


# ——— Section-aware segmentation (Polish regulations) ———

CHAPTER_RE = re.compile(r"^\s*Rozdział\s+([IVXLC\d]+)\b.*", re.IGNORECASE)
PARAGRAPH_RE = re.compile(r"^\s*§\s*(\d+[a-zA-Z]?)\b\s*(.*)$")
ATTACHMENT_RE = re.compile(r"^\s*Załącznik(\s+nr\s+\d+)?\b.*", re.IGNORECASE)
REGULAMIN_RE = re.compile(r"^\s*REGULAMIN\b.*", re.IGNORECASE)

# Enumeration (points/subpoints) markers within paragraphs
ENUM_NUM_RE = re.compile(r"^\s*(\d{1,3})[\.)]\s+(.*)$")  # e.g. '1) ...' or '2. ...'
ENUM_LIT_RE = re.compile(r"^\s*(?:lit\.)?\s*([a-ząćęłńóśźż])[\)]\s+(.*)$", re.IGNORECASE)  # 'a) ...' or 'lit. b) ...'
ENUM_ROM_RE = re.compile(r"^\s*((?:[ivx]+|[IVX]+))[\)]\s+(.*)$")  # 'i) ...' / 'IV) ...'
ENUM_DASH_RE = re.compile(r"^\s*[•\-–—]\s+(.*)$")  # bullet/tiret lines


def _segment_polish_sections(text: str) -> List[Tuple[str, str]]:
    """Heuristically segment Polish regulation-like documents into sections.

    Recognizes headers such as:
    - "Rozdział N" (optionally followed by a subtitle line)
    - Paragraph markers "§ N" (used as sub-sections under current chapter)
    - "Załącznik ..." blocks
    - "REGULAMIN" header

    Returns a list of (section_label, section_text) without crossing boundaries
    between identified sections. Falls back to a single preamble section when
    no markers are found.
    """
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

        # Attachments
        if ATTACHMENT_RE.match(line or ""):
            flush()
            current_top = line.strip()
            current_label = current_top
            i += 1
            continue

        # Chapters
        m_ch = CHAPTER_RE.match(line or "")
        if m_ch:
            flush()
            # try to capture a subtitle on the next non-empty, non-header line
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

        # Paragraphs (§)
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

        # Main header "REGULAMIN"
        if REGULAMIN_RE.match(line or ""):
            flush()
            current_top = "REGULAMIN"
            current_label = current_top
            i += 1
            continue

        # default: accumulate
        buffer.append(line)
        i += 1

    flush()

    # If no sections recognized, return single block
    if not sections:
        return [("Preambuła", text)]
    return sections


def chunk_text_by_sections(
    text: str,
    target_tokens: int = 900,
    overlap_tokens: int = 150,
) -> List[Dict[str, Any]]:
    """Split text into section-constrained chunks.

    - Detects legal-like sections (Rozdział/§/Załącznik/REGULAMIN).
    - For sections being paragraphs ("§ …"), further detects enumerations
      (points and subpoints) such as numeric points (1), letter points (a)),
      roman points (i)) and dash bullets, and splits accordingly.
    - Chunks within each (sub)section independently using token-aware packing.
    - Returns list of {"text": str, "section": str} dicts.
    """
    segments = _segment_polish_sections(text)
    out: List[Dict[str, Any]] = []
    for label, body in segments:
        # If the label corresponds to a paragraph ("§ …" appears in label),
        # attempt to split its content into enumeration-based sub-sections.
        if "§" in label:
            enum_splits = _split_enumerations_in_paragraph(body)
            if enum_splits:
                for suffix, subtext in enum_splits:
                    full_label = f"{label} {suffix}".strip()
                    parts = chunk_text(subtext, target_tokens=target_tokens, overlap_tokens=overlap_tokens)
                    for p in parts:
                        out.append({"text": p, "section": full_label})
                continue
        # Fallback for non-paragraph sections or when no enumerations found
        parts = chunk_text(body, target_tokens=target_tokens, overlap_tokens=overlap_tokens)
        for p in parts:
            out.append({"text": p, "section": label})
    return out


# ──────────────────────────────────────────────────────────────────────────────
# ENUMERATION SPLITTING — points and subpoints inside paragraph sections (§)
# ──────────────────────────────────────────────────────────────────────────────

def _match_enum_marker(line: str) -> Optional[Tuple[str, str, str]]:
    """Return tuple (kind, tag, rest) if line starts with an enumeration marker.

    kind ∈ {"num", "lit", "rom", "dash"}
    tag  = normalized label element (e.g., '3', 'b', 'iv', 'dash')
    rest = remaining text after the marker.
    """
    m = ENUM_NUM_RE.match(line)
    if m:
        return ("num", m.group(1), m.group(2))
    m = ENUM_LIT_RE.match(line)
    if m:
        return ("lit", m.group(1).lower(), m.group(2))
    m = ENUM_ROM_RE.match(line)
    if m:
        return ("rom", m.group(1), m.group(2))
    m = ENUM_DASH_RE.match(line)
    if m:
        return ("dash", "dash", m.group(1))
    return None


def _should_split(count: int, contents: List[str], min_count: int = 2, min_chars: int = 20) -> bool:
    """Heuristic: split only if there are enough items and they are not trivial."""
    if count < min_count:
        return False
    long_enough = sum(1 for s in contents if len((s or "").strip()) >= min_chars)
    return long_enough >= min_count


def _split_enumerations_in_paragraph(body: str) -> List[Tuple[str, str]]:
    """Split a paragraph ("§ …") body into logical sub-sections based on enumerations.

    Strategy (three levels):
    - Level 1: numeric points (1) 2. …) → label 'pkt X'
    - Level 2: letter points (a) b) … or 'lit. a)') → label 'lit. y)'
    - Level 3: roman (i) ii) …) or dash bullets '–'/'-'/'•' → label 'roman i)' or 'tiret N'

    Returns list of (suffix_label, text). If no sensible split is found,
    returns an empty list (caller will fall back to chunking the whole section).
    """
    lines = body.splitlines()
    # Data structures to collect hierarchy in encounter order
    nums: List[Dict[str, Any]] = []
    current_num: Optional[Dict[str, Any]] = None
    current_lit: Optional[Dict[str, Any]] = None
    current_sub: Optional[Dict[str, Any]] = None

    def new_num(tag: str, first: str):
        nonlocal current_num, current_lit, current_sub
        blk = {"tag": tag, "content": [first] if first else [], "lits": [], "subs": [], "sub_counter": 0}
        nums.append(blk)
        current_num = blk
        current_lit = None
        current_sub = None

    def new_lit(tag: str, first: str):
        nonlocal current_lit, current_sub
        if current_num is None:
            return
        blk = {"tag": tag, "content": [first] if first else [], "subs": [], "sub_counter": 0}
        current_num["lits"].append(blk)
        current_lit = blk
        current_sub = None

    def new_sub(kind: str, tag: str, first: str):
        nonlocal current_sub
        parent = current_lit if current_lit is not None else current_num
        if parent is None:
            return
        if kind == "dash":
            parent["sub_counter"] += 1
            tag = str(parent["sub_counter"])  # ordinal for tyrets
        blk = {"kind": kind, "tag": tag, "content": [first] if first else []}
        parent["subs"].append(blk)
        current_sub = blk

    for raw in lines:
        line = raw.rstrip()
        m = _match_enum_marker(line)
        if m:
            kind, tag, rest = m
            if kind == "num":
                new_num(tag, rest)
                continue
            if kind == "lit":
                new_lit(tag, rest)
                continue
            # roman/dash become subpoints of the deepest available parent
            new_sub(kind, tag, rest)
            continue
        # Non-marker line: append to the deepest open block
        if current_sub is not None:
            current_sub["content"].append(line)
        elif current_lit is not None:
            current_lit["content"].append(line)
        elif current_num is not None:
            current_num["content"].append(line)
        else:
            # No enumeration seen yet — do not accumulate preamble; prefer fallback
            pass

    # Decide splitting granularity
    results: List[Tuple[str, str]] = []

    def serialize_sub(sub: Dict[str, Any]) -> str:
        return "\n".join(sub.get("content", []))

    def serialize_lit(lit: Dict[str, Any], deep: bool) -> List[Tuple[str, str]]:
        if deep and _should_split(len(lit["subs"]), [serialize_sub(s) for s in lit["subs"]]):
            out: List[Tuple[str, str]] = []
            # Third-level split
            for sub in lit["subs"]:
                if sub.get("kind") == "rom":
                    suffix = f"lit. {lit['tag']}) roman {sub['tag']})"
                else:
                    suffix = f"lit. {lit['tag']}) tiret {sub['tag']}"
                out.append((suffix, serialize_sub(sub)))
            return out
        # No third-level split: return lit as single block
        text_parts = []
        if lit.get("content"):
            text_parts.append("\n".join(lit["content"]))
        # inline subs if present
        for sub in lit.get("subs", []):
            text_parts.append(serialize_sub(sub))
        return [(f"lit. {lit['tag']})", "\n".join([t for t in text_parts if t]))]

    def serialize_num(num: Dict[str, Any]) -> List[Tuple[str, str]]:
        # Prefer splitting by letters if there is a real list
        lit_texts = ["\n".join(l.get("content", [])) for l in num.get("lits", [])]
        if _should_split(len(num["lits"]), lit_texts):
            out: List[Tuple[str, str]] = []
            for lit in num["lits"]:
                out.extend(serialize_lit(lit, deep=True))
            return out
        # Else consider splitting by third-level under the numeric point
        sub_texts = [serialize_sub(s) for s in num.get("subs", [])]
        if _should_split(len(num["subs"]), sub_texts):
            out = []
            for sub in num["subs"]:
                if sub.get("kind") == "rom":
                    suffix = f"pkt {num['tag']} roman {sub['tag']})"
                else:
                    suffix = f"pkt {num['tag']} tiret {sub['tag']}"
                out.append((suffix, serialize_sub(sub)))
            return out
        # No inner split: numeric point as a whole
        text_parts = []
        if num.get("content"):
            text_parts.append("\n".join(num["content"]))
        for lit in num.get("lits", []):
            # inline lit content (including their subs)
            for suffix, txt in serialize_lit(lit, deep=False):
                text_parts.append(txt)
        for sub in num.get("subs", []):
            text_parts.append(serialize_sub(sub))
        return [(f"pkt {num['tag']}", "\n".join([t for t in text_parts if t]))]

    # If we found multiple numeric points, split at least by numeric level
    if _should_split(len(nums), ["\n".join(n.get("content", [])) for n in nums]):
        for num in nums:
            results.extend(serialize_num(num))
    else:
        # Not enough structure detected — return empty to signal fallback
        results = []

    # Filter out empty blocks
    final = [(suffix, txt.strip()) for (suffix, txt) in results if (txt or "").strip()]
    return final


SUMMARY_PROMPT = (
    "Streść poniższy tekst w maks. 5 zdaniach, wypisz też 'SIGNATURE' (10–20 lematów kluczowych) "
    "i 'ENTITIES' (nazwy własne/ID/zakres dat). Bez komentarzy.\n\n"
    "FORMAT:\nSUMMARY: ...\nSIGNATURE: lemma1, lemma2, ...\nENTITIES: ...\n\nTEKST:\n"  # Tekst doklejony na końcu
)

# JSON-mode instruction for models that support response_format={"type":"json_object"}
SUMMARY_PROMPT_JSON = (
    "Zwróć wyłącznie poprawny JSON bez komentarzy i bez kodu. Klucze: "
    "'summary' (string, max 5 zdań po polsku), "
    "'signature' (lista 10–20 lematów kluczowych jako strings), "
    "'entities' (string z nazwami własnymi/ID/zakresami dat)."
)


def llm_summary(text: str, model: str = settings.summary_model, max_tokens: int = 300) -> Dict[str, Any]:
    """Summarize text using chat completions. Prefer JSON mode; fallback to text parser.

    Returns a dict with keys: summary (str), signature (List[str]), entities (str).
    """
    text = text.strip()
    if len(text) > 8000:
        text = text[:8000]

    # Try JSON mode first (if enabled in settings)
    if getattr(settings, "summary_json_mode", True):
        try:
            rsp = summary_client.chat.completions.create(
                model=model,
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "Jesteś zwięzłym ekstrakcyjnym streszczaczem."},
                    {"role": "user", "content": SUMMARY_PROMPT_JSON + "\n\nTEKST:\n" + text},
                ],
                max_tokens=max_tokens,
            )
            content = rsp.choices[0].message.content or "{}"
            data = json.loads(content)
            summary_val = str(data.get("summary", "")).strip()
            signature_val = data.get("signature", [])
            if isinstance(signature_val, str):
                signature_list = [s.strip() for s in re.split(r",|;|\n", signature_val) if s.strip()]
            elif isinstance(signature_val, list):
                signature_list = [str(s).strip() for s in signature_val if str(s).strip()]
            else:
                signature_list = []
            entities_val = data.get("entities", "")
            if isinstance(entities_val, list):
                entities_str = ", ".join(str(x) for x in entities_val)
            else:
                entities_str = str(entities_val)
            if summary_val:
                return {"summary": summary_val, "signature": signature_list, "entities": entities_str}
        except Exception as _:
            # Fall back to text mode
            pass

    # Text-mode prompt and robust parsing (case/whitespace tolerant)
    rsp = summary_client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": "Jesteś zwięzłym ekstrakcyjnym streszczaczem."},
            {"role": "user", "content": SUMMARY_PROMPT + text},
        ],
        max_tokens=max_tokens,
    )
    out = rsp.choices[0].message.content or ""
    summary = ""
    signature_list: List[str] = []
    entities_str = ""
    for line in out.splitlines():
        m = re.match(r"^\s*summary\s*:\s*(.*)$", line, re.IGNORECASE)
        if m:
            summary = m.group(1).strip()
            continue
        m = re.match(r"^\s*signature\s*:\s*(.*)$", line, re.IGNORECASE)
        if m:
            sig = m.group(1).strip()
            signature_list = [s.strip() for s in re.split(r",|;", sig) if s.strip()]
            continue
        m = re.match(r"^\s*entities\s*:\s*(.*)$", line, re.IGNORECASE)
        if m:
            entities_str = m.group(1).strip()
            continue
    if not summary:
        summary = out.strip()[:600]
    return {"summary": summary, "signature": signature_list, "entities": entities_str}


# ——— Embedding i TF‑IDF ———

def get_embedding_dim() -> int:
    # Jednorazowo embedujemy prosty tekst i odczytujemy wymiar
    vec = embed_text(["test"])[0]
    return len(vec)


def embed_text(texts: List[str]) -> List[List[float]]:
    rsp = embedding_client.embeddings.create(model=settings.embedding_model, input=texts)
    return [d.embedding for d in rsp.data]


VECTORIZER_PATH = VECTOR_STORE_DIR / "tfidf_vectorizer.json"
SUMMARY_VECTORIZER_PATH = VECTOR_STORE_DIR / "tfidf_vectorizer_summary.json"


def load_vectorizer(path: pathlib.Path = VECTORIZER_PATH) -> Optional[TfidfVectorizer]:
    if path.exists():
        obj = json.loads(path.read_text())
        vec = TfidfVectorizer(**obj["params"])  # type: ignore
        vec.vocabulary_ = {k: int(v) for k, v in obj["vocab"].items()}  # type: ignore
        vec.idf_ = np.array(obj["idf"])  # type: ignore
        return vec
    return None


def save_vectorizer(vec: TfidfVectorizer, path: pathlib.Path = VECTORIZER_PATH):
    params = vec.get_params()
    payload = {
        "params": {k: v for k, v in params.items() if k in ["lowercase", "ngram_range", "min_df", "max_df"]},
        "vocab": {k: int(v) for k, v in vec.vocabulary_.items()},
        "idf": vec.idf_.tolist(),
    }
    path.write_text(json.dumps(payload))


def fit_vectorizer(corpus: List[str], path: pathlib.Path = VECTORIZER_PATH) -> TfidfVectorizer:
    vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, max_df=0.9)
    vec.fit(corpus)
    save_vectorizer(vec, path=path)
    return vec


def tfidf_vector(
    texts: List[str], vec: Optional[TfidfVectorizer], path: pathlib.Path = VECTORIZER_PATH
) -> List[Tuple[List[int], List[float]]]:
    if not vec:
        vec = load_vectorizer(path)
        if not vec:
            # jeśli brak globalnego wektoryzatora, fit tymczasowy na wejściu (słabsze, ale działa)
            vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1, max_df=1.0)
            vec.fit(texts)
    m = vec.transform(texts)
    results: List[Tuple[List[int], List[float]]] = []
    for i in range(m.shape[0]):
        row = m.getrow(i)
        indices = row.indices.astype(int).tolist()
        data = row.data.astype(float).tolist()
        results.append((indices, data))
    return results


# ──────────────────────────────────────────────────────────────────────────────
# SCORING HELPERS — normalization and hybrid ops
# ──────────────────────────────────────────────────────────────────────────────

def _normalize(values: List[float], method: str = DEFAULT_SCORE_NORM) -> List[float]:
    if not values:
        return []
    if method == "none":
        return values
    arr = np.array(values, dtype=float)
    if method == "zscore":
        mean = float(arr.mean())
        std = float(arr.std())
        if std < 1e-12:
            return [0.0 for _ in values]
        return [float((v - mean) / std) for v in arr]
    # minmax default
    vmin = float(arr.min())
    vmax = float(arr.max())
    if vmax - vmin < 1e-12:
        return [0.0 for _ in values]
    return [float((v - vmin) / (vmax - vmin)) for v in arr]


def _sparse_dot(query_lookup: Dict[int, float], indices: List[int], values: List[float]) -> float:
    if not query_lookup or not indices or not values:
        return 0.0
    acc = 0.0
    for i, v in zip(indices, values):
        q = query_lookup.get(int(i))
        if q is not None:
            acc += q * float(v)
    return float(acc)


def _sparse_pair_cos(
    a_idx: List[int], a_val: List[float], b_idx: List[int], b_val: List[float]
) -> float:
    # Assuming L2-normalized TF-IDF values from scikit (default). Then dot = cosine.
    if not a_idx or not b_idx:
        return 0.0
    i = j = 0
    sim = 0.0
    while i < len(a_idx) and j < len(b_idx):
        ai = int(a_idx[i])
        bj = int(b_idx[j])
        if ai == bj:
            sim += float(a_val[i]) * float(b_val[j])
            i += 1
            j += 1
        elif ai < bj:
            i += 1
        else:
            j += 1
    return float(sim)


def _cosine_dense(a: List[float], b: List[float]) -> float:
    va = np.array(a, dtype=float)
    vb = np.array(b, dtype=float)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(va, vb) / denom)


# ──────────────────────────────────────────────────────────────────────────────
# SEARCH HELPERS — stage-1 and stage-2 decomposition
# ──────────────────────────────────────────────────────────────────────────────

def _build_sparse_queries_for_query(query: str, use_hybrid: bool) -> Tuple[
    Optional[Tuple[List[int], List[float]]], Optional[Tuple[List[int], List[float]]]
]:
    if not (use_hybrid and SPARSE_ENABLED):
        return None, None
    content_sparse_query: Optional[Tuple[List[int], List[float]]] = None
    summary_sparse_query: Optional[Tuple[List[int], List[float]]] = None
    content_vec = load_vectorizer()
    if content_vec is not None:
        idx, val = tfidf_vector([query], content_vec)[0]
        if idx:
            content_sparse_query = (idx, val)
    summary_vec_model = load_vectorizer(path=SUMMARY_VECTORIZER_PATH)
    if summary_vec_model is not None:
        s_idx, s_val = tfidf_vector([query], summary_vec_model, path=SUMMARY_VECTORIZER_PATH)[0]
        if s_idx:
            summary_sparse_query = (s_idx, s_val)
    return content_sparse_query, summary_sparse_query


def _stage1_select_documents(
    q_vec: List[float],
    flt: Optional[qm.Filter],
    summary_sparse_query: Optional[Tuple[List[int], List[float]]],
    req: "SearchQuery",
) -> Tuple[List[str], Dict[str, Any]]:
    sum_search = qdrant.search(
        collection_name=settings.collection_name,
        query_vector=(SUMMARY_VECTOR_NAME, q_vec),
        query_filter=flt,
        limit=max(50, req.top_m),
        with_payload=True,
        with_vectors=[SUMMARY_VECTOR_NAME] if req.mmr_stage1 else False,
        score_threshold=None,
        search_params=qm.SearchParams(exact=False, hnsw_ef=128),
    )
    summary_sparse_lookup: Dict[int, float] = {}
    if summary_sparse_query is not None:
        summary_sparse_lookup = dict(zip(summary_sparse_query[0], summary_sparse_query[1]))

    doc_map: Dict[str, Dict[str, Any]] = {}
    for r in sum_search:
        payload = r.payload or {}
        did = payload.get("doc_id")
        if not did or did in doc_map:
            continue
        dense_score = float(r.score or 0.0)
        sparse_dot = 0.0
        if summary_sparse_lookup and payload.get("summary_sparse_indices") and payload.get("summary_sparse_values"):
            sparse_dot = _sparse_dot(summary_sparse_lookup, payload.get("summary_sparse_indices", []), payload.get("summary_sparse_values", []))
        vec_map = r.vector or {}
        dense_vec = vec_map.get(SUMMARY_VECTOR_NAME) or []
        doc_map[did] = {
            "doc_id": did,
            "dense_vec": dense_vec,
            "sparse_idx": payload.get("summary_sparse_indices", []) or [],
            "sparse_val": payload.get("summary_sparse_values", []) or [],
            "dense_score": dense_score,
            "sparse_score": sparse_dot,
            "path": payload.get("path"),
        }

    if not doc_map:
        return [], {}

    doc_items = list(doc_map.values())
    dense_scores = [float(x["dense_score"]) for x in doc_items]
    sparse_scores = [float(x["sparse_score"]) for x in doc_items]
    dense_norm = _normalize(dense_scores, req.score_norm)
    sparse_norm = _normalize(sparse_scores, req.score_norm)
    hybrid_rel = [req.dense_weight * d + req.sparse_weight * s for d, s in zip(dense_norm, sparse_norm)]

    if req.mmr_stage1 and len(doc_items) > 1:
        rep_alpha = req.rep_alpha if req.rep_alpha is not None else req.dense_weight
        dense_vecs = [x["dense_vec"] for x in doc_items]
        sparse_vecs = [(x["sparse_idx"], x["sparse_val"]) for x in doc_items]
        mmr_idx = mmr_diversify_hybrid(dense_vecs, sparse_vecs, hybrid_rel, min(req.top_m, len(doc_items)), req.mmr_lambda, rep_alpha)
        cand_doc_ids = [doc_items[i]["doc_id"] for i in mmr_idx]
    else:
        order = sorted(range(len(doc_items)), key=lambda i: hybrid_rel[i], reverse=True)
        order = order[: min(req.top_m, len(order))]
        cand_doc_ids = [doc_items[i]["doc_id"] for i in order]

    return cand_doc_ids, doc_map


def _stage2_select_chunks(
    cand_doc_ids: List[str],
    q_vec: List[float],
    content_sparse_query: Optional[Tuple[List[int], List[float]]],
    req: "SearchQuery",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[float]]:
    flt2 = qm.Filter(must=[qm.FieldCondition(key="doc_id", match=qm.MatchAny(any=cand_doc_ids))])
    cont_search = qdrant.search(
        collection_name=settings.collection_name,
        query_vector=(CONTENT_VECTOR_NAME, q_vec),
        query_filter=flt2,
        limit=req.top_m,
        with_payload=True,
        with_vectors=[CONTENT_VECTOR_NAME],
        search_params=qm.SearchParams(exact=False, hnsw_ef=128),
    )
    mmr_pool: List[Dict[str, Any]] = []
    for hit in cont_search:
        payload = hit.payload or {}
        dense_score = float(hit.score or 0.0)
        sparse_dot = 0.0
        if content_sparse_query is not None and payload.get("content_sparse_indices") and payload.get("content_sparse_values"):
            q_lookup = dict(zip(content_sparse_query[0], content_sparse_query[1]))
            sparse_dot = _sparse_dot(q_lookup, payload.get("content_sparse_indices", []), payload.get("content_sparse_values", []))
        mmr_pool.append({
            "hit": hit,
            "doc_id": payload.get("doc_id", ""),
            "dense_vec": (hit.vector or {}).get(CONTENT_VECTOR_NAME) or [],
            "sparse_idx": payload.get("content_sparse_indices", []) or [],
            "sparse_val": payload.get("content_sparse_values", []) or [],
            "dense_score": dense_score,
            "sparse_score": sparse_dot,
        })

    if not mmr_pool:
        return [], [], []

    dense_scores2 = [x["dense_score"] for x in mmr_pool]
    sparse_scores2 = [x["sparse_score"] for x in mmr_pool]
    dense_norm2 = _normalize(dense_scores2, req.score_norm)
    sparse_norm2 = _normalize(sparse_scores2, req.score_norm)
    rel2 = [req.dense_weight * d + req.sparse_weight * s for d, s in zip(dense_norm2, sparse_norm2)]

    rep_alpha = req.rep_alpha if req.rep_alpha is not None else req.dense_weight
    dense_vecs2 = [x["dense_vec"] for x in mmr_pool]
    sparse_vecs2 = [(x["sparse_idx"], x["sparse_val"]) for x in mmr_pool]
    doc_ids2 = [x["doc_id"] for x in mmr_pool]

    sel_idx = mmr_diversify_hybrid(
        dense_vecs2,
        sparse_vecs2,
        rel2,
        min(req.top_k, len(mmr_pool)),
        req.mmr_lambda,
        rep_alpha,
        per_doc_ids=doc_ids2,
        per_doc_limit=max(1, int(req.per_doc_limit)) if req.per_doc_limit and req.per_doc_limit > 0 else None,
    )

    selected = [mmr_pool[i] for i in sel_idx]
    final_hits: List[Dict[str, Any]] = []
    for idx, item in zip(sel_idx, selected):
        hit = item["hit"]
        payload = hit.payload or {}
        final_score = float(rel2[idx])
        final_hits.append({"hit": hit, "score": float(final_score), "payload": payload})
    final_hits.sort(key=lambda x: x["score"], reverse=True)
    return final_hits, mmr_pool, rel2


def _classify_mode(query: str, mode: str) -> str:
    """Classify retrieval mode based on query when mode=auto."""
    if mode != "auto":
        return mode
    q = query.lower()
    if re.search(r"archiw|stara|z \d{4}|wersja\s+z", q):
        return "archival"
    if re.search(r"obowiązując|aktualn|teraz|bieżąc", q):
        return "current"
    return "all"


def _build_neighbor_index(
    mmr_pool: List[Dict[str, Any]], rel2: List[float]
) -> Dict[Tuple[str, Optional[str], int], Dict[str, Any]]:
    """Build lookup for contiguous neighbor expansion when merging chunks."""
    neighbor_index: Dict[Tuple[str, Optional[str], int], Dict[str, Any]] = {}
    for idx2, item in enumerate(mmr_pool):
        payload = (item.get("hit").payload if item.get("hit") else {}) or {}
        did = payload.get("doc_id") or ""
        sec = payload.get("section")
        cid = payload.get("chunk_id")
        if not did or cid is None:
            continue
        score2 = float(rel2[idx2]) if idx2 < len(rel2) else float(item.get("dense_score", 0.0))
        neighbor_index[(did, sec, int(cid))] = {"payload": payload, "score": score2}
    return neighbor_index


def _shape_results(
    final_hits: List[Dict[str, Any]],
    doc_map: Dict[str, Any],
    mmr_pool: List[Dict[str, Any]],
    rel2: List[float],
    req: "SearchQuery",
) -> Tuple[List["SearchHit"], Optional[List[Dict[str, Any]]], Optional[List["MergedBlock"]]]:
    blocks_payload: Optional[List[MergedBlock]] = None
    if req.merge_chunks or req.result_format == "blocks":
        neighbor_index = None
        if req.expand_neighbors and req.expand_neighbors > 0:
            neighbor_index = _build_neighbor_index(mmr_pool, rel2)
        raw_blocks = build_merged_blocks(
            final_hits,
            merge_group_budget_tokens=req.merge_group_budget_tokens,
            max_merged_per_group=req.max_merged_per_group,
            join_delim=req.block_join_delimiter,
            summary_mode=req.summary_mode,
            expand_neighbors=max(0, int(req.expand_neighbors or 0)),
            neighbor_index=neighbor_index,
        )
        blocks_payload = [
            MergedBlock(
                doc_id=b["doc_id"],
                path=b.get("path", ""),
                section=b.get("section"),
                first_chunk_id=int(b.get("first_chunk_id", 0)),
                last_chunk_id=int(b.get("last_chunk_id", 0)),
                score=float(b.get("score", 0.0)),
                summary=b.get("summary"),
                text=b.get("text", ""),
                token_estimate=int(b.get("token_estimate")) if b.get("token_estimate") is not None else None,
            )
            for b in raw_blocks
        ]

    if req.result_format == "blocks":
        return [], None, blocks_payload

    results: List[SearchHit] = []
    groups_payload: Optional[List[Dict[str, Any]]] = None

    if req.result_format == "grouped":
        groups: Dict[str, Dict[str, Any]] = {}
        for fh in final_hits:
            payload = fh.get("payload") or {}
            did = payload.get("doc_id", "")
            if not did:
                continue
            grp = groups.get(did)
            if grp is None:
                grp = {
                    "doc_id": did,
                    "path": payload.get("path", ""),
                    "summary": None if req.summary_mode == "none" else payload.get("summary"),
                    "score": float(fh.get("score", 0.0)),
                    "chunks": [],
                }
                groups[did] = grp
            else:
                if float(fh.get("score", 0.0)) > float(grp.get("score", 0.0)):
                    grp["score"] = float(fh.get("score", 0.0))
            snippet = (payload.get("text") or "").strip()[:500]
            grp["chunks"].append({
                "chunk_id": payload.get("chunk_id", 0),
                "score": float(fh.get("score", 0.0)),
                "snippet": snippet,
            })
        groups_list = list(groups.values())
        for g in groups_list:
            g["chunks"].sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        groups_list.sort(key=lambda g: float(g.get("score", 0.0)), reverse=True)
        groups_payload = groups_list
        results = []
    else:
        seen_docs: set = set()
        for fh in final_hits:
            payload = fh.get("payload") or {}
            did = payload.get("doc_id", "")
            if req.summary_mode == "none":
                summary_val: Optional[str] = None
            elif req.summary_mode == "first":
                summary_val = None if did in seen_docs else payload.get("summary")
                if did not in seen_docs:
                    seen_docs.add(did)
            else:
                summary_val = payload.get("summary")
            results.append(
                SearchHit(
                    doc_id=did,
                    path=payload.get("path", ""),
                    section=payload.get("section"),
                    chunk_id=payload.get("chunk_id", 0),
                    score=float(fh.get("score", 0.0)),
                    snippet=(payload.get("text") or "").strip()[:500] if payload.get("text") else (payload.get("summary", "")[:500]),
                    summary=summary_val,
                )
            )

    return results, groups_payload, blocks_payload



# ──────────────────────────────────────────────────────────────────────────────
# MERGING HELPERS — build merged blocks from contiguous chunks
# ──────────────────────────────────────────────────────────────────────────────

def _approx_tokens(s: str) -> int:
    # Rough heuristic: ~4 chars per token
    return max(1, len(s) // 4)


def build_merged_blocks(
    final_hits: List[Dict[str, Any]],
    merge_group_budget_tokens: int,
    max_merged_per_group: int,
    join_delim: str,
    summary_mode: str = "first",
    expand_neighbors: int = 0,
    neighbor_index: Optional[Dict[Tuple[str, Optional[str], int], Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    # Group by (doc_id, section)
    groups: Dict[Tuple[str, Optional[str]], List[Dict[str, Any]]] = {}
    for fh in final_hits:
        payload = fh.get("payload") or {}
        did = payload.get("doc_id") or ""
        sec = payload.get("section")
        if not did:
            continue
        key = (did, sec)
        groups.setdefault(key, []).append(fh)

    blocks: List[Dict[str, Any]] = []
    for (doc_id, section), items in groups.items():
        # sort by chunk_id ascending
        items.sort(key=lambda x: int((x.get("payload") or {}).get("chunk_id", 0)))

        # find contiguous runs
        runs: List[List[Dict[str, Any]]] = []
        cur_run: List[Dict[str, Any]] = []
        for cur in items:
            if not cur_run:
                cur_run = [cur]
            else:
                prev = cur_run[-1]
                prev_id = int((prev.get("payload") or {}).get("chunk_id", 0))
                cur_id = int((cur.get("payload") or {}).get("chunk_id", 0))
                if cur_id == prev_id + 1:
                    cur_run.append(cur)
                else:
                    runs.append(cur_run)
                    cur_run = [cur]
        if cur_run:
            runs.append(cur_run)

        # pick top runs by max score
        runs.sort(key=lambda r: max(float(x.get("score", 0.0)) for x in r), reverse=True)
        take_n = max(1, int(max_merged_per_group))
        selected_runs = runs[:take_n]

        for run in selected_runs:
            # Optionally expand contiguous run with neighbors from candidate pool
            if expand_neighbors and neighbor_index:
                # Determine group identifiers
                p0 = (run[0].get("payload") or {})
                did = p0.get("doc_id") or ""
                sec = p0.get("section")
                # Collect existing chunk_ids in run
                existing_ids = {int((r.get("payload") or {}).get("chunk_id", -10)) for r in run}
                # Compute current bounds
                cur_first = min(existing_ids) if existing_ids else None
                cur_last = max(existing_ids) if existing_ids else None
                # Extend to the left
                if cur_first is not None:
                    for step in range(1, int(expand_neighbors) + 1):
                        cid = cur_first - step
                        key = (did, sec, cid)
                        if key in neighbor_index:
                            if cid not in existing_ids:
                                nb = neighbor_index[key]
                                run.insert(0, {"payload": nb.get("payload", {}), "score": float(nb.get("score", 0.0))})
                                existing_ids.add(cid)
                                cur_first = cid
                        else:
                            break  # keep contiguity
                # Extend to the right
                if cur_last is not None:
                    for step in range(1, int(expand_neighbors) + 1):
                        cid = cur_last + step
                        key = (did, sec, cid)
                        if key in neighbor_index:
                            if cid not in existing_ids:
                                nb = neighbor_index[key]
                                run.append({"payload": nb.get("payload", {}), "score": float(nb.get("score", 0.0))})
                                existing_ids.add(cid)
                                cur_last = cid
                        else:
                            break  # keep contiguity
            text_parts: List[str] = []
            scores: List[float] = []
            used_tokens = 0
            first_chunk_id = int((run[0].get("payload") or {}).get("chunk_id", 0))
            last_chunk_id = first_chunk_id
            for r in run:
                payload = r.get("payload") or {}
                chunk_text = (payload.get("text") or "").strip()
                if not chunk_text:
                    continue
                t = _approx_tokens(chunk_text)
                # Always allow first piece, then enforce budget
                if text_parts and used_tokens + t > int(merge_group_budget_tokens):
                    break
                text_parts.append(chunk_text)
                scores.append(float(r.get("score", 0.0)))
                last_chunk_id = int(payload.get("chunk_id", last_chunk_id))
                used_tokens += t

            if not text_parts:
                continue

            # Capture document summary, but defer inclusion policy until after sorting
            payload0 = (run[0].get("payload") or {})
            doc_summary_val = payload0.get("summary")

            blocks.append(
                {
                    "doc_id": doc_id,
                    "path": payload0.get("path", ""),
                    "section": section,
                    "first_chunk_id": first_chunk_id,
                    "last_chunk_id": last_chunk_id,
                    "score": max(scores) if scores else 0.0,
                    "summary": None,  # will be assigned after global sorting based on policy
                    "doc_summary": doc_summary_val,  # internal key used for policy
                    "text": join_delim.join(text_parts).strip(),
                    "token_estimate": used_tokens,
                }
            )

    # Sort globally by score so that summary_mode='first' lands on top-scoring block per doc
    blocks.sort(key=lambda b: float(b.get("score", 0.0)), reverse=True)

    # Apply summary inclusion policy per document (not per section)
    if summary_mode == "none":
        for b in blocks:
            b["summary"] = None
    elif summary_mode == "first":
        seen_docs: set = set()
        for b in blocks:
            did = b.get("doc_id")
            if did not in seen_docs:
                b["summary"] = b.get("doc_summary")
                seen_docs.add(did)
            else:
                b["summary"] = None
    else:  # "all"
        for b in blocks:
            b["summary"] = b.get("doc_summary")

    return blocks


# ──────────────────────────────────────────────────────────────────────────────
# QDRANT — KOLEKCJA
# ──────────────────────────────────────────────────────────────────────────────

def ensure_collection(collection: Optional[str] = None, dim: Optional[int] = None):
    collection = collection or settings.collection_name
    if dim is None:
        # Use configured embedding dimension to avoid unnecessary API calls
        dim = settings.embedding_dim
    vectors_config = {
        CONTENT_VECTOR_NAME: qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
        SUMMARY_VECTOR_NAME: qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
    }
    # Konfiguracja nazwanej macierzy rzadkiej – Qdrant 1.x: przekazujemy ją jako nazwany
    # wektor w polu `vector` punktu (wartość = SparseVector)
    sparse_config = (
        {
            CONTENT_SPARSE_NAME: qm.SparseVectorParams(),
            SUMMARY_SPARSE_NAME: qm.SparseVectorParams(),
        }
        if SPARSE_ENABLED
        else None
    )
    try:
        info = qdrant.get_collection(collection)
        # Validate dense vectors config (expect named vectors)
        try:
            vecs = getattr(getattr(info, "config", None), "params", None)
            vecs = getattr(vecs, "vectors", None)
            names: set = set()
            if isinstance(vecs, dict):
                names = set(vecs.keys())
            else:
                for attr in ("configs", "map", "vectors", "items", "data"):
                    mapping = getattr(vecs, attr, None)
                    if isinstance(mapping, dict) and mapping:
                        names = set(mapping.keys())
                        break
            expected = {CONTENT_VECTOR_NAME, SUMMARY_VECTOR_NAME}
            if not names or not expected.issubset(names):
                logger.error(
                    "Collection '%s' incompatible vectors config. Found names=%s, expected at least %s",
                    collection,
                    sorted(list(names)) if names else None,
                    sorted(list(expected)),
                )
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"Kolekcja '{collection}' ma niekompatybilną konfigurację wektorów. "
                        f"Wymagane nazwane wektory: {sorted(list(expected))}. Usuń kolekcję lub uruchom ingest z reindex=true."
                    ),
                )
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning("Could not verify vectors config for '%s': %s", collection, exc)
        # Ensure sparse vectors exist (idempotent on modern Qdrant)
        if SPARSE_ENABLED:
            try:
                qdrant.update_collection(
                    collection_name=collection,
                    sparse_vectors_config={
                        CONTENT_SPARSE_NAME: qm.SparseVectorParams(),
                        SUMMARY_SPARSE_NAME: qm.SparseVectorParams(),
                    },
                )
                logger.debug("Ensured sparse vectors for '%s' are configured", collection)
            except UnexpectedResponse as exc:
                msg = str(exc).lower()
                if "already" in msg or "exist" in msg or "conflict" in msg:
                    logger.debug("Sparse vectors already present for '%s'", collection)
                else:
                    logger.error("Failed to ensure sparse vectors for '%s': %s", collection, exc)
                    raise
        logger.debug("Collection '%s' exists and is compatible", collection)
        return
    except HTTPException:
        # bubble up incompatibility to caller
        raise
    except Exception:
        logger.debug("Collection '%s' not found; creating", collection)
        pass

    try:
        qdrant.create_collection(
            collection_name=collection,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_config,
            optimizers_config=qm.OptimizersConfigDiff(indexing_threshold=20000),
        )
        logger.debug("Collection '%s' created", collection)
    except UnexpectedResponse as exc:
        # 409 means already exists – ignorujemy, wszystko OK
        if getattr(exc, "status_code", None) == 409 or "already exists" in str(exc).lower():
            logger.debug("Collection '%s' creation returned conflict; treating as existing", collection)
            return
        raise


# ──────────────────────────────────────────────────────────────────────────────
# MODELE Pydantic — żądania/odpowiedzi
# ──────────────────────────────────────────────────────────────────────────────
class About(BaseModel):
    name: str = settings.app_name
    version: str = settings.app_version
    author: str = "Seweryn Sitarski (seweryn.sitarski@gmail.com) with support from Kat"
    description: str = "Dwustopniowy RAG ze streszczeniami + hybryda dense/TF‑IDF (Qdrant)"


class InitCollectionsRequest(BaseModel):
    collection_name: str = Field(default_factory=lambda: settings.collection_name)
    force_dim_probe: bool = False


class ScanRequest(BaseModel):
    base_dir: str
    glob: str = "**/*"
    recursive: bool = True


class ScanResponse(BaseModel):
    files: List[str]


class IngestBuildRequest(BaseModel):
    base_dir: str
    glob: str = "**/*"
    recursive: bool = True
    reindex: bool = False
    chunk_tokens: int = 900
    chunk_overlap: int = 150
    language_hint: Optional[str] = None
    collection_name: str = Field(default_factory=lambda: settings.collection_name)
    enable_sparse: bool = True
    rebuild_tfidf: bool = True


class SummariesGenerateRequest(BaseModel):
    files: List[str]


class SearchQuery(BaseModel):
    query: str = Field(..., description="User question in natural language. Keep it concise; no tool/system instructions.")
    top_m: int = Field(100, description="Stage-1 (summaries) candidate document count. Typical 50–200.")
    top_k: int = Field(10, description="Final result count after Stage-2 selection. Typical 5–10.")
    mode: str = Field("auto", description="Retrieval mode: auto|current|archival|all. 'current' filters is_active=true; 'archival' false.")
    use_hybrid: bool = Field(True, description="Enable hybrid scoring (dense + TF-IDF) for query.")
    dense_weight: float = Field(0.6, description="Weight of dense similarity in hybrid relevance [0..1].")
    sparse_weight: float = Field(0.4, description="Weight of sparse (TF-IDF) similarity in hybrid relevance [0..1].")
    mmr_lambda: float = Field(DEFAULT_MMR_LAMBDA, description="MMR relevance-vs-diversity balance [0..1]. Higher = more relevance.")
    per_doc_limit: int = Field(DEFAULT_PER_DOC_LIMIT, description="Max results per single document in Stage-2.")
    score_norm: str = Field(DEFAULT_SCORE_NORM, description="Score normalization: minmax|zscore|none.")
    rep_alpha: Optional[float] = Field(None, description="Redundancy alpha in hybrid MMR (dense contribution). Defaults to dense_weight.")
    mmr_stage1: bool = Field(True, description="Apply hybrid MMR already at Stage-1 (summaries).")
    # Controls duplication of document summary in results
    summary_mode: str = Field("first", description="Document summary duplication: none|first|all. 'first' shows once per doc.")
    # Optional merging of adjacent chunks into larger blocks
    merge_chunks: bool = Field(False, description="If true, also build merged blocks per (doc_id, section).")
    merge_group_budget_tokens: int = Field(1200, description="Approx token budget per merged block (~4 chars/token).")
    max_merged_per_group: int = Field(1, description="Max merged blocks to return for each (doc_id, section) group.")
    block_join_delimiter: str = Field("\n\n", description="Delimiter used when concatenating contiguous chunks in a merged block.")
    expand_neighbors: int = Field(0, description="When merging, also try to include up to N missing adjacent chunks from candidates (mmr_pool). 0 disables.")
    # Controls shape of response: flat list vs grouped per document vs blocks
    result_format: str = Field("flat", description="Response shape: flat|grouped|blocks. For tools, 'blocks' is recommended.")


class SearchHit(BaseModel):
    doc_id: str = Field(..., description="Stable document identifier (sha1 over absolute path).")
    path: str = Field(..., description="Absolute document path (for citation).")
    section: Optional[str] = Field(default=None, description="Optional document section identifier, if present.")
    chunk_id: int = Field(..., description="Chunk index within the document (0-based).")
    score: float = Field(..., description="Hybrid relevance score (normalized according to score_norm).")
    snippet: str = Field(..., description="Short text snippet of the chunk or summary fallback.")
    summary: Optional[str] = Field(default=None, description="Document-level summary (presence controlled by summary_mode).")


class SearchResponse(BaseModel):
    took_ms: int = Field(..., description="Total search latency in milliseconds.")
    hits: List[SearchHit] = Field(..., description="Flat hit list (chunk-level).")
    # Optional grouped representation (when requested)
    groups: Optional[List["SearchGroup"]] = Field(default=None, description="Grouped results per document (summary + chunks).")
    # Optional merged blocks representation
    blocks: Optional[List["MergedBlock"]] = Field(default=None, description="Merged blocks per (doc_id, section). Prefer for tool use.")


class SearchChunk(BaseModel):
    chunk_id: int = Field(..., description="Chunk index within the document (0-based).")
    score: float = Field(..., description="Hybrid relevance score (normalized).")
    snippet: str = Field(..., description="Short text snippet of the chunk.")


class SearchGroup(BaseModel):
    doc_id: str = Field(..., description="Stable document identifier.")
    path: str = Field(..., description="Absolute document path.")
    summary: Optional[str] = Field(default=None, description="Document-level summary (single copy per document).")
    score: float = Field(..., description="Max score among group's chunks.")
    chunks: List[SearchChunk] = Field(..., description="Chunk-level results belonging to this document.")

class MergedBlock(BaseModel):
    doc_id: str = Field(..., description="Stable document identifier.")
    path: str = Field(..., description="Absolute document path.")
    section: Optional[str] = Field(default=None, description="Optional section identifier.")
    first_chunk_id: int = Field(..., description="First chunk id (inclusive) in this merged block.")
    last_chunk_id: int = Field(..., description="Last chunk id (inclusive) in this merged block.")
    score: float = Field(..., description="Block score = max score among its member chunks.")
    summary: Optional[str] = Field(default=None, description="Document/section summary if requested by summary_mode.")
    text: str = Field(..., description="Merged textual content of the block (joined contiguous chunks).")
    token_estimate: Optional[int] = Field(default=None, description="Heuristic token length (~4 chars/token).")

# Rebuild forward refs again after defining MergedBlock
SearchResponse.model_rebuild()


# ──────────────────────────────────────────────────────────────────────────────
# FASTAPI
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(title=f"{settings.app_name} OpenAPI Tool", version=settings.app_version)


def build_admin_operations() -> List[Dict[str, Any]]:
    route_lookup: Dict[Tuple[str, str], APIRoute] = {}
    for route in app.routes:
        if isinstance(route, APIRoute):
            for method in route.methods or []:
                route_lookup[(route.path, method.upper())] = route

    operations: List[Dict[str, Any]] = []
    for spec in ADMIN_OPERATION_SPECS:
        method = spec["method"].upper()
        path = spec["path"]
        route = route_lookup.get((path, method))
        summary = route.summary if route else None
        description = route.description if route else None
        doc_parts = [part for part in [summary, description] if part]
        operations.append(
            {
                "id": spec["id"],
                "label": spec.get("label") or f"{method} {path}",
                "method": method,
                "path": path,
                "doc": "\n\n".join(doc_parts),
                "body": spec.get("body"),
            }
        )
    return operations


@app.middleware("http")
async def admin_ui_debug_middleware(request: Request, call_next):
    global _admin_debug_activated
    header_value = request.headers.get(ADMIN_UI_REQUEST_HEADER)
    if header_value and header_value.lower() in {"1", "true", "yes"} and not _admin_debug_activated:
        _admin_debug_activated = True
        settings.debug = True
        logger.setLevel(logging.DEBUG)
        logger.info("Admin UI request detected — DEBUG logging enabled")
    response = await call_next(request)
    return response


@app.get(
    "/admin",
    include_in_schema=False,
    response_class=HTMLResponse,
    summary="Panel administracyjny",
    description="Statyczny panel HTML do testowania i debugowania endpointów rags_tool.",
)
def admin_console():
    operations = build_admin_operations()
    tpl_path = pathlib.Path(__file__).parent / "templates" / "admin.html"
    try:
        html = tpl_path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.error("Failed to load admin template: %s", exc)
        return HTMLResponse(content="<html><body><p>Admin UI unavailable.</p></body></html>")
    html = html.replace("__OPERATIONS__", json.dumps(operations, ensure_ascii=False))
    return HTMLResponse(content=html)


@app.get(
    "/about",
    response_model=About,
    include_in_schema=False,
    summary="Informacje o aplikacji",
    description="Zwraca metadane serwisu rags_tool (nazwa, wersja, autor, opis).",
)
def about():
    return About()


@app.get(
    "/health",
    include_in_schema=False,
    summary="Stan usługi",
    description="Sprawdza połączenie z Qdrant i raportuje kondycję aplikacji.",
)
def health():
    try:
        qdrant.get_collections()
        return {"status": "ok", "qdrant": True}
    except Exception as e:
        return {"status": "degraded", "qdrant": False, "error": str(e)}


@app.post(
    "/collections/init",
    include_in_schema=False,
    summary="Inicjalizacja kolekcji",
    description=(
        "Tworzy kolekcję we wskazanej nazwie (jeśli nie istnieje) i opcjonalnie sondą"
        " sprawdza wymiar embeddingów przy użyciu force_dim_probe."
    ),
)
def collections_init(req: InitCollectionsRequest):
    dim = get_embedding_dim() if req.force_dim_probe else None
    ensure_collection(req.collection_name, dim)
    return {"ok": True, "collection": req.collection_name}


@app.post(
    "/ingest/scan",
    response_model=ScanResponse,
    include_in_schema=False,
    summary="Skanowanie korpusu",
    description="Zwraca listę plików w katalogu bazowym, które kwalifikują się do ingestu.",
)
def ingest_scan(req: ScanRequest):
    base = pathlib.Path(req.base_dir)
    if not base.exists():
        raise HTTPException(status_code=400, detail="base_dir nie istnieje")
    files = [str(p) for p in _scan_files(base, req.glob, req.recursive)]
    return ScanResponse(files=files)


@app.post(
    "/summaries/generate",
    include_in_schema=False,
    summary="Streszczenia wybranych plików",
    description="Generuje streszczenia oraz podpisy dla listy plików bez zapisu do Qdrant.",
)
def summaries_generate(req: SummariesGenerateRequest):
    results = {}
    for f in req.files:
        p = pathlib.Path(f)
        if not p.exists():
            results[f] = {"error": "not found"}
            continue
        text = extract_text(p)
        summ = llm_summary(text)
        results[f] = summ
    return {"results": results}


@app.post(
    "/ingest/build",
    include_in_schema=False,
    summary="Pełny ingest korpusu",
    description=(
        "Buduje indeks rags_tool: parsuje dokumenty, tworzy streszczenia,"
        " embeddingi oraz zapisuje punkty (wraz z TF-IDF) do Qdrant."
    ),
)
def ingest_build(req: IngestBuildRequest):
    t0 = time.time()
    logger.debug(
        "Starting ingest build | base_dir=%s glob=%s recursive=%s reindex=%s",
        req.base_dir,
        req.glob,
        req.recursive,
        req.reindex,
    )
    if req.reindex:
        try:
            qdrant.delete_collection(collection_name=req.collection_name)
            logger.debug("Deleted collection '%s' due to reindex request", req.collection_name)
        except Exception as exc:
            logger.debug("Delete collection '%s' skipped or failed: %s", req.collection_name, exc)
    ensure_collection(req.collection_name)

    base = pathlib.Path(req.base_dir)
    if not base.exists():
        raise HTTPException(status_code=400, detail="base_dir nie istnieje")

    file_paths = _scan_files(base, req.glob, req.recursive)
    logger.debug("Found %d files for ingest", len(file_paths))
    if not file_paths:
        return {"ok": True, "indexed": 0, "took_ms": int((time.time() - t0) * 1000)}

    # 1) Parse and chunk
    doc_records, all_chunks, summary_corpus = _collect_documents(file_paths, req.chunk_tokens, req.chunk_overlap)

    # 2) TF-IDF vectorizers (optional)
    content_vec, summary_vec = _prepare_tfidf(all_chunks, summary_corpus, req.enable_sparse, req.rebuild_tfidf)

    # 3) Build and upsert points
    point_count = _build_and_upsert_points(
        doc_records,
        content_vec,
        summary_vec,
        enable_sparse=req.enable_sparse,
        collection_name=req.collection_name,
    )

    took_ms = int((time.time() - t0) * 1000)
    logger.debug(
        "Ingest build finished | documents=%d points=%d took_ms=%d",
        len(doc_records),
        point_count,
        took_ms,
    )
    return {"ok": True, "indexed": point_count, "documents": len(doc_records), "took_ms": took_ms}


# ──────────────────────────────────────────────────────────────────────────────
# SEARCH — dwustopniowe: streszczenia → pełny tekst (+ MMR, hybryda)
# ──────────────────────────────────────────────────────────────────────────────

def mmr_diversify(vectors: np.ndarray, scores: np.ndarray, k: int, lam: float = DEFAULT_MMR_LAMBDA) -> List[int]:
    # Legacy MMR (dense only); kept for compatibility
    selected: List[int] = []
    candidates = list(range(len(scores)))
    if len(candidates) <= k:
        return candidates
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    def sims(a, b) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / (denom + 1e-9))
    while len(selected) < k and candidates:
        best_i = None
        best_score = -1e9
        for i in candidates:
            rep = 0.0
            for j in selected:
                rep = max(rep, sims(vectors[i], vectors[j]))
            mmr = lam * float(scores[i]) - (1 - lam) * rep
            if mmr > best_score:
                best_score = mmr
                best_i = i
        if best_i is None:
            break
        selected.append(best_i)
        candidates.remove(best_i)
    return selected


def mmr_diversify_hybrid(
    dense_vecs: List[List[float]],
    sparse_vecs: List[Tuple[List[int], List[float]]],
    rel_scores: List[float],
    k: int,
    lam: float,
    rep_alpha: float,
    per_doc_ids: Optional[List[str]] = None,
    per_doc_limit: Optional[int] = None,
) -> List[int]:
    # Greedy MMR with hybrid redundancy (dense+sparse) and optional per-doc cap
    n = len(rel_scores)
    candidates = list(range(n))
    if n <= k:
        return candidates
    selected: List[int] = []
    counts: Dict[str, int] = {}
    def allowed(i: int) -> bool:
        if per_doc_ids is None or per_doc_limit is None:
            return True
        did = per_doc_ids[i]
        return counts.get(did, 0) < per_doc_limit
    while len(selected) < k and candidates:
        best_i = None
        best_score = -1e12
        for i in candidates:
            if per_doc_limit is not None and per_doc_ids is not None:
                if not allowed(i):
                    continue
            rep = 0.0
            for j in selected:
                d_sim = _cosine_dense(dense_vecs[i], dense_vecs[j])
                s_sim = _sparse_pair_cos(
                    sparse_vecs[i][0], sparse_vecs[i][1], sparse_vecs[j][0], sparse_vecs[j][1]
                )
                rep = max(rep, rep_alpha * d_sim + (1.0 - rep_alpha) * s_sim)
            mmr = lam * float(rel_scores[i]) - (1.0 - lam) * rep
            if mmr > best_score:
                best_score = mmr
                best_i = i
        if best_i is None:
            break
        # if per-doc cap blocks all candidates, relax once to fill
        if per_doc_limit is not None and per_doc_ids is not None and not allowed(best_i):
            alt = None
            alt_score = -1e12
            for i in candidates:
                if allowed(i):
                    rep = 0.0
                    for j in selected:
                        d_sim = _cosine_dense(dense_vecs[i], dense_vecs[j])
                        s_sim = _sparse_pair_cos(
                            sparse_vecs[i][0], sparse_vecs[i][1], sparse_vecs[j][0], sparse_vecs[j][1]
                        )
                        rep = max(rep, rep_alpha * d_sim + (1.0 - rep_alpha) * s_sim)
                    mmr = lam * float(rel_scores[i]) - (1.0 - lam) * rep
                    if mmr > alt_score:
                        alt_score = mmr
                        alt = i
            if alt is not None:
                best_i = alt
        selected.append(best_i)
        candidates.remove(best_i)
        if per_doc_limit is not None and per_doc_ids is not None:
            did = per_doc_ids[best_i]
            counts[did] = counts.get(did, 0) + 1
    return selected


@app.post(
    "/search/query",
    response_model=SearchResponse,
    summary="rags_tool search (LLM tool)",
    operation_id="rags_tool_search",
    tags=["tools"],
    description=(
        "Two-stage retrieval for LLM tools. Stage-1 ranks document summaries; "
        "Stage-2 ranks full-text chunks within selected documents using hybrid scoring (dense+TF-IDF) and MMR.\n\n"
        "Usage guidance for tools:\n"
        "- Prefer `result_format=\"blocks\"` with `merge_chunks=true` to receive consolidated blocks.\n"
        "- Set `top_k` to 5–10 and keep defaults for other params unless you know why to change them.\n"
        "- `summary_mode=first` includes a single document summary per document.\n"
        "- `mode=auto` auto-detects current/archival, or force with `current|archival|all`.\n"
        "Returned fields: use `blocks[].text` as evidence; cite `blocks[].path` and chunk id range."
    ),
)
def search_query(req: SearchQuery):
    t0 = time.time()
    ensure_collection()

    # 0) Tryb
    mode = _classify_mode(req.query, req.mode)

    # 1) Query embedding
    q_vec = embed_text([req.query])[0]

    # 1a) Sparse queries
    content_sparse_query, summary_sparse_query = _build_sparse_queries_for_query(req.query, req.use_hybrid)

    # 1b) Filter
    flt = None
    if mode in ("current", "archival"):
        flt = qm.Filter(must=[qm.FieldCondition(key="is_active", match=qm.MatchValue(value=(mode == "current")))])

    # 1c) Stage-1 selection
    cand_doc_ids, doc_map = _stage1_select_documents(q_vec, flt, summary_sparse_query, req)
    if not cand_doc_ids:
        return SearchResponse(took_ms=int((time.time() - t0) * 1000), hits=[])

    # 2) Stage-2 re-rank
    final_hits, mmr_pool, rel2 = _stage2_select_chunks(cand_doc_ids, q_vec, content_sparse_query, req)
    if not final_hits:
        return SearchResponse(took_ms=int((time.time() - t0) * 1000), hits=[])

    # Shape response
    results, groups_payload, blocks_payload = _shape_results(final_hits, doc_map, mmr_pool, rel2, req)

    if req.result_format == "blocks":
        took_ms = int((time.time() - t0) * 1000)
        if settings.debug:
            logger.debug(
                "Search took %d ms | stage1_docs=%d stage2_candidates=%d returned_blocks=%d",
                took_ms,
                len(doc_map) if 'doc_map' in locals() else 0,
                len(mmr_pool) if 'mmr_pool' in locals() else 0,
                len(blocks_payload or []),
            )
        return SearchResponse(took_ms=took_ms, hits=[], groups=None, blocks=blocks_payload)

    took_ms = int((time.time() - t0) * 1000)
    if settings.debug:
        logger.debug(
            "Search took %d ms | stage1_docs=%d stage2_candidates=%d returned=%d",
            took_ms,
            len(doc_map) if 'doc_map' in locals() else 0,
            len(mmr_pool) if 'mmr_pool' in locals() else 0,
            len(results),
        )
    return SearchResponse(took_ms=took_ms, hits=results, groups=groups_payload, blocks=blocks_payload)
