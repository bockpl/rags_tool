"""Embedding client and TF-IDF utilities."""

from __future__ import annotations

import json
from pathlib import Path
import logging
import hashlib
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
from openai import OpenAI
from openai import BadRequestError  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer

from app.core.tokenizer import (
    TokenizerAdapter,
    count_tokens as _adapter_count_tokens,
    load_tokenizer,
    truncate_to_tokens as _adapter_truncate,
)
from app.settings import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

VECTOR_STORE_DIR: Path = settings.vector_store_dir
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

VECTORIZER_PATH = VECTOR_STORE_DIR / "tfidf_vectorizer.json"
SUMMARY_VECTORIZER_PATH = VECTOR_STORE_DIR / "tfidf_vectorizer_summary.json"

# OpenAI-compatible embedding client
embedding_client = OpenAI(base_url=settings.embedding_api_url, api_key=settings.embedding_api_key)

_TOKENIZER_ADAPTER: TokenizerAdapter = load_tokenizer(getattr(settings, "embedding_tokenizer", None))


class IterableCorpus:
    """Re-iterable corpus proxy storing size metadata for TF-IDF fitting."""

    def __init__(self, size: int, factory: Callable[[], Iterable[str]]):
        self.size = size
        self._factory = factory

    def __len__(self) -> int:
        return self.size

    def __iter__(self):
        yield from self._factory()


def _embed_raw(texts: List[str]) -> List[List[float]]:
    rsp = embedding_client.embeddings.create(model=settings.embedding_model, input=texts)
    return [d.embedding for d in rsp.data]


def _maybe_prefix(texts: List[str], prefix: str) -> List[str]:
    if not prefix:
        return texts
    pref = str(prefix)
    out: List[str] = []
    for t in texts:
        ts = str(t or "")
        if ts.startswith(pref):
            out.append(ts)
        else:
            out.append(pref + ts)
    return out


def embed_query(texts: List[str]) -> List[List[float]]:
    """Embed query strings with a model-specific query prefix.

    Minimal path + diagnostics on error. No retries/truncation beyond a single
    safety cap in _limit_for_embedding to surface the exact failing inputs.
    """
    prefixed = _maybe_prefix(texts, settings.embedding_query_prefix)
    limited = _limit_for_embedding(prefixed)
    try:
        return _embed_raw(limited)
    except BadRequestError as exc:
        _log_embedding_error(prefixed, limited, purpose="query", exc=exc)
        raise


def embed_passage(texts: List[str]) -> List[List[float]]:
    """Embed documents/passages with a model-specific passage prefix.

    Minimal path + diagnostics on error. No retries/truncation beyond a single
    safety cap in _limit_for_embedding to surface the exact failing inputs.
    """
    prefixed = _maybe_prefix(texts, settings.embedding_passage_prefix)
    limited = _limit_for_embedding(prefixed)
    try:
        return _embed_raw(limited)
    except BadRequestError as exc:
        _log_embedding_error(prefixed, limited, purpose="passage", exc=exc)
        raise


def get_embedding_dim() -> int:
    vec = _embed_raw(["test"])[0]
    return len(vec)


def _count_tokens(text: str) -> int:
    return _adapter_count_tokens(_TOKENIZER_ADAPTER, text)


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    return _adapter_truncate(_TOKENIZER_ADAPTER, text, max_tokens)


def _limit_for_embedding(texts: List[str]) -> List[str]:
    """Ensure each input respects EMBEDDING_MAX_TOKENS to avoid 400 errors."""
    limit = int(getattr(settings, "embedding_max_tokens"))
    safe_limit = max(1, limit)
    out: List[str] = []
    for t in texts:
        ts = str(t or "")
        if _count_tokens(ts) > safe_limit:
            ts = _truncate_to_tokens(ts, safe_limit)
        out.append(ts)
    return out

def _log_embedding_error(original: List[str], limited: List[str], *, purpose: str, exc: Exception) -> None:
    """Log detailed diagnostics for a failed embeddings batch.

    Includes per-item char lengths, token estimates, prefix presence, and a
    short head/tail sample along with a stable sha1 digest to identify texts.
    """
    try:
        max_tok = int(getattr(settings, "embedding_max_tokens"))
    except Exception:
        max_tok = 512
    try:
        chunk_tokens_setting = int(getattr(settings, "chunk_tokens"))
    except Exception:
        chunk_tokens_setting = None  # type: ignore[assignment]
    try:
        chunk_overlap_setting = int(getattr(settings, "chunk_overlap"))
    except Exception:
        chunk_overlap_setting = None  # type: ignore[assignment]
    model = getattr(settings, "embedding_model", "unknown")
    items: List[Dict[str, Any]] = []
    # Select only suspicious (over-limit by local estimate); if none, show top 10 by tokens
    estimates = [(_count_tokens(s or ""), i) for i, s in enumerate(original)]
    over = [i for (tok, i) in estimates if tok > max_tok]
    order = over if over else [i for (_, i) in sorted(estimates, key=lambda p: p[0], reverse=True)[:10]]
    for i in order:
        src = str(original[i] or "")
        lim = str(limited[i] or "") if i < len(limited) else ""
        items.append(
            {
                "i": i,
                "sha1": hashlib.sha1(src.encode("utf-8", "ignore")).hexdigest(),
                "char_len": len(src),
                "token_est": _count_tokens(src),
                "char_len_limited": len(lim),
                "token_est_limited": _count_tokens(lim),
                "head": src[:120].replace("\n", " "),
                "tail": src[-120:].replace("\n", " ") if len(src) > 120 else "",
            }
        )
    payload = {
        "purpose": purpose,
        "model": model,
        "items": items,
        "batch_size": len(original),
        "max_tokens_setting": max_tok,
        "chunk_tokens_setting": chunk_tokens_setting,
        "chunk_overlap_setting": chunk_overlap_setting,
    }
    logger.error("Embedding request failed | details=%s | error=%s", json.dumps(payload, ensure_ascii=False), exc)


def _embed_many(texts: List[str]) -> List[List[float]]:
    """Embed list of texts with micro-batching and robust retries.

    Uses settings.embedding_batch_size to avoid overwhelming the backend and
    to bypass per-request engine limits. Each micro-batch benefits from the
    same retry logic and fallbacks.
    """
    if not texts:
        return []
    try:
        batch_size = int(getattr(settings, "embedding_batch_size", 32) or 32)
    except Exception:
        batch_size = 32
    batch_size = max(1, min(256, batch_size))
    out: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        vecs = _embed_with_retry_batch(chunk)
        out.extend(vecs)
    return out


def load_vectorizer(path: Path = VECTORIZER_PATH) -> Optional[TfidfVectorizer]:
    if path.exists():
        obj = json.loads(path.read_text())
        vec = TfidfVectorizer(**obj["params"])  # type: ignore
        vec.vocabulary_ = {k: int(v) for k, v in obj["vocab"].items()}  # type: ignore
        vec.idf_ = np.array(obj["idf"])  # type: ignore
        return vec
    return None


def save_vectorizer(vec: TfidfVectorizer, path: Path = VECTORIZER_PATH):
    params = vec.get_params()
    payload = {
        "params": {k: v for k, v in params.items() if k in ["lowercase", "ngram_range", "min_df", "max_df"]},
        "vocab": {k: int(v) for k, v in vec.vocabulary_.items()},
        "idf": vec.idf_.tolist(),
    }
    path.write_text(json.dumps(payload))


def _vectorizer_params_for_corpus(corpus_size: int) -> Dict[str, Any]:
    if corpus_size <= 1:
        return {"lowercase": True, "ngram_range": (1, 2), "min_df": 1, "max_df": 1.0}
    if corpus_size == 2:
        return {"lowercase": True, "ngram_range": (1, 2), "min_df": 1, "max_df": 1.0}
    return {"lowercase": True, "ngram_range": (1, 2), "min_df": 2, "max_df": 0.9}


def fit_vectorizer(
    corpus: Iterable[str], corpus_size: int, path: Path = VECTORIZER_PATH
) -> TfidfVectorizer:
    vec = TfidfVectorizer(**_vectorizer_params_for_corpus(corpus_size))
    vec.fit(corpus)
    save_vectorizer(vec, path=path)
    return vec


def tfidf_vector(
    texts: List[str], vec: Optional[TfidfVectorizer], path: Path = VECTORIZER_PATH
) -> List[Tuple[List[int], List[float]]]:
    if not vec:
        vec = load_vectorizer(path)
        if not vec:
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


def _normalize_corpus(corpus: Optional[Iterable[str]]) -> Tuple[Optional[Iterable[str]], int]:
    if corpus is None:
        return None, 0
    try:
        size = len(corpus)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError("Corpus object must define __len__ for TF-IDF preparation") from exc
    if size == 0:
        return None, 0
    return corpus, int(size)


def prepare_tfidf(
    all_chunks: Optional[Iterable[str]],
    summary_corpus: Optional[Iterable[str]],
    enable_sparse: bool,
    rebuild_tfidf: bool,
) -> Tuple[Optional[TfidfVectorizer], Optional[TfidfVectorizer]]:
    if not enable_sparse:
        return None, None
    content_vec: Optional[TfidfVectorizer] = None
    summary_vec: Optional[TfidfVectorizer] = None

    chunk_iter, chunk_size = _normalize_corpus(all_chunks)
    summary_iter, summary_size = _normalize_corpus(summary_corpus)

    if chunk_iter is not None:
        if rebuild_tfidf or not VECTORIZER_PATH.exists():
            content_vec = fit_vectorizer(chunk_iter, chunk_size)
        else:
            content_vec = load_vectorizer()
            if content_vec is None:
                content_vec = fit_vectorizer(chunk_iter, chunk_size)
    else:
        content_vec = load_vectorizer()

    if summary_iter is not None:
        if rebuild_tfidf or not SUMMARY_VECTORIZER_PATH.exists():
            summary_vec = fit_vectorizer(summary_iter, summary_size, path=SUMMARY_VECTORIZER_PATH)
        else:
            summary_vec = load_vectorizer(path=SUMMARY_VECTORIZER_PATH)
            if summary_vec is None:
                summary_vec = fit_vectorizer(summary_iter, summary_size, path=SUMMARY_VECTORIZER_PATH)
    else:
        summary_vec = load_vectorizer(path=SUMMARY_VECTORIZER_PATH)

    return content_vec, summary_vec
