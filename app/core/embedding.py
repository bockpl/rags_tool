"""Embedding client and TF-IDF utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer

from app.settings import get_settings

settings = get_settings()

VECTOR_STORE_DIR: Path = settings.vector_store_dir
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

VECTORIZER_PATH = VECTOR_STORE_DIR / "tfidf_vectorizer.json"
SUMMARY_VECTORIZER_PATH = VECTOR_STORE_DIR / "tfidf_vectorizer_summary.json"

# OpenAI-compatible embedding client
embedding_client = OpenAI(base_url=settings.embedding_api_url, api_key=settings.embedding_api_key)

try:  # optional tokenizer for precise truncation
    import tiktoken  # type: ignore

    _EMB_TOKENIZER = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover
    tiktoken = None  # type: ignore
    _EMB_TOKENIZER = None  # type: ignore


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


def embed_text(texts: List[str]) -> List[List[float]]:
    """Legacy embedding helper (no prefixing). Prefer embed_query/embed_passage."""
    return _embed_raw(texts)


def embed_query(texts: List[str]) -> List[List[float]]:
    """Embed query strings with a model-specific query prefix."""
    limited = _limit_for_embedding(_maybe_prefix(texts, settings.embedding_query_prefix))
    return _embed_raw(limited)


def embed_passage(texts: List[str]) -> List[List[float]]:
    """Embed documents/passages with a model-specific passage prefix."""
    limited = _limit_for_embedding(_maybe_prefix(texts, settings.embedding_passage_prefix))
    return _embed_raw(limited)


def get_embedding_dim() -> int:
    vec = _embed_raw(["test"])[0]
    return len(vec)


def _count_tokens(text: str) -> int:
    if not text:
        return 0
    if _EMB_TOKENIZER is None:
        return max(1, len(text) // 4)
    try:
        return len(_EMB_TOKENIZER.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0 or not text:
        return ""
    if _EMB_TOKENIZER is None:
        # char-based heuristic (~4 chars/token)
        max_chars = max_tokens * 4
        return text[:max_chars]
    try:
        toks = _EMB_TOKENIZER.encode(text)
        if len(toks) <= max_tokens:
            return text
        return _EMB_TOKENIZER.decode(toks[:max_tokens])
    except Exception:
        max_chars = max_tokens * 4
        return text[:max_chars]


def _limit_for_embedding(texts: List[str]) -> List[str]:
    """Ensure each input respects EMBEDDING_MAX_TOKENS to avoid 400 errors."""
    limit = int(getattr(settings, "embedding_max_tokens", 512) or 512)
    safe_limit = max(1, limit)
    out: List[str] = []
    for t in texts:
        ts = str(t or "")
        if _count_tokens(ts) > safe_limit:
            ts = _truncate_to_tokens(ts, safe_limit)
        out.append(ts)
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
