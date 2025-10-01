"""Embedding client and TF-IDF utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


def embed_text(texts: List[str]) -> List[List[float]]:
    rsp = embedding_client.embeddings.create(model=settings.embedding_model, input=texts)
    return [d.embedding for d in rsp.data]


def get_embedding_dim() -> int:
    vec = embed_text(["test"])[0]
    return len(vec)


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


def fit_vectorizer(corpus: List[str], path: Path = VECTORIZER_PATH) -> TfidfVectorizer:
    vec = TfidfVectorizer(**_vectorizer_params_for_corpus(len(corpus)))
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


def prepare_tfidf(
    all_chunks: List[str],
    summary_corpus: List[str],
    enable_sparse: bool,
    rebuild_tfidf: bool,
) -> Tuple[Optional[TfidfVectorizer], Optional[TfidfVectorizer]]:
    if not enable_sparse:
        return None, None
    content_vec: Optional[TfidfVectorizer] = None
    summary_vec: Optional[TfidfVectorizer] = None
    if all_chunks:
        if rebuild_tfidf or not VECTORIZER_PATH.exists():
            content_vec = fit_vectorizer(all_chunks)
        else:
            content_vec = load_vectorizer()
            if content_vec is None:
                content_vec = fit_vectorizer(all_chunks)
    else:
        content_vec = load_vectorizer()

    if summary_corpus:
        if rebuild_tfidf or not SUMMARY_VECTORIZER_PATH.exists():
            summary_vec = fit_vectorizer(summary_corpus, path=SUMMARY_VECTORIZER_PATH)
        else:
            summary_vec = load_vectorizer(path=SUMMARY_VECTORIZER_PATH)
            if summary_vec is None:
                summary_vec = fit_vectorizer(summary_corpus, path=SUMMARY_VECTORIZER_PATH)
    else:
        summary_vec = load_vectorizer(path=SUMMARY_VECTORIZER_PATH)

    return content_vec, summary_vec
