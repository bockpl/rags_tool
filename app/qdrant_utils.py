"""Qdrant client and collection utilities, point upsert helpers."""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from qdrant_client.http.exceptions import UnexpectedResponse

from app.core.embedding import (
    SUMMARY_VECTORIZER_PATH,
    embed_text,
    tfidf_vector,
)
from app.core.constants import (
    CONTENT_SPARSE_NAME,
    CONTENT_VECTOR_NAME,
    SPARSE_ENABLED,
    SUMMARY_SPARSE_NAME,
    SUMMARY_VECTOR_NAME,
)
from app.settings import get_settings
from fastapi import HTTPException

settings = get_settings()

logger = logging.getLogger("rags_tool")

qdrant = QdrantClient(
    url=settings.qdrant_url,
    api_key=settings.qdrant_api_key,
    timeout=settings.qdrant_request_timeout,
)


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def ensure_collection(collection: Optional[str] = None, dim: Optional[int] = None):
    collection = collection or settings.collection_name
    if dim is None:
        dim = settings.embedding_dim
    vectors_config = {
        CONTENT_VECTOR_NAME: qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
        SUMMARY_VECTOR_NAME: qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
    }
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
        except Exception as exc:
            logger.warning("Could not verify vectors config for '%s': %s", collection, exc)
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
        if getattr(exc, "status_code", None) == 409 or "already exists" in str(exc).lower():
            logger.debug("Collection '%s' creation returned conflict; treating as existing", collection)
            return
        raise


def build_and_upsert_points(
    doc_records: Iterable[Dict[str, Any]],
    content_vec,
    summary_vec,
    *,
    enable_sparse: bool,
    collection_name: str,
) -> int:
    points: List[qm.PointStruct] = []
    batch_limit = 256
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
            sparse_chunks = tfidf_vector(content_texts, content_vec)
            if summary_vec is not None and summary_sparse_text:
                summary_sparse = tfidf_vector([summary_sparse_text], summary_vec, path=SUMMARY_VECTORIZER_PATH)[0]
            else:
                summary_sparse = ([], [])
        else:
            sparse_chunks = [([], []) for _ in chunks]
            summary_sparse = ([], [])

        summary_payload: Dict[str, Any] = {
            "doc_id": doc_id,
            "path": path,
            "is_active": True,
            "point_type": "summary",
            "summary": doc_summary,
            "signature": doc_signature,
        }
        summary_vectors: Dict[str, Any] = {
            SUMMARY_VECTOR_NAME: summary_dense_vec,
            CONTENT_VECTOR_NAME: summary_dense_vec,
        }
        if SPARSE_ENABLED and enable_sparse and summary_sparse[0]:
            summary_vectors[SUMMARY_SPARSE_NAME] = qm.SparseVector(
                indices=summary_sparse[0], values=summary_sparse[1]
            )
            summary_payload["summary_sparse_indices"] = summary_sparse[0]
            summary_payload["summary_sparse_values"] = summary_sparse[1]

        summary_pid = int(str(int(sha1(f"{doc_id}:summary")[0:12], 16))[:12])
        points.append(
            qm.PointStruct(id=summary_pid, vector=summary_vectors, payload=summary_payload)
        )
        point_count += 1

        for i, chunk_item in enumerate(chunks):
            if isinstance(chunk_item, dict):
                chunk_text_val = chunk_item.get("text", "")
                section_label = chunk_item.get("section")
            else:
                chunk_text_val = str(chunk_item)
                section_label = None
            pid = int(str(int(sha1(f"{doc_id}:{i}")[0:12], 16))[:12])
            payload: Dict[str, Any] = {
                "doc_id": doc_id,
                "path": path,
                "chunk_id": i,
                "is_active": True,
                "point_type": "chunk",
                "text": chunk_text_val,
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

            points.append(qm.PointStruct(id=pid, vector=vectors, payload=payload))
            point_count += 1

        if len(points) >= batch_limit:
            qdrant.upsert(collection_name=collection_name, points=points)
            points = []

    if points:
        qdrant.upsert(collection_name=collection_name, points=points)

    return point_count
