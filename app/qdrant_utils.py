"""Qdrant client and collection utilities, point upsert helpers."""

from __future__ import annotations

import hashlib
import logging
import pathlib
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

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


def _derive_summary_collection(base: Optional[str]) -> str:
    if base:
        return f"{base}_summaries"
    if settings.summary_collection_name:
        return settings.summary_collection_name
    return f"{settings.collection_name}_summaries"


def _derive_content_collection(base: Optional[str]) -> str:
    if base:
        return f"{base}_content"
    if settings.content_collection_name:
        return settings.content_collection_name
    return f"{settings.collection_name}_content"


def derive_collection_names(base: Optional[str] = None) -> Tuple[str, str]:
    return _derive_summary_collection(base), _derive_content_collection(base)


def _ensure_single_collection(
    collection: str,
    vectors_config: Dict[str, qm.VectorParams],
    sparse_config: Optional[Dict[str, qm.SparseVectorParams]] = None,
):
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
            expected = set(vectors_config.keys())
            if names != expected:
                logger.error(
                    "Collection '%s' incompatible vectors config. Found names=%s, expected %s",
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
        if SPARSE_ENABLED and sparse_config:
            try:
                qdrant.update_collection(
                    collection_name=collection,
                    sparse_vectors_config=sparse_config,
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


def ensure_collections(collection_base: Optional[str] = None, dim: Optional[int] = None):
    if dim is None:
        dim = settings.embedding_dim
    summary_collection, content_collection = derive_collection_names(collection_base)
    summary_vectors = {
        SUMMARY_VECTOR_NAME: qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
    }
    content_vectors = {
        CONTENT_VECTOR_NAME: qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
    }
    summary_sparse = (
        {SUMMARY_SPARSE_NAME: qm.SparseVectorParams()}
        if SPARSE_ENABLED
        else None
    )
    content_sparse = (
        {CONTENT_SPARSE_NAME: qm.SparseVectorParams()}
        if SPARSE_ENABLED
        else None
    )
    _ensure_single_collection(summary_collection, summary_vectors, summary_sparse)
    _ensure_single_collection(content_collection, content_vectors, content_sparse)


def build_and_upsert_points(
    doc_records: Iterable[Dict[str, Any]],
    content_vec,
    summary_vec,
    *,
    enable_sparse: bool,
    collection_base: Optional[str],
) -> int:
    summary_collection, content_collection = derive_collection_names(collection_base)
    batch_limit = 256
    summary_points: List[qm.PointStruct] = []
    content_points: List[qm.PointStruct] = []
    point_count = 0
    replacement_relations: Dict[str, List[str]] = {}
    doc_paths: Dict[str, str] = {}
    doc_titles: Dict[str, str] = {}

    for rec in doc_records:
        doc_id = rec["doc_id"]
        path = rec["path"]
        chunks = rec["chunks"]
        doc_summary = rec["doc_summary"]
        doc_signature = rec["doc_signature"]
        summary_sparse_text = rec["summary_sparse_text"]
        replacement_info = str(rec.get("replacement", "") or "brak").strip() or "brak"
        if replacement_info.lower() == "brak":
            replacement_info = "brak"
        doc_title = str(rec.get("doc_title") or rec.get("title") or "").strip()
        if not doc_title:
            try:
                doc_title = pathlib.Path(path).stem
            except Exception:
                doc_title = ""

        doc_paths[doc_id] = path
        if doc_title:
            doc_titles[doc_id] = doc_title
        normalized_refs = _parse_replacement_list(replacement_info)
        if normalized_refs:
            replacement_relations[doc_id] = normalized_refs

        summary_dense_vec = embed_text([doc_summary])[0]
        content_texts = [c.get("text", c) if isinstance(c, dict) else str(c) for c in chunks]
        content_vecs = embed_text(content_texts)

        if enable_sparse:
            sparse_chunks = tfidf_vector(content_texts, content_vec)
            if summary_vec is not None and summary_sparse_text:
                summary_sparse = tfidf_vector(
                    [summary_sparse_text], summary_vec, path=SUMMARY_VECTORIZER_PATH
                )[0]
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
            "title": doc_title,
            "summary": doc_summary,
            "signature": doc_signature,
        }
        summary_payload["replacement"] = replacement_info
        summary_vectors: Dict[str, Any] = {
            SUMMARY_VECTOR_NAME: summary_dense_vec,
        }
        if SPARSE_ENABLED and enable_sparse and summary_sparse[0]:
            summary_vectors[SUMMARY_SPARSE_NAME] = qm.SparseVector(
                indices=summary_sparse[0], values=summary_sparse[1]
            )
            summary_payload["summary_sparse_indices"] = summary_sparse[0]
            summary_payload["summary_sparse_values"] = summary_sparse[1]

        summary_pid = int(str(int(sha1(f"{doc_id}:summary")[0:12], 16))[:12])
        summary_points.append(
            qm.PointStruct(id=summary_pid, vector=summary_vectors, payload=summary_payload)
        )
        point_count += 1
        if len(summary_points) >= batch_limit:
            qdrant.upsert(collection_name=summary_collection, points=summary_points)
            summary_points = []

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
            }

            if SPARSE_ENABLED and enable_sparse:
                indices, values = sparse_chunks[i]
                if indices:
                    vectors[CONTENT_SPARSE_NAME] = qm.SparseVector(indices=indices, values=values)
                    payload["content_sparse_indices"] = indices
                    payload["content_sparse_values"] = values

            content_points.append(qm.PointStruct(id=pid, vector=vectors, payload=payload))
            point_count += 1
            if len(content_points) >= batch_limit:
                qdrant.upsert(collection_name=content_collection, points=content_points)
                content_points = []

    if summary_points:
        qdrant.upsert(collection_name=summary_collection, points=summary_points)
    if content_points:
        qdrant.upsert(collection_name=content_collection, points=content_points)

    _apply_replacement_statuses(
        summary_collection,
        content_collection,
        replacement_relations,
        doc_paths,
        doc_titles,
    )

    return point_count


def _parse_replacement_list(raw: str) -> List[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    if text.lower() == "brak":
        return []
    parts = re.split(r"[,;\n]", text)
    items = [part.strip() for part in parts if part.strip()]
    return items


def _normalize_reference(value: str) -> str:
    norm = value.strip().lower()
    norm = norm.replace("\\", "/")
    norm = re.sub(r"\s+", " ", norm)
    return norm


def _apply_replacement_statuses(
    summary_collection: str,
    content_collection: str,
    replacements: Dict[str, List[str]],
    doc_paths: Dict[str, str],
    doc_titles: Optional[Dict[str, str]] = None,
):
    if not replacements:
        return

    name_to_doc: Dict[str, str] = {}
    titles = doc_titles or {}
    for doc_id, path in doc_paths.items():
        norm_path = _normalize_reference(path)
        name_to_doc[norm_path] = doc_id
        path_obj = pathlib.Path(path)
        name_to_doc[_normalize_reference(path_obj.name)] = doc_id
        name_to_doc[_normalize_reference(path_obj.stem)] = doc_id
        name_to_doc[doc_id] = doc_id
        title_val = titles.get(doc_id)
        if title_val:
            name_to_doc[_normalize_reference(title_val)] = doc_id

    deactivate: Set[str] = set()
    keep_active: Set[str] = set(replacements.keys())

    for replacer_id, refs in replacements.items():
        for ref in refs:
            norm_ref = _normalize_reference(ref)
            candidate = name_to_doc.get(norm_ref)
            if not candidate and norm_ref:
                # try unique substring match across known names
                matches = {
                    doc
                    for name, doc in name_to_doc.items()
                    if norm_ref in name and doc != replacer_id
                }
                if len(matches) == 1:
                    candidate = matches.pop()
            if candidate and candidate != replacer_id:
                deactivate.add(candidate)
            elif candidate is None:
                logger.debug(
                    "Replacement reference not matched | doc_id=%s reference=%s",
                    replacer_id,
                    ref,
                )

    for doc_id in deactivate:
        _set_is_active_flag(summary_collection, content_collection, doc_id, False)

    for doc_id in keep_active:
        if doc_id in deactivate:
            continue
        _set_is_active_flag(summary_collection, content_collection, doc_id, True)


def _set_is_active_flag(
    summary_collection: str,
    content_collection: str,
    doc_id: str,
    is_active: bool,
):
    payload = {"is_active": is_active}
    for collection in (summary_collection, content_collection):
        filter_clause = qm.Filter(
            must=[qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))]
        )
        point_ids = _collect_point_ids(collection, filter_clause)
        if not point_ids:
            continue
        try:
            qdrant.set_payload(
                collection_name=collection,
                payload=payload,
                points=point_ids,
            )
        except Exception as exc:
            logger.warning(
                "Failed to update is_active flag | collection=%s doc_id=%s error=%s",
                collection,
                doc_id,
                exc,
            )


def _collect_point_ids(collection: str, flt: qm.Filter) -> List[int]:
    ids: List[int] = []
    offset = None
    try:
        while True:
            res = qdrant.scroll(
                collection_name=collection,
                scroll_filter=flt,
                limit=256,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            if isinstance(res, tuple):
                records, offset = res
            else:
                records = getattr(res, "points", None)
                offset = getattr(res, "next_page_offset", None)
                if records is None:
                    records = []
            if not records:
                break
            ids.extend([rec.id for rec in records if rec.id is not None])
            if offset is None:
                break
    except Exception as exc:
        logger.warning(
            "Failed to collect point ids | collection=%s error=%s",
            collection,
            exc,
        )
        return []
    return ids
