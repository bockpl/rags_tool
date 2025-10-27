"""Lightweight browse/analytics over the corpus (LLM-friendly).

Provides simple operations without MMR/rerank/shaping, such as:
- count how many candidate documents match a query (content scope),
- list candidate doc ids and basic metadata,
- facet counts over candidate documents.

Implementation note:
- Selection operates on document content (chunk-level points) and does not
  search through summaries. Summaries may be fetched only to enrich metadata
  (title/doc_date) after candidate selection.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from qdrant_client.http import models as qm

from app.core.embedding import embed_query
from app.core.search import (
    _build_sparse_queries_for_query,
    _classify_mode,
)
from app.core.constants import CONTENT_VECTOR_NAME, CONTENT_SPARSE_NAME, SPARSE_ENABLED
from app.qdrant_utils import qdrant
from app.core.store_access import fetch_doc_summaries
from app.settings import get_settings

settings = get_settings()


class BrowseParams:
    """Input parameters for browse operations."""

    def __init__(
        self,
        queries: List[str],
        *,
        top_m: int = 100,
        use_hybrid: bool = True,
        mode: str = "auto",
    ) -> None:
        self.queries = [str(q).strip() for q in queries if str(q).strip()]
        self.top_m = int(top_m)
        self.use_hybrid = bool(use_hybrid)
        self.mode = str(mode or "auto")


def _unified_mode_and_filter(queries: List[str], mode: str) -> Tuple[str, Optional[qm.Filter]]:
    """Return unified mode and filter for Stage‑1 based on queries and requested mode."""
    if mode != "auto":
        m = mode
    else:
        modes = {_classify_mode(q, "auto") for q in queries}
        if modes == {"current"}:
            m = "current"
        elif modes == {"archival"}:
            m = "archival"
        else:
            m = "all"
    if m == "current":
        flt = qm.Filter(must=[qm.FieldCondition(key="is_active", match=qm.MatchValue(value=True))])
    elif m == "archival":
        flt = qm.Filter(must=[qm.FieldCondition(key="is_active", match=qm.MatchValue(value=False))])
    else:
        flt = None
    return m, flt


def stage1_candidates(params: BrowseParams) -> Tuple[Set[str], Dict[str, Dict[str, Any]], bool]:
    """Return union of candidate document ids across queries (content-based).

    Selection is performed over chunk-level content vectors (dense + optional
    TF-IDF). Summaries are not used for searching. Returns (doc_ids, doc_map,
    approx) where `approx` is True when any per-query unique doc_id list hit the
    `top_m` cap.
    """
    if not params.queries:
        return set(), {}, False

    _, flt = _unified_mode_and_filter(params.queries, params.mode)
    approx = False
    union: Set[str] = set()
    merged_map: Dict[str, Dict[str, Any]] = {}

    # Base filter: point_type == chunk plus optional is_active condition
    base_must = [qm.FieldCondition(key="point_type", match=qm.MatchValue(value="chunk"))]
    if flt and getattr(flt, "must", None):
        base_must = list(flt.must) + base_must
    flt_chunks = qm.Filter(must=base_must)

    for q in params.queries:
        q_vec = embed_query([q])[0]
        content_sparse_query, _ = _build_sparse_queries_for_query(q, params.use_hybrid)

        per_query_doc_ids: List[str] = []
        seen_local: Set[str] = set()

        # Dense search over content
        try:
            dense_hits = qdrant.search(
                collection_name=settings.qdrant_content_collection,
                query_vector=(CONTENT_VECTOR_NAME, q_vec),
                query_filter=flt_chunks,
                limit=max(50, params.top_m * 3),
                with_payload=["doc_id", "is_active"],
                with_vectors=False,
                search_params=qm.SearchParams(exact=False, hnsw_ef=128),
            )
        except Exception:
            dense_hits = []

        for hit in dense_hits:
            payload = hit.payload or {}
            did = str(payload.get("doc_id") or "")
            if not did or did in seen_local:
                continue
            seen_local.add(did)
            per_query_doc_ids.append(did)
            # Record minimal metadata that we can derive without summaries
            merged_map.setdefault(did, {})
            if merged_map[did].get("is_active") is None and payload.get("is_active") is not None:
                merged_map[did]["is_active"] = bool(payload.get("is_active"))
            if len(per_query_doc_ids) >= params.top_m:
                break

        # Optional TF-IDF sparse over content to supplement recall
        if len(per_query_doc_ids) < params.top_m and params.use_hybrid and SPARSE_ENABLED and content_sparse_query is not None:
            try:
                c_idx, c_val = content_sparse_query
                sparse_hits = qdrant.search(
                    collection_name=settings.qdrant_content_collection,
                    query_vector=(CONTENT_SPARSE_NAME, qm.SparseVector(indices=c_idx, values=c_val)),
                    query_filter=flt_chunks,
                    limit=max(50, params.top_m * 3),
                    with_payload=["doc_id", "is_active"],
                    with_vectors=False,
                    search_params=qm.SearchParams(exact=False, hnsw_ef=128),
                )
            except Exception:
                sparse_hits = []
            for hit in sparse_hits:
                payload = hit.payload or {}
                did = str(payload.get("doc_id") or "")
                if not did or did in seen_local:
                    continue
                seen_local.add(did)
                per_query_doc_ids.append(did)
                merged_map.setdefault(did, {})
                if merged_map[did].get("is_active") is None and payload.get("is_active") is not None:
                    merged_map[did]["is_active"] = bool(payload.get("is_active"))
                if len(per_query_doc_ids) >= params.top_m:
                    break

        if len(per_query_doc_ids) >= params.top_m:
            approx = True

        for did in per_query_doc_ids:
            union.add(did)

    return union, merged_map, approx


def count_documents(params: BrowseParams) -> Tuple[int, bool]:
    """Count candidate documents (Stage‑1) across queries.

    Returns (total, approx).
    """
    doc_ids, _, approx = stage1_candidates(params)
    return len(doc_ids), approx


def list_document_minimal(params: BrowseParams, limit: int = 200) -> Tuple[List[Dict[str, Any]], bool]:
    """List up to `limit` candidate documents with minimal metadata.

    Returns (docs, approx). Ordering is stable by title then doc_id.
    """
    ids, doc_map, approx = stage1_candidates(params)
    selected = sorted(ids)
    if limit and len(selected) > limit:
        approx = True
        selected = selected[:limit]
    # Enrich via summaries when needed
    missing = [d for d in selected if d not in doc_map]
    if missing:
        doc_map.update(fetch_doc_summaries(missing))
    docs = [
        {
            "doc_id": did,
            "title": (doc_map.get(did, {}) or {}).get("doc_title"),
            "doc_date": (doc_map.get(did, {}) or {}).get("doc_date"),
            "is_active": (doc_map.get(did, {}) or {}).get("is_active"),
        }
        for did in selected
    ]
    docs.sort(key=lambda x: ((x.get("title") or "").lower(), x.get("doc_id") or ""))
    return docs, approx


def facet_counts(params: BrowseParams, fields: Iterable[str]) -> Tuple[Dict[str, Dict[str, int]], bool]:
    """Compute simple facets over candidate documents.

    Supported fields:
    - is_active
    - year (derived from doc_date prefix YYYY)
    """
    ids, doc_map, approx = stage1_candidates(params)
    if ids:
        missing = [d for d in ids if d not in doc_map]
        if missing:
            doc_map.update(fetch_doc_summaries(missing))
    fields_norm = [str(f).strip().lower() for f in fields]
    out: Dict[str, Dict[str, int]] = {}
    for did in ids:
        meta = doc_map.get(did, {})
        for f in fields_norm:
            if f == "is_active":
                key = "true" if bool(meta.get("is_active")) else "false"
                out.setdefault("is_active", {})[key] = out.get("is_active", {}).get(key, 0) + 1
            elif f == "year":
                dd = str(meta.get("doc_date") or "")
                year = dd[:4] if len(dd) >= 4 and dd[:4].isdigit() else "brak"
                out.setdefault("year", {})[year] = out.get("year", {}).get(year, 0) + 1
    return out, approx
