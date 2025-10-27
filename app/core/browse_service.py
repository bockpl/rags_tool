"""Lightweight browse/analytics over the corpus (LLM-friendly).

Provides simple operations without MMR/rerank/shaping, such as:
- count how many candidate documents match a query (Stage-1 scope),
- list candidate doc ids and basic metadata,
- facet counts over candidate documents.

The implementation reuses Stage-1 selection primitives to keep semantics
consistent with answer-oriented retrieval while remaining cheap to compute.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from qdrant_client.http import models as qm

from app.core.embedding import embed_query
from app.core.search import (
    _build_sparse_queries_for_query,
    _classify_mode,
    _stage1_select_documents,
)
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
    """Return union of candidate document ids across queries and a metadata map.

    Returns (doc_ids, doc_map, approx) where `approx` is True when any per-query
    candidate list hit the `top_m` cap (indicating the union may be truncated
    w.r.t. the full corpus).
    """
    if not params.queries:
        return set(), {}, False
    unified_mode, flt = _unified_mode_and_filter(params.queries, params.mode)
    approx = False
    union: Set[str] = set()
    merged_map: Dict[str, Dict[str, Any]] = {}
    for q in params.queries:
        q_vec = embed_query([q])[0]
        content_sparse_query, summary_sparse_query = _build_sparse_queries_for_query(q, params.use_hybrid)
        req_like = type("_Req", (), {
            "top_m": params.top_m,
            "score_norm": "minmax",
            "dense_weight": 0.6,
            "sparse_weight": 0.4,
            "mmr_stage1": True,
            "mmr_lambda": 0.3,
            "rep_alpha": None,
            "use_hybrid": params.use_hybrid,
            "entity_strategy": "auto",
        })()
        doc_ids, doc_map = _stage1_select_documents(q, q_vec, flt, summary_sparse_query, req_like)
        if len(doc_ids) >= max(1, params.top_m):
            approx = True
        for did in doc_ids:
            union.add(did)
            if did not in merged_map and did in doc_map:
                merged_map[did] = doc_map[did]
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

