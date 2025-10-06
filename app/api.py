"""FastAPI endpoints for rags_tool."""

from __future__ import annotations

import base64
import inspect
import io
import json
import logging
import pathlib
import re
import sys
import tempfile
import time
from threading import Lock
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.routing import APIRoute
from pydantic import ValidationError

from app.core.chunking import chunk_text_by_sections
from qdrant_client.http import models as qm
from app.core.embedding import IterableCorpus, prepare_tfidf
from app.core.embedding import embed_query
from app.core.parsing import SUPPORTED_EXT, extract_text
from app.core.search import (
    _build_sparse_queries_for_query,
    _classify_mode,
    _stage1_select_documents,
    _stage2_select_chunks,
    _shape_results,
)
from app.core.summary import llm_summary
from app.core.ranker_client import OpenAIReranker
from app.models import (
    About,
    DebugEmbedRequest,
    DebugEmbedResponse,
    DebugShapeRequest,
    DebugShapeResponse,
    DebugStage1Request,
    DebugStage1Response,
    DebugStage2Request,
    DebugStage2Response,
    CollectionsExportRequest,
    CollectionsImportRequest,
    IngestBuildRequest,
    InitCollectionsRequest,
    ScanRequest,
    ScanResponse,
    SearchQuery,
    SearchResponse,
    SummariesGenerateRequest,
)
from app.qdrant_utils import (
    build_and_upsert_points,
    derive_collection_names,
    ensure_collections,
    export_collections_bundle,
    import_collections_bundle,
    qdrant,
    sha1,
)
from app.settings import get_settings
from app.admin_routes import attach_admin_routes


settings = get_settings()

logger = logging.getLogger("rags_tool")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.DEBUG if settings.debug else logging.INFO)
logger.propagate = False


_collection_init_lock = Lock()
_initialized_collections: Set[str] = set()

# Przygotuj czytelny JSON body dla operacji ingest-build bez ryzykownego łączenia stringów
INGEST_BUILD_BODY = json.dumps(
    {
        "base_dir": "/app/data",
        "glob": "**/*",
        "recursive": True,
        "reindex": False,
        "chunk_tokens": settings.chunk_tokens,
        "chunk_overlap": settings.chunk_overlap,
        "collection_name": "rags_tool",
        "enable_sparse": True,
        "rebuild_tfidf": True,
    },
    ensure_ascii=False,
    indent=2,
)


def _collection_cache_key(base: Optional[str]) -> str:
    summary_collection, content_collection = derive_collection_names(base)
    return f"{summary_collection}::{content_collection}"


def _mark_collections_initialized(base: Optional[str]) -> None:
    key = _collection_cache_key(base)
    with _collection_init_lock:
        _initialized_collections.add(key)


def _clear_collection_cache(base: Optional[str]) -> None:
    key = _collection_cache_key(base)
    with _collection_init_lock:
        _initialized_collections.discard(key)


def _ensure_collections_cached(base: Optional[str] = None) -> None:
    key = _collection_cache_key(base)
    with _collection_init_lock:
        if key in _initialized_collections:
            return
        ensure_collections(base)
        _initialized_collections.add(key)


def _scan_files(base: pathlib.Path, pattern: str, recursive: bool) -> List[pathlib.Path]:
    iterator = base.rglob(pattern) if recursive else base.glob(pattern)
    return [p for p in iterator if p.is_file() and p.suffix.lower() in SUPPORTED_EXT]


def _iter_document_records(
    file_paths: List[pathlib.Path], chunk_tokens: int, chunk_overlap: int
) -> Iterable[Dict[str, Any]]:
    for path in file_paths:
        doc_start = time.time()
        logger.debug("Processing document %s", path)
        raw = extract_text(path)
        chunk_items = chunk_text_by_sections(raw, target_tokens=chunk_tokens, overlap_tokens=chunk_overlap, merge_up_to="ust")
        chunks = chunk_items
        if not chunks:
            logger.debug("Document %s produced no chunks; skipping", path)
            continue
        doc_sum = llm_summary(raw)
        doc_id = sha1(str(path.resolve()))
        doc_title = str(doc_sum.get("title", "") or "").strip() or path.stem
        summary_signature = doc_sum.get("signature", [])
        replacement_info = doc_sum.get("replacement", "brak") or "brak"
        if isinstance(replacement_info, str) and replacement_info.lower() == "brak":
            replacement_info = "brak"
        summary_sparse_parts = [
            doc_title,
            doc_sum.get("summary", ""),
            " ".join(summary_signature),
        ]
        if replacement_info.lower() != "brak":
            summary_sparse_parts.append(replacement_info)
        summary_sparse_text = " ".join(part for part in summary_sparse_parts if part).strip()
        rec = {
            "doc_id": doc_id,
            "path": str(path.resolve()),
            "chunks": chunks,
            "doc_title": doc_title,
            "doc_summary": doc_sum.get("summary", ""),
            "doc_signature": summary_signature,
            "replacement": replacement_info,
            "summary_sparse_text": summary_sparse_text,
        }
        logger.debug(
            "Document %s parsed | chunks=%d summary_len=%d took_ms=%d",
            path,
            len(chunks),
            len(rec["doc_summary"] or ""),
            int((time.time() - doc_start) * 1000),
        )
        yield rec


def _iter_saved_records(path: pathlib.Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _iter_chunk_texts(path: pathlib.Path) -> Iterable[str]:
    for rec in _iter_saved_records(path):
        for chunk in rec.get("chunks", []):
            if isinstance(chunk, dict):
                text = chunk.get("text", "")
            else:
                text = str(chunk)
            if text:
                yield text


def _iter_summary_texts(path: pathlib.Path) -> Iterable[str]:
    for rec in _iter_saved_records(path):
        summary = rec.get("summary_sparse_text")
        if summary:
            yield summary


ADMIN_OPERATION_SPECS: List[Dict[str, Any]] = [
    {"id": "search-debug-embed", "path": "/search/debug/embed", "method": "POST", "label": "Search Debug: 1) embed", "body": "{\"query\":\"Jak działa rags_tool?\",\"mode\":\"auto\",\"use_hybrid\":true}"},
    {"id": "search-debug-stage1", "path": "/search/debug/stage1", "method": "POST", "label": "Search Debug: 2) stage1", "body": "{\"q_text\":\"Jak działa rags_tool?\",\"q_vec\":[0.0],\"mode\":\"auto\",\"use_hybrid\":true,\"top_m\":100,\"score_norm\":\"minmax\",\"dense_weight\":0.6,\"sparse_weight\":0.4,\"mmr_stage1\":true,\"mmr_lambda\":0.3}"},
    {"id": "search-debug-stage2", "path": "/search/debug/stage2", "method": "POST", "label": "Search Debug: 3) stage2", "body": "{\"q_text\":\"Jak działa rags_tool?\",\"q_vec\":[0.0],\"cand_doc_ids\":[\"<doc_id>\"],\"doc_map\":{},\"top_k\":10,\"per_doc_limit\":2,\"score_norm\":\"minmax\",\"dense_weight\":0.6,\"sparse_weight\":0.4,\"mmr_lambda\":0.3}"},
    {"id": "search-debug-shape", "path": "/search/debug/shape", "method": "POST", "label": "Search Debug: 4) shape", "body": "{\"final_hits\":[{\"doc_id\":\"<doc_id>\",\"path\":\"/abs/path\",\"section\":null,\"chunk_id\":0,\"score\":0.5,\"snippet\":\"...\"}],\"result_format\":\"blocks\",\"merge_chunks\":true,\"merge_group_budget_tokens\":1200,\"max_merged_per_group\":1,\"block_join_delimiter\":\"\\n\\n\",\"summary_mode\":\"first\"}"},
    {"id": "about", "path": "/about", "method": "GET"},
    {"id": "health", "path": "/health", "method": "GET"},
    {
        "id": "collections-init",
        "path": "/collections/init",
        "method": "POST",
        "body": "{\n  \"collection_name\": \"rags_tool\",\n  \"force_dim_probe\": false\n}",
    },
    {
        "id": "collections-export",
        "path": "/collections/export",
        "method": "POST",
        "label": "Eksport kolekcji (plik .tar.gz)",
        "body": "{}",
    },
    {
        "id": "collections-import",
        "path": "/collections/import",
        "method": "POST",
        "label": "Import kolekcji z archiwum",
        "body": "{\n  \"archive_base64\": \"<wklej_archiwum_base64>\",\n  \"replace_existing\": true\n}",
        "accepts_file": True,
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
        "body": INGEST_BUILD_BODY,
    },
    {
        "id": "search-query",
        "path": "/search/query",
        "method": "POST",
        "body": "{\n  \"query\": [\n    \"Jak działa rags_tool?\",\n    \"architektura rags_tool\"\n  ],\n  \"top_m\": 10,\n  \"top_k\": 5,\n  \"mode\": \"auto\",\n  \"use_hybrid\": true,\n  \"dense_weight\": 0.6,\n  \"sparse_weight\": 0.4,\n  \"mmr_lambda\": 0.3,\n  \"per_doc_limit\": 2,\n  \"score_norm\": \"minmax\",\n  \"rep_alpha\": 0.6,\n  \"mmr_stage1\": true,\n  \"summary_mode\": \"first\",\n  \"merge_chunks\": true,\n  \"merge_group_budget_tokens\": 1200,\n  \"max_merged_per_group\": 1,\n  \"expand_neighbors\": 1,\n  \"result_format\": \"blocks\"\n}",
    },
]

app = FastAPI(title=f"{settings.app_name} OpenAPI Tool", version=settings.app_version)

# Attach Admin UI and debug endpoints from isolated module
attach_admin_routes(app)


# Admin UI and step-by-step debug endpoints are defined in app/admin_routes.py


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


# --- Search debug: step-by-step endpoints ---


@app.post(
    "/search/debug/embed",
    include_in_schema=False,
    summary="Search Debug: Etap 1/4 — embed + sparse",
    description=(
        "Wejście: pojedynczy tekst zapytania.\n"
        "Działanie: wykrycie trybu (current/archival) → embed zapytania → budowa zapytań TF‑IDF (gdy hybryda).\n\n"
        "Wywołuje funkcje: app.core.search._classify_mode(), app.core.embedding.embed_query(),\n"
        "app.core.search._build_sparse_queries_for_query().\n\n"
        "Zwraca: wektor zapytania, opcjonalne zapytania TF‑IDF (content/summary) oraz payload dla kolejnego etapu."
    ),
)
def search_debug_embed(req: DebugEmbedRequest):
    # Coerce query to list[str]
    if isinstance(req.query, list):
        queries = [str(x).strip() for x in req.query if str(x or "").strip()]
    else:
        qraw = str(req.query or "").strip()
        queries = [qraw] if qraw else []
    if not queries:
        raise HTTPException(status_code=422, detail="Field 'query' must contain at least one non-empty string")
    qi = max(0, min(int(req.query_index or 0), len(queries) - 1))
    q = queries[qi]

    # Mode detection like in /search/query but for a single (selected) query
    if req.mode != "auto":
        mode = req.mode
    else:
        mode = _classify_mode(q, "auto")

    # Build filter descriptor info only (UI podgląd); właściwy qm.Filter w kolejnych etapach
    filter_info = {"is_active": True} if mode == "current" else ({"is_active": False} if mode == "archival" else None)

    # Embed and (optional) sparse queries
    q_vec = embed_query([q])[0]
    content_sparse_query, summary_sparse_query = _build_sparse_queries_for_query(q, req.use_hybrid)

    def sq_pack(sq):
        if not sq:
            return None
        idx, val = sq
        return {"indices": [int(i) for i in idx], "values": [float(v) for v in val]}

    next_payload_stage1 = {
        "q_text": q,
        "q_vec": q_vec,
        "mode": mode,
        "use_hybrid": bool(req.use_hybrid),
        "top_m": int(req.top_m),
        "score_norm": str(req.score_norm),
        "dense_weight": float(req.dense_weight),
        "sparse_weight": float(req.sparse_weight),
        "mmr_stage1": bool(req.mmr_stage1),
        "mmr_lambda": float(req.mmr_lambda),
        "summary_sparse_query": sq_pack(summary_sparse_query),
        "content_sparse_query": sq_pack(content_sparse_query),
    }

    # Jeśli globalnie pomijamy Etap 1, pokieruj "Next step" bezpośrednio do Stage 2 (pełny korpus)
    if bool(settings.search_skip_stage1_default):
        next_spec = {
            "operation_id": "search-debug-stage2",
            "payload": {
                "q_text": q,
                "q_vec": q_vec,
                "cand_doc_ids": [],
                "doc_map": {},
                "top_k": int(req.top_k),
                "per_doc_limit": int(req.per_doc_limit),
                "score_norm": str(req.score_norm),
                "dense_weight": float(req.dense_weight),
                "sparse_weight": float(req.sparse_weight),
                "mmr_lambda": float(req.mmr_lambda),
                "content_sparse_query": sq_pack(content_sparse_query),
            },
            "info": {"note": "Stage 1 skipped (full corpus)"},
        }
    else:
        next_spec = {"operation_id": "search-debug-stage1", "payload": next_payload_stage1, "info": {"filter": filter_info}}

    return {
        "step": "embed",
        "queries": queries,
        "selected_query_index": qi,
        "q_text": q,
        "mode": mode,
        "q_vec": q_vec,
        "q_vec_len": len(q_vec),
        "content_sparse_query": sq_pack(content_sparse_query),
        "summary_sparse_query": sq_pack(summary_sparse_query),
        "config": {
            "skip_stage1_active": bool(settings.search_skip_stage1_default),
            "use_hybrid": bool(req.use_hybrid),
        },
        "_next": next_spec,
    }


@app.post(
    "/search/debug/stage1",
    include_in_schema=False,
    summary="Search Debug: Etap 2/4 — dokumenty (stage1)",
    description=(
        "Wejście: q_text, q_vec, (opcjonalnie) summary_sparse_query.\n"
        "Działanie: wyszukiwanie po streszczeniach → hybrydowe ważenie dense/sparse → MMR na poziomie dokumentów\n"
        "(opcjonalny reranker jeśli skonfigurowany).\n\n"
        "Wywołuje funkcje: app.core.search._stage1_select_documents() (wewnątrz: _normalize(), mmr_diversify_hybrid()).\n\n"
        "Zwraca: posortowane doc_id oraz zredukowaną mapę dokumentów do wglądu."
    ),
)
def search_debug_stage1(req: DebugStage1Request):
    # Build filter for mode
    flt = None
    if req.mode in ("current", "archival"):
        flt = qm.Filter(must=[qm.FieldCondition(key="is_active", match=qm.MatchValue(value=(req.mode == "current")))])

    # Prepare summary sparse query
    summary_sq = None
    if req.summary_sparse_query is not None:
        summary_sq = (list(map(int, req.summary_sparse_query.indices)), list(map(float, req.summary_sparse_query.values)))

    # Build temporary request-like object with needed fields
    class _R:
        def __init__(self, src: DebugStage1Request):
            self.top_m = src.top_m
            self.score_norm = src.score_norm
            self.dense_weight = src.dense_weight
            self.sparse_weight = src.sparse_weight
            self.mmr_stage1 = src.mmr_stage1
            self.mmr_lambda = src.mmr_lambda
            self.rep_alpha = src.rep_alpha

    cand_doc_ids, doc_map = _stage1_select_documents(req.q_text, req.q_vec, flt, summary_sq, _R(req))

    # Trim doc_map for UI (bez wektorów)
    ui_doc_map: Dict[str, Any] = {}
    for did, info in doc_map.items():
        ui_doc_map[did] = {
            "doc_id": did,
            "path": info.get("path"),
            "doc_summary": info.get("doc_summary"),
            "doc_signature": info.get("doc_signature"),
            "dense_score": float(info.get("dense_score", 0.0)),
            "sparse_score": float(info.get("sparse_score", 0.0)),
        }

    next_payload = {
        "q_text": req.q_text,
        "q_vec": req.q_vec,
        "cand_doc_ids": cand_doc_ids,
        "doc_map": ui_doc_map,
        "top_k": 10,
        "per_doc_limit": 2,
        "score_norm": req.score_norm,
        "dense_weight": req.dense_weight,
        "sparse_weight": req.sparse_weight,
        "mmr_lambda": req.mmr_lambda,
        "content_sparse_query": req.content_sparse_query,
    }

    return {
        "step": "stage1",
        "cand_doc_ids": cand_doc_ids,
        "doc_map": ui_doc_map,
        "config": {
            "skip_stage1_active": bool(settings.search_skip_stage1_default),
            "score_norm": req.score_norm,
            "dense_weight": req.dense_weight,
            "sparse_weight": req.sparse_weight,
            "mmr_stage1": req.mmr_stage1,
            "mmr_lambda": req.mmr_lambda,
        },
        "_next": {"operation_id": "search-debug-stage2", "payload": next_payload},
    }


@app.post(
    "/search/debug/stage2",
    include_in_schema=False,
    summary="Search Debug: Etap 3/4 — chunki (stage2)",
    description=(
        "Wejście: lista doc_id z Etapu 2, q_text, q_vec, (opcjonalnie) content_sparse_query.\n"
        "Działanie: wyszukiwanie chunków w obrębie dokumentów → hybrydowe ważenie dense/sparse → MMR z limitem per‑doc\n"
        "(opcjonalny reranker jeśli skonfigurowany).\n\n"
        "Wywołuje funkcje: app.core.search._stage2_select_chunks() (wewnątrz: _normalize(), mmr_diversify_hybrid()).\n\n"
        "Zwraca: listę wybranych chunków (hits) oraz payload dla kształtowania."
    ),
)
def search_debug_stage2(req: DebugStage2Request):
    # Rebuild content sparse query
    content_sq = None
    if req.content_sparse_query is not None:
        content_sq = (list(map(int, req.content_sparse_query.indices)), list(map(float, req.content_sparse_query.values)))

    # Build temporary request-like object with needed fields
    class _R2:
        def __init__(self, src: DebugStage2Request):
            self.top_m = src.top_m
            self.top_k = src.top_k
            self.per_doc_limit = src.per_doc_limit
            self.score_norm = src.score_norm
            self.dense_weight = src.dense_weight
            self.sparse_weight = src.sparse_weight
            self.mmr_lambda = src.mmr_lambda
            self.rep_alpha = src.rep_alpha
            # fields not used downstream but required by type
            self.merge_chunks = False
            self.result_format = "flat"
            self.expand_neighbors = 0
            self.block_join_delimiter = "\n\n"
            self.merge_group_budget_tokens = 1200
            self.max_merged_per_group = 1
            self.summary_mode = "first"

    # Note: doc_map passed in here is already trimmed (no vectors) — it's enough to enrich payloads
    final_hits, mmr_pool, rel2 = _stage2_select_chunks(req.cand_doc_ids, req.q_text, req.q_vec, content_sq, req.doc_map, _R2(req))

    # Convert to debug hits
    dbg_hits: List[Dict[str, Any]] = []
    for fh in final_hits:
        payload = fh.get("payload") or {}
        dbg_hits.append({
            "doc_id": payload.get("doc_id", ""),
            "path": payload.get("path"),
            "section": payload.get("section"),
            "chunk_id": int(payload.get("chunk_id", 0)),
            "score": float(fh.get("score", 0.0)),
            "snippet": (payload.get("text") or "").strip()[:400] if payload.get("text") else (payload.get("summary", "")[:400]),
        })

    next_payload = {
        "final_hits": dbg_hits,
        "result_format": "blocks",
        "merge_chunks": True,
        "merge_group_budget_tokens": 1200,
        "max_merged_per_group": 1,
        "block_join_delimiter": "\n\n",
        "summary_mode": "first",
    }

    return {
        "step": "stage2",
        "hits": dbg_hits,
        "pool_size": len(mmr_pool),
        "config": {
            "skip_stage1_active": bool(settings.search_skip_stage1_default),
            "score_norm": req.score_norm,
            "dense_weight": req.dense_weight,
            "sparse_weight": req.sparse_weight,
            "mmr_lambda": req.mmr_lambda,
            "per_doc_limit": req.per_doc_limit,
        },
        "_next": {"operation_id": "search-debug-shape", "payload": next_payload}
    }


@app.post(
    "/search/debug/shape",
    include_in_schema=False,
    summary="Search Debug: Etap 4/4 — kształtowanie wyników",
    description=(
        "Wejście: final_hits z Etapu 3 oraz opcje formatu.\n"
        "Działanie: budowa bloków/ grup/ listy płaskiej, łączenie chunków w bloki wg budżetu tokenów,\n"
        "(w debug domyślnie bez sąsiadów).\n\n"
        "Wywołuje funkcje: app.core.search._shape_results() oraz app.core.search.build_merged_blocks().\n\n"
        "Zwraca: results/groups/blocks zgodnie z result_format."
    ),
)
def search_debug_shape(req: DebugShapeRequest):
    # Build final_hits in expected shape for _shape_results
    final_hits = []
    for h in req.final_hits:
        payload = {
            "doc_id": h.doc_id,
            "path": h.path,
            "section": h.section,
            "chunk_id": h.chunk_id,
            "text": h.snippet or "",
        }
        final_hits.append({"payload": payload, "score": float(h.score)})

    class _R3:
        def __init__(self, src: DebugShapeRequest):
            self.merge_chunks = src.merge_chunks
            self.result_format = src.result_format
            self.expand_neighbors = 0
            self.block_join_delimiter = src.block_join_delimiter
            self.merge_group_budget_tokens = src.merge_group_budget_tokens
            self.max_merged_per_group = src.max_merged_per_group
            self.summary_mode = src.summary_mode

    results, groups, blocks = _shape_results(final_hits, {}, [], [], _R3(req))
    return {
        "step": "shape",
        "results": results,
        "groups": groups,
        "blocks": blocks,
        "config": {
            "skip_stage1_active": bool(settings.search_skip_stage1_default),
            "summary_mode": req.summary_mode,
            "result_format": req.result_format,
            "merge_chunks": req.merge_chunks,
        }
    }


@app.post(
    "/collections/init",
    include_in_schema=False,
    summary="Inicjalizacja kolekcji",
    description=(
        "Tworzy parę kolekcji (streszczenia + treść) dla wskazanej nazwy bazowej (jeśli nie istnieją) i opcjonalnie sondą sprawdza wymiar embeddingów przy użyciu force_dim_probe."
    ),
)
def collections_init(req: InitCollectionsRequest):
    from app.core.embedding import get_embedding_dim

    dim = get_embedding_dim() if req.force_dim_probe else None
    ensure_collections(req.collection_name, dim)
    _mark_collections_initialized(req.collection_name)
    summary_collection, content_collection = derive_collection_names(req.collection_name)
    return {
        "ok": True,
        "collection": req.collection_name,
        "summary_collection": summary_collection,
        "content_collection": content_collection,
    }


@app.post(
    "/collections/export",
    include_in_schema=False,
    summary="Eksport kolekcji Qdrant",
    description="Eksportuje wszystkie kolekcje do archiwum .tar.gz zawierającego snapshoty Qdrant oraz lokalne artefakty TF-IDF.",
)
def collections_export(req: CollectionsExportRequest):
    if req.collection_names:
        logger.info(
            "Parametr collection_names=%s został przesłany, ale eksport obejmuje wszystkie kolekcje.",
            req.collection_names,
        )
    bundle, meta = export_collections_bundle(req.collection_names)
    filename = meta.get("filename") or "qdrant-export.tar.gz"
    headers = {
        "Content-Disposition": f"attachment; filename={filename}",
        "X-Rags-Collections": ",".join(meta.get("collections", [])),
        "X-Rags-Vector-Store": ",".join(meta.get("vector_store_files", [])),
        "X-Rags-Snapshots": ",".join(meta.get("snapshots", [])),
    }
    return StreamingResponse(io.BytesIO(bundle), media_type="application/gzip", headers=headers)


@app.post(
    "/collections/import",
    include_in_schema=False,
    summary="Import kolekcji Qdrant",
    description=(
        "Przyjmuje archiwum .tar.gz wygenerowane przez /collections/export (plik lub base64) i odtwarza kolekcje ze snapshotów Qdrant oraz indeksy TF-IDF."
    ),
)
async def collections_import(
    request: Request,
    archive_file: UploadFile | None = File(
        default=None,
        description="Archiwum .tar.gz wygenerowane przez /collections/export.",
    ),
    replace_existing_form: bool | None = Form(
        default=None,
        description="Czy nadpisać istniejące kolekcje (gdy wysyłasz formularz).",
    ),
    replace_existing_query: bool = Query(
        default=True,
        description="Czy nadpisać istniejące kolekcje (dla zapytań bez formularza).",
    ),
):
    binary_archive: bytes | None = None
    replace_existing = replace_existing_query

    if archive_file is not None:
        binary_archive = await archive_file.read()
        await archive_file.close()
        if replace_existing_form is not None:
            replace_existing = replace_existing_form
    else:
        content_type = request.headers.get("content-type", "").lower()
        if "application/json" in content_type:
            try:
                payload_data = await request.json()
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Nie udało się wczytać JSON: {exc}") from exc

            try:
                payload = CollectionsImportRequest(**payload_data)
            except ValidationError as exc:
                raise HTTPException(status_code=422, detail=exc.errors()) from exc

            try:
                binary_archive = base64.b64decode(payload.archive_base64)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Nie udało się zdekodować base64: {exc}") from exc
            replace_existing = payload.replace_existing
        elif "application/octet-stream" in content_type or "application/gzip" in content_type:
            binary_archive = await request.body()
        elif content_type.startswith("multipart/form-data"):
            form = await request.form()
            if "archive_base64" in form:
                try:
                    payload = CollectionsImportRequest(
                        archive_base64=str(form["archive_base64"]),
                        replace_existing=replace_existing_form if replace_existing_form is not None else replace_existing_query,
                    )
                except ValidationError as exc:
                    raise HTTPException(status_code=422, detail=exc.errors()) from exc
                try:
                    binary_archive = base64.b64decode(payload.archive_base64)
                except Exception as exc:
                    raise HTTPException(status_code=400, detail=f"Nie udało się zdekodować base64: {exc}") from exc
                replace_existing = payload.replace_existing
            else:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Formularz multipart musi zawierać pole 'archive_file' z plikiem archiwum lub 'archive_base64'."
                    ),
                )
        else:
            body_bytes = await request.body()
            if body_bytes:
                binary_archive = body_bytes

    if not binary_archive:
        raise HTTPException(
            status_code=400,
            detail=(
                "Brak archiwum do importu. Wyślij JSON z polem 'archive_base64', formularz multipart z polem 'archive_file' lub surowe archiwum .tar.gz."
            ),
        )

    summary = import_collections_bundle(binary_archive, replace_existing=replace_existing)
    with _collection_init_lock:
        _initialized_collections.clear()
    return {"status": "ok", **summary}


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
        "Buduje indeks rags_tool: parsuje dokumenty, tworzy streszczenia, embeddingi oraz zapisuje punkty (wraz z TF-IDF) do Qdrant."
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
    summary_collection, content_collection = derive_collection_names(req.collection_name)

    if req.reindex:
        _clear_collection_cache(req.collection_name)
        for coll_name in (summary_collection, content_collection):
            try:
                qdrant.delete_collection(collection_name=coll_name)
                logger.debug(
                    "Deleted collection '%s' due to reindex request", coll_name
                )
            except Exception as exc:
                logger.debug(
                    "Delete collection '%s' skipped or failed: %s", coll_name, exc
                )
    ensure_collections(req.collection_name)
    _mark_collections_initialized(req.collection_name)

    base = pathlib.Path(req.base_dir)
    if not base.exists():
        raise HTTPException(status_code=400, detail="base_dir nie istnieje")

    file_paths = _scan_files(base, req.glob, req.recursive)
    logger.debug("Found %d files for ingest", len(file_paths))
    if not file_paths:
        return {"ok": True, "indexed": 0, "took_ms": int((time.time() - t0) * 1000)}

    with tempfile.TemporaryDirectory() as tmpdir:
        store_path = pathlib.Path(tmpdir) / "doc_records.jsonl"
        doc_count = 0
        chunk_count = 0
        summary_count = 0

        with store_path.open("w", encoding="utf-8") as fh:
            for record in _iter_document_records(file_paths, req.chunk_tokens, req.chunk_overlap):
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                doc_count += 1
                chunk_count += len(record.get("chunks", []))
                if record.get("summary_sparse_text"):
                    summary_count += 1

        if doc_count == 0:
            took_ms = int((time.time() - t0) * 1000)
            return {"ok": True, "indexed": 0, "documents": 0, "took_ms": took_ms}

        if req.enable_sparse:
            chunk_corpus = IterableCorpus(
                size=chunk_count,
                factory=lambda: _iter_chunk_texts(store_path),
            )
            summary_corpus = IterableCorpus(
                size=summary_count,
                factory=lambda: _iter_summary_texts(store_path),
            )
        else:
            chunk_corpus = None
            summary_corpus = None

        content_vec, summary_vec = prepare_tfidf(
            chunk_corpus,
            summary_corpus,
            req.enable_sparse,
            req.rebuild_tfidf,
        )

        point_count = build_and_upsert_points(
            _iter_saved_records(store_path),
            content_vec,
            summary_vec,
            enable_sparse=req.enable_sparse,
            collection_base=req.collection_name,
        )

    took_ms = int((time.time() - t0) * 1000)
    logger.debug(
        "Ingest build finished | documents=%d points=%d took_ms=%d",
        doc_count,
        point_count,
        took_ms,
    )
    return {"ok": True, "indexed": point_count, "documents": doc_count, "took_ms": took_ms}


@app.post(
    "/search/query",
    response_model=SearchResponse,
    summary="rags_tool search (LLM tool)",
    operation_id="rags_tool_search",
    tags=["tools"],
    description=settings.search_tool_description,
)
def search_query(req: SearchQuery):
    """
    Endpoint: /search/query (POST)

    Purpose
    -------
    Provides two‑stage hybrid retrieval for LLM‑powered tools. Stage 1 ranks document
    summaries; Stage 2 ranks full‑text chunks within the selected documents using a
    combination of dense embeddings and TF‑IDF sparse vectors, with optional hybrid MMR
    diversification and a per‑document cap.

    Parameters (SearchQuery)
    ------------------------
    - **query** (List[str]): list of focused queries (3–12 words each; prefer titles/signatures/dates). All queries are executed and results are fused.
    - **top_m** (int): candidate documents for Stage 1 (default 100; typical 50–200).
    - **top_k** (int): global number of final results after Stage 2 (typical 5–10).
      Use `per_doc_limit` to prevent dominance of a single document.
    - **mode** (str): `"auto"` (detect current/archival), `"current"`, `"archival"` or `"all"`.
    - **use_hybrid** (bool): enable dense + sparse scoring.
    - **dense_weight** / **sparse_weight** (float): weighting of dense vs sparse scores.
    - **mmr_lambda** (float): trade‑off between relevance and diversity (default 0.3).
    - **per_doc_limit** (int): max chunks per document after MMR.
    - **score_norm** (str): `"minmax"`, `"zscore"` or `"none"` for score normalisation.
    - **rep_alpha** (float): weighting between dense and sparse similarity in MMR.
    - **mmr_stage1** (bool): apply MMR already at document selection.
    - **summary_mode** (str): `"none" | "first" | "all"` (summary duplication strategy).
    - **merge_chunks** (bool): consolidate consecutive chunks into blocks.
    - **merge_group_budget_tokens** (int): token budget per merged block.
    - **max_merged_per_group** (int): max blocks per document.
    - **expand_neighbors** (int): include surrounding chunks around a block.
    - **result_format** (str): `"flat" | "grouped" | "blocks"` (`"blocks"` is default and recommended for tools).

    Recommendations for LLM callers
    --------------------------------
    * Prefer `result_format="blocks"` to obtain concise evidence blocks; blocks are
      constructed even if `merge_chunks=false`.
    * Keep `top_k` between 5‑10 unless you need finer granularity.
    * `summary_mode="first"` returns a single document summary per hit, useful for
      citation.
    * The returned payload contains `blocks` where each block includes:
        - `text` – concatenated chunk text (evidence).
        - `path` – source file path.
        - `first_chunk_id` / `last_chunk_id` – range of original chunk IDs.
        - `score` – relevance score.

    Returns
    -------
    SearchResponse containing timing, hits (if not using blocks), optional groups,
    and a list of `blocks` when `result_format="blocks"`.

    Example request (JSON):
    {
        "query": [
            "Jak działa rags_tool?",
            "architektura rags_tool"
        ],
        "top_m": 10,
        "top_k": 5,
        "mode": "auto",
        "use_hybrid": true,
        "dense_weight": 0.6,
        "sparse_weight": 0.4,
        "mmr_lambda": 0.3,
        "per_doc_limit": 2,
        "score_norm": "minmax",
        "rep_alpha": 0.6,
        "mmr_stage1": true,
        "summary_mode": "first",
        "merge_chunks": true,
        "merge_group_budget_tokens": 1200,
        "max_merged_per_group": 1,
        "expand_neighbors": 1,
        "result_format": "blocks"
    }

    The endpoint returns a JSON with `blocks` ready for citation by the LLM.
    """
    t0 = time.time()
    # --- RERANKER: konfiguracja z .env ---
    # Włączony wtedy, gdy podano zarówno BASE_URL, jak i MODEL.
    ranker_enabled = bool(settings.ranker_base_url and settings.ranker_model)
    # Minimalne parametry sterowane z .env (LLM nie może ich nadpisać):
    RERANK_TOP_N = max(1, int(settings.rerank_top_n))
    RETURN_TOP_K = max(1, int(settings.return_top_k))
    RANKER_THRESHOLD = float(settings.ranker_score_threshold)
    RANKER_MAX_LEN = max(1, int(settings.ranker_max_length))
    # Internal fusion defaults (hidden from tool schema/LLM)
    RRF_K = 60
    OVERSAMPLE = 2
    DEDUPE_BY = "chunk"  # other option could be "doc", kept internal

    queries = [q.strip() for q in (req.query or []) if str(q or "").strip()]
    if not queries:
        raise HTTPException(status_code=422, detail="Field 'query' must contain at least one non-empty string")

    query_hash = sha1(json.dumps(queries, ensure_ascii=False))
    _ensure_collections_cached()

    # Determine unified mode when 'auto' is requested
    if req.mode != "auto":
        mode = req.mode
    else:
        modes = {_classify_mode(q, "auto") for q in queries}
        if modes == {"current"}:
            mode = "current"
        elif modes == {"archival"}:
            mode = "archival"
        else:
            mode = "all"

    flt = None
    if mode in ("current", "archival"):
        flt = qm.Filter(must=[qm.FieldCondition(key="is_active", match=qm.MatchValue(value=(mode == "current")))])

    # Batch-embed queries (with model-specific query prefix)
    q_vecs = embed_query(queries)

    # Accumulators for fusion and neighbor expansion
    fused: Dict[tuple, Dict[str, Any]] = {}
    global_mmr_pool: List[Dict[str, Any]] = []
    global_rel2: List[float] = []

    # Oversampled per-query limit to keep enough candidates after dedup/fusion
    # Jeśli ranker jest włączony, budżetujemy kandydatów pod rerank zamiast polegać na top_k z API.
    if ranker_enabled:
        per_query_limit = max(1, int(RERANK_TOP_N))
    else:
        per_query_limit = max(1, int(req.top_k) * OVERSAMPLE)

    any_docs = False
    skip_stage1 = bool(settings.search_skip_stage1_default)
    for qi, (q, q_vec) in enumerate(zip(queries, q_vecs)):
        content_sparse_query, summary_sparse_query = _build_sparse_queries_for_query(q, req.use_hybrid)
        # Clone request with per-query top_k oversampled
        req_i = req.model_copy(update={"top_k": per_query_limit})
        if skip_stage1:
            # Pełny korpus: pomiń Etap 1 i wyszukuj bezpośrednio w chunkach (z zachowaniem filtra trybu)
            final_hits, mmr_pool, rel2 = _stage2_select_chunks(None, q, q_vec, content_sparse_query, {}, req_i, flt)
            if not final_hits:
                continue
            any_docs = True
        else:
            cand_doc_ids, doc_map = _stage1_select_documents(q, q_vec, flt, summary_sparse_query, req)
            if not cand_doc_ids:
                continue
            any_docs = True
            final_hits, mmr_pool, rel2 = _stage2_select_chunks(cand_doc_ids, q, q_vec, content_sparse_query, doc_map, req_i)
        # Append to global neighbor pools (used later for optional neighbor expansion)
        global_mmr_pool.extend(mmr_pool)
        global_rel2.extend(rel2)
        # RRF fusion on chunk identity
        for rank, fh in enumerate(final_hits, start=1):
            payload = fh.get("payload") or {}
            did = payload.get("doc_id") or ""
            sec = payload.get("section")
            cid = payload.get("chunk_id")
            if not did or cid is None:
                continue
            key = (did, sec, int(cid)) if DEDUPE_BY == "chunk" else (did, None, -1)
            entry = fused.get(key)
            incr = 1.0 / (RRF_K + rank)
            if entry is None:
                fused[key] = {
                    "payload": payload,
                    "score": incr,
                }
            else:
                entry["score"] += incr

    if not any_docs or not fused:
        took_ms = int((time.time() - t0) * 1000)
        logger.info(
            "Search finished | took_ms=%d query_hash=%s stage=fusion hits=0",
            took_ms,
            query_hash,
        )
        return SearchResponse(took_ms=took_ms, hits=[])

    # Build final fused list; nie ograniczamy tutaj do top_k jeśli ranker jest włączony
    fused_list = [
        {"payload": v.get("payload"), "score": float(v.get("score", 0.0))}
        for v in fused.values()
    ]
    fused_list.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    final_hits = fused_list if ranker_enabled else fused_list[: max(1, int(req.top_k))]

    # Shape results; merged blocks are built AFTER fusion, preserving the requirement
    results, groups_payload, blocks_payload = _shape_results(final_hits, {}, global_mmr_pool, global_rel2, req)

    if req.result_format == "blocks":
        took_ms = int((time.time() - t0) * 1000)
        logger.info(
            "Search finished | took_ms=%d query_hash=%s mode=%s fmt=blocks blocks=%d",
            took_ms,
            query_hash,
            mode,
            len(blocks_payload or []),
        )
        return SearchResponse(took_ms=took_ms, hits=[], groups=None, blocks=blocks_payload)

    took_ms = int((time.time() - t0) * 1000)
    logger.info(
        "Search finished | took_ms=%d query_hash=%s mode=%s fmt=%s hits=%d groups=%d blocks=%d",
        took_ms,
        query_hash,
        mode,
        req.result_format,
        len(results or []),
        len(groups_payload or []),
        len(blocks_payload or []),
    )
    return SearchResponse(took_ms=took_ms, hits=results, groups=groups_payload, blocks=blocks_payload)
