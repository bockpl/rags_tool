"""Admin UI and step-by-step debug endpoints, isolated from core API.

This module attaches all non-functional (UI/debug) routes to a provided FastAPI app.
It keeps the core/business endpoints in app/api.py clean and separate.
"""

from __future__ import annotations

import json
import logging
import pathlib
from typing import Any, Dict, List, Optional, Tuple

from fastapi import HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.routing import APIRoute
from qdrant_client.http import models as qm

from app.core.embedding import embed_query
from app.core.search import (
    _build_sparse_queries_for_query,
    _classify_mode,
    _stage1_select_documents,
    _stage2_select_chunks,
    _shape_results,
)
from app.models import (
    DebugEmbedRequest,
    DebugStage1Request,
    DebugStage2Request,
    DebugShapeRequest,
)
from app.settings import get_settings


settings = get_settings()
logger = logging.getLogger("rags_tool")


ADMIN_OPERATION_SPECS: List[Dict[str, Any]] = [
    {"id": "search-debug-embed", "path": "/search/debug/embed", "method": "POST", "label": "Search Debug: 1) embed", "body": "{\"query\":\"Jak działa rags_tool?\",\"mode\":\"auto\",\"use_hybrid\":true}"},
    {"id": "search-debug-stage1", "path": "/search/debug/stage1", "method": "POST", "label": "Search Debug: 2) stage1", "body": "{\"q_text\":\"Jak działa rags_tool?\",\"q_vec\":[0.0],\"mode\":\"auto\",\"use_hybrid\":true,\"top_m\":100,\"score_norm\":\"minmax\",\"dense_weight\":0.6,\"sparse_weight\":0.4,\"mmr_stage1\":true,\"mmr_lambda\":0.3}"},
    {"id": "search-debug-stage2", "path": "/search/debug/stage2", "method": "POST", "label": "Search Debug: 3) stage2", "body": "{\"q_text\":\"Jak działa rags_tool?\",\"q_vec\":[0.0],\"cand_doc_ids\":[\"<doc_id>\"],\"doc_map\":{},\"top_k\":10,\"per_doc_limit\":2,\"score_norm\":\"minmax\",\"dense_weight\":0.6,\"sparse_weight\":0.4,\"mmr_lambda\":0.3}"},
    {"id": "search-debug-shape", "path": "/search/debug/shape", "method": "POST", "label": "Search Debug: 4) shape", "body": "{\"final_hits\":[{\"doc_id\":\"<doc_id>\",\"path\":\"/abs/path\",\"section\":null,\"chunk_id\":0,\"score\":0.5,\"snippet\":\"...\"}],\"result_format\":\"blocks\",\"merge_chunks\":true,\"merge_group_budget_tokens\":1200,\"max_merged_per_group\":1,\"block_join_delimiter\":\"\\n\\n\",\"summary_mode\":\"first\"}"},
]

ADMIN_UI_REQUEST_HEADER = "x-admin-ui"


def _sq_pack(sq: Optional[Tuple[List[int], List[float]]]) -> Optional[Dict[str, Any]]:
    if not sq:
        return None
    idx, val = sq
    return {"indices": [int(i) for i in idx], "values": [float(v) for v in val]}


def _build_admin_operations(app) -> List[Dict[str, Any]]:
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
        handler_name = None
        handler_loc = None
        if route and getattr(route, "endpoint", None):
            try:
                handler_name = getattr(route.endpoint, "__name__", None)
                handler_loc = f"{getattr(route.endpoint, '__module__', 'app')}:{handler_name}"
            except Exception:
                handler_name = None
                handler_loc = None
        label = spec.get("label") or f"{method} {path}"
        if path == "/search/query":
            label = f"{label} ({'full-corpus' if settings.search_skip_stage1_default else 'two-stage'})"
        operations.append(
            {
                "id": spec["id"],
                "label": label,
                "method": method,
                "path": path,
                "doc": "\n\n".join(doc_parts),
                "body": spec.get("body"),
                "accepts_file": spec.get("accepts_file", False),
                "meta": {
                    "app_version": settings.app_version,
                    "skip_stage1": bool(settings.search_skip_stage1_default),
                },
                "handler": handler_name,
                "handler_loc": handler_loc,
            }
        )
    return operations


def attach_admin_routes(app) -> None:
    # Collect existing routes to avoid duplicate registration when imported alongside legacy definitions
    existing = set()
    try:
        for route in app.routes:
            if isinstance(route, APIRoute):
                for method in route.methods or []:
                    existing.add((route.path, method.upper()))
    except Exception:
        existing = set()
    # Middleware (activate DEBUG logs from Admin UI)
    async def admin_ui_debug_middleware(request: Request, call_next):
        header_value = request.headers.get(ADMIN_UI_REQUEST_HEADER)
        if header_value and header_value.lower() in {"1", "true", "yes"}:
            logger.setLevel(logging.DEBUG)
            logger.info("Admin UI request detected — DEBUG logging enabled")
        response = await call_next(request)
        return response

    app.middleware("http")(admin_ui_debug_middleware)

    # /admin console (HTML)
    def admin_console():
        operations = _build_admin_operations(app)
        tpl_path = pathlib.Path(__file__).parent.parent / "templates" / "admin.html"
        try:
            html = tpl_path.read_text(encoding="utf-8")
        except Exception as exc:
            logger.error("Failed to load admin template: %s", exc)
            return HTMLResponse(content="<html><body><p>Admin UI unavailable.</p></body></html>")
        html = html.replace("__OPERATIONS__", json.dumps(operations, ensure_ascii=False))
        return HTMLResponse(content=html)

    if ("/admin", "GET") not in existing:
        app.get("/admin", include_in_schema=False, response_class=HTMLResponse, summary="Panel administracyjny", description="Statyczny panel HTML do testowania i debugowania endpointów rags_tool.")(admin_console)

    # --- Debug endpoints ---

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
            "summary_sparse_query": _sq_pack(summary_sparse_query),
            "content_sparse_query": _sq_pack(content_sparse_query),
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
                    "content_sparse_query": _sq_pack(content_sparse_query),
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
            "content_sparse_query": _sq_pack(content_sparse_query),
            "summary_sparse_query": _sq_pack(summary_sparse_query),
            "config": {
                "skip_stage1_active": bool(settings.search_skip_stage1_default),
                "use_hybrid": bool(req.use_hybrid),
            },
            "_next": next_spec,
        }

    if ("/search/debug/embed", "POST") not in existing:
        app.post("/search/debug/embed", include_in_schema=False, summary="Search Debug: Etap 1/4 — embed + sparse")(search_debug_embed)

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

    if ("/search/debug/stage1", "POST") not in existing:
        app.post("/search/debug/stage1", include_in_schema=False, summary="Search Debug: Etap 2/4 — dokumenty (stage1)")(search_debug_stage1)

    def search_debug_stage2(req: DebugStage2Request):
        # Rebuild content sparse query
        content_sq = None
        if req.content_sparse_query is not None:
            content_sq = (list(map(int, req.content_sparse_query.indices)), list(map(float, req.content_sparse_query.values)))

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

    if ("/search/debug/stage2", "POST") not in existing:
        app.post("/search/debug/stage2", include_in_schema=False, summary="Search Debug: Etap 3/4 — chunki (stage2)")(search_debug_stage2)

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

    if ("/search/debug/shape", "POST") not in existing:
        app.post("/search/debug/shape", include_in_schema=False, summary="Search Debug: Etap 4/4 — kształtowanie wyników")(search_debug_shape)
