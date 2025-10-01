"""FastAPI endpoints for rags_tool."""

from __future__ import annotations

import json
import logging
import pathlib
import re
import sys
import tempfile
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.routing import APIRoute

from app.core.chunking import chunk_text_by_sections
from qdrant_client.http import models as qm
from app.core.embedding import IterableCorpus, prepare_tfidf
from app.core.embedding import embed_text
from app.core.parsing import SUPPORTED_EXT, extract_text
from app.core.search import (
    _build_sparse_queries_for_query,
    _classify_mode,
    _stage1_select_documents,
    _stage2_select_chunks,
    _shape_results,
)
from app.core.summary import llm_summary
from app.models import (
    About,
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
    qdrant,
)
from app.settings import get_settings


settings = get_settings()

logger = logging.getLogger("rags_tool")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.DEBUG if settings.debug else logging.INFO)
logger.propagate = False


def sha1(s: str) -> str:
    import hashlib

    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _scan_files(base: pathlib.Path, pattern: str, recursive: bool) -> List[pathlib.Path]:
    iterator = base.rglob(pattern) if recursive else base.glob(pattern)
    return [p for p in iterator if p.is_file() and p.suffix.lower() in SUPPORTED_EXT]


def _iter_document_records(
    file_paths: List[pathlib.Path], chunk_tokens: int, chunk_overlap: int
) -> Iterable[Dict[str, Any]]:
    for path in file_paths:
        logger.debug("Processing document %s", path)
        raw = extract_text(path)
        chunk_items = chunk_text_by_sections(raw, target_tokens=chunk_tokens, overlap_tokens=chunk_overlap, merge_up_to="ust")
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
        logger.debug(
            "Document %s parsed | chunks=%d summary_len=%d",
            path,
            len(chunks),
            len(rec["doc_summary"] or ""),
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

ADMIN_UI_REQUEST_HEADER = "x-admin-ui"
_admin_debug_activated = False

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
    tpl_path = pathlib.Path(__file__).parent.parent / "templates" / "admin.html"
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
        "Tworzy parę kolekcji (streszczenia + treść) dla wskazanej nazwy bazowej (jeśli nie istnieją) i opcjonalnie sondą sprawdza wymiar embeddingów przy użyciu force_dim_probe."
    ),
)
def collections_init(req: InitCollectionsRequest):
    from app.core.embedding import get_embedding_dim

    dim = get_embedding_dim() if req.force_dim_probe else None
    ensure_collections(req.collection_name, dim)
    summary_collection, content_collection = derive_collection_names(req.collection_name)
    return {
        "ok": True,
        "collection": req.collection_name,
        "summary_collection": summary_collection,
        "content_collection": content_collection,
    }


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
    description="Two-stage hybrid retrieval for LLM tools. Detailed parameter documentation is available in the function docstring.",
)
def search_query(req: SearchQuery):
    """
    Endpoint: /search/query (POST)

    Purpose
    -------
    Provides two‑stage hybrid retrieval for LLM‑powered tools. Stage 1 ranks document
    summaries, Stage 2 ranks full‑text chunks within the selected documents using a
    combination of dense embeddings and TF‑IDF sparse vectors, with optional MMR
    diversification.

    Parameters (SearchQuery)
    ------------------------
    - **query** (str): user query.
    - **top_m** (int): number of top documents for stage 1 (default 10).
    - **top_k** (int): number of top chunks per document for stage 2 (default 5).
    - **mode** (str): `"auto"` (detect current/archival), `"current"`, `"archival"` or `"all"`.
    - **use_hybrid** (bool): enable dense + sparse scoring.
    - **dense_weight** / **sparse_weight** (float): weighting of dense vs sparse scores.
    - **mmr_lambda** (float): trade‑off between relevance and diversity (default 0.3).
    - **per_doc_limit** (int): max chunks per document after MMR.
    - **score_norm** (str): `"minmax"`, `"zscore"` or `"none"` for score normalisation.
    - **rep_alpha** (float): weighting between dense and sparse similarity in MMR.
    - **mmr_stage1** (bool): apply MMR already at document selection.
    - **summary_mode** (str): `"first"` (include one summary per doc) or `"none"`.
    - **merge_chunks** (bool): consolidate consecutive chunks into blocks.
    - **merge_group_budget_tokens** (int): token budget per merged block.
    - **max_merged_per_group** (int): max blocks per document.
    - **expand_neighbors** (int): include surrounding chunks around a block.
    - **result_format** (str): `"blocks"` (recommended), `"hits"` or `"grouped"`.

    Recommendations for LLM callers
    --------------------------------
    * Use `result_format="blocks"` together with `merge_chunks=true` to obtain
      concise evidence blocks.
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
        "query": "Jak działa rags_tool?",
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
    ensure_collections()

    mode = _classify_mode(req.query, req.mode)
    q_vec = embed_text([req.query])[0]
    content_sparse_query, summary_sparse_query = _build_sparse_queries_for_query(req.query, req.use_hybrid)
    flt = None
    if mode in ("current", "archival"):
        flt = qm.Filter(must=[qm.FieldCondition(key="is_active", match=qm.MatchValue(value=(mode == "current")))])

    cand_doc_ids, doc_map = _stage1_select_documents(q_vec, flt, summary_sparse_query, req)
    if not cand_doc_ids:
        return SearchResponse(took_ms=int((time.time() - t0) * 1000), hits=[])

    final_hits, mmr_pool, rel2 = _stage2_select_chunks(
        cand_doc_ids, q_vec, content_sparse_query, doc_map, req
    )
    if not final_hits:
        return SearchResponse(took_ms=int((time.time() - t0) * 1000), hits=[])

    results, groups_payload, blocks_payload = _shape_results(final_hits, doc_map, mmr_pool, rel2, req)

    if req.result_format == "blocks":
        took_ms = int((time.time() - t0) * 1000)
        return SearchResponse(took_ms=took_ms, hits=[], groups=None, blocks=blocks_payload)

    took_ms = int((time.time() - t0) * 1000)
    return SearchResponse(took_ms=took_ms, hits=results, groups=groups_payload, blocks=blocks_payload)
