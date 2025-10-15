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
    _truncate_head_tail,
)
from app.core.summary import llm_summary
from app.core.ranker_client import OpenAIReranker
from app.models import (
    About,
    CollectionsExportRequest,
    CollectionsImportRequest,
    IngestBuildRequest,
    InitCollectionsRequest,
    ScanRequest,
    ScanResponse,
    SearchQuery,
    SearchResponse,
    SummariesGenerateRequest,
    ContradictionAnalysisRequest,
    ContradictionAnalysisResponse,
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
from app.core.summary_cache import (
    compute_file_sha256,
    load_sidecar,
    save_sidecar,
    sidecar_path_for,
)
from app.core.embedding import embed_passage
from app.core.contradictions import analyze_contradictions


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
        "force_regen_summary": False,
    },
    ensure_ascii=False,
    indent=2,
)


# Build a stable cache key for (summary, content) collections derived from base.
def _collection_cache_key(base: Optional[str]) -> str:
    """Stable cache key for a pair of (summary, content) collections."""
    summary_collection, content_collection = derive_collection_names(base)
    return f"{summary_collection}::{content_collection}"


# Mark collections for the given base as initialized in this process.
def _mark_collections_initialized(base: Optional[str]) -> None:
    """Remember that collections for `base` were ensured in this process."""
    key = _collection_cache_key(base)
    with _collection_init_lock:
        _initialized_collections.add(key)


# Clear the initialized-collections flag for the given base.
def _clear_collection_cache(base: Optional[str]) -> None:
    """Forget initialization status for collections derived from `base`."""
    key = _collection_cache_key(base)
    with _collection_init_lock:
        _initialized_collections.discard(key)


# Ensure collections exist once per process (guards repeated init).
def _ensure_collections_cached(base: Optional[str] = None) -> None:
    """Ensure Qdrant collections exist once per process using a local cache."""
    key = _collection_cache_key(base)
    with _collection_init_lock:
        if key in _initialized_collections:
            return
        ensure_collections(base)
        _initialized_collections.add(key)


# Scan filesystem for supported files matching pattern.
def _scan_files(base: pathlib.Path, pattern: str, recursive: bool) -> List[pathlib.Path]:
    """Scan `base` for files matching pattern and supported extensions."""
    iterator = base.rglob(pattern) if recursive else base.glob(pattern)
    return [p for p in iterator if p.is_file() and p.suffix.lower() in SUPPORTED_EXT]


# Iterate ingest records: chunks + LLM summary (+ cached vectors) per file.
def _iter_document_records(
    file_paths: List[pathlib.Path], chunk_tokens: int, chunk_overlap: int, *, force_regen_summary: bool
) -> Iterable[Dict[str, Any]]:
    """Yield ingest records for each document (chunks + summary + vectors).

    Uses sidecar cache unless `force_regen_summary` is set.
    """
    for path in file_paths:
        doc_start = time.time()
        logger.debug("Processing document %s", path)
        # Always extract text and chunks (not cached here)
        raw = extract_text(path)
        chunk_items = chunk_text_by_sections(
            raw,
            target_tokens=chunk_tokens,
            overlap_tokens=chunk_overlap,
        )
        chunks = chunk_items
        if not chunks:
            logger.debug("Document %s produced no chunks; skipping", path)
            continue
        # Try sidecar cache for summary + vectors (unless forced to regenerate)
        content_sha256 = compute_file_sha256(path)
        sidecar = None
        sc_path = sidecar_path_for(path)
        sc_name = sc_path.name
        if force_regen_summary:
            if sc_path.exists():
                logger.debug("Force regen: ignoring sidecar | path=%s sidecar=%s", path, sc_name)
        else:
            if sc_path.exists():
                logger.debug("Sidecar present, validating | path=%s sidecar=%s", path, sc_name)
            else:
                logger.debug("Sidecar not found | path=%s expected=%s", path, sc_name)
            sidecar = load_sidecar(path, expected_sha256=content_sha256)
        if sidecar:
            logger.debug("Using sidecar cache | path=%s sidecar=%s", path, sc_name)
            summ_block = sidecar.get("summary", {})
            vectors_block = sidecar.get("vectors", {})
            doc_sum = {
                "title": str(summ_block.get("title") or ""),
                "summary": str(summ_block.get("summary") or ""),
                "signature": list(summ_block.get("signature") or []),
                "replacement": str(summ_block.get("replacement") or "brak") or "brak",
                "doc_date": str(summ_block.get("doc_date") or "brak") or "brak",
            }
            summary_dense_vec = list(vectors_block.get("summary_dense") or [])
        else:
            # Generate with LLM and compute dense embedding for summary; then cache
            doc_sum = llm_summary(raw)
            summary_text = doc_sum.get("summary", "") or ""
            summary_dense_vec = embed_passage([summary_text])[0]
            # Persist sidecar (atomic write); best-effort, ignore failures
            try:
                save_sidecar(
                    path,
                    content_sha256=content_sha256,
                    title=str(doc_sum.get("title", "") or ""),
                    summary=summary_text,
                    signature=list(doc_sum.get("signature", []) or []),
                    replacement=str(doc_sum.get("replacement", "brak") or "brak"),
                    summary_dense=list(summary_dense_vec),
                    doc_date=str(doc_sum.get("doc_date", "brak") or "brak"),
                )
                logger.debug("Sidecar saved | path=%s sidecar=%s", path, sidecar_path_for(path).name)
            except Exception as exc:
                logger.debug("Sidecar save skipped | path=%s error=%s", path, exc)
            if not force_regen_summary and sc_path.exists():
                logger.debug("Sidecar rejected or stale; regenerated | path=%s sidecar=%s", path, sc_name)
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
        # Include doc_date in sparse summary if available and not 'brak'
        doc_date_val = str(doc_sum.get("doc_date", "") or "").strip()
        if doc_date_val and doc_date_val.lower() != "brak":
            summary_sparse_parts.append(doc_date_val)
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
            "doc_date": doc_sum.get("doc_date", "brak"),
            "summary_sparse_text": summary_sparse_text,
            # Precomputed dense summary vector (used to skip embedding in upsert stage)
            "summary_dense_vec": list(summary_dense_vec) if (sidecar or summary_dense_vec is not None) else None,
        }
        logger.debug(
            "Document %s parsed | chunks=%d summary_len=%d took_ms=%d",
            path,
            len(chunks),
            len(rec["doc_summary"] or ""),
            int((time.time() - doc_start) * 1000),
        )
        yield rec


# Iterate JSONL records saved during ingest.
def _iter_saved_records(path: pathlib.Path) -> Iterable[Dict[str, Any]]:
    """Iterate JSONL records previously written by ingest step."""
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


# Iterate chunk texts from saved ingest JSONL.
def _iter_chunk_texts(path: pathlib.Path) -> Iterable[str]:
    """Iterate chunk texts from stored ingest records."""
    for rec in _iter_saved_records(path):
        for chunk in rec.get("chunks", []):
            if isinstance(chunk, dict):
                text = chunk.get("text", "")
            else:
                text = str(chunk)
            if text:
                yield text


# Iterate summary texts from saved ingest JSONL (for TF‑IDF fitting).
def _iter_summary_texts(path: pathlib.Path) -> Iterable[str]:
    """Iterate summary texts (for TF-IDF fitting) from stored records."""
    for rec in _iter_saved_records(path):
        summary = rec.get("summary_sparse_text")
        if summary:
            yield summary


ADMIN_OPERATION_SPECS: List[Dict[str, Any]] = [
    {"id": "search-debug-embed", "path": "/search/debug/embed", "method": "POST", "label": "Search Debug: 1) embed", "body": "{\"query\":\"Jak działa rags_tool?\",\"mode\":\"auto\",\"use_hybrid\":true}"},
    {"id": "search-debug-stage1", "path": "/search/debug/stage1", "method": "POST", "label": "Search Debug: 2) stage1", "body": "{\"q_text\":\"Jak działa rags_tool?\",\"q_vec\":[0.0],\"mode\":\"auto\",\"use_hybrid\":true,\"top_m\":100,\"score_norm\":\"minmax\",\"dense_weight\":0.6,\"sparse_weight\":0.4,\"mmr_stage1\":true,\"mmr_lambda\":0.3}"},
    {"id": "search-debug-stage2", "path": "/search/debug/stage2", "method": "POST", "label": "Search Debug: 3) stage2", "body": "{\"q_text\":\"Jak działa rags_tool?\",\"q_vec\":[0.0],\"cand_doc_ids\":[\"<doc_id>\"],\"doc_map\":{},\"top_k\":10,\"per_doc_limit\":2,\"score_norm\":\"minmax\",\"dense_weight\":0.6,\"sparse_weight\":0.4,\"mmr_lambda\":0.3}"},
    {"id": "search-debug-shape", "path": "/search/debug/shape", "method": "POST", "label": "Search Debug: 4) shape", "body": "{\"final_hits\":[{\"doc_id\":\"<doc_id>\",\"path\":\"/abs/path\",\"section\":null,\"chunk_id\":0,\"score\":0.5,\"snippet\":\"...\"}],\"result_format\":\"blocks\",\"summary_mode\":\"first\"}"},
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
        "body": "{\n  \"query\": [\n    \"Jak działa rags_tool?\",\n    \"architektura rags_tool\"\n  ],\n  \"top_m\": 10,\n  \"top_k\": 5,\n  \"mode\": \"auto\",\n  \"use_hybrid\": true,\n  \"dense_weight\": 0.6,\n  \"sparse_weight\": 0.4,\n  \"mmr_lambda\": 0.3,\n  \"per_doc_limit\": 2,\n  \"score_norm\": \"minmax\",\n  \"rep_alpha\": 0.6,\n  \"mmr_stage1\": true,\n  \"summary_mode\": \"first\",\n  \"result_format\": \"blocks\"\n}",
    },
]

app = FastAPI(title=f"{settings.app_name} OpenAPI Tool", version=settings.app_version)

# Attach Admin UI and debug endpoints from isolated module
attach_admin_routes(app)


# Ensure Qdrant collections and indexes at process startup
@app.on_event("startup")
def _startup_ensure_collections() -> None:
    """Best-effort ensure of collections and TF-IDF warm-up at startup."""
    try:
        _ensure_collections_cached()
        # Pre-warm TF-IDF vectorizers (content + summaries) to avoid cold-start latency
        try:
            prepare_tfidf(force_rebuild=False)
            logger.info("TF-IDF vectorizers pre-warmed at startup")
        except Exception as exc:
            logger.warning("TF-IDF pre-warm skipped: %s", exc)
        # Log key runtime switches for clarity
        logger.info(
            "Startup config | skip_stage1=%s dual_query_sparse=%s minimal_payload=%s batch_section_fetch=%s rrf_k=%d oversample=%d dense_for_mmr=%s",
            bool(settings.search_skip_stage1_default),
            bool(settings.search_dual_query_sparse),
            bool(settings.search_minimal_payload),
            bool(settings.batch_section_fetch),
            int(settings.dual_query_rrf_k),
            int(settings.dual_query_oversample),
            bool(settings.dual_query_dense_for_mmr),
        )
        logger.info("Collections ensured at startup")
    except Exception as exc:
        # Do not block startup on failures; health endpoint will reflect real status
        logger.warning("Startup ensure_collections failed: %s", exc)


# Admin UI and step-by-step debug endpoints are defined in app/admin_routes.py


@app.get(
    "/about",
    response_model=About,
    include_in_schema=False,
    summary="Informacje o aplikacji",
    description="Zwraca metadane serwisu rags_tool (nazwa, wersja, autor, opis).",
)
# Return static service metadata (name, version, author, description).
def about():
    """Return basic service metadata (name, version, author, description)."""
    return About()


@app.get(
    "/health",
    include_in_schema=False,
    summary="Stan usługi",
    description="Sprawdza połączenie z Qdrant i raportuje kondycję aplikacji.",
)
# Lightweight health probe with Qdrant connectivity check.
def health():
    """Return health status with a light Qdrant connectivity check."""
    try:
        qdrant.get_collections()
        return {"status": "ok", "qdrant": True}
    except Exception as e:
        return {"status": "degraded", "qdrant": False, "error": str(e)}


@app.post(
    "/analysis/contradictions",
    response_model=ContradictionAnalysisResponse,
    summary="Analiza sprzeczności (sekcjami)",
    description=getattr(settings, "contradictions_tool_description", "Analiza sprzeczności dla tytułu dokumentu."),
    tags=["tools"],
)
def analysis_contradictions(req: ContradictionAnalysisRequest):
    """Przeprowadź analizę sprzeczności dla dokumentu wskazanego tytułem.

    Domyślnie przeszukuje wyłącznie dokumenty obowiązujące (is_active=true) i raportuje
    sekcjami na poziomie 'ust'. Analiza wykonywana jest on-the-fly (bez trwałej pamięci).
    """
    try:
        _ensure_collections_cached()
    except Exception:
        # kontynuuj — analiza i tak spróbuje z Qdrant i zwróci błąd w razie problemów
        pass
    return analyze_contradictions(req)


# --- Search debug: step-by-step endpoints ---





@app.post(
    "/collections/init",
    include_in_schema=False,
    summary="Inicjalizacja kolekcji",
    description=(
        "Tworzy parę kolekcji (streszczenia + treść) dla wskazanej nazwy bazowej (jeśli nie istnieją) i opcjonalnie sondą sprawdza wymiar embeddingów przy użyciu force_dim_probe."
    ),
)
# Initialize or validate summary/content collections for a base name.
def collections_init(req: InitCollectionsRequest):
    """Create or validate the pair of collections for the given base name."""
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
# Export all collections and TF‑IDF artifacts as a tar.gz bundle.
def collections_export(req: CollectionsExportRequest):
    """Export all Qdrant collections and local TF-IDF artifacts to tar.gz."""
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
# Import collections from an uploaded tar.gz bundle (multipart or raw body).
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
# List candidate files for ingest under base_dir.
def ingest_scan(req: ScanRequest):
    """Return files under base_dir that qualify for ingest (by extension)."""
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
# Generate summaries for the given files (no persistence).
def summaries_generate(req: SummariesGenerateRequest):
    """Generate summaries for the provided files without persisting results."""
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
# Full ingest: parse, summarize, embed and upsert into Qdrant.
def ingest_build(req: IngestBuildRequest):
    """Parse, summarize, embed and upsert corpus into Qdrant (full ingest)."""
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
            for record in _iter_document_records(
                file_paths, req.chunk_tokens, req.chunk_overlap, force_regen_summary=req.force_regen_summary
            ):
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
# Two‑stage hybrid search with optional reranker over merged blocks.
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
    - **result_format** (str): `"flat" | "grouped" | "blocks"` (`"blocks"` is default and recommended for tools).

    Recommendations for LLM callers
    --------------------------------
    * Prefer `result_format="blocks"` to obtain concise evidence blocks; bloki są
      budowane przez scalenie wszystkich chunków danej sekcji w pełny blok tekstu.
    * Keep `top_k` between 5‑10 unless you need finer granularity.
    * `summary_mode="first"` returns a single document summary per hit, useful for
      citation.
    * The returned payload contains `blocks` where each block includes:
        - `text` – concatenated chunk text (evidence).
        - `path` – source file path.
        - `title` – document title.
        - `doc_date` – document date (YYYY, YYYY-MM, or YYYY-MM-DD) or 'brak'.
        - `is_active` – whether the document is current (true) or archival (false).
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
            sec = payload.get("section_path")
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

    # Build final fused list (RRF over unique chunks)
    fused_list = [
        {"payload": v.get("payload"), "score": float(v.get("score", 0.0))}
        for v in fused.values()
    ]
    fused_list.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    # Shape results first; merged blocks are built AFTER fusion
    # Dalsze kroki (reranking) wykonujemy na już zmergowanych blokach
    results, groups_payload, blocks_payload = _shape_results(fused_list, {}, global_mmr_pool, global_rel2, req)

    if req.result_format == "blocks":
        # Opcjonalny reranker na zmergowanych blokach sekcyjnych
        if ranker_enabled and (blocks_payload or []):
            try:
                client = OpenAIReranker(settings.ranker_base_url or "", settings.ranker_api_key, settings.ranker_model or "")
                passages = [
                    _truncate_head_tail(str(b.get("text", "")), max(1, int(settings.ranker_max_length)))
                    for b in (blocks_payload or [])
                ]
                q_joined = " | ".join(queries)
                top_n = min(max(1, int(settings.return_top_k)), len(passages))
                rr = client.rerank(query=q_joined, documents=passages, top_n=top_n)
                # Zamapuj ranking rankera do bloków (po indeksie wejściowym)
                idx_to_block = {i: b for i, b in enumerate(blocks_payload or [])}
                rr_sorted = sorted(rr, key=lambda r: float(r.get("relevance_score", 0.0)), reverse=True)
                # Uzupełnij ranker_score i zastosuj per_doc_limit w kolejności rankera
                counts: Dict[str, int] = {}
                selected_blocks = []
                for r in rr_sorted:
                    i = int(r.get("index", -1))
                    if i < 0 or i >= len(idx_to_block):
                        continue
                    b = dict(idx_to_block[i])
                    b["ranker_score"] = float(r.get("relevance_score", 0.0))
                    did = str(b.get("doc_id", ""))
                    if did:
                        if counts.get(did, 0) >= max(1, int(req.per_doc_limit)):
                            continue
                        counts[did] = counts.get(did, 0) + 1
                    selected_blocks.append(b)
                    if len(selected_blocks) >= top_n:
                        break
                blocks_payload = selected_blocks
            except Exception as exc:
                logger.warning("Ranker failed on merged blocks: %s", exc)
                # Fallback: bez reranka, utnij do top_k po score i per_doc_limit
                counts: Dict[str, int] = {}
                trimmed = []
                for b in sorted(blocks_payload or [], key=lambda x: float(x.get("score", 0.0)), reverse=True):
                    did = str(b.get("doc_id", ""))
                    if did:
                        if counts.get(did, 0) >= max(1, int(req.per_doc_limit)):
                            continue
                        counts[did] = counts.get(did, 0) + 1
                    trimmed.append(b)
                    if len(trimmed) >= max(1, int(req.top_k)):
                        break
                blocks_payload = trimmed
        else:
            # Bez rankera: egzekwuj per_doc_limit i utnij do top_k po score
            counts: Dict[str, int] = {}
            trimmed = []
            for b in sorted(blocks_payload or [], key=lambda x: float(x.get("score", 0.0)), reverse=True):
                did = str(b.get("doc_id", ""))
                if did:
                    if counts.get(did, 0) >= max(1, int(req.per_doc_limit)):
                        continue
                    counts[did] = counts.get(did, 0) + 1
                trimmed.append(b)
                if len(trimmed) >= max(1, int(req.top_k)):
                    break
            blocks_payload = trimmed

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
