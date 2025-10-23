"""Golden QA UI and API routes.

Provides a simple HTML page under /golden and a small REST API to
generate, list, edit and regenerate QA items for a golden test set.

This integrates with existing CLI functionality in tools/golden_make.py
by importing and calling its build_golden() function. Single‑item
regeneration uses the same LLM flow as the CLI.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List, Optional
import random

from fastapi import HTTPException, Query
from fastapi.responses import HTMLResponse, FileResponse
import mimetypes

from app.models import (
    GoldenGenerateRequest,
    GoldenGenerateResponse,
    GoldenListResponse,
    GoldenUpdateRequest,
    GoldenRegenerateRequest,
)


def _golden_file_path(out_dir: str) -> pathlib.Path:
    base = pathlib.Path(out_dir)
    return base / "golden_qa.jsonl"


def _read_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if not path.exists():
        return items
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            try:
                items.append(json.loads(s))
            except Exception:
                # Skip malformed lines
                continue
    return items


def _write_jsonl_atomic(path: pathlib.Path, items: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    tmp.replace(path)


def attach_golden_routes(app) -> None:
    """Attach Golden QA routes and the HTML page to a FastAPI app."""

    # HTML UI
    def golden_ui():
        tpl_path = pathlib.Path(__file__).parent.parent / "templates" / "golden.html"
        try:
            html = tpl_path.read_text(encoding="utf-8")
        except Exception:
            return HTMLResponse(content="<html><body><p>Golden UI unavailable.</p></body></html>")
        return HTMLResponse(content=html)

    app.get("/golden", include_in_schema=False, response_class=HTMLResponse, summary="Golden QA UI")(golden_ui)

    # File download for a given absolute path (limited to supported extensions)
    def golden_file(path: str = Query(..., description="Absolutna ścieżka pliku dokumentu")):
        p = pathlib.Path(path)
        if not p.exists() or not p.is_file():
            raise HTTPException(status_code=404, detail="Plik nie istnieje")
        # Try to reuse allowed extensions from tools.golden_make; fallback to safe set
        try:
            from tools.golden_make import SUPPORTED_EXT as _SUP_EXT  # type: ignore
            allowed = set(_SUP_EXT)  # type: ignore[arg-type]
        except Exception:
            allowed = {".txt", ".md", ".markdown", ".html", ".htm", ".pdf"}
        if p.suffix.lower() not in allowed:
            raise HTTPException(status_code=400, detail="Niedozwolone rozszerzenie pliku")
        # Guess mime
        mime, _ = mimetypes.guess_type(str(p))
        return FileResponse(str(p), media_type=mime or "application/octet-stream", filename=p.name)

    app.get("/golden/file", include_in_schema=False, summary="Pobierz plik dokumentu")(golden_file)

    # API: generate dataset using tools.golden_make
    def golden_generate(req: GoldenGenerateRequest) -> GoldenGenerateResponse:
        try:
            from tools.golden_make import build_golden  # type: ignore
        except Exception as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=500, detail=f"Golden generator unavailable: {exc}")

        base = pathlib.Path(req.base_dir)
        if not base.exists():
            raise HTTPException(status_code=400, detail="base_dir nie istnieje")
        out_dir = pathlib.Path(req.out_dir)
        # Ensure LLM env (GOLDEN_*) — fallback to SUMMARY_* if not set
        import os
        from app.settings import get_settings
        set_fallback = False
        old_env = {
            "GOLDEN_LLM_BASE_URL": os.environ.get("GOLDEN_LLM_BASE_URL"),
            "GOLDEN_LLM_API_KEY": os.environ.get("GOLDEN_LLM_API_KEY"),
            "GOLDEN_LLM_MODEL": os.environ.get("GOLDEN_LLM_MODEL"),
        }
        if not (old_env["GOLDEN_LLM_BASE_URL"] and old_env["GOLDEN_LLM_API_KEY"] and old_env["GOLDEN_LLM_MODEL"]):
            s = get_settings()
            if s.summary_api_url and s.summary_api_key and s.summary_model:
                os.environ["GOLDEN_LLM_BASE_URL"] = str(s.summary_api_url)
                os.environ["GOLDEN_LLM_API_KEY"] = str(s.summary_api_key)
                os.environ["GOLDEN_LLM_MODEL"] = str(s.summary_model)
                set_fallback = True
        try:
            summary: Dict[str, Any] = build_golden(
                base_dir=base,
                out_dir=out_dir,
                glob=req.glob,
                recursive=bool(req.recursive),
                limit_docs=(int(req.limit_docs) if req.limit_docs is not None else None),
                per_doc_qa=max(1, int(req.per_doc_qa)),
                target_qa=(int(req.target_qa) if req.target_qa is not None else None),
                seed=int(req.seed),
            )
        except RuntimeError as rexc:
            # Common cases: LLM not configured (400) vs connectivity (502). Keep simple: treat as bad gateway for UI visibility.
            raise HTTPException(status_code=502, detail=str(rexc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Błąd generowania golden QA: {exc}")
        finally:
            # Restore env if fallback was applied
            if set_fallback:
                for k, v in old_env.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v

        qa_path = _golden_file_path(str(out_dir))
        return GoldenGenerateResponse(
            documents=int(summary.get("documents", 0)),
            qa_items=int(summary.get("qa_items", 0)),
            took_ms=int(summary.get("took_ms", 0)),
            seed=int(summary.get("seed", 0)),
            use_llm=bool(summary.get("use_llm", False)),
            llm_model=summary.get("llm_model"),
            qa_path=str(qa_path.resolve()),
        )

    app.post("/golden/generate", include_in_schema=False, summary="Generuj golden QA")(golden_generate)

    # API: list items
    def golden_list(out_dir: str = Query("data/golden", description="Katalog z plikiem golden_qa.jsonl")) -> GoldenListResponse:
        qa_path = _golden_file_path(out_dir)
        items = _read_jsonl(qa_path)
        # Cast to target shape (dict passthrough is fine for Pydantic)
        return GoldenListResponse(items=items, qa_path=str(qa_path.resolve()))

    app.get("/golden/list", include_in_schema=False, summary="Pobierz listę QA")(golden_list)

    # API: update one item (question/answer)
    def golden_update(req: GoldenUpdateRequest) -> Dict[str, Any]:
        qa_path = _golden_file_path(req.out_dir)
        items = _read_jsonl(qa_path)
        found = False
        for it in items:
            if str(it.get("id")) == req.id:
                it["query"] = req.query
                it["expected_answer"] = req.expected_answer
                found = True
                break
        if not found:
            raise HTTPException(status_code=404, detail="Nie znaleziono pozycji o podanym id")
        _write_jsonl_atomic(qa_path, items)
        return {"status": "ok", "updated": req.id}

    app.post("/golden/update", include_in_schema=False, summary="Aktualizuj jedną pozycję QA")(golden_update)

    # API: regenerate one item using the same doc (LLM required)
    def golden_regenerate(req: GoldenRegenerateRequest) -> Dict[str, Any]:
        qa_path = _golden_file_path(req.out_dir)
        items = _read_jsonl(qa_path)
        idx = -1
        cur: Optional[Dict[str, Any]] = None
        for i, it in enumerate(items):
            if str(it.get("id")) == req.id:
                idx = i
                cur = it
                break
        if idx < 0 or cur is None:
            raise HTTPException(status_code=404, detail="Nie znaleziono pozycji o podanym id")

        # Obtain doc path from meta
        meta = cur.get("meta") or {}
        doc_path = meta.get("doc_path") or meta.get("path")
        title = meta.get("doc_title") or ""
        as_of = meta.get("as_of") or None
        if not doc_path:
            raise HTTPException(status_code=400, detail="Brak ścieżki dokumentu w meta; nie można zregenerować")

        p = pathlib.Path(str(doc_path))
        if not p.exists():
            raise HTTPException(status_code=400, detail="Plik dokumentu nie istnieje (meta.doc_path)")

        # Import LLM helpers from the CLI module to keep single source of truth
        try:
            from tools.golden_make import (  # type: ignore
                _extract_text as _gm_extract_text,
                _llm_client_from_env as _gm_llm_client,
                _llm_generate_natural_qa_for_doc as _gm_generate,
                _extract_doc_symbol_from_path as _gm_doc_symbol,
                _infer_as_of_year as _gm_infer_year,
                _scan_files as _gm_scan_files,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Brak narzędzi LLM do regeneracji: {exc}")

        # Ensure LLM env (GOLDEN_*) — fallback to SUMMARY_* if not set
        import os
        from app.settings import get_settings
        client = _gm_llm_client()
        model = os.environ.get("GOLDEN_LLM_MODEL")
        if not (client and model):
            s = get_settings()
            if s.summary_api_url and s.summary_api_key and s.summary_model:
                # Set env and rebuild client
                os.environ["GOLDEN_LLM_BASE_URL"] = str(s.summary_api_url)
                os.environ["GOLDEN_LLM_API_KEY"] = str(s.summary_api_key)
                os.environ["GOLDEN_LLM_MODEL"] = str(s.summary_model)
                client = _gm_llm_client()
                model = os.environ.get("GOLDEN_LLM_MODEL")
        if not (client and model):
            raise HTTPException(status_code=400, detail="LLM nie jest skonfigurowany (GOLDEN_* lub SUMMARY_*). Ustaw GOLDEN_LLM_* albo SUMMARY_API_URL/KEY/MODEL.")

        # Extract current document text (base case)
        try:
            text = _gm_extract_text(p)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Nie można odczytać dokumentu: {exc}")

        # Build pool of candidate documents from base_dir/glob (preferred)
        chosen_path = p
        chosen_title = title
        chosen_symbol = meta.get("doc_symbol") or p.stem
        chosen_as_of = as_of

        if bool(req.use_random_doc):
            pool: List[pathlib.Path] = []
            # Always use provided base_dir/glob/recursive if available
            base_dir_val = req.base_dir
            if base_dir_val:
                try:
                    base_dir = pathlib.Path(str(base_dir_val))
                    if base_dir.exists():
                        g = req.glob or "**/*"
                        pool = _gm_scan_files(base_dir, g, bool(req.recursive if req.recursive is not None else True))
                except Exception:
                    pool = []
            # Fallback: use documents recorded in golden_documents.jsonl (same out_dir)
            if not pool:
                docs_path = pathlib.Path(req.out_dir) / "golden_documents.jsonl"
                if docs_path.exists():
                    try:
                        with docs_path.open("r", encoding="utf-8") as dfh:
                            for line in dfh:
                                try:
                                    rec = json.loads(line.strip())
                                except Exception:
                                    continue
                                pr = rec.get("path")
                                if pr:
                                    pool.append(pathlib.Path(pr))
                    except Exception:
                        pool = []
            # Choose random; prefer different from current
            candidates = [pp for pp in pool if pp.exists() and str(pp.resolve()) != str(p.resolve())]
            if not candidates:
                candidates = [pp for pp in pool if pp.exists()]
            if candidates:
                rnd = random.Random(int(req.seed) if req.seed is not None else None)
                chosen_path = rnd.choice(candidates)
                try:
                    text2 = _gm_extract_text(chosen_path)
                except Exception as exc:
                    raise HTTPException(status_code=500, detail=f"Nie można odczytać dokumentu: {exc}")
                # title: first non-empty line
                t2 = ""
                for line in text2.splitlines():
                    if line.strip():
                        t2 = line.strip()[:200]
                        break
                chosen_title = t2
                chosen_symbol = _gm_doc_symbol(chosen_path)
                chosen_as_of = _gm_infer_year(text2, t2) or ""
                text = text2  # type: ignore[assignment]
                # Ensure chosen document is recorded in golden_documents.jsonl
                try:
                    docs_path = pathlib.Path(req.out_dir) / "golden_documents.jsonl"
                    existing: set[str] = set()
                    if docs_path.exists():
                        with docs_path.open("r", encoding="utf-8") as dfh:
                            for line in dfh:
                                try:
                                    rec = json.loads(line.strip())
                                except Exception:
                                    continue
                                pr = str(rec.get("path") or "").strip()
                                if pr:
                                    existing.add(pr)
                    abs_path = str(chosen_path.resolve())
                    if abs_path not in existing:
                        rec = {
                            "path": abs_path,
                            "title": chosen_title,
                            "doc_symbol": chosen_symbol,
                            "as_of": chosen_as_of or "",
                        }
                        with docs_path.open("a", encoding="utf-8") as dfh:
                            dfh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                except Exception:
                    pass

        # Now call LLM to generate up to 3 items with requested temperature
        try:
            items_llm = _gm_generate(
                client,
                model,
                doc_text=text,  # type: ignore[arg-type]
                doc_title=str(chosen_title or ""),
                doc_symbol=str(chosen_symbol or pathlib.Path(chosen_path).stem),
                max_items=3,
                temperature=float(req.temperature or 0.7),
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc))
        if not items_llm:
            raise HTTPException(status_code=500, detail="LLM nie zwrócił nowej pary pytanie/odpowiedź")

        # Pick first different Q/A if requested
        def _norm(s: str) -> str:
            return " ".join((s or "").split()).strip()

        existing_pairs = {(_norm(it.get("query", "")), _norm(it.get("expected_answer", ""))) for it in items}
        q_old = _norm(str(cur.get("query", "")))
        a_old = _norm(str(cur.get("expected_answer", "")))

        chosen = None
        for cand in items_llm:
            q_c = _norm(str(cand.get("query", "")))
            a_c = _norm(str(cand.get("answer_text", "")))
            if not q_c or not a_c:
                continue
            if bool(req.ensure_different) and ((q_c, a_c) == (q_old, a_old) or (q_c, a_c) in existing_pairs):
                continue
            chosen = cand
            break
        if chosen is None:
            chosen = items_llm[0]

        q = str(chosen.get("query", ""))
        a = str(chosen.get("answer_text", ""))
        # No post-processing of the question — keep LLM output as-is
        if not q or not a:
            raise HTTPException(status_code=500, detail="Nowe pytanie/odpowiedź są puste")

        # Update record (keep id, replace content + meta)
        cur["query"] = q
        cur["expected_answer"] = a
        kv = chosen.get("key_values") or []
        if isinstance(kv, list):
            cur["key_values"] = kv
        # refresh meta
        cur_meta = cur.get("meta") or {}
        cur_meta["doc_title"] = chosen_title
        cur_meta["doc_path"] = str(chosen_path.resolve())
        cur_meta["doc_symbol"] = chosen_symbol
        cur_meta["as_of"] = chosen_as_of if chosen_as_of else None
        cur["meta"] = cur_meta

        items[idx] = cur
        _write_jsonl_atomic(qa_path, items)
        return {"status": "ok", "regenerated": req.id, "query": q, "expected_answer": a}

    app.post("/golden/regenerate", include_in_schema=False, summary="Zregeneruj pojedynczą pozycję QA")(golden_regenerate)
