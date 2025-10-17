"""Corpus-wide contradictions scanner CLI using rags_tool HTTP API.

This tool enumerates documents (via API or directly from Qdrant) and, for each document,
calls `/analysis/contradictions` to produce a per-document HTML report.

Features:
- Resume: skips documents with existing per-doc HTML (unless --overwrite).
- Concurrency: configurable worker pool (default 2).
- Live console progress: prints per-doc status and running totals.
- Run snapshots: writes run-level index.json and report.html.

Note: The tool uses only HTTP endpoints (no direct imports from app modules).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

try:
    from urllib.request import Request, urlopen
    from urllib.error import HTTPError, URLError
except Exception:  # pragma: no cover
    raise

__version__ = "0.2.0"


def _slugify(text: str, max_len: int = 48) -> str:
    """Slugify a string to a short, filesystem-friendly identifier.

    - Normalize diacritics (ASCII fold)
    - Lowercase, replace non-alnum with '-'
    - Collapse repeats, trim, cut to max_len
    """
    s = (text or "").strip()
    if not s:
        return "doc"
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    if not s:
        s = "doc"
    return s[:max_len]


def _http_get_json(base_url: str, path: str, timeout: int = 60) -> Any:
    url = f"{base_url.rstrip('/')}{path}"
    req = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            data = resp.read()
            return json.loads(data.decode("utf-8"))
    except HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"GET {url} failed: {e.code} {e.reason} | {body[:200]}")
    except URLError as e:
        raise RuntimeError(f"GET {url} failed: {e}")


def _http_post_json(base_url: str, path: str, payload: Dict[str, Any], timeout: int = 120) -> Any:
    url = f"{base_url.rstrip('/')}{path}"
    data = json.dumps(payload).encode("utf-8")
    req = Request(
        url,
        data=data,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            return json.loads(raw.decode("utf-8"))
    except HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"POST {url} failed: {e.code} {e.reason} | {body[:200]}")
    except URLError as e:
        raise RuntimeError(f"POST {url} failed: {e}")


def _http_post_json_abs(url: str, payload: Dict[str, Any], timeout: int = 60, headers: Optional[Dict[str, str]] = None) -> Any:
    data = json.dumps(payload).encode("utf-8")
    req = Request(
        url,
        data=data,
        headers={"Accept": "application/json", "Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            return json.loads(raw.decode("utf-8"))
    except HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"POST {url} failed: {e.code} {e.reason} | {body[:200]}")
    except URLError as e:
        raise RuntimeError(f"POST {url} failed: {e}")


def _now_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _elapsed_str(start_ts: float) -> str:
    sec = int(time.time() - start_ts)
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _discover_via_api(api_url: str, mode: str) -> Optional[List[Dict[str, Any]]]:
    try:
        resp = _http_get_json(api_url, f"/docs/list?mode={mode}")
        docs = resp.get("docs") or []
        if isinstance(docs, list):
            return docs
        return None
    except Exception:
        return None


def _discover_via_qdrant(qdrant_url: str, collection: str, mode: str) -> List[Dict[str, Any]]:
    """Fetch documents by scrolling Qdrant summaries collection (mode=current|archival)."""
    must = [
        {"key": "point_type", "match": {"value": "summary"}},
        {"key": "is_active", "match": {"value": True if mode == "current" else False}},
    ]
    out: List[Dict[str, Any]] = []
    next_page = None
    while True:
        body = {
            "filter": {"must": must},
            "limit": 256,
            "with_payload": ["doc_id", "title", "doc_date", "is_active", "path"],
        }
        if next_page is not None:
            body["offset"] = next_page
        res = _http_post_json_abs(f"{qdrant_url.rstrip('/')}/collections/{collection}/points/scroll", body, timeout=60)
        points = res.get("points") or res.get("result", {}).get("points") or []
        next_page = res.get("next_page_offset") or res.get("result", {}).get("next_page_offset")
        for p in points:
            payload = p.get("payload") or {}
            did = payload.get("doc_id")
            if not did:
                continue
            out.append({
                "doc_id": str(did),
                "title": payload.get("title"),
                "doc_date": payload.get("doc_date"),
                "is_active": payload.get("is_active"),
                "path": payload.get("path"),
            })
        if not points or next_page is None:
            break
    # stable ordering
    out.sort(key=lambda x: ((x.get("title") or "").lower(), x.get("doc_id") or ""))
    return out


def discover_documents(api_url: str, mode: str, *, list_mode: str, qdrant_url: str, summary_collection: str, path_prefix: Optional[str]) -> List[Dict[str, Any]]:
    """Discover documents using preferred list mode, then apply path filter if provided."""
    docs: Optional[List[Dict[str, Any]]] = None
    if list_mode == "api":
        docs = _discover_via_api(api_url, mode)
    elif list_mode == "qdrant":
        docs = _discover_via_qdrant(qdrant_url, summary_collection, mode)
    else:  # auto
        docs = _discover_via_api(api_url, mode)
        # If API didn't return, or path filter requested but no 'path' in payloads, fall back to Qdrant
        if not docs or (path_prefix and docs and not any("path" in d for d in docs)):
            docs = _discover_via_qdrant(qdrant_url, summary_collection, mode)
    if docs is None:
        docs = []
    if path_prefix:
        pfx = str(Path(path_prefix).resolve())
        def _match(d):
            p = d.get("path")
            if not p:
                return False
            try:
                return str(Path(p).resolve()).startswith(pfx)
            except Exception:
                return False
        docs = [d for d in docs if _match(d)]
    return docs


def ensure_dirs(base_out: Path, mode: str, run_id: str) -> Tuple[Path, Path]:
    """Create output directories and return (per_doc_dir, run_dir)."""
    per_doc_dir = base_out / "per_doc" / mode
    run_dir = base_out / "runs" / run_id
    per_doc_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    return per_doc_dir, run_dir


def build_filename(doc: Dict[str, Any], used: set[str], per_doc_dir: Path, *, ext: str = "html") -> str:
    """Build a unique, short filename for a document JSON report."""
    title = str(doc.get("title") or "").strip() or "document"
    slug = _slugify(title)
    y = None
    dd = str(doc.get("doc_date") or "").strip()
    m = re.match(r"^(\d{4})", dd)
    if m:
        y = m.group(1)
    base = slug + (f"-{y}" if y else "")
    # Resolve collisions by appending doc8 suffix
    name = base
    suffix = 0
    while True:
        candidate = name + f".{ext}"
        if candidate not in used and not (per_doc_dir / candidate).exists():
            used.add(candidate)
            return candidate
        # Append short doc id only once
        if suffix == 0:
            doc_id = str(doc.get("doc_id") or "")
            doc8 = doc_id[:8] if doc_id else f"{len(used):02d}"
            name = base + f"-{doc8}"
        else:
            name = base + f"-{suffix:02d}"
        suffix += 1


def analyze_document(
    api_url: str,
    mode: str,
    section_level: str,
    candidates: int,
    confidence_threshold: float,
    out_path: Path,
    doc: Dict[str, Any],
    timeout_s: int,
) -> Tuple[str, int]:
    """Call the API to analyze contradictions for a single document and persist JSON.

    Returns (output_file_name, conflicts_count).
    """
    payload = {
        "title": str(doc.get("title") or ""),
        "doc_id": str(doc.get("doc_id") or ""),
        "mode": mode,
        "section_level": section_level,
        "max_candidates_per_section": int(candidates),
        "include_archival_conflicts": True if mode == "archival" else False,
        "confidence_threshold": float(confidence_threshold),
    }
    res = _http_post_json(api_url, "/analysis/contradictions", payload, timeout=max(1, int(timeout_s)))
    # Count total conflicts
    total_conflicts = 0
    for rep in (res.get("findings") or []):
        total_conflicts += int(len(rep.get("conflicts") or []))
    # Render per-document HTML
    title = res.get("title") or (doc.get("title") if isinstance(doc, dict) else None) or "(brak tytułu)"
    doc_date = res.get("doc_date") or doc.get("doc_date") if isinstance(doc, dict) else None
    took_ms = int(res.get("took_ms" or 0)) if isinstance(res, dict) else 0
    sections_with_conflicts = len(res.get("findings") or []) if isinstance(res, dict) else 0
    rows = []
    for sec in (res.get("findings") or []):
        sec_name = sec.get("section") or "(sekcja)"
        rule = sec.get("rule") or ""
        rule_type = sec.get("rule_type") or ""
        conflicts = sec.get("conflicts") or []
        rows.append(f"<h3>{sec_name}</h3>")
        if rule:
            rows.append(f"<div class='muted'>Reguła: {rule} {f'({rule_type})' if rule_type else ''}</div>")
        if not conflicts:
            rows.append("<div class='ok'>Brak niezgodności w tej sekcji.</div>")
            continue
        rows.append("<ul>")
        for c in conflicts:
            other_t = c.get("other_title") or c.get("other_doc_id")
            other_d = c.get("other_doc_date") or ""
            conf = c.get("confidence")
            rationale = c.get("rationale") or ""
            qb = c.get("quotes_b") or []
            qa = c.get("quotes_a") or []
            li = [f"<div><b>{other_t}</b> {('('+other_d+')') if other_d else ''} — pewność: {conf:.2f}</div>"]
            if rationale:
                li.append(f"<div class='muted'>{rationale}</div>")
            if qa:
                li.append("<div><span class='tag'>A</span> " + " | ".join(qa) + "</div>")
            if qb:
                li.append("<div><span class='tag'>B</span> " + " | ".join(qb) + "</div>")
            rows.append("<li>" + "".join(li) + "</li>")
        rows.append("</ul>")
    # Processed documents block (audit)
    proc_rows = []
    for d in (res.get("processed_docs") or []):
        t = d.get("title") or d.get("doc_id")
        dd = d.get("doc_date") or ""
        ia = d.get("is_active")
        state = "obowiązujący" if ia is True else ("archiwalny" if ia is False else "?")
        proc_rows.append(f"<tr><td>{t}</td><td>{dd}</td><td>{state}</td></tr>")

    html = f"""
<!DOCTYPE html>
<html lang=\"pl\">
<head>
  <meta charset=\"utf-8\" />
  <title>Sprzeczności — {title}</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; }}
    h1 {{ font-size: 20px; margin: 0 0 6px; }}
    .meta {{ color:#555; margin-bottom: 16px; }}
    h3 {{ margin: 16px 0 6px; }}
    .muted {{ color:#666; font-size: 90%; }}
    .ok {{ color:#065f46; background:#ecfdf5; padding:6px 8px; border-radius:6px; display:inline-block; }}
    ul {{ padding-left: 18px; }}
    .tag {{ display:inline-block; font-weight:600; background:#eef; color:#334; border-radius:4px; padding:1px 6px; margin-right:4px; }}
  </style>
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <meta name=\"robots\" content=\"noindex\" />
  <meta name=\"referrer\" content=\"no-referrer\" />
  <meta name=\"generator\" content=\"contradictions-scan {__version__}\" />
  </head>
  <body>
    <h1>Sprzeczności — {title}</h1>
    <div class=\"meta\">Data: {doc_date or ''} • sekcji z konfliktami: {sections_with_conflicts} • łącznie konfliktów: {total_conflicts} • czas: {took_ms}ms</div>
    <h2>Przetworzone dokumenty</h2>
    <table>
      <thead><tr><th>Tytuł</th><th>Data</th><th>Status</th></tr></thead>
      <tbody>
        {''.join(proc_rows) if proc_rows else '<tr><td colspan="3" class="muted">Brak danych</td></tr>'}
      </tbody>
    </table>
    <hr />
    {''.join(rows) if rows else '<div class="ok">Nie znaleziono niezgodności.</div>'}
  </body>
</html>
"""
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write(html)
    return out_path.name, total_conflicts


def write_index(run_dir: Path, mode: str, run_id: str, started: str, results: List[Dict[str, Any]]) -> None:
    """Write run-level index.json with summary and per-doc entries."""
    ended = _now_iso()
    summary = {
        "tool": "contradictions-scan",
        "version": __version__,
        "mode": mode,
        "run_id": run_id,
        "started_at": started,
        "ended_at": ended,
        "documents": len(results),
        "processed": sum(1 for r in results if not r.get("skipped") and not r.get("error")),
        "skipped": sum(1 for r in results if r.get("skipped")),
        "errors": sum(1 for r in results if r.get("error")),
        "with_conflicts": sum(1 for r in results if int(r.get("conflicts", 0)) > 0),
    }
    out = {"summary": summary, "results": results}
    with (run_dir / "index.json").open("w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)


def write_html(run_dir: Path, mode: str, run_id: str, results: List[Dict[str, Any]]) -> None:
    """Write a simple HTML report with summary and a results table."""
    docs_total = len(results)
    processed = sum(1 for r in results if not r.get("skipped") and not r.get("error"))
    skipped = sum(1 for r in results if r.get("skipped"))
    errors = sum(1 for r in results if r.get("error"))
    with_conflicts = sum(1 for r in results if int(r.get("conflicts", 0)) > 0)

    rows = []
    for r in results:
        title = r.get("title") or "(brak tytułu)"
        date = r.get("doc_date") or ""
        conflicts = int(r.get("conflicts") or 0)
        link = r.get("file_rel") or ""
        status = "błąd" if r.get("error") else ("pominięty" if r.get("skipped") else "ok")
        rows.append(
            f"<tr><td>{title}</td><td>{date}</td><td class='num'>{conflicts}</td><td>{status}</td><td><a href='{link}'>JSON</a></td></tr>"
        )

    html = f"""
<!DOCTYPE html>
<html lang="pl">
<head>
  <meta charset="utf-8" />
  <title>Raport sprzeczności — {mode} — {run_id}</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 20px; }}
    h1 {{ font-size: 20px; margin-bottom: 8px; }}
    .meta {{ color: #555; margin-bottom: 16px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; }}
    th {{ background: #f8f8f8; text-align: left; }}
    td.num {{ text-align: right; }}
  </style>
  </head>
  <body>
    <h1>Raport sprzeczności — tryb: {mode}</h1>
    <div class="meta">Narzędzie: contradictions-scan {__version__} • run_id: {run_id} • dokumentów: {docs_total} • przetworzonych: {processed} • z konfliktami: {with_conflicts} • pominiętych: {skipped} • błędów: {errors}</div>
    <table>
      <thead>
        <tr><th>Tytuł</th><th>Data</th><th>Konflikty</th><th>Status</th><th>Plik</th></tr>
      </thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
  </body>
</html>
"""
    with (run_dir / "report.html").open("w", encoding="utf-8") as fh:
        fh.write(html)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Corpus-wide contradictions scanner (HTTP API)")
    p.add_argument("--api-url", default=os.environ.get("RAGS_TOOL_API_URL", "http://127.0.0.1:8000"), help="Base URL of rags_tool API (default: http://127.0.0.1:8000)")
    p.add_argument("--mode", choices=["current", "archival"], required=True, help="Which corpus to scan: current or archival")
    p.add_argument("--path-prefix", default=None, help="Only process documents with absolute path starting with this prefix")
    p.add_argument("--concurrency", type=int, default=2, help="Number of concurrent workers (default: 2)")
    p.add_argument("--candidates", type=int, default=5, help="Max candidates per section (default: 5)")
    p.add_argument("--section-level", default="ust", help="Section level for grouping (default: ust)")
    p.add_argument("--confidence-threshold", type=float, default=0.6, help="Confidence threshold (default: 0.6)")
    p.add_argument("--out-dir", default="reports/contradictions", help="Output directory (default: reports/contradictions)")
    p.add_argument("--timeout", type=int, default=600, help="Per-document analysis timeout in seconds (default: 600)")
    p.add_argument("--overwrite", action="store_true", help="Recompute even if per-doc HTML exists")
    # Discovery options
    p.add_argument("--list-mode", choices=["auto", "api", "qdrant"], default="auto", help="Document discovery: auto (try API /docs/list, fallback to Qdrant) or force specific mode")
    p.add_argument("--qdrant-url", default=os.environ.get("QDRANT_URL", "http://127.0.0.1:6333"), help="Qdrant base URL (for discovery when using qdrant/auto)")
    p.add_argument("--collection-base", default=os.environ.get("COLLECTION_NAME", "rags_tool"), help="Collection base name (default: rags_tool)")
    p.add_argument("--summary-collection", default=os.environ.get("SUMMARY_COLLECTION_NAME", None), help="Override summary collection name (defaults to '<base>_summaries')")

    args = p.parse_args(argv)

    api_url = args.api_url
    mode = args.mode
    concurrency = max(1, int(args.concurrency))
    candidates = max(1, int(args.candidates))
    section_level = str(args.section_level)
    conf_thr = float(args.confidence_threshold)
    timeout_s = max(1, int(args.timeout))
    out_base = Path(args.out_dir)
    path_prefix = args.path_prefix
    list_mode = args.list_mode
    qdrant_url = args.qdrant_url
    summary_collection = args.summary_collection or f"{args.collection_base}_summaries"

    started_iso = _now_iso()
    run_id = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    per_doc_dir, run_dir = ensure_dirs(out_base, mode, run_id)

    print(f"contradictions-scan {__version__} | mode={mode} | API={api_url}", flush=True)
    print(f"out={out_base} | per-doc={per_doc_dir} | run={run_dir}", flush=True)
    print(f"params: concurrency={concurrency} candidates={candidates} section_level={section_level} conf_thr={conf_thr} timeout={timeout_s}s", flush=True)

    try:
        docs = discover_documents(api_url, mode, list_mode=list_mode, qdrant_url=qdrant_url, summary_collection=summary_collection, path_prefix=path_prefix)
    except Exception as exc:
        print(f"[fatal] Nie udało się pobrać listy dokumentów: {exc}")
        return 2

    if not docs:
        print("Brak dokumentów do przetworzenia.")
        write_index(run_dir, mode, run_id, started_iso, [])
        write_html(run_dir, mode, run_id, [])
        return 0

    print(f"Znaleziono dokumentów: {len(docs)}. Start przetwarzania (concurrency={concurrency})...", flush=True)
    used_names: set[str] = set()

    # Plan tasks (respect resume)
    tasks: List[Tuple[Dict[str, Any], Path]] = []
    results: List[Dict[str, Any]] = []
    for d in docs:
        fname = build_filename(d, used_names, per_doc_dir, ext="html")
        out_path = per_doc_dir / fname
        if out_path.exists() and not args.overwrite:
            results.append(
                {
                    "doc_id": d.get("doc_id"),
                    "title": d.get("title"),
                    "doc_date": d.get("doc_date"),
                    "file": str(out_path),
                    "file_rel": str(Path("..").joinpath(Path("per_doc") / mode / out_path.name)),
                    "conflicts": None,
                    "skipped": True,
                }
            )
        else:
            tasks.append((d, out_path))

    total = len(docs)
    skipped = sum(1 for r in results if r.get("skipped"))
    print(f"Do analizy: {len(tasks)} | pominiętych (resume): {skipped}", flush=True)
    done = 0
    with_conf = 0
    errors = 0
    started = 0
    t0 = time.time()
    lock = threading.Lock()
    active_ids: set[str] = set()
    active_titles: Dict[str, str] = {}

    def _active_line() -> str:
        # Show up to 5 titles to keep line compact
        titles = [active_titles[k] for k in list(active_ids)[:5] if k in active_titles]
        more = len(active_ids) - len(titles)
        shown = " | ".join(titles) if titles else "(brak)"
        return f"Aktywne: {shown}{(f' +{more}' if more>0 else '')}"

    def run_task(d: Dict[str, Any], path: Path) -> Tuple[Dict[str, Any], Optional[str]]:
        # Mark start under lock
        nonlocal started
        with lock:
            started += 1
            did = str(d.get("doc_id") or "")
            ttl = d.get("title") or "(brak tytułu)"
            if did:
                active_ids.add(did)
                active_titles[did] = ttl
            print(
                f"→ [{started}/{total}] {ttl} | {_active_line()} | do uruchomienia: {len(tasks)-started}",
                flush=True,
            )
        try:
            name, cnt = analyze_document(
                api_url, mode, section_level, candidates, conf_thr, path, d, timeout_s
            )
            return (
                {
                    "doc_id": d.get("doc_id"),
                    "title": d.get("title"),
                    "doc_date": d.get("doc_date"),
                    "file": str(path),
                    "file_rel": str(Path("..").joinpath(Path("per_doc") / mode / name)),
                    "conflicts": int(cnt),
                    "skipped": False,
                },
                None,
            )
        except Exception as exc:
            return (
                {
                    "doc_id": d.get("doc_id"),
                    "title": d.get("title"),
                    "doc_date": d.get("doc_date"),
                    "file": str(path),
                    "file_rel": str(Path("..").joinpath(Path("per_doc") / mode / path.name)),
                    "conflicts": 0,
                    "skipped": False,
                    "error": str(exc),
                },
                str(exc),
            )

    # Execute with thread pool
    if not tasks:
        # Nothing to run — all skipped
        print(
            f"Wszystkie dokumenty ({total}) już posiadają wyniki. Pominięto: {skipped}.",
            flush=True,
        )
    else:
        with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
            futs = [ex.submit(run_task, d, p) for (d, p) in tasks]
            for fut in cf.as_completed(futs):
                item, err = fut.result()
                with lock:
                    results.append(item)
                    done += 1
                    done_did = str(item.get("doc_id") or "")
                    if done_did and done_did in active_ids:
                        active_ids.discard(done_did)
                        active_titles.pop(done_did, None)
                    if err:
                        errors += 1
                        print(
                            f"✗ [{done+skipped}/{total}] BŁĄD: {item.get('title')}: {err} | {_active_line()}",
                            flush=True,
                        )
                    else:
                        if int(item.get("conflicts", 0)) > 0:
                            with_conf += 1
                            print(
                                f"✓ [{done+skipped}/{total}] {item.get('title')} — konflikty: {int(item.get('conflicts',0))} | {_active_line()}",
                                flush=True,
                            )
                        else:
                            print(
                                f"✓ [{done+skipped}/{total}] {item.get('title')} — brak konfliktów | {_active_line()}",
                                flush=True,
                            )

    # Merge skipped-at-start entries with processed ones already added
    # (they are already in results list)
    # Sort results by conflicts desc, then title
    results_sorted = sorted(
        results,
        key=lambda r: (-(r.get("conflicts") or 0), (r.get("title") or "")),
    )
    write_index(run_dir, mode, run_id, started_iso, results_sorted)
    write_html(run_dir, mode, run_id, results_sorted)

    print(f"Zakończono. Raport: {run_dir / 'report.html'} | Indeks: {run_dir / 'index.json'}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
