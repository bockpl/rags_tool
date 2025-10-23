"""Golden QA generator — LLM‑only (standalone, RAG‑agnostic).

This CLI scans a directory of documents (txt/md/html/pdf), extracts readable text,
and asks an OpenAI‑compatible LLM to create natural, user‑style questions with
concise answers based solely on each document's content. The result is a decoupled
golden QA set suitable for system‑agnostic evaluation (any system that can answer Q→A).

Key properties
- No dependency on the existing RAG runtime, Qdrant, or app modules.
- Deterministic sampling via --seed.
- LLM‑only generation: the LLM reads the document and produces 1–3 natural
  Q→A items (no references to "§/ust.").

Outputs
- golden_qa.jsonl: one record per QA item with minimal fields needed by an evaluator.
- golden_documents.jsonl: audit-only list of processed files (path, title).
- summary.json: simple stats for quick inspection.

Usage
  python tools/golden_make.py \
    --base-dir /path/to/corpus \
    --out-dir reports/golden \
    --limit-docs 100 --per-doc-qa 10 --seed 123

LLM configuration (required; JSON‑mode, OpenAI‑compatible):
  export GOLDEN_LLM_BASE_URL=http://127.0.0.1:8001/v1
  export GOLDEN_LLM_API_KEY=sk-...
  export GOLDEN_LLM_MODEL=gpt-4o-mini
  python tools/golden_make.py --base-dir ... --target-qa 30

Notes
- The LLM must be configured; if not, the tool exits with an error.
- The LLM reads the provided text (truncated to a safe limit) and proposes 1–3
  natural Q→A items per document; you can cap global count with --target-qa.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import ast
import os
from pathlib import Path
import random
import re
import sys
import time
from typing import Any, Dict, List, Optional


# Optional deps: markdown2/html2text/PyPDF2; fall back gracefully
try:
    import markdown2  # type: ignore
except Exception:  # pragma: no cover
    markdown2 = None  # type: ignore

try:
    import html2text  # type: ignore
except Exception:  # pragma: no cover
    html2text = None  # type: ignore

try:
    import PyPDF2  # type: ignore
    HAS_PDF = True
except Exception:  # pragma: no cover
    HAS_PDF = False

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


SUPPORTED_EXT = {".txt", ".md", ".markdown", ".html", ".htm", ".pdf"}


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", "ignore")).hexdigest()


def _html_to_text(content_html: str) -> str:
    if html2text is None:
        return re.sub(r"<[^>]+>", " ", content_html)
    h = html2text.HTML2Text()
    h.ignore_links = True
    h.ignore_images = True
    h.ignore_emphasis = False
    h.bypass_tables = False
    return h.handle(content_html)


def _read_text_file(path: Path) -> str:
    ext = path.suffix.lower()
    data = path.read_bytes()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("latin1", errors="ignore")
    if ext in {".md", ".markdown"} and markdown2 is not None:
        try:
            html = markdown2.markdown(text)
            text = _html_to_text(html)
        except Exception:
            pass
    elif ext in {".html", ".htm"}:
        try:
            text = _html_to_text(text)
        except Exception:
            pass
    return text


def _extract_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf" and HAS_PDF:
        try:
            with path.open("rb") as fh:
                rdr = PyPDF2.PdfReader(fh)
                pages = [p.extract_text() or "" for p in rdr.pages]
            return "\n\n".join(pages)
        except Exception:
            # fall back to bytes decode
            pass
    return _read_text_file(path)


def _scan_files(base: Path, glob: str, recursive: bool) -> List[Path]:
    iterator = base.rglob(glob) if recursive else base.glob(glob)
    return [p for p in iterator if p.is_file() and p.suffix.lower() in SUPPORTED_EXT]


# --- Simple sectioning (legal-ish markers) ---

# (section detection removed in LLM-only mode)


def _mk_id(*parts: str) -> str:
    base = "::".join(parts)
    return "gq-" + _sha1(base)[:16]


def _norm_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


# (heuristic QA removed in LLM-only mode)

def _llm_client_from_env() -> Optional[Any]:  # returns OpenAI client or None
    if OpenAI is None:
        return None
    base = os.environ.get("GOLDEN_LLM_BASE_URL")
    key = os.environ.get("GOLDEN_LLM_API_KEY")
    model = os.environ.get("GOLDEN_LLM_MODEL")
    if not (base and key and model):
        return None
    try:
        # Sanitize base_url (fix common mistakes)
        try:
            b = (base or "").strip()
            if b.startswith("https:/") and not b.startswith("https://"):
                b = b.replace("https:/", "https://", 1)
            if b.startswith("http:/") and not b.startswith("http://"):
                b = b.replace("http:/", "http://", 1)
            # Ensure /v1 suffix for OpenAI-compatible endpoints
            if not b.rstrip("/").endswith("/v1"):
                b = b.rstrip("/") + "/v1"
        except Exception:
            b = base
        client = OpenAI(base_url=b, api_key=key)
        return client
    except Exception:
        return None


# (section-level LLM QA removed in LLM-only mode)


# --- Runner ---

def _extract_doc_symbol_from_path(path: Path) -> str:
    """Derive a document symbol from filename (stem).

    Strategy: use the filename stem as-is, trimmed to 128 chars. This matches
    common patterns like 'Z_45_2019_tj_15_12_2021'.
    """
    try:
        stem = path.stem.strip()
    except Exception:
        stem = str(path).strip()
    stem = re.sub(r"\s+", "_", stem)
    return stem[:128] if stem else "document"


def _llm_generate_natural_qa_for_doc(
    client: Any,
    model: str,
    *,
    doc_text: str,
    doc_title: str,
    doc_symbol: str,
    max_items: int,
    temperature: float = 0.0,
) -> List[Dict[str, Any]]:
    """Ask LLM to produce natural Q→A for the entire document.

    Returns a list of items: {query, answer_text, key_values?}
    """
    sys_prompt = (
        "Jesteś asystentem, który tworzy naturalne pytania i krótkie odpowiedzi "
        "na podstawie jednego dokumentu uczelni (po polsku). Przeczytaj dokument i "
        "zaproponuj 1–3 rónorodnych pytań użytecznych dla użytkownika (np. limity, terminy, uprawnienia, obowiązki, procedury). "
        "Pytanie powinny być ogólnie sensowne, pamiętaj że widzisz jeden dokument ale przetwarzany będzie cały korpus prawny uczelni. "
        "Odpowiedzi mają być krótkie (1–2 zdania), konkretne i bez żargonu typu '§/ust.'. "
        "Staraj sie by pytanie i odpowiedź były jednoznaczne, nie odnosiły się do podobnych aktów w rónych latach. "
        "FORMAT: Zwróć WYŁĄCZNIE JSON {\"items\":[{\"query\":str, \"answer_text\":str, \"key_values\":[{\"type\":one_of[number,date,percent,ects,duration_days], \"value\":str}]?}, ...]} bez komentarzy."
    )
    user_payload = (
        "TYTUŁ:\n" + (doc_title or "") + "\n\nTEKST:\n" + doc_text[:120_000]
    )
    def _strip_fences(s: str) -> str:
        x = (s or "").strip()
        if x.startswith("```"):
            x = x.split("\n", 1)[1] if "\n" in x else x[3:]
        if x.endswith("```"):
            x = x.rsplit("```", 1)[0]
        return x.strip()

    def _items_from_parsed(obj: Any) -> List[Dict[str, Any]]:
        if isinstance(obj, dict):
            cand = obj.get("items")
        else:
            cand = None
        return [it for it in (cand or []) if isinstance(it, dict)]

    def _scan_items_array(s: str) -> Optional[str]:
        lower = (s or "").lower()
        key_pos = lower.find("items")
        if key_pos < 0:
            return None
        colon = s.find(":", key_pos)
        if colon < 0:
            return None
        start = s.find("[", colon)
        if start < 0:
            return None
        depth = 0
        in_str: Optional[str] = None
        escape = False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == in_str:
                    in_str = None
            else:
                if ch in ('"', "'"):
                    in_str = ch
                elif ch == '[':
                    depth += 1
                elif ch == ']':
                    depth -= 1
                    if depth == 0:
                        return s[start:i+1]
        return None

    raw = ""
    # Perform request — connection errors must propagate
    try:
        rsp = client.chat.completions.create(
            model=model,
            temperature=float(temperature),
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_payload},
            ],
            max_tokens=1200,
        )
        content = rsp.choices[0].message.content or "{}"
        raw = _strip_fences(content)
    except Exception as exc:
        raise RuntimeError(f"LLM request failed: {exc}")

    # Parse strictly, then fall back
    items: List[Dict[str, Any]] = []
    try:
        items = _items_from_parsed(json.loads(raw))
    except Exception:
        try:
            arr_text = _scan_items_array(raw)
            if arr_text:
                try:
                    arr = json.loads(arr_text)
                except Exception:
                    arr = ast.literal_eval(arr_text)
                if isinstance(arr, list):
                    items = [it for it in arr if isinstance(it, dict)]
        except Exception:
            items = []
    if not items:
        return []

    out: List[Dict[str, Any]] = []
    for it in items[: max_items]:
        if not isinstance(it, dict):
            continue
        q = _norm_spaces(str(it.get("query", "")))
        a = _norm_spaces(str(it.get("answer_text", "")))
        if not q or not a:
            continue
        kv = it.get("key_values")
        kv_list: List[Dict[str, Any]] = []
        if isinstance(kv, list):
            for kvp in kv:
                if not isinstance(kvp, dict):
                    continue
                t = str(kvp.get("type", "")).strip().lower()
                v = _norm_spaces(str(kvp.get("value", "")))
                if t and v:
                    kv_list.append({"type": t, "value": v})
        out.append({"query": q, "answer_text": a, "key_values": kv_list})
    return out

def _infer_as_of_year(text: str, title: str) -> Optional[str]:
    """Infer a 4-digit year from title or text (best-effort)."""
    for src in (title, text[:4000]):
        m = re.search(r"\b(19\d{2}|20\d{2})\b", src or "")
        if m:
            return m.group(1)
    return None


def build_golden(
    base_dir: Path,
    out_dir: Path,
    *,
    glob: str = "**/*",
    recursive: bool = True,
    limit_docs: Optional[int] = None,
    per_doc_qa: int = 2,
    target_qa: Optional[int] = None,
    seed: int = 123,
) -> Dict[str, Any]:
    rnd = random.Random(seed)
    files = _scan_files(base_dir, glob, recursive)
    files.sort()
    if limit_docs is not None:
        files = files[: max(0, int(limit_docs))]

    out_dir.mkdir(parents=True, exist_ok=True)
    qa_path = out_dir / "golden_qa.jsonl"
    docs_path = out_dir / "golden_documents.jsonl"
    summary_path = out_dir / "summary.json"

    # LLM (required)
    llm_client = _llm_client_from_env()
    llm_model = os.environ.get("GOLDEN_LLM_MODEL") if llm_client else None
    if not (llm_client and llm_model):
        raise RuntimeError("LLM is not configured. Set GOLDEN_LLM_BASE_URL, GOLDEN_LLM_API_KEY, GOLDEN_LLM_MODEL.")

    total_docs = 0
    total_qa = 0
    started = time.time()

    with qa_path.open("w", encoding="utf-8") as qa_fh, docs_path.open("w", encoding="utf-8") as docs_fh:
        for p in files:
            if target_qa is not None and total_qa >= target_qa:
                break
            total_docs += 1
            try:
                text = _extract_text(p)
            except Exception:
                continue
            if not text or not text.strip():
                continue
            # title: first non-empty line (for context in queries)
            title = ""
            for line in text.splitlines():
                if line.strip():
                    title = line.strip()[:200]
                    break
            doc_symbol = _extract_doc_symbol_from_path(p)
            # Infer year to help build natural, disambiguated questions
            as_of_year = _infer_as_of_year(text, title) or ""
            doc_rec = {"path": str(p.resolve()), "title": title, "doc_symbol": doc_symbol, "as_of": as_of_year}
            docs_fh.write(json.dumps(doc_rec, ensure_ascii=False) + "\n")
            # LLM generates natural QA directly from the whole document
            items_llm = _llm_generate_natural_qa_for_doc(
                llm_client, llm_model,  # type: ignore[arg-type]
                doc_text=text,
                doc_title=title,
                doc_symbol=doc_symbol,
                max_items=per_doc_qa,
            )
            for it in items_llm:
                if target_qa is not None and total_qa >= target_qa:
                    break
                q = it.get("query")
                a = it.get("answer_text")
                kv = it.get("key_values") or []
                if not q or not a:
                    continue
                # Bez post-procesingu – pytanie pochodzi wprost z LLM
                q = str(q)
                record = {
                    "id": _mk_id(str(p), q, a),
                    "query": q,
                    "expected_answer": a,
                    "answer_type": "text",
                    "score_rule": "f1",
                    "unanswerable": False,
                    "difficulty": "medium",
                    "key_values": kv,
                    "meta": {
                        "doc_title": title,
                        "doc_path": str(p.resolve()),
                        "doc_symbol": doc_symbol,
                        "as_of": as_of_year if as_of_year else None,
                    },
                }
                qa_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_qa += 1

    took_ms = int((time.time() - started) * 1000)
    summary = {
        "documents": total_docs,
        "qa_items": total_qa,
        "took_ms": took_ms,
        "seed": seed,
        "use_llm": True,
        "llm_model": llm_model,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Golden QA generator (standalone)")
    p.add_argument("--base-dir", required=True, help="Base directory with documents")
    p.add_argument("--glob", default="**/*", help="Glob pattern (default: **/*)")
    p.add_argument("--recursive", action="store_true", help="Use recursive scan (default: false)")
    p.add_argument("--out-dir", default="reports/golden", help="Output directory (default: reports/golden)")
    p.add_argument("--limit-docs", type=int, default=None, help="Limit number of documents")
    p.add_argument("--per-doc-qa", type=int, default=2, help="Max QA per document (default: 2)")
    p.add_argument("--target-qa", type=int, default=None, help="Stop after generating approximately this many QA items")
    p.add_argument("--seed", type=int, default=123, help="Deterministic seed")

    args = p.parse_args(argv)
    base = Path(args.base_dir)
    if not base.exists():
        print(f"[error] base-dir not found: {base}")
        return 2
    out_dir = Path(args.out_dir)
    try:
        summary = build_golden(
            base, out_dir,
            glob=args.glob,
            recursive=bool(args.recursive),
            limit_docs=args.limit_docs,
            per_doc_qa=max(1, int(args.per_doc_qa)),
            target_qa=(int(args.target_qa) if args.target_qa is not None else None),
            seed=int(args.seed),
        )
    except Exception as exc:
        print(f"[error] Golden generation failed: {exc}")
        return 3
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
