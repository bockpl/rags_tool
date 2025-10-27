"""Sidecar cache for document summaries and vectors.

Stores gzipped JSON files next to source documents under a hidden
`.summary/` subdirectory. Cache contains only what is necessary to
skip expensive LLM summarization and dense embedding of the summary.

Schema (versioned by SCHEMA_VERSION):

{
  "schema_version": "1.0.0",
  "document": {
    "content_sha256": "..."
  },
    "summary": {
    "title": "...",
    "summary": "...",
    "signature": ["..."],
    "entities": ["..."],
    "replacement": "brak | ...",
    "doc_date": "YYYY-MM-DD | YYYY | brak"
  },
  "vectors": {
    "summary_dense": [float, ...]
  }
}
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


SCHEMA_VERSION = "1.0.0"


# Compute SHA256 digest over raw file bytes (stable w.r.t extractor changes).
def compute_file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA256 over raw file bytes.

    Using file bytes (not extracted text) ensures stability across
    extractor changes; this is used only to validate cache freshness.
    """
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# Compute path for sidecar cache: <dir>/.summary/<stem>_summary.json.gz
def sidecar_path_for(source: Path) -> Path:
    """Return sidecar cache path: `<dir>/.summary/<basename>_summary.json.gz`."""
    base = source.parent / ".summary"
    filename = f"{source.stem}_summary.json.gz"
    return base / filename


# Load sidecar if present and valid for the given file hash; else return None.
def load_sidecar(source: Path, expected_sha256: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Load sidecar cache if present and valid.

    Returns a dict with keys: summary (title, summary, signature, replacement)
    and vectors (summary_dense) when cache is valid, otherwise None.
    """
    sc_path = sidecar_path_for(source)
    if not sc_path.exists():
        return None
    try:
        with gzip.open(sc_path, "rt", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return None

    if not isinstance(data, dict):
        return None
    if str(data.get("schema_version")) != SCHEMA_VERSION:
        return None
    doc = data.get("document") or {}
    if expected_sha256 and doc.get("content_sha256") != expected_sha256:
        return None
    # Basic shape validation
    summ = data.get("summary") or {}
    vecs = data.get("vectors") or {}
    if not (isinstance(summ.get("summary", ""), str) and isinstance(vecs.get("summary_dense", []), list)):
        return None
    return data


# Atomically write sidecar cache for a document's summary and vectors.
def save_sidecar(
    source: Path,
    *,
    content_sha256: str,
    title: str,
    summary: str,
    signature: list[str],
    entities: list[str],
    replacement: str,
    summary_dense: list[float],
    doc_date: str = "brak",
    is_active: bool = True,
) -> Path:
    """Write sidecar cache atomically. Returns final path."""
    sc_path = sidecar_path_for(source)
    sc_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = sc_path.with_suffix(sc_path.suffix + ".tmp")

    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "document": {"content_sha256": content_sha256},
        "summary": {
            "title": title,
            "summary": summary,
            "signature": signature,
            "entities": entities,
            "replacement": replacement,
            "doc_date": doc_date or "brak",
            "is_active": bool(is_active),
        },
        "vectors": {"summary_dense": summary_dense},
    }

    # Atomic write: write to tmp then replace
    with gzip.open(tmp_path, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    os.replace(tmp_path, sc_path)
    return sc_path
