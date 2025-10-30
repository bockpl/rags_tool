"""Lightweight SQLite FTS5 index for fast full-corpus browse by chunk text.

Provides functions to build and query a local FTS index over chunk texts
stored in Qdrant. Intended for /browse/doc-ids to operate over the whole
corpus efficiently without large vector searches.
"""

from __future__ import annotations

import os
import sqlite3
from typing import Iterable, List, Optional, Tuple

from qdrant_client.http import models as qm

from app.qdrant_utils import qdrant
from app.settings import get_settings

settings = get_settings()


def _db_path() -> str:
    base = settings.vector_store_dir
    base.mkdir(parents=True, exist_ok=True)
    return str(base / "chunks_fts.sqlite")


def _connect() -> sqlite3.Connection:
    path = _db_path()
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=OFF;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Ensure FTS5 table exists with both text and ents columns.

    If an older schema without 'ents' exists, drop and recreate.
    """
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts "
        "USING fts5(text, ents, doc_id UNINDEXED, is_active UNINDEXED, tokenize='unicode61');"
    )
    # Verify presence of 'ents' column; if missing (old schema), recreate
    try:
        conn.execute("SELECT ents FROM chunks_fts LIMIT 1;")
    except sqlite3.OperationalError:
        with conn:
            conn.execute("DROP TABLE IF EXISTS chunks_fts;")
            conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts "
                "USING fts5(text, ents, doc_id UNINDEXED, is_active UNINDEXED, tokenize='unicode61');"
            )


def fts_count(conn: Optional[sqlite3.Connection] = None) -> int:
    own = False
    if conn is None:
        own = True
        conn = _connect()
    try:
        _ensure_schema(conn)
        cur = conn.execute("SELECT count(*) FROM chunks_fts;")
        row = cur.fetchone()
        return int(row[0] if row and row[0] is not None else 0)
    except sqlite3.OperationalError:
        return 0
    finally:
        if own:
            conn.close()


def rebuild_fts_from_qdrant() -> int:
    """Rebuild FTS5 index from Qdrant content collection. Returns inserted rows."""
    conn = _connect()
    _ensure_schema(conn)
    with conn:
        conn.execute("DELETE FROM chunks_fts;")
    inserted = 0
    # Scroll all chunks
    base_must = [
        qm.FieldCondition(key="point_type", match=qm.MatchValue(value="chunk")),
    ]
    flt = qm.Filter(must=base_must)
    offset = None
    while True:
        res = qdrant.scroll(
            collection_name=settings.qdrant_content_collection,
            scroll_filter=flt,
            limit=256,
            offset=offset,
            with_payload=["doc_id", "is_active", "text", "entities"],
            with_vectors=False,
        )
        if isinstance(res, tuple):
            records, offset = res
        else:
            records = getattr(res, "points", None)
            offset = getattr(res, "next_page_offset", None)
            if records is None:
                records = []
        if not records:
            break
        rows: List[Tuple[str, str, str, int]] = []
        for rec in records:
            p = rec.payload or {}
            did = p.get("doc_id")
            txt = p.get("text")
            if not did or not txt:
                continue
            is_act = 1 if bool(p.get("is_active")) else 0
            ents_val = p.get("entities") or []
            if isinstance(ents_val, list):
                ents_text = " ".join(str(x) for x in ents_val if str(x).strip())
            elif isinstance(ents_val, str):
                ents_text = ents_val
            else:
                ents_text = ""
            rows.append((str(txt), str(ents_text or ""), str(did), is_act))
        if rows:
            with conn:
                conn.executemany(
                    "INSERT INTO chunks_fts(text, ents, doc_id, is_active) VALUES (?, ?, ?, ?);",
                    rows,
                )
            inserted += len(rows)
        if offset is None:
            break
    conn.close()
    return inserted


def ensure_fts_ready(min_rows: int = 1) -> None:
    try:
        conn = _connect()
        cnt = fts_count(conn)
        if cnt >= int(min_rows):
            conn.close()
            return
    except Exception:
        # Attempt rebuild on any error
        pass
    finally:
        try:
            conn.close()
        except Exception:
            pass
    rebuild_fts_from_qdrant()


def _build_match_query(queries: List[str], mode: str) -> str:
    """Build an FTS5 MATCH query over text and ents columns.

    phrase: OR of (text:"..." OR ents:"...") for each query string
    any: OR of all token/column pairs (text:tok OR ents:tok ...)
    all: AND of groups (text:tok OR ents:tok) for each token
    """
    mode_l = (mode or "").strip().lower()
    parts: List[str] = []
    if mode_l == "phrase":
        for q in queries:
            s = (q or "").strip().replace('"', '""')
            if not s:
                continue
            parts.append(f'text:"{s}" OR ents:"{s}"')
        return " OR ".join(parts) if parts else ""
    # Token modes
    import re
    tokens: List[str] = []
    for q in queries:
        tokens.extend(re.findall(r"\w+", q or ""))
    tokens = [t for t in tokens if t]
    if not tokens:
        return ""
    if mode_l == "any":
        pairs = [f"text:{t}" for t in tokens] + [f"ents:{t}" for t in tokens]
        return " OR ".join(pairs)
    # all (default)
    groups = [f"(text:{t} OR ents:{t})" for t in tokens]
    return " AND ".join(groups)


def fts_search_doc_ids(
    queries: List[str],
    *,
    match: str = "phrase",
    status: str = "active",
    limit: int = 1000,
) -> List[str]:
    """Return distinct doc_ids matching queries according to FTS 'match' and status."""
    ensure_fts_ready(min_rows=1)
    conn = _connect()
    _ensure_schema(conn)
    try:
        q = _build_match_query(queries, match)
        params: List[object] = []
        # If no query tokens/phrase provided, return all (distinct) doc_ids under status filter
        if not q:
            where = "1=1"
        else:
            where = "chunks_fts MATCH ?"
            params.append(q)
        if status == "active":
            where += " AND is_active = 1"
        elif status == "inactive":
            where += " AND is_active = 0"
        sql = f"SELECT doc_id FROM chunks_fts WHERE {where} LIMIT ?;"
        params.append(int(max(1, limit)))
        cur = conn.execute(sql, params)
        rows = [r[0] for r in cur.fetchall()]
        # Distinct + preserve order
        seen = set()
        out: List[str] = []
        for did in rows:
            if did not in seen:
                seen.add(did)
                out.append(str(did))
        return out
    finally:
        conn.close()


def fts_doc_counts() -> Tuple[int, int, int]:
    """Return (total_docs, active_docs, inactive_docs) based on distinct doc_id in FTS."""
    ensure_fts_ready(min_rows=1)
    conn = _connect()
    _ensure_schema(conn)
    try:
        # DISTINCT doc_id counts; use subqueries to avoid scanning twice too much
        cur = conn.execute("SELECT COUNT(DISTINCT doc_id) FROM chunks_fts;")
        row = cur.fetchone()
        total = int(row[0]) if row and row[0] is not None else 0
        cur2 = conn.execute("SELECT COUNT(DISTINCT doc_id) FROM chunks_fts WHERE is_active = 1;")
        row2 = cur2.fetchone()
        active = int(row2[0]) if row2 and row2[0] is not None else 0
        cur3 = conn.execute("SELECT COUNT(DISTINCT doc_id) FROM chunks_fts WHERE is_active = 0;")
        row3 = cur3.fetchone()
        inactive = int(row3[0]) if row3 and row3[0] is not None else 0
        return total, active, inactive
    except Exception:
        return 0, 0, 0
    finally:
        try:
            conn.close()
        except Exception:
            pass


def fts_search_doc_count(queries: List[str], *, match: str = "phrase", status: str = "active") -> int:
    """Return COUNT(DISTINCT doc_id) for a query/status using FTS.

    When queries are empty, counts all documents under status filter.
    """
    ensure_fts_ready(min_rows=1)
    conn = _connect()
    _ensure_schema(conn)
    try:
        q = _build_match_query(queries, match)
        params: List[object] = []
        if not q:
            where = "1=1"
        else:
            where = "chunks_fts MATCH ?"
            params.append(q)
        if status == "active":
            where += " AND is_active = 1"
        elif status == "inactive":
            where += " AND is_active = 0"
        sql = f"SELECT COUNT(DISTINCT doc_id) FROM chunks_fts WHERE {where};"
        cur = conn.execute(sql, params)
        row = cur.fetchone()
        return int(row[0]) if row and row[0] is not None else 0
    finally:
        try:
            conn.close()
        except Exception:
            pass


def fts_search_doc_ids_all(queries: List[str], *, match: str = "phrase", status: str = "active") -> List[str]:
    """Return all distinct doc_ids for query/status (no limit)."""
    ensure_fts_ready(min_rows=1)
    conn = _connect()
    _ensure_schema(conn)
    try:
        q = _build_match_query(queries, match)
        params: List[object] = []
        if not q:
            where = "1=1"
        else:
            where = "chunks_fts MATCH ?"
            params.append(q)
        if status == "active":
            where += " AND is_active = 1"
        elif status == "inactive":
            where += " AND is_active = 0"
        sql = f"SELECT DISTINCT doc_id FROM chunks_fts WHERE {where};"
        cur = conn.execute(sql, params)
        rows = [str(r[0]) for r in cur.fetchall()]
        return rows
    finally:
        try:
            conn.close()
        except Exception:
            pass
