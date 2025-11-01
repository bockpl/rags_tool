"""Lightweight SQLite FTS5 index for fast full-corpus browse by chunk text.

Provides functions to build and query a local FTS index over chunk texts
stored in Qdrant. Intended for /browse/doc-ids to operate over the whole
corpus efficiently without large vector searches.
"""

from __future__ import annotations

import os
import time
import logging
import sqlite3
from typing import Iterable, List, Optional, Tuple, Dict

from qdrant_client.http import models as qm

from app.qdrant_utils import qdrant
from app.settings import get_settings

settings = get_settings()
logger = logging.getLogger("rags_tool.fts")


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
    """Ensure FTS5 table exists with the current schema.

    Note: service rebuilds the index at startup, so no migration logic here.
    """
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts "
        "USING fts5("
        "text, ents, doc_id UNINDEXED, is_active UNINDEXED, doc_date UNINDEXED, doc_date_ord UNINDEXED, "
        "tokenize='unicode61'"
        ");"
    )


def _doc_date_ord(value: Optional[str]) -> int:
    """Convert date string to ordinal YYYYMMDD (missing parts -> 00)."""
    if not isinstance(value, str):
        return 0
    s = value.strip()
    if not s or s.lower() == "brak":
        return 0
    try:
        parts = s.split("-")
        y = int(parts[0]) if len(parts) >= 1 and parts[0].isdigit() else 0
        m = int(parts[1]) if len(parts) >= 2 and parts[1].isdigit() else 0
        d = int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else 0
        if y <= 0:
            return 0
        m = max(0, min(12, m))
        d = max(0, min(31, d))
        return int(y) * 10000 + int(m) * 100 + int(d)
    except Exception:
        return 0


def _fetch_doc_dates_map() -> Dict[str, str]:
    """Fetch a map doc_id -> doc_date from summaries collection."""
    out: Dict[str, str] = {}
    try:
        flt = qm.Filter(
            must=[qm.FieldCondition(key="point_type", match=qm.MatchValue(value="summary"))]
        )
        offset = None
        while True:
            res = qdrant.scroll(
                collection_name=settings.qdrant_summary_collection,
                scroll_filter=flt,
                limit=256,
                offset=offset,
                with_payload=["doc_id", "doc_date"],
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
            for rec in records:
                p = rec.payload or {}
                did = p.get("doc_id")
                if not did:
                    continue
                dd = str(p.get("doc_date") or "").strip()
                out[str(did)] = dd
            if offset is None:
                break
    except Exception:
        return out
    return out


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
    """Rebuild FTS5 index from Qdrant content collection. Returns inserted rows.

    Always drops and recreates the FTS table to ensure fresh schema.
    """
    logger.info("FTS rebuild: starting (drop+create table)")
    start_ts = time.time()
    conn = _connect()
    with conn:
        conn.execute("DROP TABLE IF EXISTS chunks_fts;")
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts "
            "USING fts5("
            "text, ents, doc_id UNINDEXED, is_active UNINDEXED, doc_date UNINDEXED, doc_date_ord UNINDEXED, "
            "tokenize='unicode61'"
            ");"
        )
    inserted = 0
    pages = 0
    # Preload document dates (per doc_id) from summaries
    doc_dates = _fetch_doc_dates_map()
    logger.info("FTS rebuild: loaded doc_dates | count=%d", len(doc_dates))
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
        pages += 1
        rows: List[Tuple[str, str, str, int, str, int]] = []
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
            dd = doc_dates.get(str(did)) or ""
            dd_ord = _doc_date_ord(dd)
            rows.append((str(txt), str(ents_text or ""), str(did), is_act, dd, dd_ord))
        if rows:
            with conn:
                conn.executemany(
                    "INSERT INTO chunks_fts(text, ents, doc_id, is_active, doc_date, doc_date_ord) VALUES (?, ?, ?, ?, ?, ?);",
                    rows,
                )
            inserted += len(rows)
            # Periodic progress log
            if inserted % 25000 < len(rows):
                elapsed = time.time() - start_ts
                logger.info("FTS rebuild: progress | rows=%d pages=%d elapsed=%.1fs", inserted, pages, elapsed)
        if offset is None:
            break
    conn.close()
    elapsed = time.time() - start_ts
    logger.info("FTS rebuild: completed | rows=%d pages=%d elapsed=%.1fs", inserted, pages, elapsed)
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
    order: str = "none",  # none|date_desc
) -> List[str]:
    """Return distinct doc_ids matching queries according to FTS 'match' and status.

    When order == 'date_desc', returns doc_ids ordered by MAX(doc_date_ord) DESC.
    """
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
        if str(order or "none").lower() == "date_desc":
            sql = f"SELECT doc_id, MAX(doc_date_ord) as dd FROM chunks_fts WHERE {where} GROUP BY doc_id ORDER BY dd DESC LIMIT ?;"
            params.append(int(max(1, limit)))
            cur = conn.execute(sql, params)
            rows = [r[0] for r in cur.fetchall()]
            return [str(d) for d in rows]
        else:
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


def fts_search_doc_ids_all(
    queries: List[str], *, match: str = "phrase", status: str = "active", order: str = "none"
) -> List[str]:
    """Return all distinct doc_ids for query/status (no limit).

    When order == 'date_desc', returns doc_ids ordered by MAX(doc_date_ord) DESC.
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
        if str(order or "none").lower() == "date_desc":
            sql = f"SELECT doc_id, MAX(doc_date_ord) as dd FROM chunks_fts WHERE {where} GROUP BY doc_id ORDER BY dd DESC;"
            cur = conn.execute(sql, params)
            return [str(r[0]) for r in cur.fetchall()]
        else:
            sql = f"SELECT DISTINCT doc_id FROM chunks_fts WHERE {where};"
            cur = conn.execute(sql, params)
            rows = [str(r[0]) for r in cur.fetchall()]
            return rows
    finally:
        try:
            conn.close()
        except Exception:
            pass
