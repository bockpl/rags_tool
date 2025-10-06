"""Hybrid search logic: stage-1 summaries, stage-2 chunks, MMR, shaping."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from qdrant_client.http import models as qm

from app.core.embedding import (
    SUMMARY_VECTORIZER_PATH,
    load_vectorizer,
    tfidf_vector,
)
from app.core.constants import (
    CONTENT_SPARSE_NAME,
    CONTENT_VECTOR_NAME,
    SPARSE_ENABLED,
    SUMMARY_SPARSE_NAME,
    SUMMARY_VECTOR_NAME,
)
from app.qdrant_utils import qdrant
from app.settings import get_settings
from app.core.ranker_client import OpenAIReranker
from app.core.constants import RANKER_USE_STAGE1, RANKER_USE_STAGE2

settings = get_settings()

# Vector names and defaults
 # names imported from constants

DEFAULT_MMR_LAMBDA = 0.3
DEFAULT_PER_DOC_LIMIT = 2
DEFAULT_SCORE_NORM = "minmax"  # minmax|zscore|none


def _ranker_enabled() -> bool:
    """Czy ranker jest skonfigurowany (BASE_URL + MODEL)."""
    return bool(settings.ranker_base_url and settings.ranker_model)


def _truncate_head_tail(text: str, limit: int) -> str:
    """Przytnij tekst do limitu znaków (70% head, 30% tail)."""
    t = (text or "").strip()
    if len(t) <= max(1, int(limit)):
        return t
    head = int(limit * 0.7)
    tail = max(0, int(limit) - head)
    return (t[:head] + "\n...\n" + t[-tail:]).strip()


def _rerank_indices(query: str, passages: List[str], top_n: int) -> Dict[int, float]:
    """Wywołaj ranker i zwróć mapę index->score dla ocenionych elementów.

    Uwaga: 'top_n' ogranicza liczbę wyników zwracanych przez endpoint (nie liczbę
    dokumentów w wejściu). Niezwrócone elementy pozostają bez oceny.
    """
    client = OpenAIReranker(settings.ranker_base_url or "", settings.ranker_api_key, settings.ranker_model or "")
    rr = client.rerank(query=query, documents=passages, top_n=min(max(1, top_n), len(passages)))
    return {int(it.get("index")): float(it.get("relevance_score")) for it in rr}


def _normalize(values: List[float], method: str = DEFAULT_SCORE_NORM) -> List[float]:
    if not values:
        return []
    if method == "none":
        return values
    arr = np.array(values, dtype=float)
    if method == "zscore":
        mean = float(arr.mean())
        std = float(arr.std())
        if std < 1e-12:
            return [0.0 for _ in values]
        return [float((v - mean) / std) for v in arr]
    vmin = float(arr.min())
    vmax = float(arr.max())
    if vmax - vmin < 1e-12:
        return [0.0 for _ in values]
    return [float((v - vmin) / (vmax - vmin)) for v in arr]


def _sparse_dot(query_lookup: Dict[int, float], indices: List[int], values: List[float]) -> float:
    if not query_lookup or not indices or not values:
        return 0.0
    acc = 0.0
    for i, v in zip(indices, values):
        q = query_lookup.get(int(i))
        if q is not None:
            acc += q * float(v)
    return float(acc)


def _sparse_pair_cos(
    a_idx: List[int], a_val: List[float], b_idx: List[int], b_val: List[float]
) -> float:
    if not a_idx or not b_idx:
        return 0.0
    i = j = 0
    sim = 0.0
    while i < len(a_idx) and j < len(b_idx):
        ai = int(a_idx[i])
        bj = int(b_idx[j])
        if ai == bj:
            sim += float(a_val[i]) * float(b_val[j])
            i += 1
            j += 1
        elif ai < bj:
            i += 1
        else:
            j += 1
    return float(sim)


def _cosine_dense(a: List[float], b: List[float]) -> float:
    va = np.array(a, dtype=float)
    vb = np.array(b, dtype=float)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(va, vb) / denom)


def _with_must_condition(
    flt: Optional[qm.Filter], condition: qm.FieldCondition
) -> qm.Filter:
    if flt is None:
        return qm.Filter(must=[condition])
    must_items = list(flt.must or []) + [condition]
    should_items = list(flt.should or []) if flt.should else None
    must_not_items = list(flt.must_not or []) if flt.must_not else None
    return qm.Filter(must=must_items, should=should_items, must_not=must_not_items)


def _fetch_doc_summaries(doc_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Fetch summaries for given doc_ids from the summaries collection.

    Returns a map: doc_id -> { doc_id, doc_summary, doc_signature }
    """
    out: Dict[str, Dict[str, Any]] = {}
    if not doc_ids:
        return out
    try:
        flt = qm.Filter(
            must=[
                qm.FieldCondition(key="doc_id", match=qm.MatchAny(any=doc_ids)),
                qm.FieldCondition(key="point_type", match=qm.MatchValue(value="summary")),
            ]
        )
        offset = None
        while True:
            res = qdrant.scroll(
                collection_name=settings.qdrant_summary_collection,
                scroll_filter=flt,
                limit=256,
                offset=offset,
                with_payload=True,
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
                payload = rec.payload or {}
                did = payload.get("doc_id")
                if not did:
                    continue
                out[str(did)] = {
                    "doc_id": str(did),
                    "doc_summary": payload.get("summary"),
                    "doc_signature": payload.get("signature"),
                }
            if offset is None:
                break
    except Exception:
        # Best-effort; if scroll fails, return what we have (possibly empty)
        return out
    return out


def _build_sparse_queries_for_query(query: str, use_hybrid: bool) -> Tuple[
    Optional[Tuple[List[int], List[float]]], Optional[Tuple[List[int], List[float]]]
]:
    if not (use_hybrid and SPARSE_ENABLED):
        return None, None
    content_sparse_query: Optional[Tuple[List[int], List[float]]] = None
    summary_sparse_query: Optional[Tuple[List[int], List[float]]] = None
    content_vec = load_vectorizer()
    if content_vec is not None:
        idx, val = tfidf_vector([query], content_vec)[0]
        if idx:
            content_sparse_query = (idx, val)
    summary_vec_model = load_vectorizer(path=SUMMARY_VECTORIZER_PATH)
    if summary_vec_model is not None:
        s_idx, s_val = tfidf_vector([query], summary_vec_model, path=SUMMARY_VECTORIZER_PATH)[0]
        if s_idx:
            summary_sparse_query = (s_idx, s_val)
    return content_sparse_query, summary_sparse_query


def _stage1_select_documents(
    q_text: str,
    q_vec: List[float],
    flt: Optional[qm.Filter],
    summary_sparse_query: Optional[Tuple[List[int], List[float]]],
    req: Any,
) -> Tuple[List[str], Dict[str, Any]]:
    point_type_filter = _with_must_condition(
        flt, qm.FieldCondition(key="point_type", match=qm.MatchValue(value="summary"))
    )
    sum_search = qdrant.search(
        collection_name=settings.qdrant_summary_collection,
        query_vector=(SUMMARY_VECTOR_NAME, q_vec),
        query_filter=point_type_filter,
        limit=max(50, req.top_m),
        with_payload=True,
        with_vectors=[SUMMARY_VECTOR_NAME] if req.mmr_stage1 else False,
        score_threshold=None,
        search_params=qm.SearchParams(exact=False, hnsw_ef=128),
    )
    summary_sparse_lookup: Dict[int, float] = {}
    if summary_sparse_query is not None:
        summary_sparse_lookup = dict(zip(summary_sparse_query[0], summary_sparse_query[1]))

    doc_map: Dict[str, Dict[str, Any]] = {}
    for r in sum_search:
        payload = r.payload or {}
        if payload.get("point_type") and payload.get("point_type") != "summary":
            continue
        did = payload.get("doc_id")
        if not did or did in doc_map:
            continue
        dense_score = float(r.score or 0.0)
        sparse_dot = 0.0
        if summary_sparse_lookup and payload.get("summary_sparse_indices") and payload.get("summary_sparse_values"):
            sparse_dot = _sparse_dot(summary_sparse_lookup, payload.get("summary_sparse_indices", []), payload.get("summary_sparse_values", []))
        vec_map = r.vector or {}
        dense_vec = vec_map.get(SUMMARY_VECTOR_NAME) or []
        doc_map[did] = {
            "doc_id": did,
            "dense_vec": dense_vec,
            "sparse_idx": payload.get("summary_sparse_indices", []) or [],
            "sparse_val": payload.get("summary_sparse_values", []) or [],
            "dense_score": dense_score,
            "sparse_score": sparse_dot,
            "path": payload.get("path"),
            "doc_summary": payload.get("summary"),
            "doc_signature": payload.get("signature"),
        }

    if not doc_map:
        return [], {}

    doc_items = list(doc_map.values())
    dense_scores = [float(x["dense_score"]) for x in doc_items]
    sparse_scores = [float(x["sparse_score"]) for x in doc_items]
    dense_norm = _normalize(dense_scores, req.score_norm)
    sparse_norm = _normalize(sparse_scores, req.score_norm)
    hybrid_rel = [req.dense_weight * d + req.sparse_weight * s for d, s in zip(dense_norm, sparse_norm)]

    if req.mmr_stage1 and len(doc_items) > 1:
        rep_alpha = req.rep_alpha if req.rep_alpha is not None else req.dense_weight
        dense_vecs = [x["dense_vec"] for x in doc_items]
        sparse_vecs = [(x["sparse_idx"], x["sparse_val"]) for x in doc_items]
        mmr_idx = mmr_diversify_hybrid(dense_vecs, sparse_vecs, hybrid_rel, min(req.top_m, len(doc_items)), req.mmr_lambda, rep_alpha)
        order_idx = mmr_idx
    else:
        order_idx = sorted(range(len(doc_items)), key=lambda i: hybrid_rel[i], reverse=True)

    # Etap 1: opcjonalny rerank streszczeń (tylko jeśli ranker skonfigurowany i flaga włączona)
    if _ranker_enabled() and RANKER_USE_STAGE1 and order_idx:
        try:
            # Zbuduj krótkie passage na podstawie streszczenia (ew. podpisu)
            passages = []
            for i in order_idx:
                s = str(doc_items[i].get("doc_summary") or "")
                sig = doc_items[i].get("doc_signature") or []
                if isinstance(sig, list):
                    s = (s + "\n\n" + ", ".join(map(str, sig))) if s else ", ".join(map(str, sig))
                passages.append(_truncate_head_tail(s, settings.ranker_max_length))
            score_map = _rerank_indices(q_text, passages, settings.rerank_top_n)
            # Posortuj ocenione na początku; nieocenione zachowaj w oryginalnej kolejności za nimi
            scored = [(i, score_map[i2]) for i, i2 in enumerate(range(len(order_idx))) if i2 in score_map]
            # scored to (local_pos, score); przemapuj na globalne indeksy
            scored_global = [(order_idx[pos], sc) for (pos, sc) in scored]
            scored_global.sort(key=lambda x: float(x[1]), reverse=True)
            scored_ids = [doc_items[i]["doc_id"] for (i, _) in scored_global]
            not_scored_ids = [doc_items[i]["doc_id"] for i in order_idx if i not in {idx for (idx, _) in scored_global}]
            cand_doc_ids = scored_ids + not_scored_ids
            cand_doc_ids = cand_doc_ids[: min(req.top_m, len(cand_doc_ids))]
        except Exception:
            # W przypadku błędu rankera, fallback do bazowego porządku
            cand_doc_ids = [doc_items[i]["doc_id"] for i in order_idx[: min(req.top_m, len(order_idx))]]
    else:
        cand_doc_ids = [doc_items[i]["doc_id"] for i in order_idx[: min(req.top_m, len(order_idx))]]

    return cand_doc_ids, doc_map


def _stage2_select_chunks(
    cand_doc_ids: Optional[List[str]],
    q_text: str,
    q_vec: List[float],
    content_sparse_query: Optional[Tuple[List[int], List[float]]],
    doc_map: Dict[str, Any],
    req: Any,
    flt: Optional[qm.Filter] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[float]]:
    # Build filter for Stage-2 search
    must = [
        qm.FieldCondition(key="point_type", match=qm.MatchValue(value="chunk")),
    ]
    if cand_doc_ids:
        must.insert(0, qm.FieldCondition(key="doc_id", match=qm.MatchAny(any=cand_doc_ids)))
    if flt and getattr(flt, "must", None):
        must = list(flt.must) + must
    flt2 = qm.Filter(
        must=must,
        should=(flt.should if flt and getattr(flt, "should", None) else None),
        must_not=(flt.must_not if flt and getattr(flt, "must_not", None) else None),
    )
    cont_search = qdrant.search(
        collection_name=settings.qdrant_content_collection,
        query_vector=(CONTENT_VECTOR_NAME, q_vec),
        query_filter=flt2,
        limit=req.top_m,
        with_payload=True,
        with_vectors=[CONTENT_VECTOR_NAME],
        search_params=qm.SearchParams(exact=False, hnsw_ef=128),
    )
    mmr_pool: List[Dict[str, Any]] = []
    for hit in cont_search:
        payload = hit.payload or {}
        if payload.get("point_type") and payload.get("point_type") != "chunk":
            continue
        dense_score = float(hit.score or 0.0)
        sparse_dot = 0.0
        if content_sparse_query is not None and payload.get("content_sparse_indices") and payload.get("content_sparse_values"):
            q_lookup = dict(zip(content_sparse_query[0], content_sparse_query[1]))
            sparse_dot = _sparse_dot(q_lookup, payload.get("content_sparse_indices", []), payload.get("content_sparse_values", []))
        doc_info = doc_map.get(payload.get("doc_id", ""), {})
        if doc_info:
            payload.setdefault("summary", doc_info.get("doc_summary"))
            if doc_info.get("doc_signature") is not None:
                payload.setdefault("signature", doc_info.get("doc_signature"))
        mmr_pool.append({
            "hit": hit,
            "doc_id": payload.get("doc_id", ""),
            "dense_vec": (hit.vector or {}).get(CONTENT_VECTOR_NAME) or [],
            "sparse_idx": payload.get("content_sparse_indices", []) or [],
            "sparse_val": payload.get("content_sparse_values", []) or [],
            "dense_score": dense_score,
            "sparse_score": sparse_dot,
        })

    if not mmr_pool:
        return [], [], []

    dense_scores2 = [x["dense_score"] for x in mmr_pool]
    sparse_scores2 = [x["sparse_score"] for x in mmr_pool]
    dense_norm2 = _normalize(dense_scores2, req.score_norm)
    sparse_norm2 = _normalize(sparse_scores2, req.score_norm)
    rel2 = [req.dense_weight * d + req.sparse_weight * s for d, s in zip(dense_norm2, sparse_norm2)]

    # Etap 2: opcjonalny rerank chunków przed selekcją MMR (jeśli ranker skonfigurowany i flaga włączona)
    if _ranker_enabled() and RANKER_USE_STAGE2 and mmr_pool:
        try:
            passages2 = []
            for item in mmr_pool:
                payload = (item.get("hit") or {}).payload or {}
                txt = str(payload.get("text") or "")
                passages2.append(_truncate_head_tail(txt, settings.ranker_max_length))
            score_map2 = _rerank_indices(q_text, passages2, settings.rerank_top_n)
            # Nadpisz rel2 dla ocenionych elementów skalą [0..1] z rankera
            for idx, sc in score_map2.items():
                if 0 <= int(idx) < len(rel2):
                    rel2[int(idx)] = float(sc)
        except Exception:
            pass

    rep_alpha = req.rep_alpha if req.rep_alpha is not None else req.dense_weight
    dense_vecs2 = [x["dense_vec"] for x in mmr_pool]
    sparse_vecs2 = [(x["sparse_idx"], x["sparse_val"]) for x in mmr_pool]
    doc_ids2 = [x["doc_id"] for x in mmr_pool]

    sel_idx = mmr_diversify_hybrid(
        dense_vecs2,
        sparse_vecs2,
        rel2,
        min(req.top_k, len(mmr_pool)),
        req.mmr_lambda,
        rep_alpha,
        per_doc_ids=doc_ids2,
        per_doc_limit=max(1, int(req.per_doc_limit)) if req.per_doc_limit and req.per_doc_limit > 0 else None,
    )

    selected = [mmr_pool[i] for i in sel_idx]
    final_hits: List[Dict[str, Any]] = []
    # If Stage-1 was skipped, enrich with summaries so that summary_mode and shaping behave as before
    if not doc_map:
        try:
            missing_ids = sorted({x.get("doc_id", "") for x in selected if x.get("doc_id")})
            if missing_ids:
                doc_map.update(_fetch_doc_summaries(missing_ids))
        except Exception:
            pass
    for idx, item in zip(sel_idx, selected):
        hit = item["hit"]
        payload = hit.payload or {}
        doc_info = doc_map.get(payload.get("doc_id", ""), {})
        if doc_info:
            payload.setdefault("summary", doc_info.get("doc_summary"))
            if doc_info.get("doc_signature") is not None:
                payload.setdefault("signature", doc_info.get("doc_signature"))
        final_score = float(rel2[idx])
        final_hits.append({"hit": hit, "score": float(final_score), "payload": payload})
    final_hits.sort(key=lambda x: x["score"], reverse=True)
    return final_hits, mmr_pool, rel2


def _classify_mode(query: str, mode: str) -> str:
    if mode != "auto":
        return mode
    q = query.lower()
    if re.search(r"archiw|stara|z \d{4}|wersja\s+z", q):
        return "archival"
    if re.search(r"obowiązując|aktualn|teraz|bieżąc", q):
        return "current"
    return "all"


def mmr_diversify(vectors: np.ndarray, scores: np.ndarray, k: int, lam: float = DEFAULT_MMR_LAMBDA) -> List[int]:
    selected: List[int] = []
    candidates = list(range(len(scores)))
    if len(candidates) <= k:
        return candidates
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)

    def sims(a, b) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / (denom + 1e-9))

    while len(selected) < k and candidates:
        best_i = None
        best_score = -1e9
        for i in candidates:
            rep = 0.0
            for j in selected:
                rep = max(rep, sims(vectors[i], vectors[j]))
            mmr = lam * float(scores[i]) - (1 - lam) * rep
            if mmr > best_score:
                best_score = mmr
                best_i = i
        if best_i is None:
            break
        selected.append(best_i)
        candidates.remove(best_i)
    return selected


def mmr_diversify_hybrid(
    dense_vecs: List[List[float]],
    sparse_vecs: List[Tuple[List[int], List[float]]],
    rel_scores: List[float],
    k: int,
    lam: float,
    rep_alpha: float,
    per_doc_ids: Optional[List[str]] = None,
    per_doc_limit: Optional[int] = None,
) -> List[int]:
    n = len(rel_scores)
    candidates = list(range(n))
    if n <= k:
        return candidates
    selected: List[int] = []
    counts: Dict[str, int] = {}

    def allowed(i: int) -> bool:
        if per_doc_ids is None or per_doc_limit is None:
            return True
        did = per_doc_ids[i]
        return counts.get(did, 0) < per_doc_limit

    while len(selected) < k and candidates:
        best_i = None
        best_score = -1e12
        for i in candidates:
            if per_doc_limit is not None and per_doc_ids is not None:
                if not allowed(i):
                    continue
            rep = 0.0
            for j in selected:
                d_sim = _cosine_dense(dense_vecs[i], dense_vecs[j])
                s_sim = _sparse_pair_cos(
                    sparse_vecs[i][0], sparse_vecs[i][1], sparse_vecs[j][0], sparse_vecs[j][1]
                )
                rep = max(rep, rep_alpha * d_sim + (1.0 - rep_alpha) * s_sim)
            mmr = lam * float(rel_scores[i]) - (1.0 - lam) * rep
            if mmr > best_score:
                best_score = mmr
                best_i = i
        if best_i is None:
            break
        if per_doc_limit is not None and per_doc_ids is not None and not allowed(best_i):
            alt = None
            alt_score = -1e12
            for i in candidates:
                if allowed(i):
                    rep = 0.0
                    for j in selected:
                        d_sim = _cosine_dense(dense_vecs[i], dense_vecs[j])
                        s_sim = _sparse_pair_cos(
                            sparse_vecs[i][0], sparse_vecs[i][1], sparse_vecs[j][0], sparse_vecs[j][1]
                        )
                        rep = max(rep, rep_alpha * d_sim + (1.0 - rep_alpha) * s_sim)
                    mmr = lam * float(rel_scores[i]) - (1.0 - lam) * rep
                    if mmr > alt_score:
                        alt_score = mmr
                        alt = i
            if alt is not None:
                best_i = alt
        selected.append(best_i)
        candidates.remove(best_i)
        if per_doc_limit is not None and per_doc_ids is not None:
            did = per_doc_ids[best_i]
            counts[did] = counts.get(did, 0) + 1
    return selected


def _fetch_section_chunks(doc_id: str, section: Optional[str]) -> List[Dict[str, Any]]:
    """Pobierz wszystkie chunki danej sekcji dokumentu (posortowane po chunk_id).

    Jeśli `section` jest puste/None, zwraca pustą listę (brak bezpiecznego filtra po NULL w Qdrant).
    """
    if not doc_id or not section:
        return []
    try:
        flt = qm.Filter(
            must=[
                qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id)),
                qm.FieldCondition(key="point_type", match=qm.MatchValue(value="chunk")),
                qm.FieldCondition(key="section", match=qm.MatchValue(value=section)),
            ]
        )
        out: List[Dict[str, Any]] = []
        offset = None
        while True:
            res = qdrant.scroll(
                collection_name=settings.qdrant_content_collection,
                scroll_filter=flt,
                limit=256,
                offset=offset,
                with_payload=True,
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
                payload = rec.payload or {}
                if payload.get("point_type") and payload.get("point_type") != "chunk":
                    continue
                out.append(payload)
            if offset is None:
                break
        # Posortuj po chunk_id rosnąco
        out.sort(key=lambda p: int(p.get("chunk_id", 0)))
        return out
    except Exception:
        # Best-effort — na błąd zwróć pustą listę, aby nie blokować odpowiedzi
        return []


def _build_blocks_from_hits(
    final_hits: List[Dict[str, Any]],
    summary_mode: str = "first",
) -> List[Dict[str, Any]]:
    """Zbuduj bloki po pełnych sekcjach.

    - Grupuje trafienia po (doc_id, section).
    - Dla sekcji z nazwą dociąga pełną zawartość sekcji z Qdrant i skleja tekst.
    - Dla sekcji bez nazwy (None) pozostaje przy treści trafionych chunków.
    - Score bloku = max(score) spośród trafień należących do tej sekcji.
    - summary_mode: kontrola duplikacji streszczenia per dokument.
    """
    by_key: Dict[Tuple[str, Optional[str]], List[Dict[str, Any]]] = {}
    for fh in final_hits:
        payload = fh.get("payload") or {}
        did = payload.get("doc_id")
        if not did:
            continue
        key = (str(did), payload.get("section"))
        by_key.setdefault(key, []).append(fh)

    blocks: List[Dict[str, Any]] = []
    seen_docs: set = set()

    for (did, section), hits in by_key.items():
        # Bazowy payload do metadanych
        base_payload = (hits[0].get("payload") or {})
        path = base_payload.get("path", "")
        # Score sekcji = max score spośród jej trafień
        sect_score = max(float(h.get("score", 0.0)) for h in hits)

        # Dociągnij pełną sekcję (jeśli mamy etykietę) i zbuduj tekst + zakres id
        merged_text_parts: List[str] = []
        first_cid: Optional[int] = None
        last_cid: Optional[int] = None

        sec_chunks = _fetch_section_chunks(did, section)
        if sec_chunks:
            for ch in sec_chunks:
                text = (ch.get("text") or "").strip()
                if text:
                    merged_text_parts.append(text)
                cid = int(ch.get("chunk_id", 0))
                first_cid = cid if first_cid is None else min(first_cid, cid)
                last_cid = cid if last_cid is None else max(last_cid, cid)
        else:
            # Brak sekcji lub nie udało się pobrać — użyj tylko trafionych chunków
            for h in sorted(hits, key=lambda x: int((x.get("payload") or {}).get("chunk_id", 0))):
                hp = h.get("payload") or {}
                text = (hp.get("text") or "").strip()
                if text:
                    merged_text_parts.append(text)
                cid = int(hp.get("chunk_id", 0))
                first_cid = cid if first_cid is None else min(first_cid, cid)
                last_cid = cid if last_cid is None else max(last_cid, cid)

        merged_text = "\n\n".join(merged_text_parts).strip()
        token_estimate = max(1, len(merged_text) // 4) if merged_text else None

        # Summary kontrolowane per dokument
        doc_summary_val = base_payload.get("summary")
        if summary_mode == "none":
            sum_val = None
        elif summary_mode == "first":
            if did in seen_docs:
                sum_val = None
            else:
                sum_val = doc_summary_val
                seen_docs.add(did)
        else:
            sum_val = doc_summary_val

        blocks.append(
            {
                "doc_id": did,
                "path": path,
                "section": section,
                "first_chunk_id": int(first_cid if first_cid is not None else (base_payload.get("chunk_id", 0))),
                "last_chunk_id": int(last_cid if last_cid is not None else (base_payload.get("chunk_id", 0))),
                "score": float(sect_score),
                "summary": sum_val,
                "text": merged_text,
                "token_estimate": token_estimate,
            }
        )

    blocks.sort(key=lambda b: float(b.get("score", 0.0)), reverse=True)
    return blocks


def _shape_results(
    final_hits: List[Dict[str, Any]],
    doc_map: Dict[str, Any],
    mmr_pool: List[Dict[str, Any]],
    rel2: List[float],
    req: Any,
) -> Tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]], Optional[List[Dict[str, Any]]]]:
    if req.result_format == "blocks":
        blocks_payload = _build_blocks_from_hits(final_hits, summary_mode=req.summary_mode)
        return [], None, blocks_payload

    results: List[Dict[str, Any]] = []
    groups_payload: Optional[List[Dict[str, Any]]] = None

    if req.result_format == "grouped":
        groups: Dict[str, Dict[str, Any]] = {}
        for fh in final_hits:
            payload = fh.get("payload") or {}
            did = payload.get("doc_id", "")
            if not did:
                continue
            grp = groups.get(did)
            if grp is None:
                grp = {
                    "doc_id": did,
                    "path": payload.get("path", ""),
                    "summary": None if req.summary_mode == "none" else payload.get("summary"),
                    "score": float(fh.get("score", 0.0)),
                    "chunks": [],
                }
                groups[did] = grp
            else:
                if float(fh.get("score", 0.0)) > float(grp.get("score", 0.0)):
                    grp["score"] = float(fh.get("score", 0.0))
            snippet = (payload.get("text") or "").strip()[:500]
            grp["chunks"].append({
                "chunk_id": payload.get("chunk_id", 0),
                "score": float(fh.get("score", 0.0)),
                "snippet": snippet,
            })
        groups_list = list(groups.values())
        for g in groups_list:
            g["chunks"].sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        groups_list.sort(key=lambda g: float(g.get("score", 0.0)), reverse=True)
        groups_payload = groups_list
        results = []
    else:
        seen_docs: set = set()
        for fh in final_hits:
            payload = fh.get("payload") or {}
            did = payload.get("doc_id", "")
            if req.summary_mode == "none":
                summary_val: Optional[str] = None
            elif req.summary_mode == "first":
                summary_val = None if did in seen_docs else payload.get("summary")
                if did not in seen_docs:
                    seen_docs.add(did)
            else:
                summary_val = payload.get("summary")
            results.append(
                {
                    "doc_id": did,
                    "path": payload.get("path", ""),
                    "section": payload.get("section"),
                    "chunk_id": payload.get("chunk_id", 0),
                    "score": float(fh.get("score", 0.0)),
                    "snippet": (payload.get("text") or "").strip()[:500] if payload.get("text") else (payload.get("summary", "")[:500]),
                    "summary": summary_val,
                }
            )

    return results, groups_payload, blocks_payload
