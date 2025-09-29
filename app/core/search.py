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

settings = get_settings()

# Vector names and defaults
 # names imported from constants

DEFAULT_MMR_LAMBDA = 0.3
DEFAULT_PER_DOC_LIMIT = 2
DEFAULT_SCORE_NORM = "minmax"  # minmax|zscore|none


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
    q_vec: List[float],
    flt: Optional[qm.Filter],
    summary_sparse_query: Optional[Tuple[List[int], List[float]]],
    req: Any,
) -> Tuple[List[str], Dict[str, Any]]:
    sum_search = qdrant.search(
        collection_name=settings.collection_name,
        query_vector=(SUMMARY_VECTOR_NAME, q_vec),
        query_filter=flt,
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
        cand_doc_ids = [doc_items[i]["doc_id"] for i in mmr_idx]
    else:
        order = sorted(range(len(doc_items)), key=lambda i: hybrid_rel[i], reverse=True)
        order = order[: min(req.top_m, len(order))]
        cand_doc_ids = [doc_items[i]["doc_id"] for i in order]

    return cand_doc_ids, doc_map


def _stage2_select_chunks(
    cand_doc_ids: List[str],
    q_vec: List[float],
    content_sparse_query: Optional[Tuple[List[int], List[float]]],
    req: Any,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[float]]:
    flt2 = qm.Filter(must=[qm.FieldCondition(key="doc_id", match=qm.MatchAny(any=cand_doc_ids))])
    cont_search = qdrant.search(
        collection_name=settings.collection_name,
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
        dense_score = float(hit.score or 0.0)
        sparse_dot = 0.0
        if content_sparse_query is not None and payload.get("content_sparse_indices") and payload.get("content_sparse_values"):
            q_lookup = dict(zip(content_sparse_query[0], content_sparse_query[1]))
            sparse_dot = _sparse_dot(q_lookup, payload.get("content_sparse_indices", []), payload.get("content_sparse_values", []))
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
    for idx, item in zip(sel_idx, selected):
        hit = item["hit"]
        payload = hit.payload or {}
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


def _build_neighbor_index(
    mmr_pool: List[Dict[str, Any]], rel2: List[float]
) -> Dict[Tuple[str, Optional[str], int], Dict[str, Any]]:
    neighbor_index: Dict[Tuple[str, Optional[str], int], Dict[str, Any]] = {}
    for idx2, item in enumerate(mmr_pool):
        payload = (item.get("hit").payload if item.get("hit") else {}) or {}
        did = payload.get("doc_id") or ""
        sec = payload.get("section")
        cid = payload.get("chunk_id")
        if not did or cid is None:
            continue
        score2 = float(rel2[idx2]) if idx2 < len(rel2) else float(item.get("dense_score", 0.0))
        neighbor_index[(did, sec, int(cid))] = {"payload": payload, "score": score2}
    return neighbor_index


def _approx_tokens(s: str) -> int:
    return max(1, len(s) // 4)


def build_merged_blocks(
    final_hits: List[Dict[str, Any]],
    merge_group_budget_tokens: int,
    max_merged_per_group: int,
    join_delim: str,
    summary_mode: str = "first",
    expand_neighbors: int = 0,
    neighbor_index: Optional[Dict[Tuple[str, Optional[str], int], Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    by_group: Dict[Tuple[str, Optional[str]], List[Dict[str, Any]]] = {}
    for fh in final_hits:
        payload = fh.get("payload") or {}
        did = payload.get("doc_id") or ""
        sec = payload.get("section")
        if not did:
            continue
        by_group.setdefault((did, sec), []).append({"payload": payload, "score": float(fh.get("score", 0.0))})
    blocks: List[Dict[str, Any]] = []
    for (doc_id, section), items in by_group.items():
        items.sort(key=lambda x: int((x.get("payload") or {}).get("chunk_id", 0)))
        runs: List[List[Dict[str, Any]]] = []
        cur: List[Dict[str, Any]] = []
        last_id: Optional[int] = None
        for it in items:
            cid = int((it.get("payload") or {}).get("chunk_id", 0))
            if last_id is None or cid == last_id + 1:
                cur.append(it)
            else:
                runs.append(cur)
                cur = [it]
            last_id = cid
        if cur:
            runs.append(cur)
        runs.sort(key=lambda run: max([float(x.get("score", 0.0)) for x in run] or [0.0]), reverse=True)
        runs = runs[: max(1, int(max_merged_per_group or 1))]
        for run in runs:
            if expand_neighbors and neighbor_index:
                p0 = (run[0].get("payload") or {})
                did = p0.get("doc_id") or ""
                sec = p0.get("section")
                existing_ids = {int((r.get("payload") or {}).get("chunk_id", -10)) for r in run}
                cur_first = min(existing_ids) if existing_ids else None
                cur_last = max(existing_ids) if existing_ids else None
                if cur_first is not None:
                    for step in range(1, int(expand_neighbors) + 1):
                        cid = cur_first - step
                        key = (did, sec, cid)
                        if key in neighbor_index and cid not in existing_ids:
                            nb = neighbor_index[key]
                            run.insert(0, {"payload": nb.get("payload", {}), "score": float(nb.get("score", 0.0))})
                            existing_ids.add(cid)
                            cur_first = cid
                        else:
                            break
                if cur_last is not None:
                    for step in range(1, int(expand_neighbors) + 1):
                        cid = cur_last + step
                        key = (did, sec, cid)
                        if key in neighbor_index and cid not in existing_ids:
                            nb = neighbor_index[key]
                            run.append({"payload": nb.get("payload", {}), "score": float(nb.get("score", 0.0))})
                            existing_ids.add(cid)
                            cur_last = cid
                        else:
                            break
            text_parts: List[str] = []
            scores: List[float] = []
            used_tokens = 0
            first_chunk_id = int((run[0].get("payload") or {}).get("chunk_id", 0))
            last_chunk_id = first_chunk_id
            for r in run:
                payload = r.get("payload") or {}
                chunk_text = (payload.get("text") or "").strip()
                if not chunk_text:
                    continue
                t = _approx_tokens(chunk_text)
                if text_parts and used_tokens + t > int(merge_group_budget_tokens):
                    break
                text_parts.append(chunk_text)
                scores.append(float(r.get("score", 0.0)))
                last_chunk_id = int(payload.get("chunk_id", last_chunk_id))
                used_tokens += t
            if not text_parts:
                continue
            payload0 = (run[0].get("payload") or {})
            doc_summary_val = payload0.get("summary")
            blocks.append(
                {
                    "doc_id": doc_id,
                    "path": payload0.get("path", ""),
                    "section": section,
                    "first_chunk_id": first_chunk_id,
                    "last_chunk_id": last_chunk_id,
                    "score": max(scores) if scores else 0.0,
                    "summary": None,
                    "doc_summary": doc_summary_val,
                    "text": join_delim.join(text_parts).strip(),
                    "token_estimate": used_tokens,
                }
            )
    blocks.sort(key=lambda b: float(b.get("score", 0.0)), reverse=True)
    if summary_mode == "none":
        for b in blocks:
            b["summary"] = None
    elif summary_mode == "first":
        seen_docs: set = set()
        for b in blocks:
            did = b.get("doc_id")
            if did not in seen_docs:
                b["summary"] = b.get("doc_summary")
                seen_docs.add(did)
            else:
                b["summary"] = None
    else:
        for b in blocks:
            b["summary"] = b.get("doc_summary")
    return blocks


def _shape_results(
    final_hits: List[Dict[str, Any]],
    doc_map: Dict[str, Any],
    mmr_pool: List[Dict[str, Any]],
    rel2: List[float],
    req: Any,
) -> Tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]], Optional[List[Dict[str, Any]]]]:
    blocks_payload: Optional[List[Dict[str, Any]]] = None
    if req.merge_chunks or req.result_format == "blocks":
        neighbor_index = None
        if req.expand_neighbors and req.expand_neighbors > 0:
            neighbor_index = _build_neighbor_index(mmr_pool, rel2)
        raw_blocks = build_merged_blocks(
            final_hits,
            merge_group_budget_tokens=req.merge_group_budget_tokens,
            max_merged_per_group=req.max_merged_per_group,
            join_delim=req.block_join_delimiter,
            summary_mode=req.summary_mode,
            expand_neighbors=max(0, int(req.expand_neighbors or 0)),
            neighbor_index=neighbor_index,
        )
        blocks_payload = raw_blocks

    if req.result_format == "blocks":
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
