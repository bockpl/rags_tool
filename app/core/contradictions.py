"""On-the-fly contradictions/inconsistencies analysis across documents.

This module provides a high-level pipeline that:
- identifies a reference document by title (or doc_id),
- enumerates its sections,
- for each section retrieves top candidate sections from other documents,
- compares pairs with an LLM (JSON-mode) to classify contradictions,
- returns a structured JSON report.

All LLM calls use the same OpenAI-compatible endpoint as summaries.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import re
import time

from openai import OpenAI
from qdrant_client.http import models as qm

from app.core.embedding import embed_query
from app.core.search import (
    _canonical_section_label,
    _fetch_sections_chunks_batch,
    _fetch_doc_summaries,
    _truncate_head_tail,
)
from app.qdrant_utils import qdrant
from app.settings import get_settings
from app.core.constants import CONTENT_VECTOR_NAME, SUMMARY_VECTOR_NAME
from app.models import (
    ContradictionAnalysisRequest,
    ContradictionAnalysisResponse,
    ContradictionFinding,
    ContradictionSectionReport,
)


logger = logging.getLogger("rags_tool.contradictions")
settings = get_settings()

contr_client = OpenAI(base_url=settings.summary_api_url, api_key=settings.summary_api_key)


def _loads_json_object(raw: str) -> Dict[str, Any]:
    """Parse a JSON object robustly, tolerating code fences and preambles.

    - Accepts plain JSON
    - Strips markdown code fences ```...```
    - Extracts substring between first '{' and last '}'
    Returns empty dict on failure.
    """
    s = (raw or "").strip()
    if not s:
        return {}
    # Quick path
    try:
        return json.loads(s)
    except Exception:
        pass
    # Remove code fences if present
    s2 = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", s)
    if s2 != s:
        try:
            return json.loads(s2.strip())
        except Exception:
            pass
    # Heuristic: slice between first '{' and last '}'
    try:
        i = s.find("{")
        j = s.rfind("}")
        if i != -1 and j != -1 and j > i:
            sub = s[i : j + 1]
            return json.loads(sub)
    except Exception:
        pass
    # Failed — log short head for diagnostics
    logger.debug("Judge JSON parse failed | head=%s", s[:200].replace("\n", " "))
    return {}


def _mode_filter(mode: str) -> Optional[qm.Filter]:
    mode_l = (mode or "").strip().lower()
    if mode_l == "current":
        return qm.Filter(must=[qm.FieldCondition(key="is_active", match=qm.MatchValue(value=True))])
    if mode_l == "archival":
        return qm.Filter(must=[qm.FieldCondition(key="is_active", match=qm.MatchValue(value=False))])
    # all/auto -> no filter here (auto logic is for search queries)
    return None


def _find_document_by_title_or_id(title: str, doc_id_hint: Optional[str]) -> Tuple[str, Optional[str], Optional[str], Optional[bool]]:
    """Best-effort identification of a document given a title or doc_id.

    Returns (doc_id, title, doc_date, is_active).
    """
    if doc_id_hint:
        info = _fetch_doc_summaries([doc_id_hint])
        meta = info.get(doc_id_hint, {})
        return doc_id_hint, meta.get("doc_title"), meta.get("doc_date"), meta.get("is_active")

    qv = embed_query([title])[0]
    try:
        flt = qm.Filter(must=[qm.FieldCondition(key="point_type", match=qm.MatchValue(value="summary"))])
        res = qdrant.search(
            collection_name=settings.qdrant_summary_collection,
            query_vector=(SUMMARY_VECTOR_NAME, qv),
            query_filter=flt,
            limit=8,
            with_payload=["doc_id", "title", "doc_date", "is_active"]
        )
    except Exception:
        res = []
    if not res:
        raise ValueError("Nie znaleziono kandydata dokumentu po tytule")

    def _norm(s: Optional[str]) -> str:
        return " ".join((s or "").strip().lower().split())

    norm_title = _norm(title)
    best = None
    # Prefer exact/near-exact title match, otherwise take top-1 by score
    for r in res:
        p = r.payload or {}
        did = p.get("doc_id")
        if not did:
            continue
        t = _norm(p.get("title"))
        if t and (t == norm_title or norm_title in t or t in norm_title):
            best = r
            break
    if best is None:
        best = res[0]
    payload = best.payload or {}
    return payload.get("doc_id"), payload.get("title"), payload.get("doc_date"), payload.get("is_active")


def _list_sections_for_doc(doc_id: str, level: str) -> List[str]:
    """Return canonical section labels for a document at the given merge level."""
    must = [
        qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id)),
        qm.FieldCondition(key="point_type", match=qm.MatchValue(value="chunk")),
    ]
    flt = qm.Filter(must=must)
    labels: Set[str] = set()
    offset = None
    try:
        while True:
            res = qdrant.scroll(
                collection_name=settings.qdrant_content_collection,
                scroll_filter=flt,
                limit=256,
                offset=offset,
                with_payload=["section_path"],
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
                sec = p.get("section_path")
                canon = _canonical_section_label(sec, level)
                if canon:
                    labels.add(canon)
            if offset is None:
                break
    except Exception:
        return []
    out = sorted(labels)
    return out


def _detect_rule_type_heuristic(text: str) -> str:
    t = (text or "").lower()
    if re.search(r"\bwchodzi w życie\b|\bwejdzie w życie\b|\bz dniem\b", t):
        return "entry_into_force"
    if re.search(r"\btraci moc\b|\buchyla się\b|\bprzestaje obowiązywać\b", t):
        return "repeal"
    if re.search(r"\btermin\b|\bdo dnia\b|\bnie później niż\b", t):
        return "deadline"
    if re.search(r"\bco najmniej\b|\bnie więcej niż\b|\bminimum\b|\bmaximum\b|\bpróg\b", t):
        return "threshold"
    if re.search(r"\bstosuje się do\b|\bdotyczy\b|\bma zastosowanie do\b", t):
        return "scope"
    return "other"


def _extract_rule_struct(text: str) -> Tuple[Optional[str], Optional[str], str]:
    """Use LLM to produce (rule, subject, rule_type) from section text.

    Fallbacks:
    - subject: None
    - rule_type: heuristic detection when not provided
    """
    raw = (text or "").strip()
    if not raw:
        return None, None, "other"
    ctx = _truncate_head_tail(raw, int(getattr(settings, "contradictions_max_context_chars", 2500)))
    rule: Optional[str] = None
    subject: Optional[str] = None
    rule_type: Optional[str] = None
    try:
        rsp = contr_client.chat.completions.create(
            model=settings.summary_model,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": settings.contradictions_rule_prompt_json},
                {"role": "user", "content": ctx},
            ],
            max_tokens=220,
        )
        content = rsp.choices[0].message.content or "{}"
        data = _loads_json_object(content)
        rule = str(data.get("rule", "") or "").strip() or None
        subject_raw = str(data.get("subject", "") or "").strip() or None
        rule_type_raw = str(data.get("rule_type", "") or "").strip().lower() or None
        if subject_raw:
            subject = subject_raw
        if rule_type_raw:
            rule_type = rule_type_raw
    except Exception as exc:
        logger.debug("Rule extraction failed; will use heuristic: %s", exc)
    if not rule:
        rule = _truncate_head_tail(raw, 500)
    if not rule_type:
        rule_type = _detect_rule_type_heuristic(raw)
    return rule, subject, rule_type


def _group_candidates_by_section(hits: Iterable[Any], level: str) -> Dict[Tuple[str, Optional[str]], Dict[str, Any]]:
    """Group raw chunk hits by (doc_id, canonical section label)."""
    groups: Dict[Tuple[str, Optional[str]], Dict[str, Any]] = {}
    for h in hits:
        p = h.payload or {}
        did = p.get("doc_id")
        if not did:
            continue
        if p.get("point_type") and p.get("point_type") != "chunk":
            continue
        canon = _canonical_section_label(p.get("section_path"), level)
        key = (did, canon)
        ent = groups.get(key)
        score = float(h.score or 0.0)
        if ent is None:
            groups[key] = {"doc_id": did, "section": canon, "score": score, "payload": p}
        else:
            if score > float(ent.get("score", 0.0)):
                ent["score"] = score
    return groups


def _retrieve_candidate_sections(
    query_text: str,
    *,
    section_level: str,
    exclude_doc_id: Optional[str] = None,
    only_doc_id: Optional[str] = None,
    per_section_limit: int,
    mode: str,
) -> List[Dict[str, Any]]:
    """Retrieve top candidate sections across the corpus for comparison."""
    qv = embed_query([query_text])[0]
    base_must = [qm.FieldCondition(key="point_type", match=qm.MatchValue(value="chunk"))]
    if only_doc_id:
        base_must.append(qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=only_doc_id)))
    flt = _mode_filter(mode)
    # Merge must/must_not with our exclusion using must_not (portable across client versions)
    must = list(base_must)
    must_not = []
    should = None
    if flt:
        if getattr(flt, "must", None):
            must.extend(list(flt.must))
        if getattr(flt, "must_not", None):
            must_not.extend(list(flt.must_not))
        if getattr(flt, "should", None):
            should = list(flt.should)
    if exclude_doc_id and not only_doc_id:
        must_not.append(qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=exclude_doc_id)))
    final_filter = qm.Filter(must=must, must_not=(must_not or None), should=should)

    try:
        dense_hits = qdrant.search(
            collection_name=settings.qdrant_content_collection,
            query_vector=(CONTENT_VECTOR_NAME, qv),
            query_filter=final_filter,
            limit=max(50, per_section_limit * 5),
            with_payload=["doc_id", "path", "section_path", "is_active"],
            with_vectors=False,
            search_params=qm.SearchParams(exact=False, hnsw_ef=128),
        )
    except Exception:
        dense_hits = []

    groups = _group_candidates_by_section(dense_hits, section_level)
    if not groups:
        return []
    # Sort by score desc and clip
    ordered = sorted(groups.values(), key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return ordered[: per_section_limit]


def _judge_pair(
    rule_a: Optional[str],
    text_a: str,
    text_b: str,
    *,
    subject_a: Optional[str],
    meta_a: Dict[str, Any],
    meta_b: Dict[str, Any],
    rule_type: Optional[str],
) -> Tuple[str, float, Optional[str], List[str], List[str], Optional[str], Optional[str], Optional[bool]]:
    """Classify the relation between A and B using JSON-mode LLM.

    Returns (label, confidence, rationale, quotes_a, quotes_b).
    """
    # Prepare compact contexts
    a_ctx = _truncate_head_tail(text_a or "", int(getattr(settings, "contradictions_max_context_chars", 2500)))
    b_ctx = _truncate_head_tail(text_b or "", int(getattr(settings, "contradictions_max_context_chars", 2500)))
    # Pre‑gate: entry_into_force porównujemy tylko w obrębie tego samego dokumentu
    if (rule_type or "").strip().lower() == "entry_into_force" and (meta_a.get("doc_id") != meta_b.get("doc_id")):
        return "unrelated", 0.0, None, [], [], subject_a, None, False

    user_payload = {
        "rule_a": rule_a or "",
        "subject_a": subject_a or "",
        "title_a": meta_a.get("title") or "",
        "doc_id_a": meta_a.get("doc_id") or "",
        "context_a": a_ctx,
        "subject_b": meta_b.get("subject") or "",
        "title_b": meta_b.get("title") or "",
        "doc_id_b": meta_b.get("doc_id") or "",
        "context_b": b_ctx,
        "rule_type": (rule_type or ""),
    }
    label = "unrelated"
    conf = 0.0
    rationale = None
    qa: List[str] = []
    qb: List[str] = []
    subj_a_out: Optional[str] = subject_a
    subj_b_out: Optional[str] = None
    same_subj: Optional[bool] = None
    try:
        rsp = contr_client.chat.completions.create(
            model=settings.summary_model,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": settings.contradictions_judge_prompt_json},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            max_tokens=220,
        )
        content = rsp.choices[0].message.content or "{}"
        data = _loads_json_object(content)
        raw_label = str(data.get("label", "") or "").strip().lower()
        mapping = {
            "contradiction": "contradiction",
            "conflict": "contradiction",
            "inconsistent": "contradiction",
            "entails": "entails",
            "support": "entails",
            "overlap": "overlap",
            "related": "overlap",
            "unrelated": "unrelated",
        }
        label = mapping.get(raw_label, "unrelated")
        try:
            conf = float(data.get("confidence", 0.0))
        except Exception:
            conf = 0.0
        rationale = str(data.get("rationale", "") or "").strip() or None
        qa = [str(x) for x in (data.get("quotes_a") or []) if str(x).strip()]
        qb = [str(x) for x in (data.get("quotes_b") or []) if str(x).strip()]
        subj_a_out = (str(data.get("subject_extracted_a") or "").strip() or subj_a_out)
        subj_b_out = (str(data.get("subject_extracted_b") or "").strip() or None)
        same_val = data.get("same_subject")
        same_subj = (bool(same_val) if isinstance(same_val, bool) else None)
    except Exception as exc:
        logger.debug("Judge LLM failed; defaulting unrelated: %s", exc)
    # Post‑gate: dla entry_into_force wymagaj same_subject=true
    if (rule_type or "").strip().lower() == "entry_into_force" and (same_subj is not True):
        return "unrelated", 0.0, rationale, qa, qb, subj_a_out, subj_b_out, False
    return label, conf, rationale, qa, qb, subj_a_out, subj_b_out, same_subj


def analyze_contradictions(req: ContradictionAnalysisRequest) -> ContradictionAnalysisResponse:
    t0 = time.time()
    # Identify reference document
    did, title, doc_date, is_active = _find_document_by_title_or_id(req.title, req.doc_id)

    # Enumerate sections at requested level
    level = (req.section_level or "ust").strip().lower()
    sections = _list_sections_for_doc(did, level)
    if not sections:
        return ContradictionAnalysisResponse(doc_id=did, title=title, doc_date=doc_date, is_active=is_active, findings=[])

    # Fetch full sections of the reference doc in one go
    ref_chunks_map = _fetch_sections_chunks_batch(did, sections)
    findings: List[ContradictionSectionReport] = []

    # Prepare metadata cache for candidate docs (title/date/is_active)
    meta_cache: Dict[str, Dict[str, Any]] = {}
    # Track processed documents for audit
    processed_ids: Set[str] = set()
    # Add reference doc to processed and meta cache
    meta_cache[did] = {
        "doc_title": title,
        "doc_date": doc_date,
        "is_active": is_active,
    }
    processed_ids.add(did)

    total_candidates = 0
    for sec in sections:
        ref_chunks = ref_chunks_map.get(sec, [])
        ref_text = "\n\n".join((p.get("text") or "").strip() for p in ref_chunks if (p.get("text") or "").strip())
        if not ref_text.strip():
            continue
        rule, subject_a, rule_type = _extract_rule_struct(ref_text)
        # Dla "entry_into_force" porównujemy tylko w obrębie tego samego dokumentu
        if (rule_type or "").strip().lower() == "entry_into_force":
            # Kandydaci = inne sekcje tego samego dokumentu
            other_labels = [lab for lab in sections if lab != sec]
            candidates = [{"doc_id": did, "section": lab, "score": 0.0, "payload": {"path": None}} for lab in other_labels]
        else:
            candidates = _retrieve_candidate_sections(
                rule or ref_text[:500],
                section_level=level,
                exclude_doc_id=did,
                only_doc_id=None,
                per_section_limit=max(1, int(req.max_candidates_per_section)),
                mode=req.mode,
            )
        if not candidates:
            continue
        total_candidates += len(candidates)

        # Enrich metadata in bulk
        c_doc_ids = sorted({c.get("doc_id") for c in candidates if c.get("doc_id")})
        for chunk in range(0, len(c_doc_ids), 200):
            ids_slice = c_doc_ids[chunk : chunk + 200]
            try:
                meta_cache.update(_fetch_doc_summaries(ids_slice))
            except Exception:
                pass
        processed_ids.update([x for x in c_doc_ids if x])

        # Fetch candidate section texts per document (batch)
        labels_by_doc: Dict[str, List[str]] = {}
        for c in candidates:
            did2 = str(c.get("doc_id"))
            lab = c.get("section")
            if lab:
                labels_by_doc.setdefault(did2, []).append(lab)
        cand_texts: Dict[Tuple[str, str], str] = {}
        for did2, labs in labels_by_doc.items():
            mapping = _fetch_sections_chunks_batch(did2, labs)
            for lab, lst in mapping.items():
                text = "\n\n".join((p.get("text") or "").strip() for p in lst if (p.get("text") or "").strip())
                cand_texts[(did2, lab)] = text

        sec_conflicts: List[ContradictionFinding] = []
        for c in candidates:
            did2 = str(c.get("doc_id"))
            lab = c.get("section")
            text_b = cand_texts.get((did2, lab), "")
            if not text_b.strip():
                continue
            meta_a = {"doc_id": did, "title": title}
            meta_b = {"doc_id": did2, "title": (meta_cache.get(did2, {}) or {}).get("doc_title")}
            label, conf, rationale, qa, qb, subj_a_out, subj_b_out, same_subj = _judge_pair(
                rule,
                ref_text,
                text_b,
                subject_a=subject_a,
                meta_a=meta_a,
                meta_b=meta_b,
                rule_type=rule_type,
            )
            if label != "contradiction":
                continue
            # Respect archival filter unless explicitly included
            meta = meta_cache.get(did2, {})
            is_active_b = meta.get("is_active")
            if is_active_b is False and not req.include_archival_conflicts:
                continue
            if conf < float(req.confidence_threshold):
                continue
            sec_conflicts.append(
                ContradictionFinding(
                    other_doc_id=did2,
                    other_title=meta.get("doc_title"),
                    other_path=(meta.get("path") or c.get("payload", {}).get("path")),
                    other_doc_date=meta.get("doc_date"),
                    other_is_active=is_active_b,
                    other_section=lab,
                    label=label,
                    confidence=conf,
                    rationale=rationale,
                    quotes_a=qa or None,
                    quotes_b=qb or None,
                    subject_a=subj_a_out,
                    subject_b=subj_b_out,
                    same_subject=same_subj,
                    rule_type=rule_type,
                )
            )

        if sec_conflicts:
            findings.append(
                ContradictionSectionReport(
                    section=sec,
                    rule=rule,
                    rule_subject=subject_a,
                    rule_type=rule_type,
                    conflicts=sec_conflicts,
                )
            )

    # Build processed_docs list
    processed_docs_payload: List[Dict[str, Any]] = []
    for pid in sorted(processed_ids):
        meta = meta_cache.get(pid, {})
        processed_docs_payload.append(
            {
                "doc_id": pid,
                "title": meta.get("doc_title"),
                "doc_date": meta.get("doc_date"),
                "is_active": meta.get("is_active"),
            }
        )

    took_ms = int((time.time() - t0) * 1000)
    logger.debug(
        "Contradictions analysis finished | doc_id=%s sections=%d candidates=%d took_ms=%d",
        did,
        len(sections),
        int(total_candidates),
        took_ms,
    )

    return ContradictionAnalysisResponse(
        doc_id=did,
        title=title,
        doc_date=doc_date,
        is_active=is_active,
        findings=findings,
        took_ms=took_ms,
        processed_docs=processed_docs_payload,  # type: ignore[arg-type]
    )
