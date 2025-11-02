"""Summarization via OpenAI-compatible chat completions."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI

from app.settings import get_settings

settings = get_settings()

summary_client = OpenAI(base_url=settings.summary_api_url, api_key=settings.summary_api_key)

logger = logging.getLogger("rags_tool.summary")

MAX_DOC_TO_SUMMARY = 75_000

# Derive a default title from the first non-empty line of text.
def _default_title_from_text(text: str) -> str:
    for line in text.splitlines():
        candidate = line.strip()
        if candidate:
            return candidate[:200]
    return ""


# Minimal local summary used as a fallback when LLM is unavailable.
def _naive_local_summary(text: str, max_sentences: int = 5) -> Dict[str, Any]:
    """Produce a minimal, deterministic summary locally when LLM is unavailable.

    - Title: first non-empty line (trimmed to 200 chars)
    - Summary: first few sentences (up to max_sentences) or first 600 chars
    - Signature/entities: left minimal/empty (entities: [])
    - Replacement: "brak"
    """
    title = _default_title_from_text(text)
    # Simple sentence split; fallback to head if not enough delimiters
    parts = re.split(r"(?<=[\.!?])\s+", text)
    parts = [p.strip() for p in parts if p.strip()]
    if parts:
        summary = " ".join(parts[:max_sentences])
    else:
        summary = text[:600]
    if len(summary) > 600:
        summary = summary[:600]
    return {
        "title": title,
        "subtitle": "brak",
        "summary": summary,
        "signature": [],
        "entities": [],
        "replacement": "brak",
        "doc_date": "brak",
    }


# Produce a structured summary via OpenAI-compatible chat API (JSON-mode if enabled).
def llm_summary(
    text: str,
    model: str = settings.summary_model,
    max_tokens: int = 300,
    path: Optional[str] = None,
) -> Dict[str, Any]:
    text = text.strip()
    if len(text) > MAX_DOC_TO_SUMMARY:
        text = text[:MAX_DOC_TO_SUMMARY]

    if getattr(settings, "summary_json_mode", True):
        try:
            rsp = summary_client.chat.completions.create(
                model=model,
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": settings.summary_system_prompt},
                    {
                        "role": "user",
                        "content": (
                            settings.summary_prompt_json
                            + ("\n\nPATH:\n" + str(path) if path else "")
                            + "\n\nTEKST:\n"
                            + text
                        ),
                    },
                ],
                max_tokens=max_tokens,
            )
            content = rsp.choices[0].message.content or "{}"
            data = json.loads(content)
            title_val = data.get("title", "")
            if isinstance(title_val, list):
                title_str = " ".join(str(s).strip() for s in title_val if str(s).strip())
            else:
                title_str = str(title_val or "").strip()
            # subtitle (string; max 100 chars; default 'brak')
            subtitle_val = data.get("subtitle", "")
            if isinstance(subtitle_val, list):
                subtitle_str = " ".join(str(s).strip() for s in subtitle_val if str(s).strip())
            else:
                subtitle_str = str(subtitle_val or "").strip()
            subtitle_str = subtitle_str or "brak"
            if len(subtitle_str) > 100:
                subtitle_str = subtitle_str[:100]
            summary_val = str(data.get("summary", "")).strip()
            signature_val = data.get("signature", [])
            if isinstance(signature_val, str):
                signature_list = [s.strip() for s in re.split(r",|;|\n", signature_val) if s.strip()]
            elif isinstance(signature_val, list):
                signature_list = [str(s).strip() for s in signature_val if str(s).strip()]
            else:
                signature_list = []
            entities_val = data.get("entities", [])
            if isinstance(entities_val, str):
                entities_list = [s.strip() for s in re.split(r",|;|\n", entities_val) if s.strip()]
            elif isinstance(entities_val, list):
                entities_list = [str(s).strip() for s in entities_val if str(s).strip()]
            else:
                entities_list = []
            replacement_val = data.get("replacement", "")
            if isinstance(replacement_val, list):
                replacement_str = ", ".join(str(x).strip() for x in replacement_val if str(x).strip())
            else:
                replacement_str = str(replacement_val or "").strip()
            replacement_str = replacement_str or "brak"
            if replacement_str.lower() == "brak":
                replacement_str = "brak"
            # Try several keys for date; prefer 'doc_date'
            date_val = data.get("doc_date", data.get("date", data.get("document_date", "")))
            if isinstance(date_val, list):
                doc_date_str = " ".join(str(s).strip() for s in date_val if str(s).strip())
            else:
                doc_date_str = str(date_val or "").strip()
            doc_date_str = doc_date_str or "brak"
            # Optional is_active returned by the model; default to True when missing
            def _to_bool(v: Any) -> Optional[bool]:
                if isinstance(v, bool):
                    return v
                if isinstance(v, (int, float)):
                    return bool(v)
                if isinstance(v, str):
                    s = v.strip().lower()
                    if s in {"true", "yes", "1"}:
                        return True
                    if s in {"false", "no", "0"}:
                        return False
                return None
            is_active_val = _to_bool(data.get("is_active"))
            if is_active_val is None:
                is_active_val = True
            title_str = title_str or _default_title_from_text(text)
            if summary_val:
                return {
                    "title": title_str,
                    "subtitle": subtitle_str,
                    "summary": summary_val,
                    "signature": signature_list,
                    "entities": entities_list,
                    "replacement": replacement_str,
                    "doc_date": doc_date_str,
                    "is_active": bool(is_active_val),
                }
        except Exception as exc:
            logger.warning("JSON-mode summary failed; falling back to text parser: %s", exc)

    try:
        rsp = summary_client.chat.completions.create(
            model=model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": settings.summary_system_prompt},
                {"role": "user", "content": settings.summary_prompt + (text)},
            ],
            max_tokens=max_tokens,
        )
        out = rsp.choices[0].message.content or ""
    except Exception as exc:
        # Final safety net: avoid 500 during ingest/build if remote endpoint is misconfigured
        logger.warning(
            "Text-mode summary failed; using local fallback | base_url=%s error=%s",
            getattr(settings, "summary_api_url", ""),
            exc,
        )
        return _naive_local_summary(text)
    summary = ""
    title = ""
    signature_list: List[str] = []
    entities_list: List[str] = []
    replacement_str = "brak"
    doc_date_str = "brak"
    subtitle = "brak"
    for line in out.splitlines():
        m = re.match(r"^\s*title\s*:\s*(.*)$", line, re.IGNORECASE)
        if m:
            title_candidate = m.group(1).strip()
            if title_candidate:
                title = title_candidate
            continue
        m = re.match(r"^\s*subtitle\s*:\s*(.*)$", line, re.IGNORECASE)
        if m:
            s = (m.group(1) or "").strip()
            subtitle = s if s else "brak"
            continue
        m = re.match(r"^\s*summary\s*:\s*(.*)$", line, re.IGNORECASE)
        if m:
            summary = m.group(1).strip()
            continue
        m = re.match(r"^\s*signature\s*:\s*(.*)$", line, re.IGNORECASE)
        if m:
            sig = m.group(1).strip()
            signature_list = [s.strip() for s in re.split(r",|;", sig) if s.strip()]
            continue
        m = re.match(r"^\s*entities\s*:\s*(.*)$", line, re.IGNORECASE)
        if m:
            raw = m.group(1).strip()
            entities_list = [s.strip() for s in re.split(r",|;|\n", raw) if s.strip()]
            continue
        m = re.match(r"^\s*date\s*:\s*(.*)$", line, re.IGNORECASE)
        if m:
            dd = m.group(1).strip()
            doc_date_str = dd if dd else "brak"
            continue
        m = re.match(r"^\s*replacement\s*:\s*(.*)$", line, re.IGNORECASE)
        if m:
            replacement_candidate = m.group(1).strip()
            replacement_str = replacement_candidate or "brak"
            continue
    if not summary:
        summary = out.strip()[:600]
    replacement_str = replacement_str or "brak"
    if replacement_str.lower() == "brak":
        replacement_str = "brak"
    title = title or _default_title_from_text(text)
    if not subtitle:
        subtitle = "brak"
    if len(subtitle) > 100:
        subtitle = subtitle[:100]
    return {
        "title": title,
        "subtitle": subtitle,
        "summary": summary,
        "signature": signature_list,
        "entities": entities_list,
        "replacement": replacement_str,
        "doc_date": doc_date_str or "brak",
        # Text-mode fallback does not infer active status; default to True
        "is_active": True,
    }
