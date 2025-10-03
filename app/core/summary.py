"""Summarization via OpenAI-compatible chat completions."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

from openai import OpenAI

from app.settings import get_settings

settings = get_settings()

summary_client = OpenAI(base_url=settings.summary_api_url, api_key=settings.summary_api_key)

logger = logging.getLogger("rags_tool.summary")

MAX_DOC_TO_SUMMARY = 75_000

def _default_title_from_text(text: str) -> str:
    for line in text.splitlines():
        candidate = line.strip()
        if candidate:
            return candidate[:200]
    return ""


def llm_summary(text: str, model: str = settings.summary_model, max_tokens: int = 300) -> Dict[str, Any]:
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
                        "content": settings.summary_prompt_json + "\n\nTEKST:\n" + text,
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
            summary_val = str(data.get("summary", "")).strip()
            signature_val = data.get("signature", [])
            if isinstance(signature_val, str):
                signature_list = [s.strip() for s in re.split(r",|;|\n", signature_val) if s.strip()]
            elif isinstance(signature_val, list):
                signature_list = [str(s).strip() for s in signature_val if str(s).strip()]
            else:
                signature_list = []
            entities_val = data.get("entities", "")
            if isinstance(entities_val, list):
                entities_str = ", ".join(str(x) for x in entities_val)
            else:
                entities_str = str(entities_val)
            replacement_val = data.get("replacement", "")
            if isinstance(replacement_val, list):
                replacement_str = ", ".join(str(x).strip() for x in replacement_val if str(x).strip())
            else:
                replacement_str = str(replacement_val or "").strip()
            replacement_str = replacement_str or "brak"
            if replacement_str.lower() == "brak":
                replacement_str = "brak"
            title_str = title_str or _default_title_from_text(text)
            if summary_val:
                return {
                    "title": title_str,
                    "summary": summary_val,
                    "signature": signature_list,
                    "entities": entities_str,
                    "replacement": replacement_str,
                }
        except Exception as exc:
            logger.warning("JSON-mode summary failed; falling back to text parser: %s", exc)

    rsp = summary_client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": settings.summary_system_prompt},
            {"role": "user", "content": settings.summary_prompt + text},
        ],
        max_tokens=max_tokens,
    )
    out = rsp.choices[0].message.content or ""
    summary = ""
    title = ""
    signature_list: List[str] = []
    entities_str = ""
    replacement_str = "brak"
    for line in out.splitlines():
        m = re.match(r"^\s*title\s*:\s*(.*)$", line, re.IGNORECASE)
        if m:
            title_candidate = m.group(1).strip()
            if title_candidate:
                title = title_candidate
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
            entities_str = m.group(1).strip()
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
    return {
        "title": title,
        "summary": summary,
        "signature": signature_list,
        "entities": entities_str,
        "replacement": replacement_str,
    }
