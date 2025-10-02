"""Summarization via OpenAI-compatible chat completions."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from openai import OpenAI

from app.settings import get_settings

settings = get_settings()

summary_client = OpenAI(base_url=settings.summary_api_url, api_key=settings.summary_api_key)

SUMMARY_PROMPT = (
    "Streść poniższy tekst w maks. 5 zdaniach, wypisz też 'SIGNATURE' (10–20 lematów kluczowych), "
    "'ENTITIES' (nazwy własne/ID/zakres dat) oraz 'REPLACEMENT' (jakie akty zastępuje lub przez co jest "
    "zastąpiony, wpisz 'brak' jeśli brak informacji). Bez komentarzy.\n\n"
    "FORMAT:\nSUMMARY: ...\nSIGNATURE: lemma1, lemma2, ...\nENTITIES: ...\nREPLACEMENT: ...\n\nTEKST:\n"
)

SUMMARY_PROMPT_JSON = (
    "Zwróć wyłącznie poprawny JSON bez komentarzy i bez kodu. Klucze: "
    "'summary' (string, max 5 zdań po polsku), "
    "'signature' (lista 10–20 lematów kluczowych jako strings), "
    "'entities' (string z nazwami własnymi/ID/zakresami dat), "
    "'replacement' (string opisujący listę aktów zastąpionych lub słowo 'brak')."
)


def llm_summary(text: str, model: str = settings.summary_model, max_tokens: int = 300) -> Dict[str, Any]:
    text = text.strip()
    if len(text) > 8000:
        text = text[:8000]

    if getattr(settings, "summary_json_mode", True):
        try:
            rsp = summary_client.chat.completions.create(
                model=model,
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "Jesteś zwięzłym ekstrakcyjnym streszczaczem."},
                    {"role": "user", "content": SUMMARY_PROMPT_JSON + "\n\nTEKST:\n" + text},
                ],
                max_tokens=max_tokens,
            )
            content = rsp.choices[0].message.content or "{}"
            data = json.loads(content)
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
            if summary_val:
                return {
                    "summary": summary_val,
                    "signature": signature_list,
                    "entities": entities_str,
                    "replacement": replacement_str,
                }
        except Exception:
            pass

    rsp = summary_client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": "Jesteś zwięzłym ekstrakcyjnym streszczaczem."},
            {"role": "user", "content": SUMMARY_PROMPT + text},
        ],
        max_tokens=max_tokens,
    )
    out = rsp.choices[0].message.content or ""
    summary = ""
    signature_list: List[str] = []
    entities_str = ""
    replacement_str = "brak"
    for line in out.splitlines():
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
    return {
        "summary": summary,
        "signature": signature_list,
        "entities": entities_str,
        "replacement": replacement_str,
    }
