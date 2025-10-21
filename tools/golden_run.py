"""Predictions runner for golden QA (LLM with optional native tool-calling).

This CLI reads a golden_qa.jsonl file, queries an OpenAI‑compatible endpoint
with each `query`, and writes predictions.jsonl in the shape expected by the
evaluator (one JSON per line: {"id","answer"}).

The runner is intentionally simple and system‑agnostic: it assumes your LLM
endpoint handles tools/plugin orchestration internally. If you need custom
headers/params, extend the call site below.

Env (OpenAI‑compatible):
  EVAL_BASE_URL  (e.g., http://127.0.0.1:8001/v1)
  EVAL_API_KEY   (e.g., sk-...)
  EVAL_MODEL     (e.g., gpt-4o-mini)
  EVAL_SYSTEM_PROMPT (optional; preferred; system message to send) — alias: GOLDEN_RUN_SYSTEM_PROMPT
  EVAL_FINAL_JSON_KEY (optional; preferred; read this key from JSON content) — alias: GOLDEN_RUN_FINAL_JSON_KEY
  EVAL_CLEAN (optional; preferred; "1"/"true" to strip tool JSON/code fences) — alias: GOLDEN_RUN_CLEAN

Tool-calling (optional, --use-tools):
  EVAL_SEARCH_URL  (default: http://127.0.0.1:8000)
  EVAL_SEARCH_PATH (default: /search/query)
  EVAL_SEARCH_TOP_K (default: 3) — how many blocks to summarize for the tool result
  EVAL_TOOL_MAX_CHARS (default: 3000) — max chars passed back as tool result

Usage
  python tools/golden_run.py \
    --golden /path/to/golden_qa.jsonl \
    --out /path/to/predictions.jsonl \
    --sleep-ms 20
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            try:
                out.append(json.loads(s))
            except Exception:
                # tolerate JSONish
                i = s.find("{")
                j = s.rfind("}")
                if i != -1 and j != -1 and j > i:
                    try:
                        out.append(json.loads(s[i : j + 1]))
                    except Exception:
                        pass
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Golden predictions runner (OpenAI‑compatible)")
    p.add_argument("--golden", required=True, help="Path to golden_qa.jsonl")
    p.add_argument("--out", required=True, help="Output predictions.jsonl path")
    p.add_argument("--sleep-ms", type=int, default=20, help="Sleep between calls (ms)")
    p.add_argument("--use-tools", action="store_true", help="Enable native tool-calling (search) and tool loop")
    p.add_argument("--debug", action="store_true", help="Print step-by-step debug for first N items")
    p.add_argument("--debug-limit", type=int, default=1, help="How many items to debug (default: 1)")

    args = p.parse_args(argv)
    golden_path = Path(args.golden)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    base = os.getenv("EVAL_BASE_URL", "http://127.0.0.1:8001/v1")
    key = os.getenv("EVAL_API_KEY", "sk-no-key")
    model = os.getenv("EVAL_MODEL", "gpt-4o-mini")
    if OpenAI is None:
        raise RuntimeError("openai package is not available. Install `openai` first.")
    client = OpenAI(base_url=base, api_key=key)

    def _parse_jsonish(s: str) -> Optional[Dict[str, Any]]:
        s = (s or "").strip()
        if not s:
            return None
        # try direct
        try:
            val = json.loads(s)
            if isinstance(val, dict):
                return val
        except Exception:
            pass
        # try slice between first { and last }
        try:
            i = s.find("{")
            j = s.rfind("}")
            if i != -1 and j != -1 and j > i:
                sub = s[i:j+1]
                val = json.loads(sub)
                if isinstance(val, dict):
                    return val
        except Exception:
            pass
        return None

    def _clean_answer(raw: str) -> str:
        import re as _re
        s = raw or ""
        # Remove code fences (multiline)
        s = _re.sub(r"```.*?```", "", s, flags=_re.S)
        # Remove common English planning/preamble lines (agent traces)
        s = _re.sub(r"^\s*We\s+(will|need to|are going to)[^\n]*\n?", "", s, flags=_re.I|_re.M)
        s = _re.sub(r"^\s*Search\s+query:[^\n]*\n?", "", s, flags=_re.I|_re.M)
        # Iteratively strip JSON-like blocks to handle nesting (aggressive but effective)
        prev = None
        while prev != s:
            prev = s
            s = _re.sub(r"\{[^{}]*\}", "", s, flags=_re.S)
        # Prefer the content after a labeled answer marker (Polish)
        m = _re.search(r"(Odpowiedź\s*:|\*\*Odpowiedź\s*:\s*\*\*)\s*(.*)$", s, flags=_re.I|_re.S)
        if m:
            s = m.group(2)
        # Drop citation/source sections if present
        s = _re.split(r"\*\*\s*(Cytat|Źródło)\s*:|Cytat\s*:|Źródło\s*:", s, maxsplit=1, flags=_re.I)[0]
        # Final whitespace collapse
        s = _re.sub(r"\s+", " ", s).strip()
        return s or "nie wiem"

    SYS_PROMPT = os.getenv("EVAL_SYSTEM_PROMPT") or os.getenv("GOLDEN_RUN_SYSTEM_PROMPT")
    FINAL_KEY = os.getenv("EVAL_FINAL_JSON_KEY") or os.getenv("GOLDEN_RUN_FINAL_JSON_KEY")
    CLEAN = (os.getenv("EVAL_CLEAN", "") or os.getenv("GOLDEN_RUN_CLEAN", "")).lower() in {"1", "true", "yes"}

    # --- Plain ask (no tools) ---
    def _dbg(enabled: bool, *parts: Any) -> None:
        if enabled:
            print("[DEBUG]", *parts, file=sys.stderr, flush=True)

    def ask_plain(text: str, *, debug: bool = False) -> str:
        msgs = []
        if SYS_PROMPT and SYS_PROMPT.strip():
            msgs.append({"role": "system", "content": SYS_PROMPT})
        msgs.append({"role": "user", "content": text})
        _dbg(debug, "ask_plain.messages", json.dumps(msgs, ensure_ascii=False))
        rsp = client.chat.completions.create(
            model=model,
            temperature=0.0,
            messages=msgs,
        )
        content = (rsp.choices[0].message.content or "").strip()
        _dbg(debug, "ask_plain.raw_content", content)
        # optional JSON extraction of final answer
        if FINAL_KEY:
            obj = _parse_jsonish(content)
            if isinstance(obj, dict):
                val = obj.get(FINAL_KEY)
                if isinstance(val, (str, int, float)):
                    _dbg(debug, "ask_plain.final_key_hit", FINAL_KEY, "=", val)
                    return str(val).strip()
        # optional cleaning
        if CLEAN:
            cleaned = _clean_answer(content)
            _dbg(debug, "ask_plain.cleaned", cleaned)
            return cleaned
        return content

    # --- Tool-calling support (search) ---
    def _http_post_json(url: str, payload: Dict[str, Any], timeout: int = 60) -> Any:
        data = json.dumps(payload).encode("utf-8")
        req = Request(url, data=data, headers={"Accept": "application/json", "Content-Type": "application/json"}, method="POST")
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            try:
                return json.loads(raw.decode("utf-8"))
            except Exception:
                return {}

    SEARCH_BASE = os.getenv("EVAL_SEARCH_URL", "http://127.0.0.1:8000")
    SEARCH_PATH = os.getenv("EVAL_SEARCH_PATH", "/search/query")
    SEARCH_TOP_K = int(os.getenv("EVAL_SEARCH_TOP_K", "3") or 3)
    TOOL_MAX_CHARS = int(os.getenv("EVAL_TOOL_MAX_CHARS", "3000") or 3000)

    def _tool_search(arguments: Dict[str, Any], *, debug: bool = False) -> str:
        # Accept string or list
        q = arguments.get("query")
        if isinstance(q, str):
            q_list = [q]
        elif isinstance(q, list):
            q_list = [str(x) for x in q if str(x).strip()]
        else:
            q_list = []
        body = {
            "query": q_list[:3] if q_list else [],
            "top_m": max(10, SEARCH_TOP_K * 5),
            "top_k": max(1, SEARCH_TOP_K),
            "mode": "auto",
            "use_hybrid": True,
            "per_doc_limit": 2,
            "result_format": "blocks",
            "summary_mode": "first",
        }
        _dbg(debug, "tool_search.request", SEARCH_BASE.rstrip("/") + SEARCH_PATH, json.dumps(body, ensure_ascii=False))
        try:
            res = _http_post_json(SEARCH_BASE.rstrip("/") + SEARCH_PATH, body, timeout=60)
        except Exception as exc:
            _dbg(debug, "tool_search.error", str(exc))
            return f"(search_error: {exc})"
        # Summarize top blocks into compact textual context for the model
        blocks = res.get("blocks") if isinstance(res, dict) else None
        if not isinstance(blocks, list):
            _dbg(debug, "tool_search.raw_response_no_blocks", str(res)[:400])
            return json.dumps(res, ensure_ascii=False)[:TOOL_MAX_CHARS]
        parts: List[str] = []
        for b in blocks[:SEARCH_TOP_K]:
            title = str(b.get("title") or "")
            doc_date = str(b.get("doc_date") or "")
            section = str(b.get("section") or "")
            text = str(b.get("text") or "")
            head = text[: min(len(text), TOOL_MAX_CHARS // SEARCH_TOP_K - 128)]
            parts.append(f"Tytuł: {title} | Data: {doc_date} | Sekcja: {section}\n{head}")
        payload = "\n\n---\n\n".join(parts)
        _dbg(debug, "tool_search.payload_len", len(payload))
        return payload[:TOOL_MAX_CHARS]

    TOOLS_SPEC = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Przeszukaj korpus dokumentów i zwróć najtrafniejsze fragmenty do odpowiedzi.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"oneOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}]},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        }
    ]

    def _extract_inline_search(content: str, *, debug: bool = False) -> Optional[Dict[str, Any]]:
        """Heuristically extract search arguments from a textual tool intent.

        Supports forms like:
        - "...{\"search\": {\"query\": "..."}}"
        - "...{\"query\": ["...", "..."]}"
        - lines like: "Search query: \"...\""
        Returns a dict with key 'query' when detected, else None.
        """
        import re as _re
        s = content or ""
        # Try JSON object inside content (first {...})
        i = s.find("{")
        j = s.rfind("}")
        if i != -1 and j != -1 and j > i:
            try:
                obj = json.loads(s[i : j + 1])
                if isinstance(obj, dict):
                    if "search" in obj and isinstance(obj["search"], dict):
                        sub = obj["search"]
                    else:
                        sub = obj
                    q = sub.get("query")
                    if isinstance(q, str) and q.strip():
                        _dbg(debug, "inline_search.detected", q)
                        return {"query": q}
                    if isinstance(q, list):
                        lst = [str(x) for x in q if str(x).strip()]
                        if lst:
                            _dbg(debug, "inline_search.detected_list", lst)
                            return {"query": lst}
            except Exception:
                pass
        # Try "Search query: ..."
        m = _re.search(r"Search\s+query\s*:\s*\"(.+?)\"", s, flags=_re.I)
        if m:
            _dbg(debug, "inline_search.detected_heuristic", m.group(1))
            return {"query": m.group(1)}
        return None

    def ask_with_tools(text: str, *, debug: bool = False) -> str:
        msgs: List[Dict[str, Any]] = []
        if SYS_PROMPT and SYS_PROMPT.strip():
            msgs.append({"role": "system", "content": SYS_PROMPT})
        msgs.append({"role": "user", "content": text})
        _dbg(debug, "ask_with_tools.messages.0", json.dumps(msgs, ensure_ascii=False))

        # First call (tool_choice=auto)
        rsp = client.chat.completions.create(
            model=model,
            temperature=0.0,
            messages=msgs,
            tools=TOOLS_SPEC,
            tool_choice="auto",
        )
        choice = rsp.choices[0]
        msg = choice.message
        tool_calls = getattr(msg, "tool_calls", None)
        _dbg(debug, "ask_with_tools.first_reply", getattr(msg, "content", None), "tool_calls:", bool(tool_calls))
        if not tool_calls:
            content = (msg.content or "").strip()
            # Fallback: inline textual tool intent (JSON or 'Search query:')
            inline = _extract_inline_search(content, debug=debug)
            if inline:
                # Execute and ask follow-up with context
                result = _tool_search(inline, debug=debug)
                # Append assistant response and a user follow-up with tool context
                msgs.append({"role": "assistant", "content": content})
                msgs.append({
                    "role": "user",
                    "content": f"Kontekst wyszukiwania (skrót):\n{result}\n\nNa podstawie powyższego udziel teraz zwięzłej odpowiedzi po polsku.",
                })
                _dbg(debug, "ask_with_tools.followup.messages", json.dumps(msgs[-2:], ensure_ascii=False))
                rspf = client.chat.completions.create(
                    model=model,
                    temperature=0.0,
                    messages=msgs,
                )
                out = (rspf.choices[0].message.content or "").strip()
                _dbg(debug, "ask_with_tools.followup.raw", out)
                if FINAL_KEY:
                    obj = _parse_jsonish(out)
                    if isinstance(obj, dict):
                        v = obj.get(FINAL_KEY)
                        if isinstance(v, (str, int, float)):
                            _dbg(debug, "ask_with_tools.followup.final_key_hit", FINAL_KEY, "=", v)
                            return str(v).strip()
                if CLEAN:
                    cleaned = _clean_answer(out)
                    _dbg(debug, "ask_with_tools.followup.cleaned", cleaned)
                    return cleaned
                return out
            if FINAL_KEY:
                obj = _parse_jsonish(content)
                if isinstance(obj, dict):
                    v = obj.get(FINAL_KEY)
                    if isinstance(v, (str, int, float)):
                        return str(v).strip()
            return _clean_answer(content) if CLEAN else content

        # Execute each tool call once; then final follow-up
        msgs.append({"role": "assistant", "tool_calls": [
            {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in tool_calls
        ]})
        for tc in tool_calls:
            name = tc.function.name
            args_raw = tc.function.arguments or "{}"
            try:
                args = json.loads(args_raw)
            except Exception:
                args = {"query": args_raw}
            _dbg(debug, "ask_with_tools.tool_call", name, args)
            if name == "search":
                result = _tool_search(args, debug=debug)
            else:
                result = f"(unknown_tool: {name})"
            msgs.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": name,
                "content": result,
            })

        # Follow-up to obtain final answer
        rsp2 = client.chat.completions.create(
            model=model,
            temperature=0.0,
            messages=msgs,
        )
        content = (rsp2.choices[0].message.content or "").strip()
        _dbg(debug, "ask_with_tools.final.raw", content)
        if FINAL_KEY:
            obj = _parse_jsonish(content)
            if isinstance(obj, dict):
                v = obj.get(FINAL_KEY)
                if isinstance(v, (str, int, float)):
                    _dbg(debug, "ask_with_tools.final.final_key_hit", FINAL_KEY, "=", v)
                    return str(v).strip()
        if CLEAN:
            cleaned = _clean_answer(content)
            _dbg(debug, "ask_with_tools.final.cleaned", cleaned)
            return cleaned
        return content

    items = _read_jsonl(golden_path)
    written = 0
    with out_path.open("w", encoding="utf-8") as fo:
        for idx, it in enumerate(items):
            qid = it.get("id")
            q = it.get("query")
            if not qid or not q:
                continue
            try:
                dbg = bool(args.debug) and (idx < int(args.debug_limit))
                if args.use_tools:
                    ans = ask_with_tools(str(q), debug=dbg)
                else:
                    ans = ask_plain(str(q), debug=dbg)
            except Exception as exc:
                ans = f"(error: {exc})"
            fo.write(json.dumps({"id": qid, "answer": ans}, ensure_ascii=False) + "\n")
            written += 1
            if args.sleep_ms:
                time.sleep(max(0, int(args.sleep_ms)) / 1000.0)
    print(json.dumps({"predictions": written, "out": str(out_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(main())
