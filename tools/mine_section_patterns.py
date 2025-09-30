#!/usr/bin/env python3
# mine_section_patterns.py
"""
Section pattern miner for Polish documents.

Cel:
- Samodzielnie wykrywać hierarchię jednostek redakcyjnych w polskich aktach prawnych i dokumentach.
- Wydobywać na podstawie heurystyk powtarzające się markery sekcji (np. Rozdział, Art., §, ust., pkt, lit.).
- LLM (OpenAI-compatible API) generuje poprawne regexy dla wykrytych markerów.
- Wynik: snippet Python SECTION_LEVELS i LEVEL_PATTERNS.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import pathlib
import re
import sys
from collections import Counter
from typing import Any, Dict, List, Tuple

# Reuse ingest scanning and extraction
from app.core.parsing import SUPPORTED_EXT, extract_text
from app.settings import get_settings

settings = get_settings()

# Hierarchia referencyjna (stała kolejność)
HIERARCHY_ORDER = ["doc", "chapter", "art", "par", "ust", "pkt", "lit", "dash"]

# === I/O helpers ===
def sample_doc(text: str, max_chars: int = 120_000) -> str:
    if len(text) <= max_chars:
        return text
    lines = text.splitlines()
    pat = re.compile(r"(§\s*\d|art\.|ust\.|pkt\.|lit\.|rozdział|dział)", re.IGNORECASE)
    head = "\n".join(lines[:400])
    sig = [ln for ln in lines if pat.search(ln)]
    sig = sig[:4000]
    tail = "\n".join(lines[-300:])
    out = "\n".join([head, "\n...\n", "\n".join(sig[:3000]), "\n...\n", tail])
    return out[:max_chars]


def scan_files(base: pathlib.Path, glob: str, recursive: bool) -> List[pathlib.Path]:
    iterator = base.rglob(glob) if recursive else base.glob(glob)
    return [p for p in iterator if p.is_file() and p.suffix.lower() in SUPPORTED_EXT]


def _read_for_mining(path: pathlib.Path, max_chars: int) -> str:
    try:
        txt = extract_text(path)
        return sample_doc(txt, max_chars=max_chars)
    except Exception:
        return ""


# === Heurystyka: odkrywanie markerów ===
def heuristic_detect(files: List[pathlib.Path], max_chars: int) -> Tuple[List[str], Dict[str, str]]:
    marker_rx = re.compile(r"^\s*([A-ZŁŚŻŹĆÓa-z0-9§\-–—\(\)]+)", re.MULTILINE)
    counter = Counter()

    for p in files:
        text = _read_for_mining(p, max_chars=max_chars)
        if not text:
            continue
        for m in marker_rx.findall(text):
            tok = m.strip().lower()
            if not tok:
                continue
            if tok.startswith("rozdział"):
                tok = "rozdział"
            elif tok.startswith("dział"):
                tok = "dział"
            elif tok.startswith("art"):
                tok = "art"
            elif tok.startswith("§"):
                tok = "§"
            elif tok.startswith("ust"):
                tok = "ust"
            elif tok.startswith("pkt"):
                tok = "pkt"
            elif tok.startswith("lit"):
                tok = "lit"
            elif tok.startswith("-") or tok.startswith("–") or tok.startswith("—"):
                tok = "dash"
            counter[tok] += 1

    common = [tok for tok, cnt in counter.most_common() if cnt >= 3]
    levels, patterns = [], {}
    for candidate in HIERARCHY_ORDER:
        if candidate == "chapter" and ("rozdział" in common or "dział" in common):
            levels.append("chapter")
            patterns["chapter"] = r"^\s*(rozdział|dział)\s+([IVXLCDM]+|\d+)"
        if candidate == "art" and "art" in common:
            levels.append("art")
            patterns["art"] = r"^\s*art\.?\s*(\d+[a-z]?)"
        if candidate == "par" and "§" in common:
            levels.append("par")
            patterns["par"] = r"^\s*§\s*(\d+[a-z]?)"
        if candidate == "ust" and "ust" in common:
            levels.append("ust")
            patterns["ust"] = r"\bust\.?\s*(\d+[a-z]?)\b"
        if candidate == "pkt" and "pkt" in common:
            levels.append("pkt")
            patterns["pkt"] = r"(\bpkt\.?\s*(\d+[a-z]?)|^\s*[\(\[]?\d+[a-z]?[\)\]])"
        if candidate == "lit" and "lit" in common:
            levels.append("lit")
            patterns["lit"] = r"\blit\.?\s*([a-z])[\)\.]"
        if candidate == "dash" and "dash" in common:
            levels.append("dash")
            patterns["dash"] = r"^\s*[-–—]\s+"
    return levels, patterns


# === LLM fallback ===
SYSTEM_PROMPT = (
    "Jesteś analitykiem redakcji prawnej. "
    "Na podstawie TEKSTU wykryj hierarchię sekcji i podaj regexy (Python, MULTILINE) do ich wykrywania. "
    "Zwróć czysty JSON, bez ``` ani komentarzy."
)

USER_INSTRUCTIONS = (
    "Format JSON:\n"
    "{\n"
    '  "levels_order": ["doc","chapter","art","par","ust","pkt","lit","dash"],\n'
    '  "levels": [\n'
    '    {"id":"chapter","regex":"^\\\\s*(rozdział|dział)\\\\s+([IVXLCDM]+|\\\\d+)"},\n'
    '    {"id":"art","regex":"^\\\\s*art\\\\.?\\\\s*(\\\\d+[a-z]?)"}\n'
    "  ]\n"
    "}\n"
    "Pomiń poziomy, które realnie nie występują."
)


def have_openai() -> bool:
    try:
        import openai  # noqa: F401
        return True
    except Exception:
        return False


def ask_llm_batch(files: List[pathlib.Path], model: str, max_chars: int, concurrency: int) -> List[Dict[str, Any]]:
    from openai import OpenAI
    client = OpenAI(base_url=settings.summary_api_url, api_key=settings.summary_api_key)

    def _one(p: pathlib.Path) -> Dict[str, Any]:
        text = _read_for_mining(p, max_chars=max_chars)
        rsp = client.chat.completions.create(
            model=model,
            temperature=0.0,
            max_tokens=600,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_INSTRUCTIONS + "\n\nTEKST:\n" + text},
            ],
        )
        content = rsp.choices[0].message.content or "{}"
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```[a-zA-Z]*", "", content)
            content = re.sub(r"```$", "", content)
            content = content.strip()
        try:
            return json.loads(content)
        except Exception:
            sys.stderr.write(f"[WARN] Invalid JSON from LLM for {p.name}: {content[:200]}\n")
            return {}

    out: List[Dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
        futs = [ex.submit(_one, p) for p in files]
        for f in concurrent.futures.as_completed(futs):
            try:
                out.append(f.result())
            except Exception as e:
                sys.stderr.write(f"[WARN] LLM file error: {e}\n")
    return out


def merge_llm_results(per_file: List[Dict[str, Any]]) -> Tuple[List[str], Dict[str, str]]:
    agg: Dict[str, Dict[str, int]] = {}
    for res in per_file:
        lvls = res.get("levels", []) or []
        for d in lvls:
            lvl = d.get("id")
            rgx = d.get("regex")
            if not lvl or not rgx:
                continue
            agg.setdefault(lvl, {})
            agg[lvl][rgx] = agg[lvl].get(rgx, 0) + 1

    patterns: Dict[str, str] = {}
    for lvl, bag in agg.items():
        best = sorted(bag.items(), key=lambda kv: (-kv[1], len(kv[0])))[0][0]
        patterns[lvl] = best

    levels = [lvl for lvl in HIERARCHY_ORDER if lvl in patterns]
    return levels, patterns


# === Output ===
SNIPPET_TEMPLATE = """# Auto-generated by tools/mine_section_patterns.py
SECTION_LEVELS = {levels!r}
LEVEL_PATTERNS = {{
{body}
}}
"""


def format_snippet(levels: List[str], patterns: Dict[str, str]) -> str:
    body_lines = [f'  "{lvl}": r"{patterns[lvl]}",' for lvl in levels if lvl in patterns]
    return SNIPPET_TEMPLATE.format(levels=levels, body="\n".join(body_lines))


# === CLI ===
def main() -> None:
    ap = argparse.ArgumentParser(description="Wydobądź hierarchię sekcji i regexy z korpusu.")
    ap.add_argument("root", type=pathlib.Path, help="Katalog bazowy z plikami")
    ap.add_argument("--glob", default="**/*", help="Wzorzec plików")
    ap.add_argument("--recursive", action="store_true", help="Użyj rglob")
    ap.add_argument("--max-chars", type=int, default=120_000, help="Limit znaków na plik")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("section_patterns.py"))
    ap.add_argument("--model", default=settings.summary_model)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--force-llm", action="store_true")
    args = ap.parse_args()

    if not args.root.exists():
        sys.exit(f"Brak katalogu: {args.root}")

    files = scan_files(args.root, args.glob, args.recursive)
    if not files:
        sys.exit("Nie znaleziono plików w obsługiwanych formatach.")

    levels_h, patterns_h = heuristic_detect(files, max_chars=args.max_chars)
    print("[heur] wykryte poziomy:", ", ".join(levels_h) or "(brak)")

    use_llm = args.force_llm or not levels_h
    levels_final, patterns_final = levels_h, patterns_h

    if use_llm and have_openai() and settings.summary_api_key:
        llm_res = ask_llm_batch(files, model=args.model, max_chars=args.max_chars, concurrency=args.concurrency)
        levels_l, patterns_l = merge_llm_results(llm_res)
        merged_levels, merged_patterns = [], {}
        for lvl in HIERARCHY_ORDER:
            if lvl in patterns_l:
                merged_levels.append(lvl)
                merged_patterns[lvl] = patterns_l[lvl]
            elif lvl in patterns_h:
                merged_levels.append(lvl)
                merged_patterns[lvl] = patterns_h[lvl]
        levels_final, patterns_final = merged_levels, merged_patterns
        print("[LLM] Wynik scalony (LLM + heurystyka).")

    if not patterns_final:
        print("[warn] Brak wykrytych wzorców – zapisuję domyślne.")
        levels_final = ["doc"]
        patterns_final = {"doc": r".*"}

    if "doc" not in levels_final:
        levels_final.insert(0, "doc")

    snippet = format_snippet(levels_final, patterns_final)
    try:
        args.out.write_text(snippet, encoding="utf-8")
        print(snippet)
        print(f"[OK] Zapisano: {args.out}")
    except Exception as exc:
        print(snippet)
        print(f"[WARN] Nie udało się zapisać pliku: {exc}")


if __name__ == "__main__":
    main()

