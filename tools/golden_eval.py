"""Golden QA evaluator — LLM‑only (semantic judging).

This evaluator compares model predictions against a golden QA set using an
LLM‑as‑judge. It does NOT use deterministic string metrics (exact/F1/numeric).
The judge assesses semantic correctness and completeness; paraphrases are OK.

Environment (OpenAI‑compatible; same as generator):
  GOLDEN_LLM_BASE_URL
  GOLDEN_LLM_API_KEY
  GOLDEN_LLM_MODEL

Usage
  python tools/golden_eval.py \
    --golden /path/to/golden_qa.jsonl \
    --pred /path/to/predictions.jsonl \
    --out /path/to/eval_report.json \
    --max-judge 1000 --judge-sleep-ms 20 --preview 20

Input formats
- golden_qa.jsonl: lines with at least {id, query, expected_answer, meta?}
- predictions.jsonl: lines {id, answer}

Output (JSON)
- judge: enabled, total, averages (overall/correctness/completeness),
  citation_ok_rate, preview (first N items with rationale).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

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
                i = s.find("{")
                j = s.rfind("}")
                if i != -1 and j != -1 and j > i:
                    try:
                        out.append(json.loads(s[i : j + 1]))
                    except Exception:
                        pass
    return out


def _llm_client_from_env() -> Optional[Any]:
    if OpenAI is None:
        return None
    base = os.environ.get("GOLDEN_LLM_BASE_URL")
    key = os.environ.get("GOLDEN_LLM_API_KEY")
    model = os.environ.get("GOLDEN_LLM_MODEL")
    if not (base and key and model):
        return None
    try:
        return OpenAI(base_url=base, api_key=key)
    except Exception:
        return None


_JUDGE_PROMPT = (
    "Jesteś sędzią odpowiedzi. Oceń semantyczną zgodność i kompletność odpowiedzi "
    "modelu względem złotej odpowiedzi. Parafrazy są akceptowalne. Liczby, daty i jednostki "
    "muszą pozostać równoważne. Weź pod uwagę kontekst czasowy 'as_of' (jeśli podany) — odpowiedź powinna być zgodna ze stanem na ten moment. "
    "Jeśli w odpowiedzi występuje symbol dokumentu, możesz to odnotować. "
    "Zwróć WYŁĄCZNIE JSON bez komentarzy: {\n"
    "  \"verdict\": one_of[\"correct\",\"partial\",\"incorrect\",\"nonanswer\"],\n"
    "  \"score_overall\": number,\n"
    "  \"score_correctness\": number,\n"
    "  \"score_completeness\": number,\n"
    "  \"citation_ok\": boolean,\n"
    "  \"rationale\": string\n"
    "}"
)


def _normalize_symbol(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s = s.strip()
    return s or None


def _judge_one(
    client: Any,
    model: str,
    *,
    query: str,
    golden_ans: str,
    pred_ans: str,
    doc_symbol: Optional[str],
    as_of: Optional[str],
) -> Dict[str, Any]:
    try:
        payload = {
            "query": query,
            "golden_answer": golden_ans,
            "predicted_answer": pred_ans,
            "doc_symbol": doc_symbol or None,
            "as_of": as_of or None,
        }
        rsp = client.chat.completions.create(
            model=model,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _JUDGE_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            max_tokens=220,
        )
        content = rsp.choices[0].message.content or "{}"
        data = json.loads(content)
        out = {
            "score_overall": float(data.get("score_overall", 0.0) or 0.0),
            "score_correctness": float(data.get("score_correctness", 0.0) or 0.0),
            "score_completeness": float(data.get("score_completeness", 0.0) or 0.0),
            "citation_ok": bool(data.get("citation_ok", False)),
            "verdict": str(data.get("verdict", "")).strip() or None,
            "rationale": str(data.get("rationale", "")).strip() or None,
        }
        # Clamp
        for k in ("score_overall", "score_correctness", "score_completeness"):
            v = out[k]
            if v < 0.0:
                out[k] = 0.0
            elif v > 1.0:
                out[k] = 1.0
        return out
    except Exception:
        return {"score_overall": 0.0, "score_correctness": 0.0, "score_completeness": 0.0, "citation_ok": False, "verdict": None, "rationale": None}


def evaluate_llm(
    golden_path: Path,
    pred_path: Path,
    *,
    max_judge: Optional[int] = None,
    judge_sleep_ms: int = 0,
    preview: int = 20,
) -> Dict[str, Any]:
    client = _llm_client_from_env()
    model = os.environ.get("GOLDEN_LLM_MODEL") if client else None
    if not (client and model):
        raise RuntimeError("LLM is not configured. Set GOLDEN_LLM_BASE_URL, GOLDEN_LLM_API_KEY, GOLDEN_LLM_MODEL.")

    golden = _read_jsonl(golden_path)
    preds_list = _read_jsonl(pred_path)
    preds: Dict[str, str] = {str(it.get("id")): str(it.get("answer", "") or "") for it in preds_list if it.get("id") is not None}

    judged = 0
    sum_overall = 0.0
    sum_corr = 0.0
    sum_comp = 0.0
    cit_ok = 0
    cit_total = 0
    preview_items: List[Dict[str, Any]] = []

    for g in golden:
        if max_judge is not None and judged >= max_judge:
            break
        qid = str(g.get("id", "") or "")
        if not qid:
            continue
        query = str(g.get("query", "") or "")
        gold = str(g.get("expected_answer", "") or "")
        guess = preds.get(qid, "")
        if not query:
            continue
        meta = g.get("meta") if isinstance(g.get("meta"), dict) else {}
        sym = _normalize_symbol(meta.get("doc_symbol")) if isinstance(meta, dict) else None
        as_of = str(meta.get("as_of")) if isinstance(meta, dict) and meta.get("as_of") else None

        res = _judge_one(client, model, query=query, golden_ans=gold, pred_ans=guess, doc_symbol=sym, as_of=as_of)
        judged += 1
        sum_overall += float(res.get("score_overall", 0.0))
        sum_corr += float(res.get("score_correctness", 0.0))
        sum_comp += float(res.get("score_completeness", 0.0))
        if res.get("citation_ok"):
            cit_ok += 1
        if sym:
            cit_total += 1

        if len(preview_items) < max(0, int(preview)):
            preview_items.append(
                {
                    "id": qid,
                    "query": query,
                    "golden": gold,
                    "pred": guess,
                    "judge": res,
                }
            )

        if judge_sleep_ms:
            import time as _t

            _t.sleep(max(0, int(judge_sleep_ms)) / 1000.0)

    report = {
        "summary": {
            "total_questions": len(golden),
            "total_predictions": len(preds),
            "total_judged": judged,
            "avg_overall": (sum_overall / judged) if judged else None,
            "avg_correctness": (sum_corr / judged) if judged else None,
            "avg_completeness": (sum_comp / judged) if judged else None,
            "citation_ok_rate": (cit_ok / cit_total) if cit_total else None,
        },
        "judge": {
            "enabled": True,
            "total": judged,
            "avg_overall": (sum_overall / judged) if judged else None,
            "avg_correctness": (sum_corr / judged) if judged else None,
            "avg_completeness": (sum_comp / judged) if judged else None,
            "citation_ok_rate": (cit_ok / cit_total) if cit_total else None,
            "preview": preview_items,
        },
    }
    return report


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Golden QA evaluator — LLM‑only")
    p.add_argument("--golden", required=True, help="Path to golden_qa.jsonl")
    p.add_argument("--pred", required=True, help="Path to predictions.jsonl")
    p.add_argument("--out", required=True, help="Where to save JSON report")
    p.add_argument("--max-judge", type=int, default=None, help="Max items to judge (cost control)")
    p.add_argument("--judge-sleep-ms", type=int, default=0, help="Sleep between judge calls (ms)")
    p.add_argument("--preview", type=int, default=20, help="Preview item count in report")

    args = p.parse_args(argv)
    golden_path = Path(args.golden)
    pred_path = Path(args.pred)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rep = evaluate_llm(
        golden_path,
        pred_path,
        max_judge=(int(args.max_judge) if args.max_judge is not None else None),
        judge_sleep_ms=max(0, int(args.judge_sleep_ms)),
        preview=max(0, int(args.preview)),
    )
    out_path.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "out": str(out_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(main())
