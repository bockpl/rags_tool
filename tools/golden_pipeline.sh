#!/usr/bin/env bash
set -euo pipefail

# Golden QA end-to-end pipeline (LLM-only generation → predictions → LLM-judge eval)
#
# Konfiguracja odbywa się przez zmienne środowiskowe. Poniżej wartości domyślne.

# === LLM do generacji pytań i sędziego (ten sam endpoint) ===
: "${GOLDEN_LLM_BASE_URL:=https://ai.p.lodz.pl/api}"
: "${GOLDEN_LLM_API_KEY:=sk-92137ab5f8964f00a1a903d15344c04f}"
: "${GOLDEN_LLM_MODEL:=gpt-oss:120b}"
export GOLDEN_LLM_BASE_URL GOLDEN_LLM_API_KEY GOLDEN_LLM_MODEL

# === Badany model (LLM do odpowiedzi) ===
: "${EVAL_BASE_URL:=https://ai.test.p.lodz.pl/api}"
: "${EVAL_API_KEY:=sk-c1cab53cccfb47bea1328fd7b4be9c74}"
: "${EVAL_MODEL:=bielikp-duy}"
export EVAL_BASE_URL EVAL_API_KEY EVAL_MODEL

# Czyszczenie odpowiedzi i (opcjonalnie) klucz JSON z finalną odpowiedzią
: "${EVAL_CLEAN:=1}"
: "${EVAL_FINAL_JSON_KEY:=}"
export EVAL_CLEAN EVAL_FINAL_JSON_KEY

# === (Opcjonalne) narzędzia: search ===
: "${USE_TOOLS:=1}"
: "${EVAL_SEARCH_URL:=http://172.22.1.5:7001}"
: "${EVAL_SEARCH_PATH:=/search/query}"
: "${EVAL_SEARCH_TOP_K:=3}"
: "${EVAL_TOOL_MAX_CHARS:=3000}"
export EVAL_SEARCH_URL EVAL_SEARCH_PATH EVAL_SEARCH_TOP_K EVAL_TOOL_MAX_CHARS

# === Dane i parametry ===
: "${BASE_DIR:=data/Zarządzenia_JM_Rektora}"
: "${OUT_DIR:=data/golden}"
: "${GLOB:=**/*.txt}"
: "${PER_DOC_QA:=2}"
: "${TARGET_QA:=30}"
: "${SLEEP_MS:=20}"
# Limit kosztu sędziego (opcjonalnie)
: "${MAX_JUDGE:=}"

mkdir -p "$OUT_DIR"

echo "== [1/3] Golden set (LLM-only) =="
python tools/golden_make.py \
  --base-dir "$BASE_DIR" \
  --glob "$GLOB" \
  --recursive \
  --out-dir "$OUT_DIR" \
  --per-doc-qa "$PER_DOC_QA" \
  --target-qa "$TARGET_QA" \
  --seed 123

GOLDEN_FILE="$OUT_DIR/golden_qa.jsonl"
PRED_FILE="$OUT_DIR/predictions.jsonl"
REPORT_FILE="$OUT_DIR/eval_report.json"

echo "== [2/3] Predykcje badanego modelu =="
if [[ "$USE_TOOLS" == "1" ]]; then
  python tools/golden_run.py \
    --golden "$GOLDEN_FILE" \
    --out "$PRED_FILE" \
    --sleep-ms "$SLEEP_MS" \
    --use-tools
else
  python tools/golden_run.py \
    --golden "$GOLDEN_FILE" \
    --out "$PRED_FILE" \
    --sleep-ms "$SLEEP_MS"
fi

echo "== [3/3] Ewaluacja (LLM-as-judge) =="
JUDGE_ARGS=()
if [[ -n "${MAX_JUDGE}" ]]; then
  JUDGE_ARGS+=(--max-judge "$MAX_JUDGE")
fi
python tools/golden_eval.py \
  --golden "$GOLDEN_FILE" \
  --pred "$PRED_FILE" \
  --out "$REPORT_FILE" \
  "${JUDGE_ARGS[@]}"

echo "\nGotowe. Raport (zawiera summary): $REPORT_FILE"
echo "Predykcje: $PRED_FILE"
echo "Golden: $GOLDEN_FILE"
