"""
RAGS_tool — dwustopniowy RAG ze streszczeniami (FastAPI + Qdrant)
Version: 0.4.0
Author: Seweryn Sitarski (seweryn.sitarski@gmail.com) with support from Kat
License: MIT

ZAŁOŻENIA
- Narzędzie OpenAPI dla LLM (OpenWebUI / OpenAI Tools):
  • /about — metadane
  • /health — zdrowie serwisu
  • /collections/init — utworzenie/aktualizacja kolekcji w Qdrant
  • /ingest/scan — skan katalogu
  • /ingest/build — przetworzenie korpusu: parsowanie → streszczenia → wektory → indeks
  • /summaries/generate — streszczenia dla pojedynczych plików
  • /search/query — główne odpytywanie (dwustopniowe): streszczenia → pełny tekst (z hybrydą dense+tfidf)

- Wejście: katalog z dokumentami (TXT/MD/HTML; PDF opcjonalnie jeśli dostępny PyPDF2) + instancja Qdrant.
- Embedding i streszczenia: modele dostępne przez protokół OpenAI (np. vLLM/Ollama z endpointem OpenAI).
- Indeks Qdrant jako multivektor: "content_dense", "summary_dense" + (opcjonalnie) sparse TF‑IDF.
- Brak re-rankera cross-encoder (opcjonalny TODO) — używamy fuzji wyników i MMR.

URUCHOMIENIE
  export QDRANT_URL=http://127.0.0.1:6333
  export QDRANT_API_KEY=""
  export EMBEDDING_API_URL=http://127.0.0.1:8000/v1   # Twój endpoint embedding
  export EMBEDDING_API_KEY=sk-embed-xxx
  export EMBEDDING_MODEL="BAAI/bge-m3"               # lub inny zgodny z /v1/embeddings
  export SUMMARY_API_URL=http://127.0.0.1:8001/v1     # Twój endpoint do streszczeń
  export SUMMARY_API_KEY=sk-summary-xxx
  export SUMMARY_MODEL="gpt-4o-mini"                 # lub lokalny chat model zgodny z /v1/chat/completions
  export COLLECTION_NAME="rags_tool"
  export VECTOR_STORE_DIR=".rags_tool_store"         # gdzie zapisze się TF‑IDF, cache itp.
  export DEBUG=false                                  # ustaw na true, by zobaczyć logi ingestu

  # (alternatywnie) wpisz te wartości do pliku .env — aplikacja wczyta go automatycznie

  pip install fastapi uvicorn qdrant-client openai pydantic pydantic-settings tiktoken scikit-learn markdown2 beautifulsoup4 html2text
  # PyPDF2 (opcjonalnie): pip install pypdf2

  uvicorn app:app --host 0.0.0.0 --port 8080

OpenAPI będzie pod /openapi.json, Swagger UI pod /docs — gotowe do wpięcia w OpenWebUI Tools (OpenAPI).
"""

import json
import time
import uuid
import hashlib
import pathlib
import logging
import sys
from typing import List, Optional, Dict, Any, Tuple

from fastapi import FastAPI, Body, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, Field

# Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from qdrant_client.http.exceptions import UnexpectedResponse

# OpenAI‑compatible klienci
from openai import OpenAI

# Tekst / parsowanie
import re
import html
import html2text
import markdown2

# PDF (opcjonalne)
try:
    import PyPDF2  # type: ignore
    HAS_PDF = True
except Exception:
    HAS_PDF = False

# TF‑IDF (sparse hybryda)
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

from settings import get_settings

# ──────────────────────────────────────────────────────────────────────────────
# KONFIG
# ──────────────────────────────────────────────────────────────────────────────
settings = get_settings()
VECTOR_STORE_DIR = settings.vector_store_dir
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

# Stałe
CONTENT_VECTOR_NAME = "content_dense"
SUMMARY_VECTOR_NAME = "summary_dense"
CONTENT_SPARSE_NAME = "content_sparse"
SUMMARY_SPARSE_NAME = "summary_sparse"
SPARSE_ENABLED = True  # TF‑IDF hybryda

# Logger skonfigurowany pod konsolę kontenera
logger = logging.getLogger("rags_tool")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )
    logger.addHandler(handler)
logger.setLevel(logging.DEBUG if settings.debug else logging.INFO)
logger.propagate = False

ADMIN_UI_REQUEST_HEADER = "x-admin-ui"
_admin_debug_activated = False

ADMIN_UI_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"utf-8\" />
    <title>SummRAG Admin Console</title>
    <style>
        :root { color-scheme: light dark; font-family: system-ui, sans-serif; }
        body { margin: 0; padding: 24px; background: #f5f5f5; }
        .container { max-width: 900px; margin: 0 auto; background: #ffffff; padding: 24px; border-radius: 12px; box-shadow: 0 10px 30px rgba(0,0,0,0.05); }
        h1 { margin-top: 0; }
        label { font-weight: 600; display: block; margin-bottom: 8px; }
        select, textarea, button { width: 100%; font-size: 15px; }
        select { padding: 10px; border-radius: 8px; border: 1px solid #ccc; margin-bottom: 16px; }
        textarea { min-height: 200px; padding: 12px; border-radius: 8px; border: 1px solid #ccc; resize: vertical; font-family: ui-monospace, SFMono-Regular, Consolas, \"Liberation Mono\", Menlo, monospace; }
        button { padding: 12px; border: none; border-radius: 8px; background: #0060df; color: #fff; font-weight: 600; cursor: pointer; margin-top: 16px; }
        button:disabled { background: #9ca3af; cursor: not-allowed; }
        .meta { display: flex; gap: 12px; margin-bottom: 16px; font-size: 14px; color: #4b5563; flex-wrap: wrap; }
        .meta span { background: #f3f4f6; padding: 6px 10px; border-radius: 999px; }
        pre { background: #0f172a; color: #f8fafc; padding: 16px; border-radius: 10px; overflow: auto; margin-top: 24px; white-space: pre-wrap; word-wrap: break-word; }
        .note { font-size: 14px; color: #6b7280; margin-bottom: 16px; }
        .doc { background: #fef3c7; color: #78350f; border-radius: 10px; padding: 12px 16px; margin-bottom: 18px; font-size: 14px; line-height: 1.5; white-space: pre-line; }
    </style>
</head>
<body>
    <div class=\"container\">
        <h1>SummRAG Admin Console</h1>
        <p class=\"note\">Operacje wysyłają nagłówek <code>X-Admin-UI: 1</code>, aby wymusić tryb debug na serwerze.</p>
        <label for=\"operation\">Wybierz operację</label>
        <select id=\"operation\"></select>
        <div class=\"meta\">
            <span id=\"method\"></span>
            <span id=\"path\"></span>
        </div>
        <div class=\"doc\" id=\"doc\"></div>
        <div id=\"body-wrapper\">
            <label for=\"payload\">Treść żądania (JSON)</label>
            <textarea id=\"payload\" spellcheck=\"false\"></textarea>
        </div>
        <button id=\"send\">Wyślij żądanie</button>
        <pre id=\"result\">Odpowiedź pojawi się tutaj.</pre>
    </div>
    <script>
        const operations = __OPERATIONS__;

        const selectEl = document.getElementById("operation");
        const methodEl = document.getElementById("method");
        const pathEl = document.getElementById("path");
        const bodyWrapper = document.getElementById("body-wrapper");
        const docEl = document.getElementById("doc");
        const payloadEl = document.getElementById("payload");
        const resultEl = document.getElementById("result");
        const sendBtn = document.getElementById("send");

        function populateOptions() {
            operations.forEach((op, idx) => {
                const option = document.createElement("option");
                option.value = op.id;
                option.textContent = op.label;
                if (idx === 0) option.selected = true;
                selectEl.appendChild(option);
            });
        }

        function renderOperation(id) {
            const op = operations.find(item => item.id === id);
            if (!op) return;
            methodEl.textContent = op.method;
            pathEl.textContent = op.path;
            docEl.textContent = op.doc || "Brak dokumentacji dla tej operacji.";
            const hasBody = op.method !== "GET";
            if (hasBody) {
                bodyWrapper.style.display = "block";
                payloadEl.value = op.body && op.body.length ? op.body : "{}";
            } else {
                bodyWrapper.style.display = "none";
                payloadEl.value = "";
            }
            resultEl.textContent = "Odpowiedź pojawi się tutaj.";
            sendBtn.disabled = false;
            sendBtn.dataset.op = op.id;
        }

        async function runRequest() {
            const opId = sendBtn.dataset.op;
            const op = operations.find(item => item.id === opId);
            if (!op) return;
            sendBtn.disabled = true;
            resultEl.textContent = "Wysyłanie...";

            let bodyPayload = undefined;
            if (op.method !== "GET") {
                const raw = payloadEl.value.trim();
                if (raw.length) {
                    try {
                        bodyPayload = JSON.stringify(JSON.parse(raw));
                    } catch (err) {
                        resultEl.textContent = `Błąd parsowania JSON: ${err}`;
                        sendBtn.disabled = false;
                        return;
                    }
                } else {
                    bodyPayload = "{}";
                }
            }

            try {
                const response = await fetch(op.path, {
                    method: op.method,
                    headers: {
                        "Content-Type": "application/json",
                        "X-Admin-UI": "1"
                    },
                    body: op.method === "GET" ? undefined : bodyPayload
                });

                const ct = response.headers.get("content-type") || "";
                if (ct.includes("application/json")) {
                    const data = await response.json();
                    resultEl.textContent = JSON.stringify(data, null, 2);
                } else {
                    const text = await response.text();
                    resultEl.textContent = text;
                }
                if (!response.ok) {
                    resultEl.textContent = `HTTP ${response.status}\n\n${resultEl.textContent}`;
                }
            } catch (err) {
                resultEl.textContent = `Błąd wywołania: ${err}`;
            }

            sendBtn.disabled = false;
        }

        populateOptions();
        renderOperation(selectEl.value);
        selectEl.addEventListener("change", event => renderOperation(event.target.value));
        sendBtn.addEventListener("click", runRequest);
    </script>
</body>
</html>
"""

ADMIN_OPERATION_SPECS: List[Dict[str, Any]] = [
    {"id": "about", "path": "/about", "method": "GET"},
    {"id": "health", "path": "/health", "method": "GET"},
    {
        "id": "collections-init",
        "path": "/collections/init",
        "method": "POST",
        "body": "{\n  \"collection_name\": \"rags_tool\",\n  \"force_dim_probe\": false\n}",
    },
    {
        "id": "ingest-scan",
        "path": "/ingest/scan",
        "method": "POST",
        "body": "{\n  \"base_dir\": \"/app/data\",\n  \"glob\": \"**/*\",\n  \"recursive\": true\n}",
    },
    {
        "id": "summaries-generate",
        "path": "/summaries/generate",
        "method": "POST",
        "body": "{\n  \"files\": [\n    \"/app/data/example.md\"\n  ]\n}",
    },
    {
        "id": "ingest-build",
        "path": "/ingest/build",
        "method": "POST",
        "body": "{\n  \"base_dir\": \"/app/data\",\n  \"glob\": \"**/*\",\n  \"recursive\": true,\n  \"reindex\": false,\n  \"chunk_tokens\": 900,\n  \"chunk_overlap\": 150,\n  \"collection_name\": \"rags_tool\",\n  \"enable_sparse\": true,\n  \"rebuild_tfidf\": true\n}",
    },
    {
        "id": "search-query",
        "path": "/search/query",
        "method": "POST",
        "body": "{\n  \"query\": \"Jak działa SummRAG?\",\n  \"top_m\": 10,\n  \"top_k\": 5,\n  \"mode\": \"auto\",\n  \"use_hybrid\": true,\n  \"dense_weight\": 0.6,\n  \"sparse_weight\": 0.4,\n  \"mmr_lambda\": 0.3,\n  \"per_doc_limit\": 2,\n  \"score_norm\": \"minmax\",\n  \"rep_alpha\": 0.6,\n  \"mmr_stage1\": true,\n  \"summary_mode\": \"first\",\n  \"result_format\": \"flat\"\n}",
    },
]

# MMR
DEFAULT_MMR_LAMBDA = 0.3
DEFAULT_PER_DOC_LIMIT = 2
DEFAULT_SCORE_NORM = "minmax"  # minmax|zscore|none

# ──────────────────────────────────────────────────────────────────────────────
# INICJALIZACJA KLIENTÓW
# ──────────────────────────────────────────────────────────────────────────────
qdrant = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
embedding_client = OpenAI(base_url=settings.embedding_api_url, api_key=settings.embedding_api_key)
summary_client = OpenAI(base_url=settings.summary_api_url, api_key=settings.summary_api_key)

# ──────────────────────────────────────────────────────────────────────────────
# UTYLITKI
# ──────────────────────────────────────────────────────────────────────────────

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def now_ts() -> int:
    return int(time.time())


# ——— Parsowanie plików ———
SUPPORTED_EXT = {".txt", ".md", ".markdown", ".html", ".htm", ".pdf"}


def read_text_file(path: pathlib.Path) -> str:
    ext = path.suffix.lower()
    data = path.read_bytes()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("latin1", errors="ignore")
    if ext in {".md", ".markdown"}:
        # MD → tekst
        return markdown2.markdown(text)  # HTML
    return text


def html_to_text(content_html: str) -> str:
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.body_width = 0
    return h.handle(content_html)


def extract_text(path: pathlib.Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        if not HAS_PDF:
            raise HTTPException(status_code=400, detail="PDF wymaga PyPDF2 — zainstaluj lub pomiń PDF")
        txt = []
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                txt.append(page.extract_text() or "")
        return "\n".join(txt)
    elif ext in {".html", ".htm"}:
        raw = read_text_file(path)
        return html_to_text(raw)
    else:
        raw = read_text_file(path)
        # Jeśli markdown2 zwrócił HTML, zamień na tekst
        if raw.strip().lower().startswith("<") and "</" in raw:
            return html_to_text(raw)
        return raw


# ——— Chunking i streszczenia ———

def split_into_paragraphs(text: str) -> List[str]:
    # Split na podwójne nowe linie / długie akapity
    chunks = re.split(r"\n\s*\n", text)
    return [c.strip() for c in chunks if c.strip()]


def chunk_text(text: str, target_tokens: int = 900, overlap_tokens: int = 150) -> List[str]:
    # Prosty chunking po akapitach z pakietowaniem do ~tokenów (heurystyka po znakach)
    # Przy braku tokenizerów używamy znaki ~ 4 char/token → target_chars
    token_to_char = 4
    target_chars = target_tokens * token_to_char
    overlap_chars = overlap_tokens * token_to_char

    paras = split_into_paragraphs(text)
    chunks = []
    buf = ""
    for p in paras:
        if len(buf) + len(p) + 2 <= target_chars:
            buf = (buf + "\n\n" + p) if buf else p
        else:
            if buf:
                chunks.append(buf)
                # overlap na końcu poprzedniego chunku
                buf_tail = buf[-overlap_chars:]
                buf = buf_tail + "\n\n" + p
            else:
                # bardzo długi pojedynczy akapit
                start = 0
                while start < len(p):
                    end = start + target_chars
                    chunk = p[start:end]
                    if chunk:
                        chunks.append(chunk)
                    start = end - overlap_chars
                    if start < 0:
                        start = 0
                buf = ""
    if buf:
        while len(buf) > target_chars:
            chunk = buf[:target_chars]
            chunks.append(chunk)
            buf = buf[target_chars - overlap_chars:]
        if buf:
            chunks.append(buf)
    return chunks


SUMMARY_PROMPT = (
    "Streść poniższy tekst w maks. 5 zdaniach, wypisz też 'SIGNATURE' (10–20 lematów kluczowych) "
    "i 'ENTITIES' (nazwy własne/ID/zakres dat). Bez komentarzy.\n\n"
    "FORMAT:\nSUMMARY: ...\nSIGNATURE: lemma1, lemma2, ...\nENTITIES: ...\n\nTEKST:\n"  # Tekst doklejony na końcu
)


def llm_summary(text: str, model: str = settings.summary_model, max_tokens: int = 300) -> Dict[str, Any]:
    text = text.strip()
    if len(text) > 8000:
        text = text[:8000]
    rsp = summary_client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[{"role": "system", "content": "Jesteś zwięzłym ekstrakcyjnym streszczaczem."},
                  {"role": "user", "content": SUMMARY_PROMPT + text}],
        max_tokens=max_tokens,
    )
    out = rsp.choices[0].message.content or ""
    # Parsowanie prostego formatu
    summary = ""
    signature = []
    entities = ""
    for line in out.splitlines():
        if line.startswith("SUMMARY:"):
            summary = line[len("SUMMARY:") :].strip()
        elif line.startswith("SIGNATURE:"):
            sig = line[len("SIGNATURE:") :].strip()
            signature = [s.strip() for s in re.split(r",|;", sig) if s.strip()]
        elif line.startswith("ENTITIES:"):
            entities = line[len("ENTITIES:") :].strip()
    if not summary:
        summary = out.strip()[:600]
    return {"summary": summary, "signature": signature, "entities": entities}


# ——— Embedding i TF‑IDF ———

def get_embedding_dim() -> int:
    # Jednorazowo embedujemy prosty tekst i odczytujemy wymiar
    vec = embed_text(["test"])[0]
    return len(vec)


def embed_text(texts: List[str]) -> List[List[float]]:
    rsp = embedding_client.embeddings.create(model=settings.embedding_model, input=texts)
    return [d.embedding for d in rsp.data]


VECTORIZER_PATH = VECTOR_STORE_DIR / "tfidf_vectorizer.json"
SUMMARY_VECTORIZER_PATH = VECTOR_STORE_DIR / "tfidf_vectorizer_summary.json"


def load_vectorizer(path: pathlib.Path = VECTORIZER_PATH) -> Optional[TfidfVectorizer]:
    if path.exists():
        obj = json.loads(path.read_text())
        vec = TfidfVectorizer(**obj["params"])  # type: ignore
        vec.vocabulary_ = {k: int(v) for k, v in obj["vocab"].items()}  # type: ignore
        vec.idf_ = np.array(obj["idf"])  # type: ignore
        return vec
    return None


def save_vectorizer(vec: TfidfVectorizer, path: pathlib.Path = VECTORIZER_PATH):
    params = vec.get_params()
    payload = {
        "params": {k: v for k, v in params.items() if k in ["lowercase", "ngram_range", "min_df", "max_df"]},
        "vocab": {k: int(v) for k, v in vec.vocabulary_.items()},
        "idf": vec.idf_.tolist(),
    }
    path.write_text(json.dumps(payload))


def fit_vectorizer(corpus: List[str], path: pathlib.Path = VECTORIZER_PATH) -> TfidfVectorizer:
    vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, max_df=0.9)
    vec.fit(corpus)
    save_vectorizer(vec, path=path)
    return vec


def tfidf_vector(
    texts: List[str], vec: Optional[TfidfVectorizer], path: pathlib.Path = VECTORIZER_PATH
) -> List[Tuple[List[int], List[float]]]:
    if not vec:
        vec = load_vectorizer(path)
        if not vec:
            # jeśli brak globalnego wektoryzatora, fit tymczasowy na wejściu (słabsze, ale działa)
            vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1, max_df=1.0)
            vec.fit(texts)
    m = vec.transform(texts)
    results: List[Tuple[List[int], List[float]]] = []
    for i in range(m.shape[0]):
        row = m.getrow(i)
        indices = row.indices.astype(int).tolist()
        data = row.data.astype(float).tolist()
        results.append((indices, data))
    return results


# ──────────────────────────────────────────────────────────────────────────────
# SCORING HELPERS — normalization and hybrid ops
# ──────────────────────────────────────────────────────────────────────────────

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
    # minmax default
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
    # Assuming L2-normalized TF-IDF values from scikit (default). Then dot = cosine.
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


# ──────────────────────────────────────────────────────────────────────────────
# QDRANT — KOLEKCJA
# ──────────────────────────────────────────────────────────────────────────────

def ensure_collection(collection: Optional[str] = None, dim: Optional[int] = None):
    collection = collection or settings.collection_name
    if dim is None:
        dim = get_embedding_dim()
    vectors_config = {
        CONTENT_VECTOR_NAME: qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
        SUMMARY_VECTOR_NAME: qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
    }
    # Konfiguracja nazwanej macierzy rzadkiej – Qdrant 1.x: przekazujemy ją jako nazwany
    # wektor w polu `vector` punktu (wartość = SparseVector)
    sparse_config = (
        {
            CONTENT_SPARSE_NAME: qm.SparseVectorParams(),
            SUMMARY_SPARSE_NAME: qm.SparseVectorParams(),
        }
        if SPARSE_ENABLED
        else None
    )
    try:
        qdrant.get_collection(collection)
        logger.debug("Collection '%s' already exists; skipping creation", collection)
        return
    except Exception:
        logger.debug("Collection '%s' not found; creating", collection)
        pass

    try:
        qdrant.create_collection(
            collection_name=collection,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_config,
            optimizers_config=qm.OptimizersConfigDiff(indexing_threshold=20000),
        )
        logger.debug("Collection '%s' created", collection)
    except UnexpectedResponse as exc:
        # 409 means already exists – ignorujemy, wszystko OK
        if getattr(exc, "status_code", None) == 409 or "already exists" in str(exc).lower():
            logger.debug("Collection '%s' creation returned conflict; treating as existing", collection)
            return
        raise


# ──────────────────────────────────────────────────────────────────────────────
# MODELE Pydantic — żądania/odpowiedzi
# ──────────────────────────────────────────────────────────────────────────────
class About(BaseModel):
    name: str = settings.app_name
    version: str = settings.app_version
    author: str = "Seweryn Sitarski (seweryn.sitarski@gmail.com) with support from Kat"
    description: str = "Dwustopniowy RAG ze streszczeniami + hybryda dense/TF‑IDF (Qdrant)"


class InitCollectionsRequest(BaseModel):
    collection_name: str = Field(default_factory=lambda: settings.collection_name)
    force_dim_probe: bool = False


class ScanRequest(BaseModel):
    base_dir: str
    glob: str = "**/*"
    recursive: bool = True


class ScanResponse(BaseModel):
    files: List[str]


class IngestBuildRequest(BaseModel):
    base_dir: str
    glob: str = "**/*"
    recursive: bool = True
    reindex: bool = False
    chunk_tokens: int = 900
    chunk_overlap: int = 150
    language_hint: Optional[str] = None
    collection_name: str = Field(default_factory=lambda: settings.collection_name)
    enable_sparse: bool = True
    rebuild_tfidf: bool = True


class SummariesGenerateRequest(BaseModel):
    files: List[str]


class SearchQuery(BaseModel):
    query: str
    top_m: int = 100  # etap streszczeń
    top_k: int = 10   # wynik końcowy
    mode: str = "auto"  # auto|current|archival|all
    use_hybrid: bool = True
    dense_weight: float = 0.6
    sparse_weight: float = 0.4
    mmr_lambda: float = DEFAULT_MMR_LAMBDA
    per_doc_limit: int = DEFAULT_PER_DOC_LIMIT
    score_norm: str = DEFAULT_SCORE_NORM  # minmax|zscore|none
    rep_alpha: Optional[float] = None  # fallback to dense_weight
    mmr_stage1: bool = True
    # Controls duplication of document summary in results
    summary_mode: str = "first"  # none|first|all
    # Controls shape of response: flat list vs grouped per document
    result_format: str = "flat"  # flat|grouped


class SearchHit(BaseModel):
    doc_id: str
    path: str
    section: Optional[str] = None
    chunk_id: int
    score: float
    snippet: str
    summary: Optional[str] = None


class SearchResponse(BaseModel):
    took_ms: int
    hits: List[SearchHit]
    # Optional grouped representation (when requested)
    groups: Optional[List["SearchGroup"]] = None


class SearchChunk(BaseModel):
    chunk_id: int
    score: float
    snippet: str


class SearchGroup(BaseModel):
    doc_id: str
    path: str
    summary: Optional[str] = None
    score: float
    chunks: List[SearchChunk]

# Rebuild forward refs for Pydantic v2
SearchResponse.model_rebuild()


# ──────────────────────────────────────────────────────────────────────────────
# FASTAPI
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(title=f"{settings.app_name} OpenAPI Tool", version=settings.app_version)


def build_admin_operations() -> List[Dict[str, Any]]:
    route_lookup: Dict[Tuple[str, str], APIRoute] = {}
    for route in app.routes:
        if isinstance(route, APIRoute):
            for method in route.methods or []:
                route_lookup[(route.path, method.upper())] = route

    operations: List[Dict[str, Any]] = []
    for spec in ADMIN_OPERATION_SPECS:
        method = spec["method"].upper()
        path = spec["path"]
        route = route_lookup.get((path, method))
        summary = route.summary if route else None
        description = route.description if route else None
        doc_parts = [part for part in [summary, description] if part]
        operations.append(
            {
                "id": spec["id"],
                "label": spec.get("label") or f"{method} {path}",
                "method": method,
                "path": path,
                "doc": "\n\n".join(doc_parts),
                "body": spec.get("body"),
            }
        )
    return operations


@app.middleware("http")
async def admin_ui_debug_middleware(request: Request, call_next):
    global _admin_debug_activated
    header_value = request.headers.get(ADMIN_UI_REQUEST_HEADER)
    if header_value and header_value.lower() in {"1", "true", "yes"} and not _admin_debug_activated:
        _admin_debug_activated = True
        settings.debug = True
        logger.setLevel(logging.DEBUG)
        logger.info("Admin UI request detected — DEBUG logging enabled")
    response = await call_next(request)
    return response


@app.get(
    "/admin",
    include_in_schema=False,
    response_class=HTMLResponse,
    summary="Panel administracyjny",
    description="Statyczny panel HTML do testowania i debugowania endpointów SummRAG.",
)
def admin_console():
    operations = build_admin_operations()
    html = ADMIN_UI_HTML_TEMPLATE.replace(
        "__OPERATIONS__", json.dumps(operations, ensure_ascii=False)
    )
    return html


@app.get(
    "/about",
    response_model=About,
    include_in_schema=False,
    summary="Informacje o aplikacji",
    description="Zwraca metadane serwisu SummRAG (nazwa, wersja, autor, opis).",
)
def about():
    return About()


@app.get(
    "/health",
    include_in_schema=False,
    summary="Stan usługi",
    description="Sprawdza połączenie z Qdrant i raportuje kondycję aplikacji.",
)
def health():
    try:
        qdrant.get_collections()
        return {"status": "ok", "qdrant": True}
    except Exception as e:
        return {"status": "degraded", "qdrant": False, "error": str(e)}


@app.post(
    "/collections/init",
    include_in_schema=False,
    summary="Inicjalizacja kolekcji",
    description=(
        "Tworzy kolekcję we wskazanej nazwie (jeśli nie istnieje) i opcjonalnie sondą"
        " sprawdza wymiar embeddingów przy użyciu force_dim_probe."
    ),
)
def collections_init(req: InitCollectionsRequest):
    dim = get_embedding_dim() if req.force_dim_probe else None
    ensure_collection(req.collection_name, dim)
    return {"ok": True, "collection": req.collection_name}


@app.post(
    "/ingest/scan",
    response_model=ScanResponse,
    include_in_schema=False,
    summary="Skanowanie korpusu",
    description="Zwraca listę plików w katalogu bazowym, które kwalifikują się do ingestu.",
)
def ingest_scan(req: ScanRequest):
    base = pathlib.Path(req.base_dir)
    if not base.exists():
        raise HTTPException(status_code=400, detail="base_dir nie istnieje")
    pattern = req.glob
    files = [str(p) for p in base.glob(pattern) if p.is_file() and p.suffix.lower() in SUPPORTED_EXT]
    return ScanResponse(files=files)


@app.post(
    "/summaries/generate",
    include_in_schema=False,
    summary="Streszczenia wybranych plików",
    description="Generuje streszczenia oraz podpisy dla listy plików bez zapisu do Qdrant.",
)
def summaries_generate(req: SummariesGenerateRequest):
    results = {}
    for f in req.files:
        p = pathlib.Path(f)
        if not p.exists():
            results[f] = {"error": "not found"}
            continue
        text = extract_text(p)
        summ = llm_summary(text)
        results[f] = summ
    return {"results": results}


@app.post(
    "/ingest/build",
    include_in_schema=False,
    summary="Pełny ingest korpusu",
    description=(
        "Buduje indeks SummRAG: parsuje dokumenty, tworzy streszczenia,"
        " embeddingi oraz zapisuje punkty (wraz z TF-IDF) do Qdrant."
    ),
)
def ingest_build(req: IngestBuildRequest):
    t0 = time.time()
    logger.debug(
        "Starting ingest build | base_dir=%s glob=%s recursive=%s reindex=%s",
        req.base_dir,
        req.glob,
        req.recursive,
        req.reindex,
    )
    ensure_collection(req.collection_name)

    base = pathlib.Path(req.base_dir)
    if not base.exists():
        raise HTTPException(status_code=400, detail="base_dir nie istnieje")

    file_paths = [
        p for p in base.glob(req.glob) if p.is_file() and p.suffix.lower() in SUPPORTED_EXT
    ]
    logger.debug("Found %d files for ingest", len(file_paths))
    if not file_paths:
        return {"ok": True, "indexed": 0, "took_ms": int((time.time() - t0) * 1000)}

    # Parsowanie i chunking
    all_chunks: List[str] = []
    summary_corpus: List[str] = []
    doc_records: List[Dict[str, Any]] = []

    for path in file_paths:
        logger.debug("Processing document %s", path)
        raw = extract_text(path)
        chunks = chunk_text(raw, target_tokens=req.chunk_tokens, overlap_tokens=req.chunk_overlap)
        if not chunks:
            logger.debug("Document %s produced no chunks; skipping", path)
            continue
        # streszczenie całości (dla indeksu streszczeń dokumentu)
        doc_sum = llm_summary(raw[:12000])  # limit ochronny
        doc_id = sha1(str(path.resolve()))
        summary_signature = doc_sum.get("signature", [])
        summary_sparse_text = " ".join(
            [doc_sum.get("summary", ""), " ".join(summary_signature)]
        ).strip()
        rec = {
            "doc_id": doc_id,
            "path": str(path.resolve()),
            "chunks": chunks,
            "doc_summary": doc_sum["summary"],
            "doc_signature": summary_signature,
            "summary_sparse_text": summary_sparse_text,
        }
        doc_records.append(rec)
        all_chunks.extend(chunks)
        if summary_sparse_text:
            summary_corpus.append(summary_sparse_text)
        logger.debug(
            "Document %s parsed | chunks=%d summary_len=%d",
            path,
            len(chunks),
            len(doc_sum["summary"] or ""),
        )

    # TF‑IDF (globalny)
    content_vec: Optional[TfidfVectorizer] = None
    summary_vec: Optional[TfidfVectorizer] = None
    if req.enable_sparse:
        logger.debug("Sparse mode enabled; rebuild=%s", req.rebuild_tfidf)
        if all_chunks:
            if req.rebuild_tfidf or not VECTORIZER_PATH.exists():
                content_vec = fit_vectorizer(all_chunks)
                logger.debug("Fitted new TF-IDF vectorizer over %d chunks", len(all_chunks))
            else:
                content_vec = load_vectorizer()
                if content_vec is None:
                    content_vec = fit_vectorizer(all_chunks)
                    logger.debug("Rebuilt TF-IDF vectorizer (cache missing) over %d chunks", len(all_chunks))
                else:
                    logger.debug("Loaded existing TF-IDF vectorizer")
        else:
            logger.debug("No chunks available for TF-IDF fitting; reusing existing vectorizer if present")
            content_vec = load_vectorizer()

        if summary_corpus:
            if req.rebuild_tfidf or not SUMMARY_VECTORIZER_PATH.exists():
                summary_vec = fit_vectorizer(summary_corpus, path=SUMMARY_VECTORIZER_PATH)
                logger.debug("Fitted new summary TF-IDF vectorizer over %d summaries", len(summary_corpus))
            else:
                summary_vec = load_vectorizer(path=SUMMARY_VECTORIZER_PATH)
                if summary_vec is None:
                    summary_vec = fit_vectorizer(summary_corpus, path=SUMMARY_VECTORIZER_PATH)
                    logger.debug(
                        "Rebuilt summary TF-IDF vectorizer (cache missing) over %d summaries",
                        len(summary_corpus),
                    )
                else:
                    logger.debug("Loaded existing summary TF-IDF vectorizer")
        else:
            logger.debug("No summaries available for TF-IDF fitting; reusing existing summary vectorizer if present")
            summary_vec = load_vectorizer(path=SUMMARY_VECTORIZER_PATH)

    # Budowa punktów Qdrant
    points: List[qm.PointStruct] = []
    point_count = 0

    for rec in doc_records:
        doc_id = rec["doc_id"]
        path = rec["path"]
        chunks = rec["chunks"]
        doc_summary = rec["doc_summary"]
        doc_signature = rec["doc_signature"]
        summary_sparse_text = rec["summary_sparse_text"]

        # Embedding streszczenia dokumentu (użyjemy go też jako wektor sekcji głównej)
        summary_dense_vec = embed_text([doc_summary])[0]

        # Embedding chunków
        content_vecs = embed_text(chunks)
        # TF‑IDF sparse
        if req.enable_sparse:
            sparse_chunks = tfidf_vector(chunks, content_vec)
            if summary_vec is not None and summary_sparse_text:
                summary_sparse = tfidf_vector(
                    [summary_sparse_text], summary_vec, path=SUMMARY_VECTORIZER_PATH
                )[0]
            else:
                summary_sparse = ([], [])
        else:
            sparse_chunks = [([], []) for _ in chunks]
            summary_sparse = ([], [])

        for i, chunk in enumerate(chunks):
            pid = int(str(int(sha1(f"{doc_id}:{i}")[0:12], 16))[:12])  # stabilny int z sha1
            payload = {
                "doc_id": doc_id,
                "path": path,
                "chunk_id": i,
                "is_active": True,
                "summary": doc_summary,
                "text": chunk,
                "signature": doc_signature,
            }

            vectors = {
                CONTENT_VECTOR_NAME: content_vecs[i],
                SUMMARY_VECTOR_NAME: summary_dense_vec,
            }

            if SPARSE_ENABLED and req.enable_sparse:
                indices, values = sparse_chunks[i]
                if indices:
                    vectors[CONTENT_SPARSE_NAME] = qm.SparseVector(indices=indices, values=values)
                    payload["content_sparse_indices"] = indices
                    payload["content_sparse_values"] = values
                if summary_sparse[0]:
                    vectors[SUMMARY_SPARSE_NAME] = qm.SparseVector(
                        indices=summary_sparse[0], values=summary_sparse[1]
                    )
                    payload["summary_sparse_indices"] = summary_sparse[0]
                    payload["summary_sparse_values"] = summary_sparse[1]

            points.append(qm.PointStruct(id=pid, vector=vectors, payload=payload))
            point_count += 1

        # Upsert w partiach, by nie trzymać wszystkiego w RAM
        if len(points) >= 1024:
            qdrant.upsert(collection_name=req.collection_name, points=points)
            logger.debug("Upserted batch of %d points into %s", len(points), req.collection_name)
            points = []

    if points:
        qdrant.upsert(collection_name=req.collection_name, points=points)
        logger.debug("Upserted final batch of %d points into %s", len(points), req.collection_name)

    took_ms = int((time.time() - t0) * 1000)
    logger.debug(
        "Ingest build finished | documents=%d points=%d took_ms=%d",
        len(doc_records),
        point_count,
        took_ms,
    )
    return {"ok": True, "indexed": point_count, "documents": len(doc_records), "took_ms": took_ms}


# ──────────────────────────────────────────────────────────────────────────────
# SEARCH — dwustopniowe: streszczenia → pełny tekst (+ MMR, hybryda)
# ──────────────────────────────────────────────────────────────────────────────

def mmr_diversify(vectors: np.ndarray, scores: np.ndarray, k: int, lam: float = DEFAULT_MMR_LAMBDA) -> List[int]:
    # Legacy MMR (dense only); kept for compatibility
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
    # Greedy MMR with hybrid redundancy (dense+sparse) and optional per-doc cap
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
        # if per-doc cap blocks all candidates, relax once to fill
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


@app.post(
    "/search/query",
    response_model=SearchResponse,
    summary="Zapytanie SummRAG",
    description=(
        "Dwustopniowe wyszukiwanie: najpierw streszczenia, następnie pełne teksty"
        " dokumentów, z opcjonalną hybrydą TF-IDF i prostym MMR."
    ),
)
def search_query(req: SearchQuery):
    t0 = time.time()
    ensure_collection()

    # 0) Klasyfikacja trybu (uproszczona)
    mode = req.mode
    if mode == "auto":
        q = req.query.lower()
        if re.search(r"archiw|stara|z \d{4}|wersja\s+z", q):
            mode = "archival"
        elif re.search(r"obowiązując|aktualn|teraz|bieżąc", q):
            mode = "current"
        else:
            mode = "all"

    # 1) Etap streszczeń — embedding zapytania
    q_vec = embed_text([req.query])[0]

    # Hybryda: dense + tfidf
    content_sparse_query: Optional[Tuple[List[int], List[float]]] = None
    summary_sparse_query: Optional[Tuple[List[int], List[float]]] = None
    if req.use_hybrid and SPARSE_ENABLED:
        content_vec = load_vectorizer()
        if content_vec is not None:
            idx, val = tfidf_vector([req.query], content_vec)[0]
            if idx:
                content_sparse_query = (idx, val)
        summary_vec_model = load_vectorizer(path=SUMMARY_VECTORIZER_PATH)
        if summary_vec_model is not None:
            s_idx, s_val = tfidf_vector([req.query], summary_vec_model, path=SUMMARY_VECTORIZER_PATH)[0]
            if s_idx:
                summary_sparse_query = (s_idx, s_val)

    # Filtry payload
    flt = None
    if mode in ("current", "archival"):
        flt = qm.Filter(
            must=[
                qm.FieldCondition(key="is_active", match=qm.MatchValue(value=(mode == "current")))
            ]
        )

    # Search w Qdrant po summary_dense (opcjonalnie z hybrydą)
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

    # Zbierz kandydatów po doc_id (unikalne) i przygotuj dane do hybrydowego MMR
    doc_map: Dict[str, Dict[str, Any]] = {}
    for r in sum_search:
        payload = r.payload or {}
        did = payload.get("doc_id")
        if not did:
            continue
        if did in doc_map:
            continue
        dense_score = float(r.score or 0.0)
        sparse_dot = 0.0
        if summary_sparse_lookup and payload.get("summary_sparse_indices") and payload.get("summary_sparse_values"):
            sparse_dot = _sparse_dot(summary_sparse_lookup, payload.get("summary_sparse_indices", []), payload.get("summary_sparse_values", []))
        vec_map = r.vector or {}
        dense_vec = vec_map.get(SUMMARY_VECTOR_NAME)
        if dense_vec is None:
            dense_vec = []
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
        return SearchResponse(took_ms=int((time.time() - t0) * 1000), hits=[])

    doc_items = list(doc_map.values())
    dense_scores = [float(x["dense_score"]) for x in doc_items]
    sparse_scores = [float(x["sparse_score"]) for x in doc_items]
    dense_norm = _normalize(dense_scores, req.score_norm)
    sparse_norm = _normalize(sparse_scores, req.score_norm)
    hybrid_rel = [req.dense_weight * d + req.sparse_weight * s for d, s in zip(dense_norm, sparse_norm)]

    # Opcjonalny hybrydowy MMR na Etapie 1
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

    if not cand_doc_ids:
        return SearchResponse(took_ms=int((time.time() - t0) * 1000), hits=[])

    # 2) Etap pełnego tekstu — ograniczamy do kandydackich doc_id
    flt2 = qm.Filter(
        must=[qm.FieldCondition(key="doc_id", match=qm.MatchAny(any=cand_doc_ids))]
    )

    cont_search = qdrant.search(
        collection_name=settings.collection_name,
        query_vector=(CONTENT_VECTOR_NAME, q_vec),
        query_filter=flt2,
        limit=req.top_m,
        with_payload=True,
        with_vectors=[CONTENT_VECTOR_NAME],
        search_params=qm.SearchParams(exact=False, hnsw_ef=128),
    )

    # Przygotuj kandydatów do hybrydowego MMR z limitem per-doc
    mmr_pool = []
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
        return SearchResponse(took_ms=int((time.time() - t0) * 1000), hits=[])

    # normalizacja i hybrydowy relevance
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
    # finalne przeliczenie hybrydowego score i sortowanie malejąco
    final_hits = []
    for idx, item in zip(sel_idx, selected):
        hit = item["hit"]
        payload = hit.payload or {}
        final_score = float(rel2[idx])
        final_hits.append({
            "hit": hit,
            "score": float(final_score),
            "payload": payload,
        })
    final_hits.sort(key=lambda x: x["score"], reverse=True)

    # Build response according to requested format
    results: List[SearchHit] = []
    groups_payload: Optional[List[Dict[str, Any]]] = None

    if req.result_format == "grouped":
        # Group results per document, keep only one summary at doc level
        groups: Dict[str, Dict[str, Any]] = {}
        for fh in final_hits:
            payload = fh["payload"] or {}
            did = payload.get("doc_id", "")
            if not did:
                # Skip items without doc_id
                continue
            grp = groups.get(did)
            if grp is None:
                grp = {
                    "doc_id": did,
                    "path": payload.get("path", ""),
                    "summary": None if req.summary_mode == "none" else payload.get("summary"),
                    "score": float(fh["score"]),  # will be max over chunks
                    "chunks": [],
                }
                groups[did] = grp
            else:
                # update group score with max
                if float(fh["score"]) > float(grp.get("score", 0.0)):
                    grp["score"] = float(fh["score"])
            # Append chunk (never include summary at chunk level)
            snippet = (payload.get("text") or "").strip()[:500]
            grp["chunks"].append({
                "chunk_id": payload.get("chunk_id", 0),
                "score": float(fh["score"]),
                "snippet": snippet,
            })
        # Sort groups by score desc, and chunks inside each group by score desc
        groups_list = list(groups.values())
        for g in groups_list:
            g["chunks"].sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        groups_list.sort(key=lambda g: float(g.get("score", 0.0)), reverse=True)
        groups_payload = groups_list
        results = []  # keep flat list empty in grouped mode
    else:
        # Flat format: control duplication of summary per document
        seen_docs: set = set()
        for fh in final_hits:
            payload = fh["payload"]
            did = payload.get("doc_id", "")
            # Decide summary according to summary_mode
            summary_val: Optional[str]
            if req.summary_mode == "none":
                summary_val = None
            elif req.summary_mode == "first":
                if did in seen_docs:
                    summary_val = None
                else:
                    summary_val = payload.get("summary")
                    seen_docs.add(did)
            else:  # "all"
                summary_val = payload.get("summary")

            results.append(
                SearchHit(
                    doc_id=did,
                    path=payload.get("path", ""),
                    section=payload.get("section"),
                    chunk_id=payload.get("chunk_id", 0),
                    score=float(fh["score"]),
                    snippet=(payload.get("text") or "").strip()[:500] if payload.get("text") else (payload.get("summary", "")[:500]),
                    summary=summary_val,
                )
            )

    took_ms = int((time.time() - t0) * 1000)
    if settings.debug:
        logger.debug(
            "Search took %d ms | stage1_docs=%d stage2_candidates=%d returned=%d",
            took_ms,
            len(doc_map) if 'doc_map' in locals() else 0,
            len(mmr_pool) if 'mmr_pool' in locals() else 0,
            len(results),
        )
    return SearchResponse(took_ms=took_ms, hits=results, groups=groups_payload)
