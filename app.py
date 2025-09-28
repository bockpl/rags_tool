"""
SummRAG — dwustopniowy RAG ze streszczeniami (FastAPI + Qdrant)
Version: 0.1.2
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
  • /documents/upsert — pojedynczy dokument w/gotowych danych
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
  export COLLECTION_NAME="summrag"
  export VECTOR_STORE_DIR=".summrag_store"         # gdzie zapisze się TF‑IDF, cache itp.
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
        "body": "{\n  \"collection_name\": \"summrag\",\n  \"force_dim_probe\": false\n}",
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
        "body": "{\n  \"base_dir\": \"/app/data\",\n  \"glob\": \"**/*\",\n  \"recursive\": true,\n  \"reindex\": false,\n  \"chunk_tokens\": 900,\n  \"chunk_overlap\": 150,\n  \"collection_name\": \"summrag\",\n  \"enable_sparse\": true,\n  \"rebuild_tfidf\": true\n}",
    },
    {
        "id": "documents-upsert",
        "path": "/documents/upsert",
        "method": "POST",
        "body": "{\n  \"doc_id\": \"manual-doc-1\",\n  \"path\": \"/app/data/manual.txt\",\n  \"chunks\": [\n    \"Pierwszy fragment\",\n    \"Drugi fragment\"\n  ],\n  \"summaries\": [\n    \"Streszczenie pierwszego fragmentu\",\n    \"Streszczenie drugiego fragmentu\"\n  ]\n}",
    },
    {
        "id": "search-query",
        "path": "/search/query",
        "method": "POST",
        "body": "{\n  \"query\": \"Jak działa SummRAG?\",\n  \"top_m\": 10,\n  \"top_k\": 5,\n  \"mode\": \"auto\"\n}",
    },
]

# MMR
DEFAULT_MMR_LAMBDA = 0.3

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


def load_vectorizer() -> Optional[TfidfVectorizer]:
    if VECTORIZER_PATH.exists():
        obj = json.loads(VECTORIZER_PATH.read_text())
        vec = TfidfVectorizer(**obj["params"])  # type: ignore
        vec.vocabulary_ = {k: int(v) for k, v in obj["vocab"].items()}  # type: ignore
        vec.idf_ = np.array(obj["idf"])  # type: ignore
        return vec
    return None


def save_vectorizer(vec: TfidfVectorizer):
    params = vec.get_params()
    payload = {
        "params": {k: v for k, v in params.items() if k in ["lowercase", "ngram_range", "min_df", "max_df"]},
        "vocab": {k: int(v) for k, v in vec.vocabulary_.items()},
        "idf": vec.idf_.tolist(),
    }
    VECTORIZER_PATH.write_text(json.dumps(payload))


def fit_vectorizer(corpus: List[str]) -> TfidfVectorizer:
    vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, max_df=0.9)
    vec.fit(corpus)
    save_vectorizer(vec)
    return vec


def tfidf_vector(texts: List[str], vec: Optional[TfidfVectorizer]) -> List[Tuple[List[int], List[float]]]:
    if not vec:
        vec = load_vectorizer()
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
    sparse_config = None
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


class UpsertDocumentRequest(BaseModel):
    doc_id: str
    path: str
    chunks: List[str]
    summaries: List[str]
    signatures: List[List[str]] = Field(default_factory=list)
    is_active: Optional[bool] = True
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None


class SearchQuery(BaseModel):
    query: str
    top_m: int = 100  # etap streszczeń
    top_k: int = 10   # wynik końcowy
    mode: str = "auto"  # auto|current|archival|all
    use_hybrid: bool = True
    dense_weight: float = 0.6
    sparse_weight: float = 0.4
    mmr_lambda: float = DEFAULT_MMR_LAMBDA


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
        rec = {
            "doc_id": doc_id,
            "path": str(path.resolve()),
            "chunks": chunks,
            "doc_summary": doc_sum["summary"],
            "doc_signature": doc_sum.get("signature", []),
        }
        doc_records.append(rec)
        all_chunks.extend(chunks)
        logger.debug(
            "Document %s parsed | chunks=%d summary_len=%d",
            path,
            len(chunks),
            len(doc_sum["summary"] or ""),
        )

    # TF‑IDF (globalny)
    vec = None
    if req.enable_sparse:
        logger.debug("Sparse mode enabled; rebuild=%s", req.rebuild_tfidf)
        if req.rebuild_tfidf or not VECTORIZER_PATH.exists():
            vec = fit_vectorizer(all_chunks)
            logger.debug("Fitted new TF-IDF vectorizer over %d chunks", len(all_chunks))
        else:
            vec = load_vectorizer()
            if vec is None:
                vec = fit_vectorizer(all_chunks)
                logger.debug("Rebuilt TF-IDF vectorizer (cache missing) over %d chunks", len(all_chunks))
            else:
                logger.debug("Loaded existing TF-IDF vectorizer")

    # Budowa punktów Qdrant
    points: List[qm.PointStruct] = []
    point_count = 0

    for rec in doc_records:
        doc_id = rec["doc_id"]
        path = rec["path"]
        chunks = rec["chunks"]
        doc_summary = rec["doc_summary"]

        # Embedding streszczenia dokumentu (użyjemy go też jako wektor sekcji głównej)
        summary_vec = embed_text([doc_summary])[0]

        # Embedding chunków
        content_vecs = embed_text(chunks)
        # TF‑IDF sparse
        if req.enable_sparse:
            sparse_chunks = tfidf_vector(chunks, vec)
        else:
            sparse_chunks = [([], []) for _ in chunks]

        for i, chunk in enumerate(chunks):
            pid = int(str(int(sha1(f"{doc_id}:{i}")[0:12], 16))[:12])  # stabilny int z sha1
            payload = {
                "doc_id": doc_id,
                "path": path,
                "chunk_id": i,
                "is_active": True,
                "summary": doc_summary,
            }
            if req.enable_sparse:
                indices, values = sparse_chunks[i]
                if indices:
                    payload["sparse_indices"] = indices
                    payload["sparse_values"] = values

            vectors = {
                CONTENT_VECTOR_NAME: content_vecs[i],
                SUMMARY_VECTOR_NAME: summary_vec,
            }

            points.append(
                qm.PointStruct(
                    id=pid,
                    vector=vectors,
                    payload=payload,
                )
            )
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
    # Prosta implementacja MMR na macierzy (kandydaci × dim)
    selected: List[int] = []
    candidates = list(range(len(scores)))
    if len(candidates) <= k:
        return candidates
    # Normalizacja
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    sims = lambda a, b: float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    # Greedy
    while len(selected) < k and candidates:
        best_i = None
        best_score = -1e9
        for i in candidates:
            rep = 0.0
            for j in selected:
                rep = max(rep, sims(vectors[i], vectors[j]))
            mmr = lam * scores[i] - (1 - lam) * rep
            if mmr > best_score:
                best_score = mmr
                best_i = i
        selected.append(best_i)  # type: ignore
        candidates.remove(best_i)  # type: ignore
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
    sparse_query = None
    if req.use_hybrid and SPARSE_ENABLED and VECTORIZER_PATH.exists():
        vec = load_vectorizer()
        if vec is not None:
            idx, val = tfidf_vector([req.query], vec)[0]
            if idx:
                sparse_query = dict(zip(idx, val))

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
        with_vectors=False,
        score_threshold=None,
        search_params=qm.SearchParams(exact=False, hnsw_ef=128),
    )

    # Zbierz kandydatów po doc_id (unikalne)
    cand_doc_ids: List[str] = []
    for r in sum_search:
        did = r.payload.get("doc_id")  # type: ignore
        if did and did not in cand_doc_ids:
            cand_doc_ids.append(did)
        if len(cand_doc_ids) >= req.top_m:
            break

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
        with_vectors=False,
        search_params=qm.SearchParams(exact=False, hnsw_ef=128),
    )

    # Opcjonalna dywersyfikacja MMR (po samych wynikach — w przybliżeniu użyjemy score)
    # Bez wektorów punktów trudno policzyć rep; użyjemy tylko score, co zadziała jako zwykłe Top‑K
    # Dla prostego MMR potrzebowalibyśmy wektorów kandydatów — uproszczenie: Top‑K bez MMR

    results: List[SearchHit] = []
    for r in cont_search[: req.top_k]:
        payload = r.payload or {}
        sparse_boost = 0.0
        if sparse_query and payload.get("sparse_indices") and payload.get("sparse_values"):
            for idx_val, val_val in zip(payload["sparse_indices"], payload["sparse_values"]):
                q_val = sparse_query.get(idx_val)
                if q_val is not None:
                    sparse_boost += q_val * val_val

        results.append(
            SearchHit(
                doc_id=payload.get("doc_id", ""),
                path=payload.get("path", ""),
                section=payload.get("section"),
                chunk_id=payload.get("chunk_id", 0),
                score=float(r.score or 0.0) + sparse_boost,
                snippet=(payload.get("text") or "").strip()[:500] if payload.get("text") else payload.get("summary", "")[:500],
                summary=payload.get("summary"),
            )
        )

    took_ms = int((time.time() - t0) * 1000)
    return SearchResponse(took_ms=took_ms, hits=results)


# ──────────────────────────────────────────────────────────────────────────────
# DODATKOWE: /documents/upsert — jeśli chcesz sam podawać gotowe chunk/summary
# ──────────────────────────────────────────────────────────────────────────────
@app.post(
    "/documents/upsert",
    include_in_schema=False,
    summary="Ręczny upsert dokumentu",
    description="Aktualizuje lub dodaje dokument w Qdrant z dostarczonymi chunkami i streszczeniami.",
)
def documents_upsert(req: UpsertDocumentRequest):
    ensure_collection()
    # Embedding streszczeń i chunków
    sum_vec = embed_text([" ".join(req.summaries)[:12000] or "summary"])[0]
    cont_vecs = embed_text(req.chunks)
    sparse_vectors = tfidf_vector(req.chunks, load_vectorizer()) if SPARSE_ENABLED else [([], [])] * len(req.chunks)

    points: List[qm.PointStruct] = []
    for i, chunk in enumerate(req.chunks):
        pid = int(str(int(sha1(f"{req.doc_id}:{i}")[0:12], 16))[:12])
        payload = {
            "doc_id": req.doc_id,
            "path": req.path,
            "chunk_id": i,
            "is_active": req.is_active,
            "summary": (req.summaries[i] if i < len(req.summaries) else req.summaries[-1]) if req.summaries else None,
            "text": chunk,
            "valid_from": req.valid_from,
            "valid_to": req.valid_to,
        }
        vectors = {CONTENT_VECTOR_NAME: cont_vecs[i], SUMMARY_VECTOR_NAME: sum_vec}
        if SPARSE_ENABLED:
            idx, val = sparse_vectors[i]
            if idx:
                payload["sparse_indices"] = idx
                payload["sparse_values"] = val
        points.append(
            qm.PointStruct(
                id=pid,
                vector=vectors,
                payload=payload,
            )
        )

    qdrant.upsert(collection_name=settings.collection_name, points=points)
    return {"ok": True, "points": len(points)}
