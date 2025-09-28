# SummRAG (0.4.0)

Dwustopniowy serwis RAG zbudowany na FastAPI. System wspiera streszczanie dokumentów, indeksowanie w Qdrant oraz wyszukiwanie hybrydowe (dense + TF-IDF).

## Nowości w 0.4.0

- Kontrola duplikacji streszczeń w wynikach: nowy parametr `summary_mode` (`none` | `first` | `all`). Domyślnie `first` — streszczenie dokumentu pojawia się tylko przy pierwszym trafieniu z danego dokumentu (eliminuje powtarzanie).
- Nowy format wyników: `result_format` (`flat` | `grouped`). Domyślnie `flat`. W trybie `grouped` wyniki są grupowane per dokument (jedno streszczenie na dokument + lista trafionych fragmentów).
- Zaktualizowany panel Admin UI ma domyślnie `summary_mode: "first"` i `result_format: "flat"` w predefiniowanym żądaniu `search-query`.

## Nowości w 0.3.0

- MMR i ranking liczone w przestrzeni hybrydowej (dense + TF‑IDF) z normalizacją wyników; końcowe sortowanie po score hybrydowym.
- Limit per‑doc w Etapie 2 (domyślnie 2) przeciwdziała dominacji jednego dokumentu.
- Opcje normalizacji: `minmax` (domyślna), `zscore`, `none` — stabilizują wagi dense/sparse.
- Opcjonalny MMR na Etapie 1 (streszczenia) z re‑pulsywnością hybrydową.
- Nowe parametry zapytania: `per_doc_limit`, `score_norm`, `rep_alpha`, `mmr_stage1` (szczegóły niżej).

## Wymagania

- Python 3.11+
- Zewnętrzna instancja Qdrant (SaaS lub self-hosted)
- Endpoint zgodny z protokołem OpenAI zapewniający embeddingi i model konwersacyjny (np. vLLM, Ollama, OpenAI)

## Konfiguracja środowiska

Ustaw wymagane zmienne środowiskowe przed uruchomieniem aplikacji:

```bash
export QDRANT_URL="http://127.0.0.1:6333"
export QDRANT_API_KEY=""
export EMBEDDING_API_URL="http://127.0.0.1:8000/v1"
export EMBEDDING_API_KEY="sk-embed-xxx"
export EMBEDDING_MODEL="BAAI/bge-m3"
export SUMMARY_API_URL="http://127.0.0.1:8001/v1"
export SUMMARY_API_KEY="sk-summary-xxx"
export SUMMARY_MODEL="gpt-4o-mini"
export COLLECTION_NAME="rags_tool"
export VECTOR_STORE_DIR=".rags_tool_store"
export DEBUG="false"  # ustaw na "true", aby włączyć logi debugujące
```

Możesz także umieścić te wartości w pliku `.env`; aplikacja wczyta je automatycznie dzięki `pydantic-settings`. W repo znajdziesz przykładowy plik `.env`, który możesz skopiować i dostosować. Flaga `DEBUG=true` włącza szczegółowe logi z przebiegu ingestu (po jednym wpisie na dokument).

## Uruchomienie lokalne

1. Utwórz i aktywuj wirtualne środowisko.
2. Zainstaluj zależności:
   ```bash
   pip install fastapi uvicorn qdrant-client openai pydantic pydantic-settings tiktoken scikit-learn markdown2 beautifulsoup4 html2text PyPDF2
   ```
3. Start serwera:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8080
   ```
4. Dokumentacja API jest dostępna pod `/docs` (Swagger UI) oraz `/openapi.json`.

## Uruchomienie w Dockerze

1. Zbuduj obraz:
   ```bash
   docker build -t rags_tool:latest .
   ```
2. Uruchom kontener, przekazując wymagane zmienne środowiskowe:
   ```bash
   docker run --rm -p 8080:8080 \
     --env-file .env \
    -v $(pwd)/.rags_tool_store:/app/.rags_tool_store \
    rags_tool:latest
   ```

> **Uwaga**: Qdrant i modele LLM muszą działać poza kontenerem i być dostępne pod adresami przekazanymi w zmiennych środowiskowych.

## Najważniejsze endpointy

- `GET /about` – metadane usługi
- `GET /health` – sprawdzenie dostępności Qdrant
- `POST /collections/init` – utworzenie/aktualizacja kolekcji w Qdrant
- `POST /ingest/scan` – skanowanie katalogu z dokumentami
- `POST /ingest/build` – pełny pipeline ingestu (streszczenia + embedding + indeks)
- `POST /summaries/generate` – generowanie streszczeń pojedynczych plików
- `POST /search/query` – zapytania dwustopniowe z hybrydowym rankingiem

### Parametry wyszukiwania (`POST /search/query`)

- `top_m`: liczba docelowych dokumentów po Etapie 1 (streszczenia).
- `top_k`: końcowa liczba wyników (po Etapie 2).
- `use_hybrid`: włącza TF‑IDF po stronie zapytania (domyślnie true).
- `dense_weight`, `sparse_weight`: wagi składników hybrydowych.
- `mmr_lambda`: balans trafności vs. dywersyfikacji (MMR).
- `per_doc_limit`: maksymalna liczba wyników z jednego dokumentu (domyślnie 2).
- `score_norm`: `minmax` | `zscore` | `none` — sposób normalizacji przed fuzją.
- `rep_alpha`: udział dense w repulsji MMR (domyślnie = `dense_weight`).
- `mmr_stage1`: MMR po stronie streszczeń (domyślnie true).
- `summary_mode`: `none` | `first` | `all` — kontrola tego, czy i kiedy dołączać streszczenie dokumentu do trafień (domyślnie `first`).
- `result_format`: `flat` | `grouped` — kształt odpowiedzi; w `grouped` otrzymasz listę grup dokumentów z fragmentami.

Przykładowe zapytanie (flat, bez duplikacji streszczeń):

```json
{
  "query": "Jak działa SummRAG?",
  "top_m": 10,
  "top_k": 5,
  "mode": "auto",
  "use_hybrid": true,
  "dense_weight": 0.6,
  "sparse_weight": 0.4,
  "mmr_lambda": 0.3,
  "per_doc_limit": 2,
  "score_norm": "minmax",
  "rep_alpha": 0.6,
  "mmr_stage1": true,
  "summary_mode": "first",
  "result_format": "flat"
}
```

Przykładowe zapytanie (grouped):

```json
{
  "query": "Jak działa SummRAG?",
  "top_m": 10,
  "top_k": 5,
  "result_format": "grouped",
  "summary_mode": "first"
}
```

## Licencja

MIT
