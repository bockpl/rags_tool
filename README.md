# rags_tool (0.6.1)

Dwustopniowy serwis RAG zbudowany na FastAPI. System wspiera streszczanie dokumentów, indeksowanie w Qdrant oraz wyszukiwanie hybrydowe (dense + TF-IDF).

## Nowości w 0.6.1

- Dokładne chunkowanie po tokenach: `chunk_text` wykorzystuje teraz `tiktoken` do liczenia tokenów, dzięki czemu fragmenty lepiej mieszczą się w budżecie modeli i są bardziej spójne (mniej zbyt małych/zbyt dużych chunków). Jeśli `tiktoken` nie jest dostępny, stosowany jest bezpieczny fallback heurystyczny (~4 znaki/token). Zmiana wpływa tylko na sposób wyznaczania granic chunków — API pozostaje bez zmian.

## Nowości w 0.6.0

- Scalanie z rozszerzaniem sąsiadów: nowy parametr `expand_neighbors` pozwala dołączyć brakujące, sąsiadujące chunki z puli kandydatów (Stage‑2) podczas łączenia bloków. Działa w ramach tego samego `(doc_id, section)` i respektuje budżet tokenów.

## Nowości w 0.5.3

- Stabilne OpenAPI dla narzędzi: dodano `operation_id` (`rags_tool_search`) i `tags: ["tools"]` dla `/search/query`, aby uniknąć problemów z importerami narzędzi (np. OpenWebUI).

## Nowości w 0.5.2

- Rozszerzone opisy w OpenAPI dla narzędzia LLM (parametry `SearchQuery`, pola odpowiedzi i opis endpointu `/search/query`).

## Nowości w 0.5.1

- Poprawka uruchomienia Pydantic: przesunięto `model_rebuild()` po definicji `MergedBlock` (naprawia błąd `PydanticUndefinedAnnotation`).

## Nowości w 0.5.0

- Scalanie sąsiadujących chunków w większe bloki w odpowiedzi wyszukiwania.
- Nowe parametry zapytania do kontroli scalania:
  - `merge_chunks` (bool, domyślnie false) – włącza scalanie.
  - `merge_group_budget_tokens` (int, domyślnie 1200) – budżet tokenów na blok w grupie `(doc_id, section)`.
  - `max_merged_per_group` (int, domyślnie 1) – maksymalna liczba złożonych bloków zwracanych z każdej grupy.
  - `block_join_delimiter` (string, domyślnie `"\n\n"`) – separator łączący treści chunków.
- Rozszerzony `result_format` o `blocks` – pozwala zwrócić wyłącznie listę scalonych bloków.

## Nowości w 0.4.0

- Kontrola duplikacji streszczeń w wynikach: nowy parametr `summary_mode` (`none` | `first` | `all`). Domyślnie `first` — streszczenie dokumentu pojawia się tylko przy pierwszym trafieniu z danego dokumentu (eliminuje powtarzanie).
- Nowy format wyników: `result_format` (`flat` | `grouped`). Domyślnie `flat`. W trybie `grouped` wyniki są grupowane per dokument (jedno streszczenie na dokument + lista trafionych fragmentów).
- Zaktualizowany panel Admin UI ma domyślnie `summary_mode: "first"` i `result_format: "blocks"` w predefiniowanym żądaniu `search-query`.

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

## Parametry LLM (OpenAI‑compatible)

Serwis korzysta z dwóch endpointów zgodnych z protokołem OpenAI:

- Embedding API — do wektoryzacji treści i streszczeń
  - `EMBEDDING_API_URL` (np. `http://127.0.0.1:8000/v1`)
  - `EMBEDDING_API_KEY` (token; może być pusty jeśli serwer nie wymaga)
  - `EMBEDDING_MODEL` (np. `BAAI/bge-m3` lub inny zgodny z `/v1/embeddings`)
  - Wymagania: endpoint `/v1/embeddings` przyjmuje `{"model": str, "input": List[str]}` i zwraca `{"data": [{"embedding": List[float]}, ...]}`.

- Summary (Chat) API — do generowania streszczeń dokumentów
  - `SUMMARY_API_URL` (np. `http://127.0.0.1:8001/v1`)
  - `SUMMARY_API_KEY`
  - `SUMMARY_MODEL` (np. `gpt-4o-mini` lub kompatybilny model czatowy)
  - Wymagania: endpoint `/v1/chat/completions` przyjmuje `{"model": str, "messages": [{role, content}, ...], "temperature": float, "max_tokens": int}` i zwraca `{"choices": [{"message": {"content": str}}]}`.

### Jak działają wywołania

- Embedding:
  - Aplikacja odpytuje `/v1/embeddings` batchowo (`input: List[str]`).
  - Wymiar wektora (dim) wykrywany jest sondą z tekstem `"test"` – musi być stały między wywołaniami; zmiana modelu = zmiana wymiaru.
  - W przypadku zmiany modelu embedującego warto użyć `POST /collections/init` z `force_dim_probe=true` i/lub przebudować indeks.

- Streszczenia:
  - Aplikacja woła `/v1/chat/completions` z `temperature=0.0` i `max_tokens=300`.
  - Wysyłany jest polski prompt proszący o format:
    - `SUMMARY: ...`
    - `SIGNATURE: lemma1, lemma2, ...`
    - `ENTITIES: ...`
  - Parser oczekuje powyższych prefiksów linii; jeśli model ich nie zwróci, używa fallbacku (pierwsze ~600 znaków odpowiedzi jako `summary`).

### Przykłady cURL

- Embeddings:

```bash
curl -sS "$EMBEDDING_API_URL/embeddings" \
  -H "Authorization: Bearer $EMBEDDING_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "'"${EMBEDDING_MODEL}"'", "input": ["hello", "world"]}'
```

- Chat (streszczenia):

```bash
curl -sS "$SUMMARY_API_URL/chat/completions" \
  -H "Authorization: Bearer $SUMMARY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"${SUMMARY_MODEL}"'",
    "temperature": 0.0,
    "max_tokens": 300,
    "messages": [
      {"role": "system", "content": "Jesteś zwięzłym ekstrakcyjnym streszczaczem."},
      {"role": "user", "content": "SUMMARY: ...\nSIGNATURE: lemma1, lemma2, ...\nENTITIES: ...\n\nTEKST:\nTo jest przykładowy tekst do streszczenia."}
    ]
  }'
```

### Dobór modeli i wskazówki

- Embedding:
  - `BAAI/bge-m3` — uniwersalny, wielojęzyczny, dobry baseline (COSINE).
  - Inne modele są ok, jeśli zwracają spójną długość wektora i wspierają `/v1/embeddings`.
- Streszczenia:
  - Model czatowy musi wspierać `/v1/chat/completions` i generację w języku polskim.
  - Ustaw `temperature=0.0` dla deterministycznych, ekstrakcyjnych streszczeń.

### Rozwiązywanie problemów

- „Api key is used with an insecure connection.” — komunikat ostrzegawczy z klienta Qdrant przy użyciu HTTP z kluczem API. Najlepiej przejść na HTTPS (`QDRANT_URL`).
- „Dimension mismatch” po zmianie modelu embedującego — uruchom `POST /collections/init` z `force_dim_probe=true` i/lub przebuduj indeks (`/ingest/build`).
- Model czatowy nie zwraca prefiksów `SUMMARY:`/`SIGNATURE:`/`ENTITIES:` — sprawdź, czy endpoint respektuje wiadomość systemową i format prośby; w razie czego parser użyje fallbacku.

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
- `result_format`: `flat` | `grouped` | `blocks` — kształt odpowiedzi; w `grouped` otrzymasz listę grup dokumentów z fragmentami; w `blocks` zwracane są wyłącznie scalone bloki.
- `merge_chunks`: włącza scalanie sąsiadujących chunków do bloków (działa dla `flat`, `grouped` – jako dodatkowe pole `blocks` oraz dla `blocks` – jako główna odpowiedź).
- `merge_group_budget_tokens`: budżet na łączony tekst bloku w ramach grupy `(doc_id, section)` (heurystyka ~4 znaki/token).
- `max_merged_per_group`: maksymalna liczba bloków na grupę.
- `block_join_delimiter`: separator przy łączeniu chunków.
- `expand_neighbors`: liczba sąsiadów po obu stronach, których spróbujemy dołączyć z puli kandydatów (mmr_pool) podczas scalania (0 = wył.).

Przykładowe zapytanie (flat, bez duplikacji streszczeń):

```json
{
  "query": "Jak działa rags_tool?",
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
  "query": "Jak działa rags_tool?",
  "top_m": 10,
  "top_k": 5,
  "result_format": "grouped",
  "summary_mode": "first"
}
```

Przykładowe zapytanie (blocks):

```json
{
  "query": "Jak działa rags_tool?",
  "top_m": 10,
  "top_k": 5,
  "merge_chunks": true,
  "merge_group_budget_tokens": 1200,
  "max_merged_per_group": 1,
  "result_format": "blocks"
}
```

## Licencja

MIT
