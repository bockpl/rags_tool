# rags_tool (1.9.1)

Dwustopniowy serwis RAG zbudowany na FastAPI. System wspiera streszczanie dokumentów, indeksowanie w Qdrant oraz wyszukiwanie hybrydowe (dense + TF-IDF). Administrator może globalnie pominąć Etap 1 (streszczenia) i wyszukiwać bezpośrednio w całym korpusie chunków — patrz `SEARCH_SKIP_STAGE1_DEFAULT`.

## Nowości w 1.9.0

- Globalny przełącznik w `.env`: `SEARCH_SKIP_STAGE1_DEFAULT` (domyślnie `false`).
  - Gdy `true`, endpoint `/search/query` pomija selekcję dokumentów po streszczeniach (Etap 1) i od razu wyszukuje w całej kolekcji chunków.
  - Zachowane są wszystkie pozostałe mechanizmy: hybryda dense/TF‑IDF, MMR, `per_doc_limit`, `summary_mode`, `result_format`, scalanie bloków oraz (jeśli skonfigurowany) reranker.
  - Uwaga dla UI/testów: w tym trybie `top_m` ogranicza początkową pulę chunków (zamiast liczby dokumentów po Etapie 1). Testy, które oczekują wywołania Etapu 1, powinny uwzględnić nowy tryb.

## Nowości w 1.8.2

- Admin UI: widoczny teraz jest „handler” (np. `app.api:search_debug_stage2`) dla każdej operacji.
- Bogatszy opis etapów debugowania w panelu (wskazuje, które funkcje wewnętrzne są wywoływane).

## Nowości w 1.8.1

- Poprawka: `/search/debug/stage2` — brakujący parametr `top_m` w obiekcie konfiguracji powodował błąd 500. Dodano pole z domyślną wartością (100).

## Nowości w 1.8.0

- Tryb debugowania wyszukiwania krok po kroku (UI + API):
  - Nowe endpointy (ukryte w OpenAPI, dostępne z panelu `/admin`):
    - `POST /search/debug/embed` – embeduje pojedyncze zapytanie i buduje zapytania TF‑IDF (hybryda).
    - `POST /search/debug/stage1` – selekcja dokumentów na streszczeniach (MMR, hybryda).
    - `POST /search/debug/stage2` – selekcja chunków w obrębie wybranych dokumentów (MMR, hybryda).
    - `POST /search/debug/shape` – kształtowanie odpowiedzi (`flat|grouped|blocks`).
  - Panel Admin UI: po wykonaniu etapu pojawia się przycisk „→ Następny etap”, który
    automatycznie przełącza operację i wstawia przygotowane body na podstawie poprzedniej odpowiedzi.
  - Proste, JSON‑owe formaty — wyniki etapów zawierają minimalne pola do inspekcji i kontynuacji.

## Nowości w 1.7.1

- Poprawka zgodności wejścia: pole `query` w `/search/query` akceptuje teraz również zagnieżdżone listy
  (np. `[["a","b","c"]]`). Wejście jest spłaszczane do `["a","b","c"]`, co eliminuje 422 przy takim formacie.

## Nowości w 1.7.0

- Reranker (OpenAI‑compatible): dodano opcjonalny krok rerankingu po wyszukiwaniu wektorowym.
  - Minimalne zmienne w `.env`: `RANKER_BASE_URL`, `RANKER_API_KEY`, `RANKER_MODEL`,
    `RERANK_TOP_N`, `RETURN_TOP_K`, `RANKER_SCORE_THRESHOLD`, `RANKER_MAX_LENGTH`.
  - Integracja w endpointzie `/search/query`: jeżeli ranker jest włączony, wyniki w formacie `blocks`
    są sortowane i filtrowane wg `ranker_score` po wywołaniu `POST {RANKER_BASE_URL}/v1/rerank`.
  - Wielozapytaniowość: lista zapytań jest łączona w jeden ciąg (separator ` || `) na potrzeby rankera.
  - Fallback: na błąd/timeout lub brak konfiguracji rankera zwracane są wyniki wektorowe (bez `ranker_score`).

## Nowości w 1.6.0

- Embedding: dodano konfigurowalne prefiksy dla modeli retrievalujących (instruction-style). Nowe zmienne: `EMBEDDING_QUERY_PREFIX` i `EMBEDDING_PASSAGE_PREFIX`. Domyślne wartości odpowiadają modelowi sdadas/mmlw-retrieval-roberta-large-v2 (`"query: "` i `"passage: "`).
- API: zapytania (dense) embedowane są z prefiksem `query`, a streszczenia i treść dokumentów z prefiksem `passage`.
- Chunking: rozmiar chunku i overlap są konfigurowalne w `.env` (zmienne `CHUNK_TOKENS`, `CHUNK_OVERLAP`) i domyślnie dostosowane do modeli z limitem ~512 tokenów. Endpoint `/ingest/build` domyślnie korzysta z tych wartości, ale można je nadpisać w żądaniu.

## Nowości w 1.5.0

- Wyszukiwanie: pole `query` przyjmuje teraz listę zapytań (`List[str]`). Każde zapytanie jest wykonywane kolejno, a wyniki są łączone metodą RRF (Reciprocal Rank Fusion) i ograniczane globalnym `top_k`.
- Parametry łączenia są wewnętrzne (brak nowych pól w modelu) i mają sensowne domyślne wartości; domyślna strategia to `rrf`.
- Scalanie chunków do pełnych bloków/sekcji odbywa się po zakończeniu wszystkich zapytań i po uporządkowaniu połączonych wyników (lepsza spójność bloków w odpowiedzi `blocks`).

## Nowości w 1.4.2

- Import: czyszczenie `VECTOR_STORE_DIR` nie usuwa już samego katalogu (co kończyło się błędem „Device or resource busy” przy montażu jako wolumen Dockera). Czyścimy zawartość katalogu, a brakujące pliki są nadpisywane podczas importu.

## Nowości w 1.4.1

- Dodano zależność `python-multipart` wymaganą do obsługi importu pliku w endpointzie `/collections/import` (multipart/form-data). Obraz Dockera i instrukcja lokalnej instalacji zostały zaktualizowane.

## Nowości w 1.4.0

- Endpoint `/collections/import` przyjmuje teraz archiwum `.tar.gz` bezpośrednio jako przesyłany plik (multipart/form-data) albo surowe body HTTP, co upraszcza wykorzystanie plików wygenerowanych przez eksport.
- Panel Admin UI pozwala wskazać plik eksportu z dysku i wysłać go do API bez ręcznego kodowania base64; wciąż można użyć JSON-a z `archive_base64` dla automatyzacji.

## Nowości w 1.3.1

- Systemowy prompt używany przy generowaniu streszczeń (`SUMMARY_SYSTEM_PROMPT`) przeniesiono do konfiguracji `.env`, co pozwala sterować tonem i rolą modelu bez edycji kodu.

## Nowości w 1.3.0

- Prompty do streszczeń (`SUMMARY_PROMPT`, `SUMMARY_PROMPT_JSON`) są teraz konfigurowalne przez zmienne środowiskowe, dzięki czemu można łatwo dostosować instrukcje dla modeli LLM bez zmian w kodzie.

## Nowości w 1.2.1

- Poprawiono eksport/import snapshotów: jeśli klient Python nie udostępnia metod snapshot, używamy bezpośrednich endpointów REST (tworzenie, pobieranie, upload). Gdy snapshoty nie są dostępne, eksport automatycznie przełącza się na tryb JSONL.

## Nowości w 1.2.0

- Eksport kolekcji wykorzystuje natywne snapshoty Qdrant; archiwum zawiera pliki snapshotów (`snapshots/<kolekcja>/<plik>.snapshot`) gotowe do ponownego wgrania.
- Import odtwarza kolekcje przez upload i recovery snapshotów (z zachowaniem opcji `replace_existing`).
- Zachowana kompatybilność wstecz: archiwa 1.1.x oparte na JSON-ach są nadal obsługiwane.
- W odpowiedzi eksportu pojawia się nagłówek `X-Rags-Snapshots` z listą wygenerowanych plików snapshotów.

## Nowości w 1.1.2

- Naprawiono eksport punktów Qdrant: stronicowanie wykorzystuje teraz iterator odporny na różnice w typie zwracanym przez `qdrant.scroll`, dzięki czemu pliki `points.jsonl` zawierają pełną zawartość nawet dla dużych kolekcji.

## Nowości w 1.1.1

- Eksport kolekcji działa strumieniowo (plik `points.jsonl` per kolekcja) i nie buforuje już całej zawartości w pamięci, dzięki czemu obsługuje duże zbiory.
- Import wspiera zarówno nowe archiwum `.tar.gz`, jak i format z wersji 1.1.0; pliki TF-IDF są odtwarzane po stronie serwera, a istniejące zasoby mogą zostać zachowane lub nadpisane.
- W odpowiedzi eksportu dodano nagłówek `X-Rags-Vector-Store` z listą plików TF-IDF do szybkiej inspekcji archiwum.

## Nowości w 1.1.0

- Eksport zawsze obejmuje wszystkie kolekcje Qdrant oraz artefakty TF-IDF z katalogu `VECTOR_STORE_DIR`; dane trafiają do archiwum `.tar.gz` kompatybilnego z nowym importem.
- Import odtwarza kolekcje i pliki indeksów, opcjonalnie zastępując istniejące zasoby (w tym katalog TF-IDF) po ustawieniu `replace_existing=true`.
- Panel Admin UI aktualizuje helper eksportu/importu do nowego formatu (archiwum `.tar.gz`).

## Nowości w 1.0.0

- Panel Admin UI otrzymał dwa helpery: eksport wszystkich kolekcji Qdrant do pojedynczego archiwum JSON.gz (`/collections/export`) oraz import plików wygenerowanych w ten sposób (`/collections/import`) z opcją zastąpienia istniejących kolekcji.
- UI automatycznie pobiera plik eksportu (bezpośredni download z przeglądarki) i przyjmuje dane importu w formacie base64.

## Nowości w 0.9.5

- Naprawiono błąd `Unknown arguments: ['timeout']` podczas `upsert` w klientach Qdrant bez wsparcia parametru per-zapytanie; limit czasu ustawiany jest teraz wyłącznie globalnie.

## Nowości w 0.9.4

- Rozbito upserty do Qdrant na mniejsze batch-e (256 punktów) i dodano ustawialny timeout, co zapobiega błędom `ResponseHandlingException: timed out` przy dużych dokumentach.
- Nowa zmienna środowiskowa `QDRANT_TIMEOUT` (domyślnie 60 s) pozwala dostosować limit czasu na operacje HTTP do konfiguracji klastra.

## Nowości w 0.9.3

- Ujednolicono etykiety sekcji generowane przez spaCy (np. § 5 ust. 3 pkt 2), dzięki czemu payload Qdrant zachowuje pełną hierarchię.
- Uproszczony fallback chunk_text_by_sections gwarantuje pary {text, section} także bez spaCy, co zabezpiecza ingest.

## Nowości w 0.9.2

- Nowe narzędzie CLI: wydobywanie wzorców sekcjonowania z korpusu (`tools/mine_section_patterns.py`).
  - Skanowanie plików zgodne z ingest/scan (`SUPPORTED_EXT`, ten sam parser `extract_text`).
  - Heurystyka + fallback do LLM z użyciem tego samego modelu co streszczenia (`SUMMARY_*`).
  - Wynikiem jest snippet Pythona z `SECTION_LEVELS` i `LEVEL_PATTERNS` (unia poziomów z całego korpusu).

## Nowości w 0.9.1

- Refaktor: podział kodu na logiczne moduły (`app/api.py`, `app/core/*`, `app/models.py`, `app/qdrant_utils.py`).
- Zmieniony sposób uruchomienia: `uvicorn main:app` (wcześniej `uvicorn app:app`).
- README i Dockerfile dostosowane do nowej struktury.

## Nowości w 0.9.0

- Sekcjony podział punktów i podpunktów w obrębie paragrafów (§): wykrywamy listy numerowane (`1)`, `2.`), literowe (`a)`, `lit. b)`), rzymskie (`i)`, `IV)`), a także tirety (`-`, `–`, `•`). Każdy wykryty element staje się osobną podsekcją z etykietą np. „§ 7 pkt 3 lit. b)”, która trafia do payloadu jako `section`.
- Heurystyki unikające szumu: segmentujemy tylko gdy poziom ma co najmniej 2 elementy i elementy mają sensowną długość; krótkie i pojedyncze pozycje pozostają w rodzicu.
- Integracja z dotychczasowym chunkingiem: podsekcje są dalej dzielone tokenowo, a grupowanie `blocks` i `grouped` zyskuje bardziej precyzyjne granice.

## Nowości w 0.8.0

- Sekcyjny chunking dla dokumentów regulaminowych i prawnych: parser rozpoznaje nagłówki „Rozdział …”, paragrafy „§ …”, a także bloki „Załącznik …”. Tekst jest dzielony w granicach sekcji i paragrafów, bez ich przecinania.
- Payloady Qdrant zawierają teraz pole `section` dla każdego chunku (np. „Rozdział 1 — Informacje ogólne § 1”), co poprawia scalanie bloków (`merge_chunks`) i prezentację wyników (`blocks`, `grouped`).
- Lepsze cytowanie: odpowiedzi zawierają `section`, co ułatwia odwołania do konkretnych fragmentów dokumentu.

## Nowości w 0.7.2

- Streszczenia w JSON: funkcja streszczeń preferuje teraz tryb JSON (`response_format={"type":"json_object"}`) i oczekuje kluczy `summary`, `signature`, `entities`. Jeśli serwer nie wspiera JSON‑mode, automatycznie używany jest dotychczasowy parser tekstowy. Przełącznik: `SUMMARY_JSON_MODE` (domyślnie `true`).

## Nowości w 0.7.1

- Refaktor: dalszy podział długich funkcji — osobne helpery dla klasyfikacji trybu, kształtowania odpowiedzi i skanowania plików; `ingest_scan` oraz `ingest_build` respektują teraz flagę `recursive` przy wyszukiwaniu plików.

## Nowości w 0.7.0

- Wymiar embeddingu z konfiguracji: kolekcja Qdrant jest tworzona na podstawie `EMBEDDING_DIM` (zamiast wykonywać zapytanie do API, by odczytać wymiar). Zmniejsza to koszty i przyspiesza start. Dla zgodności nadal można wymusić sondowanie ustawiając `force_dim_probe=true` w `POST /collections/init`.
- Poprawka startu: uniknięto błędu `NameError: SearchQuery` poprzez użycie forward refs w typach helperów.

## Nowości w 0.6.2

- Refaktor: podział długich funkcji (`ingest_build`, `search_query`) na mniejsze helpery ułatwiające utrzymanie i testowanie.
- Panel Admin UI: szablon przeniesiony do pliku `templates/admin.html` (łatwiejsza edycja). Serwer wczytuje szablon z pliku i wstrzykuje operacje przez prostą podmianę tokenu `__OPERATIONS__`.

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
export QDRANT_TIMEOUT="60"
export EMBEDDING_API_URL="http://127.0.0.1:8000/v1"
export EMBEDDING_API_KEY="sk-embed-xxx"
export EMBEDDING_MODEL="BAAI/bge-m3"
export EMBEDDING_DIM="1024"          # stały wymiar wektora dla używanego modelu
export SUMMARY_API_URL="http://127.0.0.1:8001/v1"
export SUMMARY_API_KEY="sk-summary-xxx"
export SUMMARY_MODEL="gpt-4o-mini"
export SUMMARY_SYSTEM_PROMPT="Jesteś zwięzłym ekstrakcyjnym streszczaczem."  # opcjonalna rola systemowa
export SUMMARY_PROMPT="..."         # opcjonalnie nadpisz instrukcję tekstowego promptu
export SUMMARY_PROMPT_JSON="..."     # opcjonalnie nadpisz instrukcję promptu JSON
export SEARCH_TOOL_DESCRIPTION="..." # opcjonalny opis endpointu /search/query widoczny w OpenAPI i narzędziu LLM
export COLLECTION_NAME="rags_tool"
export VECTOR_STORE_DIR=".rags_tool_store"
export DEBUG="false"  # ustaw na "true", aby włączyć logi debugujące

# Reranker (OpenAI‑compatible). Pozostaw puste, aby wyłączyć.
# Ustaw bazę BEZ '/v1' — klient sam dołączy '/v1/rerank' (lub '/rerank', jeśli baza kończy się na '/v1').
export RANKER_BASE_URL="http://127.0.0.1:8002"
export RANKER_API_KEY="sk-ranker-xxx"
export RANKER_MODEL="sdadas/polish-reranker-roberta-v3"
export RERANK_TOP_N="50"           # ilu kandydatów przekazać do rankera
export RETURN_TOP_K="5"            # ilu wyników zwrócić po rankingu
export RANKER_SCORE_THRESHOLD="0.2"# próg minimalnego score
export RANKER_MAX_LENGTH="2048"    # przybliżony limit znaków na jeden passage
```

Możesz także umieścić te wartości w pliku `.env`; aplikacja wczyta je automatycznie dzięki `pydantic-settings`. W repo znajdziesz przykładowy plik `.env`, który możesz skopiować i dostosować. Flaga `DEBUG=true` włącza szczegółowe logi z przebiegu ingestu (po jednym wpisie na dokument).

## Uruchomienie lokalne

1. Utwórz i aktywuj wirtualne środowisko.
2. Zainstaluj zależności:
   ```bash
   pip install fastapi uvicorn qdrant-client openai pydantic pydantic-settings python-multipart tiktoken scikit-learn markdown2 beautifulsoup4 html2text PyPDF2
   ```
3. Start serwera:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8080
   ```
4. Dokumentacja API jest dostępna pod `/docs` (Swagger UI) oraz `/openapi.json`.

## Struktura projektu

```
.
├── app/
│   ├── __init__.py
│   ├── api.py              # endpointy FastAPI
│   ├── core/
│   │   ├── __init__.py
│   │   ├── chunking.py     # chunk_text, count_tokens, segmentacja sekcji
│   │   ├── embedding.py    # embed_text, TF-IDF (load/save/fit/vector)
│   │   ├── parsing.py      # extract_text, html_to_text, split_into_paragraphs
│   │   ├── search.py       # _stage1, _stage2, MMR, shaping
│   │   └── summary.py      # llm_summary i prompty
│   ├── models.py           # modele Pydantic (Request/Response)
│   ├── qdrant_utils.py     # ensure_collections, upsert punktów, klient Qdrant
│   └── settings.py         # konfiguracja aplikacji
├── templates/
│   └── admin.html
├── .env
├── main.py                 # wejście startowe (uvicorn main:app)
└── Dockerfile
```

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
  - `EMBEDDING_QUERY_PREFIX` (prefiks dla zapytań; domyślnie `"query: "`)
  - `EMBEDDING_PASSAGE_PREFIX` (prefiks dla dokumentów/fragmentów; domyślnie `"passage: "`)
  - `CHUNK_TOKENS` (domyślny docelowy rozmiar chunku w tokenach)
  - `CHUNK_OVERLAP` (domyślny overlap chunków w tokenach)
  - Wymagania: endpoint `/v1/embeddings` przyjmuje `{"model": str, "input": List[str]}` i zwraca `{"data": [{"embedding": List[float]}, ...]}`.

- Summary (Chat) API — do generowania streszczeń dokumentów
  - `SUMMARY_API_URL` (np. `http://127.0.0.1:8001/v1`)
  - `SUMMARY_API_KEY`
  - `SUMMARY_MODEL` (np. `gpt-4o-mini` lub kompatybilny model czatowy)
  - Wymagania: endpoint `/v1/chat/completions` przyjmuje `{"model": str, "messages": [{role, content}, ...], "temperature": float, "max_tokens": int}` i zwraca `{"choices": [{"message": {"content": str}}]}`.

## Konfiguracja wyszukiwania

- `SEARCH_SKIP_STAGE1_DEFAULT` (bool; domyślnie `false`)
  - Jeśli `true`, Etap 1 (wyszukiwanie po streszczeniach) jest globalnie wyłączony. `/search/query` wyszukuje bezpośrednio w całej kolekcji chunków, respektując filtr trybu (`current`/`archival`), hybrydę dense/TF‑IDF, MMR, limity per‑doc, `summary_mode`, formatowanie wyników oraz ewentualny reranker.
  - W tym trybie `top_m` działa jako limit początkowej puli chunków do rozważenia w Etapie 2.

### Jak działają wywołania

- Embedding:
  - Aplikacja odpytuje `/v1/embeddings` batchowo (`input: List[str]`).
  - Dla modeli oczekujących instrukcji (np. sdadas/mmlw-retrieval-roberta-large-v2) dołącza prefiks `EMBEDDING_QUERY_PREFIX` dla zapytań oraz `EMBEDDING_PASSAGE_PREFIX` dla streszczeń i treści.
  - Rozmiar chunków kontrolują `CHUNK_TOKENS` i `CHUNK_OVERLAP`. Dla modeli z limitem ~512 zalecane wartości startowe to `400` i `64`; dla modeli 1k–2k można rozważyć większe okna (np. `900`/`150`).
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
- Qdrant 500 „Service internal error: No such file or directory (os error 2)” podczas `upsert` — aplikacja próbuje automatycznie odzyskać działanie: ponawia po `ensure_collections`, dzieli batch na mniejsze (64), a jeśli to nie pomaga, wysyła punkty bez składników TF‑IDF (tylko dense). Jeśli błąd się utrzymuje, sprawdź uprawnienia/zapisywalność storage Qdrant oraz wersję serwera (zalecana 1.7+ z obsługą named sparse vectors).

## Narzędzie: wzorce sekcji

Narzędzie CLI do wydobywania poziomów sekcji i bezpiecznych regexów na podstawie całego korpusu.

- Plik: `tools/mine_section_patterns.py`
- Skanuje te same formaty co ingest (`SUPPORTED_EXT`), używa identycznego parsera treści (`extract_text`).
- Heurystyka (szybka) + LLM fallback (ten sam model co streszczenia: `SUMMARY_API_URL`, `SUMMARY_API_KEY`, `SUMMARY_MODEL`).

Przykład użycia:

```bash
python tools/mine_section_patterns.py /app/data \
  --glob "**/*" \
  --recursive \
  --out section_patterns.py
```

Przykładowy wynik (snippet Pythona):

```python
SECTION_LEVELS = ["doc", "chapter", "par", "ust", "pkt", "lit", "dash"]
LEVEL_PATTERNS = {
  "chapter": r"^\s*(rozdział|dział)\s+([IVXLCDM]+|\d+)",
  "par":     r"^\s*§\s*(\d+[a-z]?)",
  "ust":     r"\bust\.?\s*(\d+[a-z]?)\b",
  "pkt":     r"(\bpkt\.?\s*(\d+[a-z]?)|^\s*\(?\d+[a-z]?\))",
  "lit":     r"\blit\.?\s*([a-z])\)",
  "dash":    r"^\s*[-–—]\s+",
}
```

Wartości `SECTION_LEVELS` to unia poziomów znalezionych we wszystkich dokumentach; `doc` jest zawsze dołączany na górze hierarchii. `LEVEL_PATTERNS` zawiera tylko te poziomy, dla których wykryto wzorce.

## Najważniejsze endpointy

- `GET /about` – metadane usługi
- `GET /health` – sprawdzenie dostępności Qdrant
- `POST /collections/init` – utworzenie/aktualizacja kolekcji w Qdrant
- `POST /ingest/scan` – skanowanie katalogu z dokumentami
- `POST /ingest/build` – pełny pipeline ingestu (streszczenia + embedding + indeks)
- `POST /summaries/generate` – generowanie streszczeń pojedynczych plików
- `POST /search/query` – zapytania dwustopniowe z hybrydowym rankingiem

### Parametry wyszukiwania (`POST /search/query`)

- `query`: lista zapytań (`List[str]`), każde krótkie i konkretne (3–12 słów). Dodanie synonimów/wariantów zwiększa recall; wyniki są łączone i sortowane globalnie.
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
- `result_format`: `flat` | `grouped` | `blocks` — kształt odpowiedzi; domyślnie `blocks` (zalecane dla narzędzi); w `grouped` otrzymasz listę grup dokumentów z fragmentami; w `blocks` zwracane są wyłącznie scalone bloki.
- `merge_chunks`: włącza scalanie sąsiadujących chunków do bloków (działa dla `flat`, `grouped` – jako dodatkowe pole `blocks` oraz dla `blocks` – jako główna odpowiedź).
- `merge_group_budget_tokens`: budżet na łączony tekst bloku w ramach grupy `(doc_id, section)` (heurystyka ~4 znaki/token).
- `max_merged_per_group`: maksymalna liczba bloków na grupę.
- `block_join_delimiter`: separator przy łączeniu chunków.
- `expand_neighbors`: liczba sąsiadów po obu stronach, których spróbujemy dołączyć z puli kandydatów (mmr_pool) podczas scalania (0 = wył.).

Przykładowe zapytanie (flat, bez duplikacji streszczeń):

```json
{
  "query": ["Jak działa rags_tool?", "architektura rags_tool"],
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
  "query": ["Jak działa rags_tool?"],
  "top_m": 10,
  "top_k": 5,
  "result_format": "grouped",
  "summary_mode": "first"
}
```

Przykładowe zapytanie (blocks):

```json
{
  "query": ["Jak działa rags_tool?", "architektura rags_tool"],
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
## Nowości w 1.9.1

- Refaktor: wydzielono kod Admin UI i endpointy debug (`/admin`, `/search/debug/*`) do oddzielnego modułu `app/admin_routes.py` i podpinane są przez `attach_admin_routes(app)`. Kod funkcjonalny pozostaje w `app/api.py`.
